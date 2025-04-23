# Main application file
# -*- coding: utf-8 -*-
"""
Gemini Chat Agent v9.8: Codespaces Ready, Clean Console, Increased Context

Core Features:
- Model: gemini/gemini-2.0-flash
- Reads API Keys from .env file.
- Console Output: Shows only chat flow (Thinking, Tool Use, Response) & critical errors.
- Tools: Weather, Search, Firecrawl, GitHub, CodeExec(Ack), DateTime, VectorSearch
- Increased Context Window (1M tokens).
- Vector DB (in-memory), Conversation Memory Pruning.
- Streaming Responses.
- Structured Logging (Detailed logs -> file).
- File Handling.
"""

# --- Installations ---
import subprocess
import sys

def install_packages():
    packages = [
        "litellm",
        "python-dotenv",
        "requests",
        "duckduckgo-search",
        "firecrawl-py",
        "sentence-transformers",
        "numpy",
        "matplotlib",
        "PyGithub",
        "upstash-vector",
        "upstash-redis",
        "PyPDF2", # Added for PDF tool
        "mss", # Added for Screenshot tool
        "Pillow",
        "pyjwt",           # For JWT handling
        "pyyaml",          # For OpenAPI spec parsing
        "jsonschema",      # For response validation
        "requests-oauthlib", # For OAuth2 support
        "kubernetes",
        "psycopg2-binary",  # PostgreSQL
        "pymongo",          # MongoDB
        "aiosqlite"    
    ]
    # Add python-magic if needed by process_file_input's fallback
    try:
         import magic
    except ImportError:
         packages.append("python-magic-bin") # Or python-magic depending on OS/install method

    try:
        print("Checking and installing necessary libraries...") # Console print for setup
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
        print("Libraries installed/verified.") # Console print for setup
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}", file=sys.stderr) # Console print for setup error
        sys.exit(1)
    except FileNotFoundError:
         print("Error: 'pip' command not found. Please ensure Python environment is set up correctly.", file=sys.stderr) # Console print for setup error
         sys.exit(1)

install_packages()
# Re-import magic if it was just installed
try:
    import magic
except ImportError:
    magic = None # Flag if still not available

# --- Imports ---
# Ensure ALL necessary top-level imports are present
import litellm, os, base64, json, mimetypes, requests, traceback, logging, numpy as np, time, io 
from pathlib import Path 
from datetime import datetime 
from collections import defaultdict 
from dotenv import load_dotenv 
from raiden_agents import tools 
from raiden_agents.tools.base_tool import ToolExecutionError, VectorDBError, GitHubToolError, APIKeyError, AgentException 
from raiden_agents.memory.persistent_memory import RedisPersistentMemory 
from upstash_vector import Index as UpstashVectorIndex # Need this specific import

# --- Load Environment Variables ---
load_dotenv()
print("Attempted to load API keys from .env file.") 

# --- Setup Logging ---
def setup_logging():
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("gemini_agent") # Back to original logger name
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("gemini_agent_v9.8.log") # Original log file name
    file_handler.setFormatter(log_formatter); file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(); stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.WARNING) 
    logger.addHandler(file_handler); logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
logger = setup_logging()
logger.info("--- Starting Gemini Chat Agent v9.8 (Restored Console Version) ---") 

# --- Custom Exceptions ---
# AgentException is imported from base_tool now

# --- API Key Setup ---
logger.info("Setting up API Keys from environment...") 
def get_required_key(env_var):
    # 'os' is imported at the top level now
    key = os.environ.get(env_var)
    if not key:
        logger.error(f"CRITICAL: Environment variable {env_var} not set.")
        raise APIKeyError(f"Missing required API key: {env_var}")
    logger.info(f"Found required key: {env_var}") 
    return key

def get_optional_key(env_var):
    # 'os' is imported at the top level now
    key = os.environ.get(env_var)
    if not key:
        logger.warning(f"Optional environment variable {env_var} not set.")
    else:
        logger.info(f"Found optional key: {env_var}") 
    return key

try:
    gemini_api_key = get_required_key("GEMINI_API_KEY")
    openweathermap_api_key = get_optional_key("OPENWEATHERMAP_API_KEY")
    firecrawl_api_key = get_optional_key("FIRECRAWL_API_KEY")
    github_api_key = get_optional_key("GITHUB_API_KEY")
    stability_api_key = get_optional_key("STABILITY_API_KEY")
    aws_access_key = get_optional_key("AWS_ACCESS_KEY_ID")
    aws_secret_key = get_optional_key("AWS_SECRET_ACCESS_KEY")
    get_required_key("UPSTASH_VECTOR_REST_URL")
    get_required_key("UPSTASH_VECTOR_REST_TOKEN")
    get_required_key("UPSTASH_REDIS_REST_URL")
    get_required_key("UPSTASH_REDIS_REST_TOKEN")
except APIKeyError as e:
    print(f"Error: {e}. Please ensure it's set in your .env file or environment.", file=sys.stderr) 
    sys.exit(1)
except Exception as e:
     logger.critical(f"Unexpected error during API key setup: {e}", exc_info=True) 
     print(f"Unexpected critical error during API key setup: {e}", file=sys.stderr) 
     sys.exit(1)
logger.info("API Keys configured.") 

# --- Vector Database Class ---
class VectorDB:
    def __init__(self):
        self.initialized = False
        self.index = None
        self.logger = logging.getLogger("gemini_agent") 
        try:
            self.index = UpstashVectorIndex.from_env()
            self.initialized = True
            self.logger.info("Upstash Vector DB initialized successfully")
        except ImportError as e:
            self.logger.error(f"Upstash packages not installed: {e}")
        except Exception as e:
            self.logger.error(f"Upstash Vector DB initialization failed: {e}", exc_info=True)

    def add(self, text, metadata=None):
        if not self.is_ready():
            self.logger.warning("VDB add skipped: Not initialized.")
            return False
        if not text or not isinstance(text, str):
            self.logger.warning(f"VDB add skipped: Invalid text.")
            return False
        try:
            import uuid
            vector_id = str(uuid.uuid4())
            self.index.upsert([{
                "id": vector_id,
                "data": text,
                "metadata": metadata or {}
            }])
            self.logger.debug(f"Added VDB entry: {text[:50]}...")
            return True
        except Exception as e:
            self.logger.error(f"VDB add error: {e}", exc_info=True)
            return False

    def search(self, query, top_k=3):
        if not self.is_ready():
            self.logger.error("VDB search fail: Not initialized.")
            raise VectorDBError("VDB not initialized")
        try:
            results = self.index.query(data=query, top_k=top_k, include_metadata=True)
            formatted_results = [{"text": getattr(match, "data", ""), "similarity": getattr(match, "score", 0.0), "metadata": getattr(match, "metadata", {})} for match in results]
            self.logger.info(f"VDB search '{query[:30]}...' returned {len(formatted_results)} results.")
            return formatted_results
        except Exception as e:
            self.logger.error(f"VDB search error: {e}", exc_info=True)
            raise VectorDBError(f"VDB search failed: {e}")

    def is_ready(self):
        return self.initialized and self.index is not None

vector_db = VectorDB() 

# --- Initialize Tools ---
def initialize_tools():
    logger.info("Initializing tools...")
    tools_list = [
        tools.WeatherTool(),
        tools.SearchTool(),
        tools.WebScraperTool(),
        tools.CodeExecutionTool(),
        tools.DateTimeTool(),
        tools.GitHubTool(),
        tools.DataVisualizationTool(), 
        tools.AWSRekognitionTool(),
        tools.ImageGenerationTool(),
        tools.FileSystemTool(), 
        tools.PdfTool(), 
        tools.ScreenshotTool(), 
        tools.ImageUnderstandingTool(), 
        tools.VideoUnderstandingTool(), 
        tools.AudioUnderstandingTool(),
        tools.APIIntegrationTool(),
        tools.KubernetesTool(),
        tools.DatabaseTool(),
        tools.EmailIntegrationTool(),
        tools.NewsAPITool(),
        tools.CalendarSchedulingTool(),
        tools.NaturalLanguageProcessingTool(),
        tools.TaskAutomationTool(),
    ]
    if vector_db.is_ready():
        if hasattr(tools, 'VectorSearchTool'):
            tools_list.append(tools.VectorSearchTool())
            logger.info("Vector search tool initialized.")
        else:
             logger.error("VectorSearchTool class not found in tools package!")
    else:
        logger.warning("Vector search tool NOT initialized (Vector DB not ready).")

    schemas = [t.get_schema() for t in tools_list]
    tool_map = {t.name: t for t in tools_list}
    logger.info(f"Tools initialized: {list(tool_map.keys())}")
    return schemas, tool_map
active_tool_schemas, tool_map = initialize_tools()

# --- System Message Definition ---
def build_system_message():
     content = (
         "You are OmniAgent. Use tools proactively. **Never** say you lack access; state you will use a tool."
         "You **MUST NEVER** identify as an AI model or assistant.\n\n"
         "You are a multi-functional agent capable of performing various tasks. "
         "Available Tools:\n" + 
         "".join([f"- `{name}`: {tool.description}\n" for name, tool in tool_map.items()]) + 
         "\nUse `code_execution` to acknowledge Python code requests (execution handled externally). "
         "Use `github_operations` for GitHub tasks. "
         "Use `semantic_memory_search` for past info if available. "
         "Use `file_system_operations` for comprehensive local file/directory management (read, write, append, list, create, delete, move, copy, info, check existence). "
         "Use `pdf_operations` for comprehensive PDF manipulation (extract text/metadata, page count, merge, split, rotate, watermark, encrypt/decrypt). "
         "Use `take_screenshot` to capture the screen. "
         "Use `image_understanding` to analyze images (caption, ask questions, detect/segment objects). "
         "Use `video_understanding` to analyze videos from paths or URLs (summarize, ask questions, transcribe). "
         "Use `audio_understanding` to analyze audio from paths or URLs (summarize, ask questions, transcribe). "
         "Use `visualize_data` to create various plots (line, bar, scatter, hist, pie, etc.) from JSON data or CSV/JSON files, with customization options.\n\n"
         "**File Handling Capabilities:**\n"
         "- You can analyze images (using `image_understanding`), videos (using `video_understanding`), audio (using `audio_understanding`), documents (including comprehensive PDF operations), and tabular data files (CSV/JSON for `visualize_data`), provided via URL or direct upload/path.\n"
         "- When a user shares a file, carefully examine its contents and provide insights\n"
         "- For images, describe what you see in detail\n"
         "- For documents, summarize and analyze the content\n\n"
         "Answer based on tool outputs and file analysis.\n"
         "Your goal is to assist users in achieving their objectives efficiently.\n\n"
         # Persona
         "You work as a part time LinkedIn and X (Twitter) content creator for a tech company, they are called Raiden Agents.. they build AI agents for business and companies to automate their workflows.\n"
         "You are also a part time developer for a company called Raiden Agents, they build AI agents for business and companies to automate their workflows.\n"
         "You are a senior developer with over 10 years of experience in Python, JavaScript, and C++. You are also a senior data scientist with over 10 years of experience in data science and machine learning.\n"
     )
     return {"role": "system", "content": content}

SYSTEM_MESSAGE = build_system_message()
logger.info(f"System message generated. Approx tokens: {len(SYSTEM_MESSAGE['content']) // 4}")

# --- File Processing Helper ---
# Define MAX_INLINE_SIZE_BYTES needed by process_file_input
MAX_INLINE_SIZE_BYTES = 19 * 1024 * 1024 
def process_file_input(file_identifier):
    """Process file input from URL, GCS, or local path - enhanced version with better error handling."""
    logger.info(f"Processing file: {file_identifier}")
    content_part = {} 
    file_data_dict = {}
    mime_type = None
    
    # URL-based file handling
    if file_identifier.startswith("http://") or file_identifier.startswith("https://"):
        try:
            r = requests.head(file_identifier, allow_redirects=True, timeout=10)
            r.raise_for_status()
            ct = r.headers.get('Content-Type')
            mime_type = ct.split(';')[0].strip() if ct else None
            content_length = int(r.headers.get('Content-Length', -1))

            # Try downloading common media types if size is reasonable
            is_media = mime_type and (mime_type.startswith('image/') or mime_type.startswith('video/') or mime_type.startswith('audio/'))
            
            if is_media and (content_length == -1 or content_length < MAX_INLINE_SIZE_BYTES): 
                 r_get = requests.get(file_identifier, timeout=30) 
                 r_get.raise_for_status()
                 file_content = r_get.content
                 if not mime_type: mime_type = mimetypes.guess_type(file_identifier)[0] or "application/octet-stream"
                 
                 encoded_content = base64.b64encode(file_content).decode("utf-8")
                 data_uri = f"data:{mime_type};base64,{encoded_content}"
                 # Use a generic structure litellm might understand for base64 data
                 content_part = {"type": "media_url", "media_url": {"url": data_uri, "media_type": mime_type}} 
                 logger.info(f"Downloaded URL content: {len(file_content)} bytes with mime: {mime_type}")
            else:
                 if content_length > MAX_INLINE_SIZE_BYTES:
                      logger.error(f"URL content too large ({content_length} bytes) for inline processing.")
                      return None
                 else: # Not media or size unknown/large
                      logger.warning(f"Could not process URL {file_identifier} as inline data (Type: {mime_type}, Size: {content_length}). File API not implemented.")
                      return None 
        except requests.exceptions.RequestException as e:
            logger.warning(f" URL request failed: {e}")
            return None
            
    # GCS handling (Placeholder)
    elif file_identifier.startswith("gs://"):
         logger.warning("GCS file processing not implemented.")
         return None

    # Local file handling
    else:
        lp = Path(file_identifier)
        if not lp.is_file():
            logger.error(f"Local file not found: {lp}")
            return None
        try:
            file_size = lp.stat().st_size
            if file_size > MAX_INLINE_SIZE_BYTES:
                 logger.error(f"Local file '{lp}' too large ({file_size} bytes) for inline processing.")
                 return None

            fb = lp.read_bytes()
            ed = base64.b64encode(fb).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(lp)
            if not mime_type:
                if magic:
                    try: mime_type = magic.from_buffer(fb, mime=True)
                    except Exception: mime_type = "application/octet-stream"
                else: mime_type = "application/octet-stream"
                logger.warning(f" Guessed MIME: {mime_type}")
            else:
                logger.info(f" Local MIME: {mime_type}")
            
            data_uri = f"data:{mime_type};base64,{ed}"
            content_part = {"type": "media_url", "media_url": {"url": data_uri, "media_type": mime_type}} 

            # Log file provision to VectorDB
            if vector_db and vector_db.is_ready():
                fn = lp.name
                vector_db.add(f"User file provided: {fn} ({mime_type})", {
                    "type": "file_provided", "source": "local", 
                    "filename": fn, "mime_type": mime_type, 
                    "time": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Read local file error {lp}: {e}", exc_info=True)
            return None
            
    return content_part 


# --- Tool Execution Wrapper ---
def execute_tool_call(tool_call_data):
    """Wrapper for executing tool calls using dictionary input."""
    function_name = tool_call_data.get('function', {}).get('name')
    arguments_str = tool_call_data.get('function', {}).get('arguments')
    if not function_name: 
        error_msg = "Error: Tool call missing function name."
        logger.error(error_msg)
        return error_msg
    try:
        function_args = json.loads(arguments_str) if arguments_str else {}
        logger.info(f"Attempting execution: '{function_name}' args: {function_args}") 
    except json.JSONDecodeError: 
        error_msg = f"Error: Invalid JSON args for {function_name}: {arguments_str}"
        logger.error(error_msg)
        return error_msg 
    
    if function_name not in tool_map: 
        error_msg = f"Error: Unknown function '{function_name}'"
        logger.error(error_msg)
        return error_msg 
        
    try:
        tool = tool_map[function_name]
        result = tool.execute(**function_args)
        logger.info(f"Tool '{function_name}' executed successfully.")
        result_str = str(result)
        logger.debug(f"Tool '{function_name}' result snippet: {result_str[:500]}{'...' if len(result_str) > 500 else ''}")
        return result
    except ToolExecutionError as e: 
        logger.error(f"Tool execution failed '{function_name}': {e}")
        return f"Error executing tool {function_name}: {e}" 
    except Exception as e: 
        logger.critical(f"Unexpected critical error executing tool '{function_name}'", exc_info=True)
        return f"Critical Error executing tool {function_name}."


# --- Handle Streaming Response ---
def handle_streaming_response(stream):
    """Handles and prints streaming response, aggregating tool calls."""
    full_response_content = ""; tool_calls_agg = defaultdict(lambda: {"id": None, "name": None, "arguments": ""}); final_tool_calls_list = []; completed_tool_call_indices = set()
    print("\nAgent: ", end="", flush=True) 
    try:
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content: 
                print(delta_content, end="", flush=True)
                full_response_content += delta_content
            
            delta_tool_calls = chunk.choices[0].delta.tool_calls
            if delta_tool_calls:
                for tc_chunk in delta_tool_calls:
                    idx = tc_chunk.index 
                    if tc_chunk.id: 
                        tool_calls_agg[idx]["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name: 
                            tool_calls_agg[idx]["name"] = tc_chunk.function.name
                        if tc_chunk.function.arguments: 
                            tool_calls_agg[idx]["arguments"] += tc_chunk.function.arguments
                    
                    current_call = tool_calls_agg[idx]
                    if current_call["id"] and current_call["name"] and idx not in completed_tool_call_indices:
                         args_str = current_call["arguments"]
                         is_complete_json = False
                         try: 
                             json.loads(args_str)
                             is_complete_json = True
                         except json.JSONDecodeError: 
                             pass 
                         
                         if is_complete_json:
                              logger.debug(f"Stream: Finalizing tool call {idx}...") 
                              final_tool_calls_list.append({
                                  "id": current_call["id"], 
                                  "type": "function", 
                                  "function": {"name": current_call["name"], "arguments": args_str}
                              })
                              completed_tool_call_indices.add(idx)
                              
    except Exception as e: 
        logger.error(f"Stream error: {e}", exc_info=True)
        print(f"\n[Stream Error: {e}]") 
    finally: 
        print() 
        
    final_message_dict = {
        "role": "assistant", 
        "content": full_response_content if full_response_content else None, 
        "tool_calls": final_tool_calls_list if final_tool_calls_list else None
    }
    return final_message_dict


# --- Main Chat Loop ---
def chat_agent():
    model_name = "gemini/gemini-2.0-flash" 
    # Initialize Redis memory
    memory = RedisPersistentMemory(vector_db_client=vector_db, system_message=SYSTEM_MESSAGE, max_tokens=1_048_576) 

    logger.info("\n--- Raiden Agent Console Initialized ---") 
    print(f"Raiden Agent Console Initialized. Model: {model_name}. Memory: Redis. Type 'quit' to exit.") 
    print(f"Vector DB Status: {'Ready' if vector_db.is_ready() else 'Unavailable'}") 
    print(f"Redis Memory Status: {'Ready' if memory.is_ready() else 'Unavailable'}")
    print("-" * 65 + "\n") 

    if not memory.is_ready():
         print("CRITICAL: Redis Memory failed to initialize. Check Upstash Redis credentials. Exiting.", file=sys.stderr)
         sys.exit(1)

    while True:
        try:
            user_input = input("You: ") 
            if user_input.lower() == "quit": 
                logger.info("User quit.")
                break 

            user_message_content = []
            # Basic file command handling for console
            if user_input.lower().startswith("file:") and len(user_input.split(' ', 1)) > 1:
                file_id = user_input.split(' ', 1)[1].strip()
                if file_id:
                    print("Agent: Processing file reference (basic)...", flush=True)  
                    file_part = process_file_input(file_id)
                    if not file_part:
                        print("Agent: [Error processing file. Check path/URL and support.]")
                        continue 
                    
                    prompt = input("You (prompt for file): ") 
                    if prompt:
                        user_message_content.append({"type": "text", "text": prompt})
                        user_message_content.append(file_part) 
                    else:
                        user_message_content.append({"type": "text", "text": "Analyze this file."})
                        user_message_content.append(file_part)
                else:
                    print("Agent: [File command needs path/URL.]")
                    continue 
            else:
                user_message_content.append({"type": "text", "text": user_input})
            
            if not user_message_content:
                continue

            # Add user message to memory
            user_message = {"role": "user", "content": user_message_content}
            if not memory.add_message(user_message): 
                print("Agent: [Message too long or memory error.]")
                continue 

            # --- Agent Processing ---
            print("\nAgent: Thinking...", flush=True)
            logger.info("Agent: Thinking...") 

            current_messages = memory.get_messages()
            response_stream = litellm.completion(
                model=model_name, 
                messages=current_messages, 
                tools=active_tool_schemas, 
                tool_choice="auto", 
                stream=True
            )

            response_message_dict = handle_streaming_response(response_stream)
            memory.add_message(response_message_dict)

            if response_message_dict.get("tool_calls"):
                print(f"\nAgent: Using {len(response_message_dict['tool_calls'])} tool(s)...", flush=True)
                logger.info(f"LLM requested {len(response_message_dict['tool_calls'])} tool(s)...") 

                tool_results = []
                for tc_data in response_message_dict["tool_calls"]:
                    result_content = execute_tool_call(tc_data) 
                    if isinstance(result_content, str) and result_content.lower().startswith("error"):
                        logger.warning(f"Tool '{tc_data.get('function', {}).get('name')}' failed. Error: {result_content}") 
                    
                    result_msg = {"role": "tool", "tool_call_id": tc_data.get('id'), "content": str(result_content)}
                    tool_results.append(result_msg)
                    memory.add_message(result_msg) 

                print("\nAgent: Processing tool results...", flush=True)
                logger.info("Agent: Processing tool results...") 

                messages_with_results = memory.get_messages()
                final_stream = litellm.completion(
                    model=model_name, 
                    messages=messages_with_results, 
                    stream=True
                )
                
                final_response_dict = handle_streaming_response(final_stream)
                memory.add_message(final_response_dict)

                if not final_response_dict.get("content"):
                     logger.warning("Agent final response after tool use had no text content.")

            elif not response_message_dict.get("content"):
                 logger.warning("Agent received empty initial response (no content or tools).")
            
            print("-" * 65 + "\n") # Separator after agent turn

        # --- Error Handling ---
        except litellm.exceptions.APIError as e:
             logger.error(f"LiteLLM API Error: {e}", exc_info=True)
             print(f"\n!!! Agent Error: API Failure {getattr(e, 'status_code', '')} !!!", file=sys.stderr) 
             try: logger.error(f"API Error Body: {json.dumps(e.response.json(), indent=2)}") 
             except: logger.error(f"Raw API Error: {getattr(e, 'response', '')}") 
        except AgentException as e: 
             logger.error(f"Agent Error: {e}", exc_info=True)
             print(f"\n!!! Agent Error: {e} !!!", file=sys.stderr) 
        except KeyboardInterrupt: 
             logger.info("User interrupted.")
             print("\nExiting...")
             break 
        except Exception as e: 
             logger.critical(f"Critical error in main loop!", exc_info=True)
             print(f"\n!!! Critical Error: {e} !!!", file=sys.stderr)
             traceback.print_exc() # Print traceback for debugging console
             break 


# --- Start the Agent ---
if __name__ == "__main__":
    # Perform checks before starting
    if not vector_db.is_ready():
         print("CRITICAL: Vector DB failed to initialize. Check Upstash Vector credentials in .env", file=sys.stderr)
         sys.exit(1)
    # Memory readiness is checked inside chat_agent now

    try:
        chat_agent()
    except APIKeyError as e: 
        print(f"Execution stopped: {e}. Please set required API keys in .env", file=sys.stderr) 
    except Exception as main_e:
        logger.critical(f"Critical agent startup error: {main_e}", exc_info=True) 
        print(f"\n!!! Critical startup error: {main_e}. Check logs for details. !!!", file=sys.stderr)
