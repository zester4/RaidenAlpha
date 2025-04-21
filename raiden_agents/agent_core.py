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
        "Pillow" # Added for Image Understanding tool
    ]
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

# --- Imports ---
import litellm, os, base64, json, mimetypes, requests, traceback, logging, numpy as np, time, io 
from pathlib import Path 
from datetime import datetime 
from collections import defaultdict 
from dotenv import load_dotenv 
from raiden_agents import tools 
from raiden_agents.tools.base_tool import ToolExecutionError, VectorDBError, GitHubToolError, APIKeyError 
from raiden_agents.memory.persistent_memory import RedisPersistentMemory 

# --- Load Environment Variables ---
load_dotenv()
print("Attempted to load API keys from .env file.") # Console print for setup

# --- Setup Logging (Clean Console) ---
def setup_logging():
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("gemini_agent")
    logger.setLevel(logging.INFO)
    # File Handler - Detailed logs
    file_handler = logging.FileHandler("gemini_agent_v9.8.log")
    file_handler.setFormatter(log_formatter); file_handler.setLevel(logging.INFO)
    # Stream Handler - Only critical info to console
    stream_handler = logging.StreamHandler(); stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.WARNING) # Console only gets WARNING+
    logger.addHandler(file_handler); logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
logger = setup_logging()
logger.info("--- Starting Gemini Chat Agent v9.8 (Codespaces Ready) ---") # To file only

# --- Custom Exceptions ---
class AgentException(Exception): pass
# ToolExecutionError, APIKeyError, VectorDBError, GitHubToolError are now imported from base_tool.py

# --- API Key Setup (from Environment) ---
logger.info("Setting up API Keys from environment...") # To file only
def get_required_key(env_var):
    key = os.environ.get(env_var)
    if not key:
        logger.error(f"CRITICAL: Environment variable {env_var} not set. Please set it in .env or system environment.")
        raise APIKeyError(f"Missing required API key: {env_var}")
    logger.info(f"Found required key: {env_var}") #To file only
    return key

def get_optional_key(env_var):
    key = os.environ.get(env_var)
    if not key:
        logger.warning(f"Optional environment variable {env_var} not set. Related tools may not function.") # Console WARN
    else:
        logger.info(f"Found optional key: {env_var}") # To file only
    return key

try:
    gemini_api_key = get_required_key("GEMINI_API_KEY")
    openweathermap_api_key = get_optional_key("OPENWEATHERMAP_API_KEY")
    firecrawl_api_key = get_optional_key("FIRECRAWL_API_KEY")
    github_api_key = get_optional_key("GITHUB_API_KEY")
    stability_api_key = get_optional_key("STABILITY_API_KEY")
    aws_access_key = get_optional_key("AWS_ACCESS_KEY_ID")
    aws_secret_key = get_optional_key("AWS_SECRET_ACCESS_KEY")
except APIKeyError as e:
    print(f"Error: {e}. Please ensure it's set in your .env file or environment.", file=sys.stderr) # Console Error
    sys.exit(1)
except Exception as e:
     logger.critical(f"Unexpected error during API key setup: {e}", exc_info=True) # File CRITICAL
     print(f"Unexpected critical error during API key setup: {e}", file=sys.stderr) # Console CRITICAL
     sys.exit(1)
logger.info("API Keys configured.") # To file only


# --- Memory Management ---
# The ConversationMemory class is now removed.
# RedisPersistentMemory is imported and used below.

# --- Vector Database for Semantic Memory ---
class VectorDB:
    def __init__(self):
        self.initialized = False
        self.index = None
        try:
            from upstash_vector import Index
            
            # Initialize using environment variables
            self.index = Index.from_env()
            
            self.initialized = True
            logger.info("Upstash Vector DB initialized successfully")
        except ImportError as e:
            logger.error(f"Upstash packages not installed: {e}")
        except Exception as e:
            logger.error(f"Upstash Vector DB initialization failed: {e}", exc_info=True)

    def add(self, text, metadata=None):
        if not self.is_ready():
            logger.warning("VDB add skipped: Not initialized.")
            return False
        if not text or not isinstance(text, str):
            logger.warning(f"VDB add skipped: Invalid text.")
            return False

        try:
            # Create a unique ID for the vector
            import uuid
            vector_id = str(uuid.uuid4())

            # Add to vector store with proper format for Upstash Vector
            self.index.upsert([{
                "id": vector_id,
                "data": text,  # Using data field instead of values
                "metadata": metadata or {}
            }])
            
            logger.debug(f"Added VDB entry: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"VDB add error: {e}", exc_info=True)
            return False

    def search(self, query, top_k=3):
        if not self.is_ready():
            logger.error("VDB search fail: Not initialized.")
            raise VectorDBError("VDB not initialized")

        try:
            # Perform vector search
            results = self.index.query(
                data=query,  # FIX: use 'data' instead of 'query'
                top_k=top_k,
                include_metadata=True
            )

            formatted_results = []
            for match in results:
                formatted_results.append({
                    "text": getattr(match, "data", ""),
                    "similarity": getattr(match, "score", 0.0),
                    "metadata": getattr(match, "metadata", {})
                })

            logger.info(f"VDB search '{query[:30]}...' returned {len(formatted_results)} results.")
            return formatted_results

        except Exception as e:
            logger.error(f"VDB search error: {e}", exc_info=True)
            raise VectorDBError(f"VDB search failed: {e}")

    def is_ready(self):
        return self.initialized and self.index is not None
vector_db = VectorDB()

# --- Tool Implementation (Object-Oriented) ---
# Tool definitions are now imported from raiden_agents.tools

# --- Initialize Tools ---
def initialize_tools():
    logger.info("Initializing tools...")
    # Instantiate tools using the imported package
    tools_list = [
        tools.WeatherTool(),
        tools.SearchTool(),
        tools.WebScraperTool(),
        tools.CodeExecutionTool(),
        tools.DateTimeTool(),
        tools.GitHubTool(),
        tools.DataVisualizationTool(), # Enhanced
        tools.AWSRekognitionTool(),
        tools.ImageGenerationTool(),
        tools.FileSystemTool(), # Added
        tools.PdfTool(), # Added
        tools.ScreenshotTool(), # Added
        tools.ImageUnderstandingTool(), # Added
        tools.VideoUnderstandingTool(), # Added
        tools.AudioUnderstandingTool() # Added
    ]
    if vector_db.is_ready():
        # Ensure VectorSearchTool is imported in raiden_agents/tools/__init__.py
        if hasattr(tools, 'VectorSearchTool'):
            tools_list.append(tools.VectorSearchTool())
            logger.info("Vector search tool initialized.")
        else:
             logger.error("VectorSearchTool not found in tools package! Check raiden_agents/tools/__init__.py")
    else:
        logger.warning("Vector search tool NOT initialized (Vector DB not ready).")

    schemas = [t.get_schema() for t in tools_list]
    tool_map = {t.name: t for t in tools_list}
    logger.info(f"Tools initialized: {list(tool_map.keys())}")
    return schemas, tool_map
active_tool_schemas, tool_map = initialize_tools()

# --- System Message Definition ---
SYSTEM_MESSAGE = { 
    "role": "system", 
    "content": (
        "You are OmniAgent. Use tools proactively. **Never** say you lack access; state you will use a tool."
        "You **MUST NEVER** identify as an AI model or assistant.\n\n"
        "You are a multi-functional agent capable of performing various tasks. "
        "Use `code_execution` to acknowledge Python code requests (execution handled externally). "
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
        "**Available Tools:**\n" + 
        "".join([f"- `{tool.name}`: {tool.description}\n" for tool in tool_map.values()]) + 
        "\nAnswer based on tool outputs and file analysis.\n"
        "Your goal is to assist users in achieving their objectives efficiently.\n\n"
        "You work as a part time LinkedIn and X (Twitter) content creator for a tech company, they are called Raiden Agents.. they build AI agents for business and companies to automate their workflows.\n"
        "You are also a part time developer for a company called Raiden Agents, they build AI agents for business and companies to automate their workflows.\n"
        "You are a senior developer with over 10 years of experience in Python, JavaScript, and C++. You are also a senior data scientist with over 10 years of experience in data science and machine learning.\n"

    )
}
logger.info(f"System message generated. Approx tokens: {len(SYSTEM_MESSAGE['content']) // 4}")

# --- File Processing Helper (Updated) ---
def process_file_input(file_identifier):
    """Process file input from URL, GCS, or local path - enhanced version with better error handling."""
    logger.info(f"Processing file: {file_identifier}")
    content_part = {"type": "file"}
    file_data_dict = {}
    mime_type = None
    
    # URL-based file handling
    if file_identifier.startswith("http://") or file_identifier.startswith("https://"):
        file_data_dict["file_id"] = file_identifier
        mime_type, _ = mimetypes.guess_type(file_identifier)
        if not mime_type:
            try:
                r = requests.head(file_identifier, allow_redirects=True, timeout=10)
                r.raise_for_status()
                ct = r.headers.get('Content-Type')
                mime_type = ct.split(';')[0].strip() if ct else None
                # If HEAD request worked but didn't get mime type, try with GET to fetch content
                if not mime_type:
                    r = requests.get(file_identifier, timeout=15)
                    r.raise_for_status()
                    ct = r.headers.get('Content-Type')
                    mime_type = ct.split(';')[0].strip() if ct else "application/octet-stream"
                    # Try to download the content directly
                    file_content = r.content
                    encoded_content = base64.b64encode(file_content).decode("utf-8")
                    file_data_dict["file_data"] = f"data:{mime_type};base64,{encoded_content}"
                    logger.info(f" Downloaded URL content: {len(file_content)} bytes with mime: {mime_type}")
                    # Skip file_id approach and use direct content
                    del file_data_dict["file_id"]
            except requests.exceptions.RequestException as e:
                logger.warning(f" URL request failed: {e}")
                # Still try to proceed with best effort
        
        if "file_id" in file_data_dict and mime_type:
            file_data_dict["format"] = mime_type
            logger.info(f" URL MIME: {mime_type}")
        elif "file_id" in file_data_dict:
            # Default mime type if we couldn't detect it
            mime_type = "application/octet-stream"
            file_data_dict["format"] = mime_type
            logger.warning(f" Default MIME for URL: {mime_type}")
    
    # Google Cloud Storage handling
    elif file_identifier.startswith("gs://"):
        file_data_dict["file_id"] = file_identifier
        mime_type, _ = mimetypes.guess_type(file_identifier)
        if mime_type:
            file_data_dict["format"] = mime_type
            logger.info(f" GCS MIME: {mime_type}")
        else:
            # Default mime type if we couldn't detect it
            mime_type = "application/octet-stream"
            file_data_dict["format"] = mime_type
            logger.warning(f" Default MIME for GCS: {mime_type}")
    
    # Local file handling
    else:
        lp = Path(file_identifier)
        if not lp.is_file():
            logger.error(f"Local file not found: {lp}")
            return None
        
        try:
            fb = lp.read_bytes()
            ed = base64.b64encode(fb).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(lp)
            if not mime_type:
                # Try to guess from content
                import magic  # This might require installation
                try:
                    mime_type = magic.from_buffer(fb, mime=True)
                except (ImportError, AttributeError):
                    # Fallback to some basic checks
                    if fb.startswith(b'\xff\xd8\xff'):
                        mime_type = "image/jpeg"
                    elif fb.startswith(b'\x89PNG\r\n\x1a\n'):
                        mime_type = "image/png"
                    elif fb.startswith(b'GIF87a') or fb.startswith(b'GIF89a'):
                        mime_type = "image/gif"
                    elif fb.startswith(b'%PDF-'):
                        mime_type = "application/pdf"
                    else:
                        mime_type = "application/octet-stream"
                logger.warning(f" Guessed MIME: {mime_type}")
            else:
                logger.info(f" Local MIME: {mime_type}")
            
            file_data_dict["file_data"] = f"data:{mime_type};base64,{ed}"
            if vector_db.is_ready():
                fn = lp.name
                vector_db.add(f"User file: {fn} ({mime_type})", {
                    "type": "file_provided", 
                    "source": "local", 
                    "filename": fn, 
                    "mime_type": mime_type, 
                    "time": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Read local file error {lp}: {e}")
            traceback.print_exc()
            return None
    
    content_part["file"] = file_data_dict
    return content_part

# --- Tool Execution Wrapper (Accepts Dict) ---
def execute_tool_call(tool_call_data):
    """Wrapper for executing tool calls using dictionary input."""
    function_name = tool_call_data.get('function', {}).get('name')
    arguments_str = tool_call_data.get('function', {}).get('arguments')
    if not function_name: error_msg = "Error: Tool call missing function name."; logger.error(error_msg); return error_msg
    try:
        function_args = json.loads(arguments_str) if arguments_str else {}
        logger.info(f"Attempting execution: '{function_name}' args: {function_args}") # File only
    except json.JSONDecodeError: error_msg = f"Error: Invalid JSON args for {function_name}"; logger.error(error_msg); return error_msg # Console ERROR
    if function_name not in tool_map: error_msg = f"Error: Unknown function '{function_name}'"; logger.error(error_msg); return error_msg # Console ERROR
    try:
        tool = tool_map[function_name]; result = tool.execute(**function_args)
        logger.info(f"Tool '{function_name}' executed successfully.") # File only
        logger.debug(f"Tool '{function_name}' result snippet: {str(result)[:200]}...") # File only
        return result
    except ToolExecutionError as e: logger.error(f"Tool execution failed '{function_name}': {e}"); return f"Error executing tool {function_name}: {e}" # Console ERROR + return error msg
    except Exception as e: logger.critical(f"Unexpected critical error executing tool '{function_name}'", exc_info=True); return f"Critical Error executing tool {function_name}." # Console CRITICAL + return error msg


# --- Handle Streaming Response ---
# (Unchanged)
def handle_streaming_response(stream):
    full_response_content = ""; tool_calls_agg = defaultdict(lambda: {"id": None, "name": None, "arguments": ""}); final_tool_calls_list = []; completed_tool_call_indices = set(); current_tool_call_index = -1
    print("\nOmniBot: ", end="", flush=True) # Console output
    try:
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content: print(delta_content, end="", flush=True); full_response_content += delta_content
            delta_tool_calls = chunk.choices[0].delta.tool_calls
            if delta_tool_calls:
                for tc_chunk in delta_tool_calls:
                    idx = tc_chunk.index
                    if tc_chunk.id: tool_calls_agg[idx]["id"] = tc_chunk.id
                    if tc_chunk.function and tc_chunk.function.name: tool_calls_agg[idx]["name"] = tc_chunk.function.name
                    if tc_chunk.function and tc_chunk.function.arguments: tool_calls_agg[idx]["arguments"] += tc_chunk.function.arguments
                    current_call = tool_calls_agg[idx]
                    if current_call["id"] and current_call["name"] and idx not in completed_tool_call_indices:
                         args_str = current_call["arguments"]; is_complete_json = False
                         try: json.loads(args_str); is_complete_json = True
                         except json.JSONDecodeError: pass
                         if is_complete_json:
                              logger.debug(f"Stream: Finalizing tool {idx}...") # File only
                              final_tool_calls_list.append({"id": current_call["id"], "type": "function", "function": {"name": current_call["name"], "arguments": args_str}})
                              completed_tool_call_indices.add(idx)
    except Exception as e: logger.error(f"Stream error: {e}", exc_info=True); print(f"\n[Stream Error: {e}]") # Console ERROR
    finally: print()
    final_message_dict = {"role": "assistant", "content": full_response_content if full_response_content else None, "tool_calls": final_tool_calls_list if final_tool_calls_list else None}
    return final_message_dict


# --- Main Chat Loop ---
def chat_agent():
    model_name = "gemini/gemini-2.5-flash-preview-04-17"
    # Initialize the new Redis-based memory manager, passing the vector_db instance
    memory = RedisPersistentMemory(vector_db_client=vector_db, system_message=SYSTEM_MESSAGE, max_tokens=1_048_576) # Use specified token limit

    logger.info("\n--- OmniBot Initialized (v9.8 - Codespaces Ready, Redis Memory) ---") # File only
    print(f"OmniBot v9.8 Initialized. Model: {model_name}. Memory: Redis. Type 'quit' to exit.") # Console output
    print(f"Vector DB Status: {'Ready' if vector_db.is_ready() else 'Unavailable'}") # Console output
    print("-" * 65 + "\n") # Console output

    while True:
        try:
            user_input = input("You: ") # Console output
            if user_input.lower() == "quit": logger.info("User quit."); break # File only

            user_message_content = []
            # Enhanced file handling
            if user_input.lower().startswith("file:") and len(user_input.split(' ', 1)) > 1:
                file_id = user_input.split(' ', 1)[1].strip()
                if file_id:
                    print("OmniBot: Processing file, please wait...", flush=True)  # Console output
                    file_part = process_file_input(file_id)
                    if not file_part:
                        print("OmniBot: [Error processing file. Please check if the file exists or URL is accessible.]")
                        continue  # Console output
                    
                    prompt = input("You (prompt for file): ")  # Console output
                    if prompt:
                        user_message_content.extend([{"type": "text", "text": prompt}, file_part])
                    else:
                        # Allow file without prompt - default prompt
                        user_message_content.extend([{"type": "text", "text": "Analyze this file for me."}, file_part])
                else:
                    print("OmniBot: [File command needs path/URL.]")
                    continue  # Console output
            else:
                user_message_content.append({"type": "text", "text": user_input})
            
            if not user_message_content:
                continue

            user_message = {"role": "user", "content": user_message_content}
            if not memory.add_message(user_message): print("OmniBot: [Message too long.]"); continue # Console output

            user_text = user_input
            if isinstance(user_message_content, list):
                for item in user_message_content:
                    if item.get("type") == "text": user_text = item.get("text",""); break
            if vector_db.is_ready(): vector_db.add(f"User said: {user_text}", {"type": "user_message", "time": datetime.now().isoformat()})

            # --- CONSOLE OUTPUT: Thinking ---
            print("\nOmniBot: Thinking...", flush=True)
            logger.info("OmniBot: Thinking...") # File only

            current_messages = memory.get_messages()
            response_stream = litellm.completion(model=model_name, messages=current_messages, tools=active_tool_schemas, tool_choice="auto", stream=True)

            # Prints stream to console via handle_streaming_response
            response_message_dict = handle_streaming_response(response_stream)
            memory.add_message(response_message_dict)

            if response_message_dict.get("tool_calls"):
                # --- CONSOLE OUTPUT: Tool Usage ---
                print(f"OmniBot: Using {len(response_message_dict['tool_calls'])} tool(s)...", flush=True)
                logger.info(f"LLM requested {len(response_message_dict['tool_calls'])} tool(s)...") # File only

                tool_results = []
                for tc_data in response_message_dict["tool_calls"]:
                    # Pass dictionary directly to corrected execute_tool_call
                    result_content = execute_tool_call(tc_data)
                    if isinstance(result_content, str) and result_content.lower().startswith("error"):
                        logger.warning(f"Tool '{tc_data.get('function', {}).get('name')}' failed. Error: {result_content}") # Console WARN + File
                    result_msg = {"role": "tool", "tool_call_id": tc_data.get('id'), "content": str(result_content)}
                    tool_results.append(result_msg)
                    memory.add_message(result_msg) # Add result to memory

                # --- CONSOLE OUTPUT: Processing ---
                print("\nOmniBot: Processing tool results...", flush=True)
                logger.info("OmniBot: Processing tool results...") # File only

                messages_with_results = memory.get_messages()
                final_stream = litellm.completion(model=model_name, messages=messages_with_results, stream=True)

                # Prints stream to console via handle_streaming_response
                final_response_dict = handle_streaming_response(final_stream)
                memory.add_message(final_response_dict)

                final_content = final_response_dict.get("content", "")
                # Logging of final content happens below
                if final_content and vector_db.is_ready(): vector_db.add(f"OmniBot response: {final_content}", {"type": "assistant_response", "after_tool_use": True, "time": datetime.now().isoformat()})
                if final_content: logger.info(f"OmniBot final response (after tools): {final_content}") # File only
                elif response_message_dict.get("tool_calls"): logger.info("OmniBot final response after tool use had no text.") # File only
                else: logger.warning("OmniBot final response was empty.") # Console WARN

            elif response_message_dict.get("content"):
                assistant_content = response_message_dict["content"]
                logger.info(f"OmniBot response: {assistant_content}") # File only
                if vector_db.is_ready(): vector_db.add(f"OmniBot response: {assistant_content}", {"type": "assistant_response", "after_tool_use": False, "time": datetime.now().isoformat()})
            else:
                logger.warning("OmniBot received empty initial response (no content or tools).") # Console WARN

        # --- Error Handling ---
        except litellm.exceptions.APIError as e:
             logger.error(f"LiteLLM API Error: {e}", exc_info=True); print(f"\n!!! OmniBot Error: API Failure {e.status_code if hasattr(e, 'status_code') else ''} !!!", file=sys.stderr) # Console ERROR
             try: logger.error(f"API Error Body: {json.dumps(e.response.json(), indent=2)}") # Console ERROR
             except: logger.error(f"Raw API Error: {e.response.text if hasattr(e, 'response') else 'N/A'}") # Console ERROR
        except AgentException as e: logger.error(f"Agent Error: {e}", exc_info=True); print(f"\n!!! OmniBot Error: {e} !!!", file=sys.stderr) # Console ERROR
        except KeyboardInterrupt: logger.info("User interrupted."); print("\nOmniBot: Exiting..."); break # File info, Console output
        except Exception as e: logger.critical(f"Critical error in main loop!", exc_info=True); print(f"\n!!! OmniBot Critical Error: {e} !!!", file=sys.stderr); break # Console CRITICAL


# --- Start the Agent ---
if __name__ == "__main__":
    try:
        chat_agent()
    except APIKeyError:
        print("Execution stopped: Missing critical API key. Please set it in .env", file=sys.stderr) # Console output
    except Exception as main_e:
        logger.critical(f"Critical agent startup error: {main_e}", exc_info=True) # File CRITICAL
        print(f"\n!!! Critical startup error: {main_e}. Check logs for details. !!!", file=sys.stderr) # Console CRITICAL