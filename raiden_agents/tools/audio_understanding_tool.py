import logging
import os
import base64
import json
import mimetypes
import requests
import io
from pathlib import Path
import traceback
from .base_tool import Tool, ToolExecutionError

# Try importing litellm (should be available from app.py context)
try:
    import litellm
except ImportError:
    litellm = None

logger = logging.getLogger("gemini_agent")

# Supported MIME types based on Gemini documentation
SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"
}
# Max size for inline data (conservative estimate under 20MB)
MAX_INLINE_SIZE_BYTES = 19 * 1024 * 1024

class AudioUnderstandingTool(Tool):
    def __init__(self):
        super().__init__(
            name="audio_understanding",
            description="Analyzes audio from local paths or URLs to summarize, answer questions, or transcribe.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["summarize", "ask_question", "transcribe"],
                        "description": "The audio analysis operation to perform."
                    },
                    "audio_path": {
                        "type": "string",
                        "description": "Required. Local file path or URL of the audio file."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The text query or instruction. Required for 'ask_question', optional for others (defaults will be used)."
                    },
                    "timestamp_range": {
                        "type": "string",
                        "description": "Optional timestamp range (MM:SS-MM:SS) for 'ask_question' or 'transcribe' to query/process a specific segment."
                    }
                },
                "required": ["operation", "audio_path"]
            }
        )
        if not litellm:
             logger.error("litellm library is not available. AudioUnderstandingTool requires it.")

    def _process_audio_input(self, audio_identifier):
        """Processes audio path/URL, returns content part for litellm."""
        logger.info(f"Processing audio identifier: {audio_identifier}")
        content_part = {}
        audio_bytes = None
        mime_type = None

        # Handle standard URLs
        if audio_identifier.startswith("http://") or audio_identifier.startswith("https://"):
            try:
                response = requests.get(audio_identifier, timeout=30, stream=True)
                response.raise_for_status()
                content_type = response.headers.get('Content-Type')
                mime_type = content_type.split(';')[0].strip() if content_type else None

                if not mime_type or mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
                     mime_type, _ = mimetypes.guess_type(audio_identifier)
                     if mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
                          raise ToolExecutionError(f"Unsupported or undetermined audio MIME type for URL: {mime_type or 'None'}")

                # Read content, checking size limit
                audio_bytes = b""
                for chunk in response.iter_content(chunk_size=8192):
                    audio_bytes += chunk
                    if len(audio_bytes) > MAX_INLINE_SIZE_BYTES:
                        raise ToolExecutionError(f"Audio file from URL exceeds inline size limit ({MAX_INLINE_SIZE_BYTES / (1024*1024):.1f} MB).")

                logger.info(f"Fetched audio from URL. MIME: {mime_type}. Size: {len(audio_bytes)} bytes.")

            except requests.exceptions.RequestException as e:
                raise ToolExecutionError(f"Failed to fetch audio from URL '{audio_identifier}': {e}")
            except ToolExecutionError as e:
                 raise e
            except Exception as e:
                 raise ToolExecutionError(f"Error processing audio URL '{audio_identifier}': {e}")

        # Handle local file paths
        else:
            local_path = Path(audio_identifier).resolve()
            if not local_path.is_file():
                raise ToolExecutionError(f"Local audio file not found: {local_path}")
            try:
                file_size = local_path.stat().st_size
                if file_size > MAX_INLINE_SIZE_BYTES:
                     raise ToolExecutionError(f"Local audio file exceeds inline size limit ({MAX_INLINE_SIZE_BYTES / (1024*1024):.1f} MB).")

                audio_bytes = local_path.read_bytes()
                mime_type, _ = mimetypes.guess_type(local_path)
                if mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
                     # Basic check if mimetypes fails
                     ext = local_path.suffix.lower()
                     simple_mime_map = {'.wav': 'audio/wav', '.mp3': 'audio/mp3', '.aiff': 'audio/aiff', '.aac': 'audio/aac', '.ogg': 'audio/ogg', '.flac': 'audio/flac'}
                     if ext in simple_mime_map:
                          mime_type = simple_mime_map[ext]
                          logger.warning(f"Guessed MIME type '{mime_type}' based on extension for {local_path}")
                     else:
                          raise ToolExecutionError(f"Unsupported audio MIME type: {mime_type or 'Unknown'}")

                logger.info(f"Read local audio file. MIME: {mime_type}. Size: {len(audio_bytes)} bytes.")
            except ToolExecutionError as e:
                 raise e
            except Exception as e:
                raise ToolExecutionError(f"Failed to read local audio file '{local_path}': {e}")

        # Encode and format for litellm (assuming similar structure to image/video)
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        data_uri = f"data:{mime_type};base64,{base64_audio}"
        content_part = {
            "type": "audio_url", # Using a distinct type name
            "audio_url": {"url": data_uri}
        }
        return content_part

    def execute(self, **kwargs):
        if not litellm:
             raise ToolExecutionError("litellm library is not available. AudioUnderstandingTool requires it.")

        operation = kwargs.get("operation")
        audio_path = kwargs.get("audio_path")
        prompt = kwargs.get("prompt", "")
        timestamp_range = kwargs.get("timestamp_range") # e.g., "02:30-03:29"

        if not audio_path:
            raise ToolExecutionError("'audio_path' (local path or URL) is required.")
        if not prompt and operation == "ask_question":
             raise ToolExecutionError("'prompt' is required for 'ask_question' operation.")

        # --- Process Audio Input ---
        try:
            audio_content_part = self._process_audio_input(audio_path)
        except ToolExecutionError as e:
            raise e # Propagate errors

        # --- Construct Prompt and Content List ---
        final_prompt = prompt
        if operation == "summarize" and not prompt:
            final_prompt = "Summarize this audio."
        elif operation == "transcribe" and not prompt:
            final_prompt = "Provide a transcript for this audio."

        if timestamp_range:
             # Validate format roughly MM:SS-MM:SS
             if re.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", timestamp_range):
                  final_prompt += f" (referring to time range {timestamp_range})"
             else:
                  logger.warning(f"Invalid timestamp_range format '{timestamp_range}'. Ignoring.")
                  # Optionally raise ToolExecutionError here

        # Structure for litellm (text part + audio part)
        content_list = [
            audio_content_part,
            {"type": "text", "text": final_prompt.strip()}
        ]

        # --- Call the LLM via litellm ---
        # Use a model known to support audio, like gemini-pro or flash if updated
        # Let's assume gemini-pro for now, adjust if needed based on litellm/Gemini updates
        model_name = "gemini/gemini-2.0-flash" # Or potentially gemini-flash if it supports audio
        logger.info(f"Sending request to {model_name} for audio '{audio_path}' with prompt: '{final_prompt[:100]}...'")

        try:
            messages = [{"role": "user", "content": content_list}]
            response = litellm.completion(
                model=model_name,
                messages=messages
            )
            response_text = response.choices[0].message.content
            if not response_text:
                 logger.warning(f"Received empty response from {model_name} for audio analysis.")
                 return "Model returned an empty response for the audio analysis."

            logger.info(f"Received audio analysis response from {model_name}.")
            return response_text

        except Exception as e:
            logger.error(f"Error calling LLM via litellm for audio understanding: {e}", exc_info=True)
            traceback.print_exc()
            err_str = str(e)
            if "API key" in err_str: raise ToolExecutionError("API key error calling Gemini via litellm.")
            elif "Deadline Exceeded" in err_str: raise ToolExecutionError("API call timed out.")
            elif "does not support audio" in err_str.lower(): raise ToolExecutionError(f"Model '{model_name}' reported error: {err_str}")
            else: raise ToolExecutionError(f"Failed to get response from model for audio: {e}")
