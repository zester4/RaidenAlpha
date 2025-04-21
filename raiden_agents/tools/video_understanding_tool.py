import logging
import os
import base64
import json
import mimetypes
import requests
import io
from dotenv import load_dotenv
from pathlib import Path
import traceback
from .base_tool import Tool, ToolExecutionError

load_dotenv()
# Try importing litellm (should be available from app.py context)
try:
    import litellm
except ImportError:
    litellm = None

logger = logging.getLogger("gemini_agent")

# Supported MIME types based on Gemini documentation
SUPPORTED_VIDEO_MIME_TYPES = {
    "video/mp4", "video/mpeg", "video/mov", "video/avi",
    "video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/3gpp"
}
# Max size for inline data (conservative estimate under 20MB)
MAX_INLINE_SIZE_BYTES = 19 * 1024 * 1024

class VideoUnderstandingTool(Tool):
    def __init__(self):
        super().__init__(
            name="video_understanding",
            description="Analyzes video from local paths, standard URLs, or YouTube URLs to summarize, answer questions, or transcribe.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["summarize", "ask_question", "transcribe"],
                        "description": "The video analysis operation to perform."
                    },
                    "video_path": {
                        "type": "string",
                        "description": "Required. Local file path, standard URL, or YouTube URL of the video."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The text query or instruction. Required for 'ask_question', optional for others (defaults will be used)."
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "Optional timestamp (MM:SS) for 'ask_question' to query a specific point in the video."
                    }
                },
                "required": ["operation", "video_path"]
            }
        )
        if not litellm:
             logger.error("litellm library is not available. VideoUnderstandingTool requires it.")

    def _process_video_input(self, video_identifier):
        """Processes video path/URL, returns content part for litellm."""
        logger.info(f"Processing video identifier: {video_identifier}")
        content_part = {}

        # Handle YouTube URLs directly (assuming litellm supports this format)
        if "youtube.com/watch?v=" in video_identifier or "youtu.be/" in video_identifier:
            logger.info("Identified YouTube URL.")
            # Format based on Google SDK example, hoping litellm understands similar structure
            # This part is experimental with litellm - might need adjustment
            content_part = {
                "type": "file_data", # Hypothetical type for litellm based on Google SDK
                "file_data": {
                    "mime_type": "video/youtube", # Custom type to indicate YouTube URL
                    "file_uri": video_identifier
                }
            }
            # Alternative simpler approach: just pass the URL in text? Less likely to work.
            # content_part = {"type": "text", "text": f"Analyze YouTube video: {video_identifier}"}
            return content_part, "youtube" # Indicate type

        # Handle standard URLs
        elif video_identifier.startswith("http://") or video_identifier.startswith("https://"):
            try:
                response = requests.get(video_identifier, timeout=30, stream=True) # Stream for potentially large files
                response.raise_for_status()
                content_type = response.headers.get('Content-Type')
                mime_type = content_type.split(';')[0].strip() if content_type else None

                if not mime_type or mime_type not in SUPPORTED_VIDEO_MIME_TYPES:
                     # Try guessing from URL if header is missing/wrong
                     mime_type, _ = mimetypes.guess_type(video_identifier)
                     if mime_type not in SUPPORTED_VIDEO_MIME_TYPES:
                          raise ToolExecutionError(f"Unsupported or undetermined video MIME type for URL: {mime_type or 'None'}")

                # Read content, checking size limit for inline data
                video_bytes = b""
                for chunk in response.iter_content(chunk_size=8192):
                    video_bytes += chunk
                    if len(video_bytes) > MAX_INLINE_SIZE_BYTES:
                        raise ToolExecutionError(f"Video file from URL exceeds inline size limit ({MAX_INLINE_SIZE_BYTES / (1024*1024):.1f} MB). Use File API if available or provide smaller video.")

                logger.info(f"Fetched video from URL. MIME: {mime_type}. Size: {len(video_bytes)} bytes.")

            except requests.exceptions.RequestException as e:
                raise ToolExecutionError(f"Failed to fetch video from URL '{video_identifier}': {e}")
            except ToolExecutionError as e:
                 raise e
            except Exception as e:
                 raise ToolExecutionError(f"Error processing video URL '{video_identifier}': {e}")

        # Handle local file paths
        else:
            local_path = Path(video_identifier).resolve()
            if not local_path.is_file():
                raise ToolExecutionError(f"Local video file not found: {local_path}")
            try:
                file_size = local_path.stat().st_size
                if file_size > MAX_INLINE_SIZE_BYTES:
                     raise ToolExecutionError(f"Local video file exceeds inline size limit ({MAX_INLINE_SIZE_BYTES / (1024*1024):.1f} MB). Use File API if available or provide smaller video.")

                video_bytes = local_path.read_bytes()
                mime_type, _ = mimetypes.guess_type(local_path)
                if mime_type not in SUPPORTED_VIDEO_MIME_TYPES:
                    # Add basic check based on extension if mimetypes fails
                    ext = local_path.suffix.lower()
                    simple_mime_map = {'.mp4': 'video/mp4', '.mov': 'video/mov', '.avi': 'video/avi', '.webm': 'video/webm', '.mpg': 'video/mpg', '.wmv': 'video/wmv'}
                    if ext in simple_mime_map:
                         mime_type = simple_mime_map[ext]
                         logger.warning(f"Guessed MIME type '{mime_type}' based on extension for {local_path}")
                    else:
                         raise ToolExecutionError(f"Unsupported video MIME type: {mime_type or 'Unknown'}")

                logger.info(f"Read local video file. MIME: {mime_type}. Size: {len(video_bytes)} bytes.")
            except ToolExecutionError as e:
                 raise e
            except Exception as e:
                raise ToolExecutionError(f"Failed to read local video file '{local_path}': {e}")

        # Encode and format for litellm (assuming data URI works for video like image)
        base64_video = base64.b64encode(video_bytes).decode('utf-8')
        data_uri = f"data:{mime_type};base64,{base64_video}"
        content_part = {
            "type": "video_url", # Using a distinct type name, hoping litellm handles it
            "video_url": {"url": data_uri}
        }
        return content_part, "inline" # Indicate type

    def execute(self, **kwargs):
        if not litellm:
             raise ToolExecutionError("litellm library is not available. VideoUnderstandingTool requires it.")

        operation = kwargs.get("operation")
        video_path = kwargs.get("video_path")
        prompt = kwargs.get("prompt", "")
        timestamp = kwargs.get("timestamp")

        if not video_path:
            raise ToolExecutionError("'video_path' (local path, URL, or YouTube URL) is required.")
        if not prompt and operation == "ask_question":
             raise ToolExecutionError("'prompt' is required for 'ask_question' operation.")

        # --- Process Video Input ---
        try:
            video_content_part, input_type = self._process_video_input(video_path)
        except ToolExecutionError as e:
            raise e # Propagate errors from processing

        # --- Construct Prompt and Content List ---
        final_prompt = prompt
        if operation == "summarize" and not prompt:
            final_prompt = "Summarize this video."
        elif operation == "transcribe" and not prompt:
            final_prompt = "Provide a transcript for this video."
        elif operation == "ask_question" and timestamp:
            final_prompt = f"{prompt} (referring to timestamp {timestamp})"

        # Structure for litellm (text part + video part)
        # Place text *after* video for potentially better results as per Gemini docs
        content_list = [
            video_content_part,
            {"type": "text", "text": final_prompt.strip()}
        ]

        # --- Call the LLM via litellm ---
        # Use gemini-pro-vision or the latest equivalent supporting video
        # Note: Model capabilities for video might differ from images (e.g., length limits)
        model_name = "gemini/gemini-2.0-flash"
        logger.info(f"Sending request to {model_name} for video '{video_path}' with prompt: '{final_prompt[:100]}...'")

        try:
            messages = [{"role": "user", "content": content_list}]
            response = litellm.completion(
                model=model_name,
                messages=messages
            )
            response_text = response.choices[0].message.content
            if not response_text:
                 logger.warning(f"Received empty response from {model_name} for video analysis.")
                 return "Model returned an empty response for the video analysis."

            logger.info(f"Received video analysis response from {model_name}.")
            return response_text

        except Exception as e:
            logger.error(f"Error calling LLM via litellm for video understanding: {e}", exc_info=True)
            traceback.print_exc()
            err_str = str(e)
            if "API key" in err_str: raise ToolExecutionError("API key error calling Gemini via litellm.")
            elif "Deadline Exceeded" in err_str: raise ToolExecutionError("API call timed out.")
            # Check for specific errors related to video processing if possible
            elif "does not support video" in err_str.lower(): raise ToolExecutionError(f"Model '{model_name}' reported error: {err_str}")
            elif "Invalid file URI" in err_str: raise ToolExecutionError(f"Error processing video URI '{video_path}'. Check format/accessibility. Detail: {err_str}")
            else: raise ToolExecutionError(f"Failed to get response from vision model for video: {e}")
