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

# Try importing Pillow, handle error if not installed
try:
    from PIL import Image
except ImportError:
    Image = None # Flag that Pillow is not available

# Try importing litellm (should be available from app.py context)
try:
    import litellm
except ImportError:
    litellm = None

logger = logging.getLogger("gemini_agent")

# Supported MIME types based on Gemini documentation
SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}

class ImageUnderstandingTool(Tool):
    def __init__(self):
        super().__init__(
            name="image_understanding",
            description="Analyzes images from local paths or URLs, describe the images, answer questions, detect objects (with bounding boxes), or segment objects (with masks). Can process multiple images.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["caption", "ask_question", "detect_objects", "segment_objects", "compare_images"],
                        "description": "The image analysis operation to perform."
                    },
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required. A list containing one or more local file paths or URLs pointing to the images."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Required (except sometimes for 'caption'). The text query or instruction for the analysis (e.g., 'What color is the car?', 'Detect vehicles', 'Segment furniture', 'Compare these images')."
                    }
                },
                "required": ["operation", "image_paths"] # Prompt is almost always required
            }
        )
        if not Image:
             logger.error("Pillow library is not installed. ImageUnderstandingTool requires it (`pip install Pillow`).")
        if not litellm:
             logger.error("litellm library is not available. ImageUnderstandingTool requires it.")

    def _process_image_path(self, image_identifier):
        """Processes a single image path (local or URL) and returns bytes and MIME type."""
        logger.info(f"Processing image identifier: {image_identifier}")
        image_bytes = None
        mime_type = None

        if image_identifier.startswith("http://") or image_identifier.startswith("https://"):
            try:
                response = requests.get(image_identifier, timeout=15)
                response.raise_for_status()
                image_bytes = response.content
                # Try to get MIME type from headers first
                content_type = response.headers.get('Content-Type')
                if content_type:
                    mime_type = content_type.split(';')[0].strip()
                logger.info(f"Fetched image from URL. Declared MIME: {mime_type}. Size: {len(image_bytes)} bytes.")
            except requests.exceptions.RequestException as e:
                raise ToolExecutionError(f"Failed to fetch image from URL '{image_identifier}': {e}")
        else:
            # Assume local path
            local_path = Path(image_identifier)
            if not local_path.is_file():
                raise ToolExecutionError(f"Local image file not found: {local_path}")
            try:
                image_bytes = local_path.read_bytes()
                # Guess MIME type from file extension first
                mime_type, _ = mimetypes.guess_type(local_path)
                logger.info(f"Read local image file. Guessed MIME: {mime_type}. Size: {len(image_bytes)} bytes.")
            except Exception as e:
                raise ToolExecutionError(f"Failed to read local image file '{local_path}': {e}")

        # Verify image and get accurate MIME type using Pillow
        if image_bytes and Image:
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    img.verify() # Verify it's a valid image file
                    # Pillow format often maps directly to MIME type (e.g., 'JPEG' -> 'image/jpeg')
                    pillow_format = img.format
                    if pillow_format:
                         guessed_mime = f"image/{pillow_format.lower()}"
                         # Prefer Pillow's detected format if available and supported
                         if guessed_mime in SUPPORTED_MIME_TYPES:
                              mime_type = guessed_mime
                         elif mime_type not in SUPPORTED_MIME_TYPES: # If header/extension guess wasn't supported either
                              # Fallback if Pillow format isn't directly mappable or supported
                              logger.warning(f"Pillow format '{pillow_format}' not directly mapped or supported. Using previous guess: {mime_type}")
                              if mime_type not in SUPPORTED_MIME_TYPES:
                                   # If still no valid MIME, raise error
                                   raise ToolExecutionError(f"Unsupported image format detected by Pillow: {pillow_format}. Path: {image_identifier}")
                         logger.info(f"Verified image with Pillow. Final MIME type: {mime_type}")
                    elif mime_type not in SUPPORTED_MIME_TYPES: # If Pillow gave no format and guess is bad
                         raise ToolExecutionError(f"Could not determine supported MIME type for image. Path: {image_identifier}")

            except Exception as e:
                raise ToolExecutionError(f"Invalid or unsupported image file '{image_identifier}'. Error: {e}")
        elif not Image:
             logger.warning("Pillow not installed, cannot verify image format or accurately determine MIME type.")
             # Proceed with guessed mime_type if available, otherwise fail
             if not mime_type or mime_type not in SUPPORTED_MIME_TYPES:
                  raise ToolExecutionError(f"Cannot process image: Pillow not installed and could not determine a supported MIME type for '{image_identifier}'.")

        if mime_type not in SUPPORTED_MIME_TYPES:
             raise ToolExecutionError(f"Unsupported image MIME type '{mime_type}' for file '{image_identifier}'. Supported types: {', '.join(SUPPORTED_MIME_TYPES)}")

        return image_bytes, mime_type

    def execute(self, **kwargs):
        if not Image or not litellm:
             missing = []
             if not Image: missing.append("Pillow")
             if not litellm: missing.append("litellm")
             raise ToolExecutionError(f"Missing required libraries for ImageUnderstandingTool: {', '.join(missing)}. Please install them.")

        operation = kwargs.get("operation")
        image_paths = kwargs.get("image_paths")
        prompt = kwargs.get("prompt", "") # Default to empty prompt

        if not image_paths or not isinstance(image_paths, list):
            raise ToolExecutionError("'image_paths' (a list of strings) is required.")
        if not prompt and operation not in ["caption"]: # Prompt required for most ops
             raise ToolExecutionError(f"'prompt' is required for operation '{operation}'.")
        if operation == "compare_images" and len(image_paths) < 2:
             raise ToolExecutionError("'compare_images' operation requires at least two image paths.")

        # --- Construct the message content list for litellm ---
        content_list = []
        # Add the text prompt first
        full_prompt = prompt
        if operation == "detect_objects" and "bounding box" not in prompt.lower():
             full_prompt += "\nDetect the prominent items. Return bounding boxes as [ymin, xmin, ymax, xmax] normalized to 0-1000."
        elif operation == "segment_objects" and "segmentation mask" not in prompt.lower():
             full_prompt += "\nOutput a JSON list of segmentation masks where each entry contains 'box_2d', 'mask' (base64 png), and 'label'."
        elif operation == "caption" and not prompt:
             full_prompt = "Describe this image." # Default caption prompt

        content_list.append({"type": "text", "text": full_prompt.strip()})

        # Process and add each image
        total_size = 0
        max_inline_size = 19 * 1024 * 1024 # Leave some room under 20MB limit

        for img_path in image_paths:
            try:
                img_bytes, mime_type = self._process_image_path(img_path)
                total_size += len(img_bytes)
                if total_size > max_inline_size:
                     raise ToolExecutionError(f"Total size of inline images exceeds limit ({max_inline_size / (1024*1024):.1f} MB). Consider using fewer/smaller images or a File API if available.")

                # Encode image bytes as base64
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                # Construct the data URI format expected by Gemini (via litellm)
                data_uri = f"data:{mime_type};base64,{base64_image}"

                # Add image part to content list (format might vary slightly based on litellm version)
                # Common format: dictionary with type and source/data
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                    # Alternative format sometimes seen (check litellm docs if needed):
                    # "type": "image",
                    # "source": {
                    #     "type": "base64",
                    #     "media_type": mime_type,
                    #     "data": base64_image
                    # }
                })
                logger.info(f"Added image '{img_path}' ({mime_type}) to request content.")

            except ToolExecutionError as e:
                raise e # Re-raise processing errors
            except Exception as e:
                raise ToolExecutionError(f"Unexpected error processing image '{img_path}': {e}")

        # --- Call the LLM via litellm ---
        model_name = "gemini/gemini-2.0-flash" # Use the vision model
        logger.info(f"Sending request to {model_name} with {len(image_paths)} image(s) and prompt: '{full_prompt[:100]}...'")

        try:
            # Construct the message structure for litellm
            messages = [{"role": "user", "content": content_list}]
            response = litellm.completion(
                model=model_name,
                messages=messages,
                # Add other parameters like temperature if needed
            )

            # Extract the response text
            response_text = response.choices[0].message.content
            if not response_text:
                 logger.warning(f"Received empty response from {model_name}.")
                 return "Model returned an empty response."

            logger.info(f"Received response from {model_name}.")
            logger.debug(f"Response snippet: {response_text[:200]}...")

            # Attempt to parse JSON if detection/segmentation was requested
            if operation in ["detect_objects", "segment_objects"]:
                 try:
                      # Check if response looks like JSON before parsing
                      if response_text.strip().startswith('[') and response_text.strip().endswith(']'):
                           parsed_json = json.loads(response_text)
                           logger.info("Successfully parsed response as JSON for detection/segmentation.")
                           # Return the parsed JSON directly
                           return json.dumps(parsed_json, indent=2)
                      else:
                           logger.warning("Response for detection/segmentation did not appear to be valid JSON. Returning raw text.")
                           return response_text
                 except json.JSONDecodeError:
                      logger.warning("Failed to parse detection/segmentation response as JSON. Returning raw text.")
                      return response_text # Return raw text if JSON parsing fails
            else:
                 # For other operations, return the text directly
                 return response_text

        except Exception as e:
            logger.error(f"Error calling LLM via litellm for image understanding: {e}", exc_info=True)
            traceback.print_exc()
            # Provide more specific error info if possible
            err_str = str(e)
            if "API key" in err_str:
                 raise ToolExecutionError("API key error calling Gemini via litellm.")
            elif "Deadline Exceeded" in err_str:
                 raise ToolExecutionError("API call timed out.")
            else:
                 raise ToolExecutionError(f"Failed to get response from vision model: {e}")
