import logging
import requests
import os
from pathlib import Path
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

# Get API key from environment
stability_api_key = os.environ.get("STABILITY_API_KEY")

class ImageGenerationTool(Tool):
    def __init__(self):
        super().__init__(
            name="generate_image",
            description="Generate images using Stability AI's image generation API",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Name for the output file (without extension)"
                    }
                },
                "required": ["prompt", "file_name"]
            }
        )
    
    def execute(self, **kwargs):
        self.validate_args(kwargs)
        
        try:
            prompt = kwargs.get("prompt")
            file_name = kwargs.get("file_name")
            output_dir = "generated_images"  # Fixed output directory
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            if not stability_api_key:
                raise ToolExecutionError("Stability AI API key missing")
            
            logger.info(f"Generating image for prompt: {prompt}")
            
            # Generate image using Stability AI API
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/core",
                headers={
                    "authorization": f"Bearer {stability_api_key}",
                    "accept": "image/*"
                },
                files={"none": ""},
                data={
                    "prompt": prompt,
                    "output_format": "jpeg"
                }
            )
            
            if response.status_code != 200:
                raise ToolExecutionError(f"API Error: {response.json()}")
            
            # Save the image
            output_path = Path(output_dir) / f"{file_name}.jpeg"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Image saved to: {output_path}")
            
            # Store in vector DB if available
            try:
                from __main__ import vector_db
                if vector_db.is_ready():
                    vector_db.add(
                        f"Generated image from prompt: {prompt}",
                        {
                            "type": "generated_image",
                            "prompt": prompt,
                            "file_path": str(output_path),
                            "time": datetime.now().isoformat()
                        }
                    )
            except ImportError:
                pass
            
            return f"Image generated and saved to {output_path}"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            raise ToolExecutionError(f"Failed to connect to Stability AI API: {e}")
        except Exception as e:
            logger.error(f"Image generation error: {e}", exc_info=True)
            raise ToolExecutionError(f"Failed to generate image: {e}")
