import logging
import os
from pathlib import Path
import traceback
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

# Try importing mss, handle error if not installed
try:
    import mss
    import mss.tools
except ImportError:
    mss = None # Flag that mss is not available

logger = logging.getLogger("gemini_agent")

class ScreenshotTool(Tool):
    def __init__(self):
        super().__init__(
            name="take_screenshot",
            description="Captures a screenshot of the primary monitor and saves it to a file.",
            parameters={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Optional. The desired path (including filename, e.g., 'screenshots/capture.png') to save the screenshot. If omitted, a timestamped filename will be generated in the current directory."
                    }
                },
                "required": [] # output_path is optional
            }
        )

    def execute(self, **kwargs):
        if not mss:
            raise ToolExecutionError("Screenshot tool dependency 'mss' is not installed. Please install it (`pip install mss`).")

        output_path_str = kwargs.get("output_path")

        try:
            if output_path_str:
                # Resolve user-provided path
                target_path = Path(output_path_str).resolve()
                # Ensure parent directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Ensure it has a common image extension, default to png if not
                if target_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                     logger.warning(f"Output path '{output_path_str}' lacks image extension. Appending '.png'.")
                     target_path = target_path.with_suffix('.png')
                output_filename = str(target_path)
            else:
                # Generate timestamped filename in CWD
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"screenshot_{timestamp}.png"
                target_path = Path(output_filename).resolve() # Save in CWD
                logger.info(f"No output path provided, saving screenshot to: {target_path}")

            # Capture the screenshot
            with mss.mss() as sct:
                # Get information of the primary monitor
                monitor_number = 1 # Typically the primary monitor
                mon = sct.monitors[monitor_number]

                # The screen part to capture
                monitor = {
                    "top": mon["top"],
                    "left": mon["left"],
                    "width": mon["width"],
                    "height": mon["height"],
                    "mon": monitor_number,
                }
                logger.info(f"Capturing monitor {monitor_number}: {monitor}")

                # Grab the data
                sct_img = sct.grab(monitor)

                # Save to the file
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=output_filename)

            logger.info(f"Screenshot saved successfully to: {output_filename}")
            return f"Screenshot saved successfully to: {output_filename}"

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}", exc_info=True)
            traceback.print_exc()
            raise ToolExecutionError(f"Failed to take screenshot: {e}")