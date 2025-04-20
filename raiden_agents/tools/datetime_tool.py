import logging
from datetime import datetime
from .base_tool import Tool

logger = logging.getLogger("gemini_agent")

class DateTimeTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_current_datetime",
            description="Returns current date and time.",
            parameters={"type": "object", "properties": {}}
        )

    def execute(self, **kwargs):
        now = datetime.now()
        formatted = now.strftime("%A, %d %B %Y, %H:%M:%S %Z")
        return f"Current date and time: {formatted}"
