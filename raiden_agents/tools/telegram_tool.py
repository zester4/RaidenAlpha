import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from telegram import Bot, Update, Message, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class TelegramTool(Tool):
    """Tool for Telegram bot operations and messaging"""
    
    def __init__(self, bot_token: str):
        super().__init__(
            name="telegram",
            description="Interact with Telegram for messaging, sending photos, files, etc and bot operations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": [
                            "SEND_MESSAGE",
                            "SEND_PHOTO",
                            "SEND_DOCUMENT",
                            "CREATE_POLL",
                            "CREATE_BUTTON",
                            "GET_CHAT_INFO",
                            "GET_CHAT_MEMBERS",
                            "PIN_MESSAGE",
                            "UNPIN_MESSAGE",
                            "DELETE_MESSAGE",
                            "GET_BOT_COMMANDS",
                            "SET_BOT_COMMANDS",
                            "CREATE_CHAT_INVITE",
                            "GET_CHAT_MEMBER_COUNT",
                            "GET_CHAT_ADMINISTRATORS",
                            "SET_CHAT_TITLE",
                            "SET_CHAT_DESCRIPTION"
                        ]
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Target chat ID for the operation",
                        "optional": False
                    },
                    "message_text": {
                        "type": "string",
                        "description": "Text content for messages",
                        "optional": True
                    },
                    "photo_path": {
                        "type": "string",
                        "description": "Path to photo file",
                        "optional": True
                    },
                    "document_path": {
                        "type": "string",
                        "description": "Path to document file",
                        "optional": True
                    },
                    "poll_question": {
                        "type": "string",
                        "description": "Question for poll creation",
                        "optional": True
                    },
                    "poll_options": {
                        "type": "array",
                        "description": "Options for poll",
                        "items": {"type": "string"},
                        "optional": True
                    },
                    "button_text": {
                        "type": "string",
                        "description": "Text for inline button",
                        "optional": True
                    },
                    "button_callback": {
                        "type": "string",
                        "description": "Callback data for button",
                        "optional": True
                    },
                    "message_id": {
                        "type": "integer",
                        "description": "Message ID for operations like pin/unpin",
                        "optional": True
                    },
                    "commands": {
                        "type": "array",
                        "description": "List of bot commands",
                        "items": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        },
                        "optional": True
                    }
                },
                "required": ["operation", "chat_id"]
            }
        )
        self.bot_token = bot_token
        self.bot = Bot(token=bot_token)
        
    async def _init_bot(self):
        """Initialize bot instance asynchronously"""
        if not hasattr(self, '_bot'):
            self._bot = await Bot(self.bot_token).initialize()
        return self._bot

    async def send_message(self, chat_id: str, text: str, 
                          reply_markup: Optional[InlineKeyboardMarkup] = None) -> Message:
        """Send a text message to a chat"""
        bot = await self._init_bot()
        try:
            return await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
        except Exception as e:
            raise ToolExecutionError(f"Failed to send message: {str(e)}")

    async def send_photo(self, chat_id: str, photo_path: str, 
                        caption: Optional[str] = None) -> Message:
        """Send a photo to a chat"""
        bot = await self._init_bot()
        try:
            with open(photo_path, 'rb') as photo:
                return await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=caption
                )
        except Exception as e:
            raise ToolExecutionError(f"Failed to send photo: {str(e)}")

    async def send_document(self, chat_id: str, document_path: str, 
                          caption: Optional[str] = None) -> Message:
        """Send a document to a chat"""
        bot = await self._init_bot()
        try:
            with open(document_path, 'rb') as document:
                return await bot.send_document(
                    chat_id=chat_id,
                    document=document,
                    caption=caption
                )
        except Exception as e:
            raise ToolExecutionError(f"Failed to send document: {str(e)}")

    async def create_poll(self, chat_id: str, question: str, 
                         options: List[str], is_anonymous: bool = True) -> Message:
        """Create a poll in a chat"""
        bot = await self._init_bot()
        try:
            return await bot.send_poll(
                chat_id=chat_id,
                question=question,
                options=options,
                is_anonymous=is_anonymous
            )
        except Exception as e:
            raise ToolExecutionError(f"Failed to create poll: {str(e)}")

    async def create_button(self, text: str, callback_data: str) -> InlineKeyboardMarkup:
        """Create an inline keyboard button"""
        try:
            button = InlineKeyboardButton(text=text, callback_data=callback_data)
            return InlineKeyboardMarkup([[button]])
        except Exception as e:
            raise ToolExecutionError(f"Failed to create button: {str(e)}")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat"""
        bot = await self._init_bot()
        try:
            chat = await bot.get_chat(chat_id)
            return {
                "id": chat.id,
                "type": chat.type,
                "title": chat.title,
                "description": chat.description,
                "member_count": await bot.get_chat_member_count(chat_id)
            }
        except Exception as e:
            raise ToolExecutionError(f"Failed to get chat info: {str(e)}")

    async def pin_message(self, chat_id: str, message_id: int) -> bool:
        """Pin a message in a chat"""
        bot = await self._init_bot()
        try:
            return await bot.pin_chat_message(
                chat_id=chat_id,
                message_id=message_id
            )
        except Exception as e:
            raise ToolExecutionError(f"Failed to pin message: {str(e)}")

    async def unpin_message(self, chat_id: str, message_id: int) -> bool:
        """Unpin a message in a chat"""
        bot = await self._init_bot()
        try:
            return await bot.unpin_chat_message(
                chat_id=chat_id,
                message_id=message_id
            )
        except Exception as e:
            raise ToolExecutionError(f"Failed to unpin message: {str(e)}")

    async def set_bot_commands(self, commands: List[Dict[str, str]]) -> bool:
        """Set bot commands"""
        bot = await self._init_bot()
        try:
            command_list = [
                (command["command"], command["description"])
                for command in commands
            ]
            return await bot.set_my_commands(command_list)
        except Exception as e:
            raise ToolExecutionError(f"Failed to set bot commands: {str(e)}")

    async def create_chat_invite(self, chat_id: str, 
                               expire_date: Optional[int] = None) -> str:
        """Create chat invite link"""
        bot = await self._init_bot()
        try:
            invite = await bot.create_chat_invite_link(
                chat_id=chat_id,
                expire_date=expire_date
            )
            return invite.invite_link
        except Exception as e:
            raise ToolExecutionError(f"Failed to create invite link: {str(e)}")

    def execute(self, **kwargs) -> str:
        """Execute the Telegram tool based on provided parameters"""
        self.validate_args(kwargs)
        operation = kwargs.get("operation")
        chat_id = kwargs.get("chat_id")
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if operation == "SEND_MESSAGE":
                if not kwargs.get("message_text"):
                    raise ToolExecutionError("message_text is required for SEND_MESSAGE operation")
                result = loop.run_until_complete(
                    self.send_message(chat_id, kwargs["message_text"])
                )
                return f"Message sent successfully. Message ID: {result.message_id}"

            elif operation == "SEND_PHOTO":
                if not kwargs.get("photo_path"):
                    raise ToolExecutionError("photo_path is required for SEND_PHOTO operation")
                result = loop.run_until_complete(
                    self.send_photo(
                        chat_id,
                        kwargs["photo_path"],
                        kwargs.get("message_text")
                    )
                )
                return f"Photo sent successfully. Message ID: {result.message_id}"

            elif operation == "CREATE_POLL":
                if not (kwargs.get("poll_question") and kwargs.get("poll_options")):
                    raise ToolExecutionError("poll_question and poll_options are required for CREATE_POLL operation")
                result = loop.run_until_complete(
                    self.create_poll(
                        chat_id,
                        kwargs["poll_question"],
                        kwargs["poll_options"]
                    )
                )
                return f"Poll created successfully. Message ID: {result.message_id}"

            elif operation == "GET_CHAT_INFO":
                result = loop.run_until_complete(self.get_chat_info(chat_id))
                return f"Chat info retrieved: {result}"

            elif operation == "PIN_MESSAGE":
                if not kwargs.get("message_id"):
                    raise ToolExecutionError("message_id is required for PIN_MESSAGE operation")
                result = loop.run_until_complete(
                    self.pin_message(chat_id, kwargs["message_id"])
                )
                return "Message pinned successfully" if result else "Failed to pin message"

            elif operation == "SET_BOT_COMMANDS":
                if not kwargs.get("commands"):
                    raise ToolExecutionError("commands are required for SET_BOT_COMMANDS operation")
                result = loop.run_until_complete(
                    self.set_bot_commands(kwargs["commands"])
                )
                return "Bot commands set successfully" if result else "Failed to set bot commands"

            else:
                raise ToolExecutionError(f"Unsupported operation: {operation}")

        except Exception as e:
            raise ToolExecutionError(f"Error executing Telegram tool: {str(e)}")
        finally:
            loop.close()

    def _format_result(self, result: Any) -> str:
        """Format the operation result for output"""
        if isinstance(result, dict):
            return "\n".join(f"{k}: {v}" for k, v in result.items())
        return str(result)