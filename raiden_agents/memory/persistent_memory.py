import logging
import json
import os
from datetime import datetime
from upstash_redis import Redis
from raiden_agents.tools.base_tool import VectorDBError # Assuming VectorDBError might be needed for VDB logging errors
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger("gemini_agent")

# Removed the problematic import from __main__

class RedisPersistentMemory:
    CHARS_PER_TOKEN_ESTIMATE = 4 # Estimate for token calculation
    CONVERSATION_KEY = "raiden_agent_conversation_history" # Redis key for the list

    def __init__(self, vector_db_client=None, max_tokens=1_048_576, system_message=None): # Add vector_db_client parameter
        self.redis_client = None
        self.initialized = False
        self.system_message = system_message
        self.vector_db = vector_db_client # Store the passed vector_db client
        self.max_tokens = max_tokens
        self.current_token_count = 0 # Track tokens for the *current context window*, not total history

        try:
            # Initialize Upstash Redis client from environment variables
            redis_url = os.environ.get("UPSTASH_REDIS_REST_URL")
            redis_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

            if not redis_url or not redis_token:
                raise ValueError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN environment variables must be set.")

            self.redis_client = Redis(url=redis_url, token=redis_token)
            # Test connection
            self.redis_client.ping()
            self.initialized = True
            logger.info(f"RedisPersistentMemory initialized successfully. Max context tokens: {self.max_tokens}")

            # Store system message if not already the first item (or if list is empty)
            if self.system_message:
                try:
                    first_item_json = self.redis_client.lindex(self.CONVERSATION_KEY, 0)
                    if not first_item_json:
                         # List is empty, add system message
                         logger.info("Adding system message as first item in Redis history.")
                         self.redis_client.lpush(self.CONVERSATION_KEY, json.dumps(self.system_message))
                    else:
                         # Check if first item *is* the system message
                         first_item = json.loads(first_item_json)
                         if first_item != self.system_message:
                              # Prepend system message if it's different or missing
                              logger.warning("Prepending system message to Redis history as it differs from the first element.")
                              self.redis_client.lpush(self.CONVERSATION_KEY, json.dumps(self.system_message))
                         # else: logger.debug("System message already present as first item.") # Optional debug log
                except Exception as e:
                     logger.error(f"Error checking/adding system message in Redis: {e}", exc_info=True)


        except ImportError:
            logger.error("upstash-redis package not installed. RedisPersistentMemory cannot function.")
        except Exception as e:
            logger.error(f"Upstash Redis initialization failed: {e}", exc_info=True)

    def _estimate_tokens(self, message):
        """Estimates token count for a message dictionary."""
        return len(json.dumps(message)) // self.CHARS_PER_TOKEN_ESTIMATE

    def add_message(self, message):
        """Adds a message to the persistent Redis history and logs to VectorDB."""
        if not self.is_ready():
            logger.error("Cannot add message: Redis client not initialized.")
            return False # Indicate failure

        try:
            message_json = json.dumps(message)
            self.redis_client.rpush(self.CONVERSATION_KEY, message_json)
            logger.debug(f"Added message to Redis. Role: {message.get('role')}")

            # Also log to VectorDB for semantic search capability (use the instance variable)
            if self.vector_db and self.vector_db.is_ready():
                log_content = ""
                if message.get("role") == "user":
                    # Extract text content if it's a list (potentially with file parts)
                    content = message.get("content")
                    if isinstance(content, list):
                         text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                         log_content = f"User said: {' '.join(text_parts)}"
                    elif isinstance(content, str):
                         log_content = f"User said: {content}"
                elif message.get("role") == "assistant":
                     log_content = f"OmniBot response: {message.get('content', '')}" # Handle potential None content
                     # Could add tool call info here if desired
                elif message.get("role") == "tool":
                     log_content = f"Tool result ({message.get('tool_call_id')}): {str(message.get('content', ''))[:200]}..." # Log snippet

                if log_content:
                    self.vector_db.add( # Use self.vector_db
                        log_content,
                        {
                            "type": f"{message.get('role')}_message",
                            "time": datetime.now().isoformat()
                            # Add other relevant metadata if needed
                        }
                    )
            return True # Indicate success
        except Exception as e:
            logger.error(f"Failed to add message to Redis or VectorDB: {e}", exc_info=True)
            return False # Indicate failure

    def get_messages(self):
        """Retrieves recent messages from Redis history, respecting max_tokens."""
        if not self.is_ready():
            logger.error("Cannot get messages: Redis client not initialized.")
            return [self.system_message] if self.system_message else []

        messages_to_return = []
        current_tokens = 0

        try:
            # Always include system message if it exists
            if self.system_message:
                messages_to_return.append(self.system_message)
                current_tokens += self._estimate_tokens(self.system_message)

            # Fetch messages from Redis, starting from the most recent
            # Fetch a large chunk initially, assuming history might be long
            # Adjust the range if performance becomes an issue with extremely long histories
            history_len = self.redis_client.llen(self.CONVERSATION_KEY)
            # Fetch up to ~2x max_tokens worth of characters, assuming 4 chars/token, plus buffer
            # This aims to fetch enough history without fetching millions of items unnecessarily
            estimated_items_needed = (self.max_tokens // self.CHARS_PER_TOKEN_ESTIMATE) * 5
            start_index = max(0, history_len - estimated_items_needed) # Fetch recent chunk
            # If system message is the first item, skip it in this fetch
            if self.system_message and start_index == 0:
                 start_index = 1

            recent_history_json = self.redis_client.lrange(self.CONVERSATION_KEY, start_index, -1) # Get from start_index to end

            # Iterate backwards through the fetched recent history
            for msg_json in reversed(recent_history_json):
                try:
                    message = json.loads(msg_json)
                    message_tokens = self._estimate_tokens(message)

                    # Check if adding this message exceeds the token limit
                    if current_tokens + message_tokens <= self.max_tokens:
                        messages_to_return.insert(1, message) # Insert after system message
                        current_tokens += message_tokens
                    else:
                        # Stop adding messages once the limit is reached
                        logger.debug(f"Token limit ({self.max_tokens}) reached. Returning {len(messages_to_return)} messages.")
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode message from Redis history: {msg_json}")
                except Exception as e:
                     logger.error(f"Error processing message from Redis history: {e}", exc_info=True)


            self.current_token_count = current_tokens # Update tracked token count for context window
            logger.info(f"Retrieved {len(messages_to_return)} messages from Redis history ({self.current_token_count} estimated tokens).")
            return messages_to_return

        except Exception as e:
            logger.error(f"Failed to retrieve messages from Redis: {e}", exc_info=True)
            # Fallback to just system message if retrieval fails
            return [self.system_message] if self.system_message else []

    def is_ready(self):
        """Checks if the Redis client is initialized and connected."""
        if not self.initialized or not self.redis_client:
            return False
        try:
            # Perform a quick check like PING to ensure connectivity
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis connection check failed: {e}")
            self.initialized = False # Mark as not ready if ping fails
            return False
