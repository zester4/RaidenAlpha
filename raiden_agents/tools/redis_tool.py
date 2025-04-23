from typing import Any, Dict, List, Optional
from redis import Redis
from .base_tool import Tool, ToolExecutionError

class RedisTool(Tool):
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        super().__init__(
            name="redis",
            description="Redis cache and data structure operations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["SET", "GET", "DELETE", "EXPIRE", "LIST_PUSH", "LIST_POP", "HASH_SET", "HASH_GET"]
                    },
                    "key": {
                        "type": "string",
                        "description": "Redis key"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to store",
                        "optional": True
                    },
                    "expire": {
                        "type": "integer",
                        "description": "Expiration time in seconds",
                        "optional": True
                    }
                },
                "required": ["operation", "key"]
            }
        )
        self.redis = Redis(host=host, port=port, db=db)

    def execute(self, **kwargs) -> str:
        self.validate_args(kwargs)
        operation = kwargs["operation"]
        key = kwargs["key"]

        try:
            if operation == "SET":
                value = kwargs.get("value")
                expire = kwargs.get("expire")
                self.redis.set(key, value, ex=expire)
                return f"Successfully set {key}"

            elif operation == "GET":
                value = self.redis.get(key)
                return str(value) if value else "Key not found"

            elif operation == "DELETE":
                result = self.redis.delete(key)
                return f"Successfully deleted {key}" if result else "Key not found"

            elif operation == "EXPIRE":
                expire = kwargs.get("expire", 3600)
                result = self.redis.expire(key, expire)
                return f"Set expiry for {key}" if result else "Key not found"

            elif operation == "LIST_PUSH":
                value = kwargs.get("value")
                self.redis.lpush(key, value)
                return f"Pushed to list {key}"

            elif operation == "LIST_POP":
                value = self.redis.lpop(key)
                return str(value) if value else "List empty"

            elif operation == "HASH_SET":
                field = kwargs.get("field")
                value = kwargs.get("value")
                self.redis.hset(key, field, value)
                return f"Set hash field {field} in {key}"

            elif operation == "HASH_GET":
                field = kwargs.get("field")
                value = self.redis.hget(key, field)
                return str(value) if value else "Field not found"

            else:
                raise ToolExecutionError(f"Unsupported operation: {operation}")

        except Exception as e:
            raise ToolExecutionError(f"Redis operation failed: {str(e)}")