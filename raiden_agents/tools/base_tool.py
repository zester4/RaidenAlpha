import logging

logger = logging.getLogger("gemini_agent")

class AgentException(Exception): pass
class ToolExecutionError(AgentException): pass
class APIKeyError(AgentException): pass
class VectorDBError(AgentException): pass
class GitHubToolError(ToolExecutionError): pass

class Tool:
    def __init__(self, name, description, parameters=None, required=None):
        self.name = name
        self.description = description
        if parameters and not isinstance(parameters, dict): raise ValueError("Params must be dict.")
        self.parameters = parameters or {"type": "object", "properties": {}}
        if required and not isinstance(required, list): raise ValueError("Required must be list.")
        self.required = required or []
        if self.required: self.parameters["required"] = self.required

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def validate_args(self, args):
        if not isinstance(args, dict):
            raise ToolExecutionError("Arguments must be a dictionary.")
        missing = [p for p in self.required if p not in args or args[p] is None]
        if missing:
            raise ToolExecutionError(f"Missing required parameters: {', '.join(missing)}")
        return True

    def execute(self, **kwargs):
        """
        Execute the tool's function.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclass must implement 'execute' method")
