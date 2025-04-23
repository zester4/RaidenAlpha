import firebase_admin
from firebase_admin import credentials, firestore, auth
from typing import Dict, Any, Optional
from .base_tool import Tool, ToolExecutionError

class FirebaseTool(Tool):
    def __init__(self, credentials_path: str):
        super().__init__(
            name="firebase",
            description="Firebase operations for Authentication and Firestore",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["CREATE_USER", "GET_USER", "UPDATE_USER", "DELETE_USER", 
                                "SET_DOCUMENT", "GET_DOCUMENT", "UPDATE_DOCUMENT", "DELETE_DOCUMENT"]
                    },
                    "collection": {
                        "type": "string",
                        "description": "Firestore collection name",
                        "optional": True
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Document ID",
                        "optional": True
                    },
                    "data": {
                        "type": "object",
                        "description": "Data to write",
                        "optional": True
                    },
                    "email": {
                        "type": "string",
                        "description": "User email",
                        "optional": True
                    },
                    "password": {
                        "type": "string",
                        "description": "User password",
                        "optional": True
                    },
                    "uid": {
                        "type": "string",
                        "description": "User ID",
                        "optional": True
                    }
                },
                "required": ["operation"]
            }
        )
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def execute(self, **kwargs) -> str:
        self.validate_args(kwargs)
        operation = kwargs["operation"]

        try:
            if operation.startswith(("CREATE_", "GET_", "UPDATE_", "DELETE_")):
                if "USER" in operation:
                    return self._handle_auth_operation(operation, **kwargs)
                else:
                    return self._handle_firestore_operation(operation, **kwargs)
            else:
                raise ToolExecutionError(f"Unsupported operation: {operation}")

        except Exception as e:
            raise ToolExecutionError(f"Firebase operation failed: {str(e)}")

    def _handle_auth_operation(self, operation: str, **kwargs) -> str:
        if operation == "CREATE_USER":
            user = auth.create_user(
                email=kwargs.get("email"),
                password=kwargs.get("password")
            )
            return f"User created: {user.uid}"

        elif operation == "GET_USER":
            user = auth.get_user(kwargs.get("uid"))
            return f"User: {user.email}"

        elif operation == "UPDATE_USER":
            user = auth.update_user(
                kwargs.get("uid"),
                email=kwargs.get("email")
            )
            return f"User updated: {user.uid}"

        elif operation == "DELETE_USER":
            auth.delete_user(kwargs.get("uid"))
            return f"User deleted"

    def _handle_firestore_operation(self, operation: str, **kwargs) -> str:
        collection = kwargs.get("collection")
        document_id = kwargs.get("document_id")
        data = kwargs.get("data")

        if operation == "SET_DOCUMENT":
            self.db.collection(collection).document(document_id).set(data)
            return f"Document set: {document_id}"

        elif operation == "GET_DOCUMENT":
            doc = self.db.collection(collection).document(document_id).get()
            return f"Document data: {doc.to_dict()}"

        elif operation == "UPDATE_DOCUMENT":
            self.db.collection(collection).document(document_id).update(data)
            return f"Document updated: {document_id}"

        elif operation == "DELETE_DOCUMENT":
            self.db.collection(collection).document(document_id).delete()
            return f"Document deleted: {document_id}"