#api_tools

# raiden_agents/tools/api_integration_tool.py

import requests
import json
import yaml
import jwt
import time
import logging
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urlparse, urljoin
from pydantic import BaseModel, Field
from .base_tool import BaseTool, ToolExecutionError

class APIConfig(BaseModel):
    """API Configuration Model"""
    base_url: str
    auth_type: str = "none"  # none, basic, bearer, oauth2, apikey
    auth_location: str = "header"  # header, query, cookie
    auth_key_name: str = "Authorization"
    headers: Dict[str, str] = Field(default_factory=dict)
    rate_limit: int = 0  # requests per second, 0 for unlimited
    timeout: int = 30
    verify_ssl: bool = True
    retry_attempts: int = 3
    retry_delay: int = 1

class APIResponse(BaseModel):
    """API Response Model"""
    status_code: int
    headers: Dict[str, str]
    content: Any
    response_time: float
    success: bool
    error: Optional[str] = None

class APIIntegrationTool(BaseTool):
    """
    Comprehensive API Integration Tool supporting multiple authentication methods,
    request types, and advanced features like rate limiting and retry logic.
    """

    def __init__(self):
        super().__init__()
        self.name = "api_integration"
        self.description = "Universal API integration tool supporting multiple HTTP methods, auth types, and advanced features"
        self.logger = logging.getLogger("gemini_agent.api_tool")
        self.configs: Dict[str, APIConfig] = {}
        self._last_request_time = {}
        
    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                        "description": "HTTP method for the request"
                    },
                    "endpoint": {
                        "type": "string",
                        "description": "API endpoint (full URL or path to append to base_url)"
                    },
                    "config_id": {
                        "type": "string",
                        "description": "Identifier for pre-configured API settings"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Additional headers for the request",
                        "additionalProperties": {"type": "string"}
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters",
                        "additionalProperties": True
                    },
                    "data": {
                        "type": "object",
                        "description": "Request body data",
                        "additionalProperties": True
                    },
                    "auth": {
                        "type": "object",
                        "description": "Authentication details if not using config_id",
                        "properties": {
                            "type": {"type": "string", "enum": ["none", "basic", "bearer", "oauth2", "apikey"]},
                            "credentials": {"type": "object", "additionalProperties": True}
                        }
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds",
                        "default": 30
                    },
                    "validate_schema": {
                        "type": "boolean",
                        "description": "Whether to validate response against OpenAPI schema if available",
                        "default": True
                    }
                },
                "required": ["method", "endpoint"]
            }
        }

    def configure_api(self, config_id: str, config: Dict[str, Any]) -> None:
        """
        Configure an API integration with specific settings
        """
        try:
            api_config = APIConfig(**config)
            self.configs[config_id] = api_config
            self.logger.info(f"API configuration '{config_id}' registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to configure API '{config_id}': {str(e)}")
            raise ToolExecutionError(f"API configuration error: {str(e)}")

    def _handle_rate_limiting(self, config_id: str, rate_limit: int) -> None:
        """
        Implement rate limiting logic
        """
        if rate_limit > 0:
            current_time = time.time()
            if config_id in self._last_request_time:
                elapsed = current_time - self._last_request_time[config_id]
                if elapsed < (1.0 / rate_limit):
                    sleep_time = (1.0 / rate_limit) - elapsed
                    time.sleep(sleep_time)
            self._last_request_time[config_id] = time.time()

    def _prepare_auth(self, config: APIConfig, auth_override: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Prepare authentication headers or parameters based on configuration
        """
        auth_headers = {}
        
        # Use override if provided
        if auth_override:
            auth_type = auth_override.get("type", "none")
            credentials = auth_override.get("credentials", {})
        else:
            auth_type = config.auth_type
            credentials = {}  # Would be loaded from secure storage in production

        try:
            if auth_type == "basic":
                import base64
                auth_str = base64.b64encode(
                    f"{credentials['username']}:{credentials['password']}".encode()
                ).decode()
                auth_headers["Authorization"] = f"Basic {auth_str}"
                
            elif auth_type == "bearer":
                auth_headers["Authorization"] = f"Bearer {credentials['token']}"
                
            elif auth_type == "oauth2":
                # Implement OAuth2 flow (simplified version)
                token = self._get_oauth2_token(credentials)
                auth_headers["Authorization"] = f"Bearer {token}"
                
            elif auth_type == "apikey":
                if config.auth_location == "header":
                    auth_headers[config.auth_key_name] = credentials['apikey']
                # Handle query and cookie auth in the request preparation
                
        except Exception as e:
            self.logger.error(f"Authentication preparation failed: {str(e)}")
            raise ToolExecutionError(f"Authentication error: {str(e)}")
            
        return auth_headers

    def _get_oauth2_token(self, credentials: Dict[str, Any]) -> str:
        """
        Implement OAuth2 token acquisition
        """
        try:
            # Simplified OAuth2 implementation - extend based on your needs
            token_url = credentials.get("token_url")
            client_id = credentials.get("client_id")
            client_secret = credentials.get("client_secret")
            scope = credentials.get("scope", "")

            response = requests.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": scope
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["access_token"]
        except Exception as e:
            raise ToolExecutionError(f"OAuth2 token acquisition failed: {str(e)}")

    def _validate_response(self, response: requests.Response, schema: Optional[Dict] = None) -> None:
        """
        Validate response against schema if provided
        """
        if not schema:
            return

        try:
            from jsonschema import validate
            validate(instance=response.json(), schema=schema)
        except Exception as e:
            self.logger.warning(f"Response validation failed: {str(e)}")
            # Don't raise error, just log warning

    def _prepare_request_url(self, config: APIConfig, endpoint: str) -> str:
        """
        Prepare the full request URL
        """
        if endpoint.startswith(('http://', 'https://')):
            return endpoint
        return urljoin(config.base_url, endpoint.lstrip('/'))

    def execute(self, method: str, endpoint: str, config_id: Optional[str] = None,
                headers: Optional[Dict[str, str]] = None, params: Optional[Dict] = None,
                data: Optional[Dict] = None, auth: Optional[Dict] = None,
                timeout: Optional[int] = None, validate_schema: bool = True) -> APIResponse:
        """
        Execute an API request with comprehensive error handling and features
        """
        start_time = time.time()
        
        try:
            # Get configuration
            config = self.configs.get(config_id, APIConfig(base_url="")) if config_id else APIConfig(base_url="")
            
            # Handle rate limiting
            self._handle_rate_limiting(config_id or "default", config.rate_limit)
            
            # Prepare request components
            url = self._prepare_request_url(config, endpoint)
            timeout = timeout or config.timeout
            
            # Merge headers
            request_headers = config.headers.copy()
            if headers:
                request_headers.update(headers)
            
            # Add authentication
            auth_headers = self._prepare_auth(config, auth)
            request_headers.update(auth_headers)
            
            # Prepare request kwargs
            request_kwargs = {
                "headers": request_headers,
                "params": params,
                "timeout": timeout,
                "verify": config.verify_ssl
            }
            
            # Add body data if present
            if data:
                if method in ["POST", "PUT", "PATCH"]:
                    content_type = request_headers.get("Content-Type", "").lower()
                    if "application/json" in content_type:
                        request_kwargs["json"] = data
                    else:
                        request_kwargs["data"] = data

            # Execute request with retry logic
            for attempt in range(config.retry_attempts):
                try:
                    response = requests.request(method, url, **request_kwargs)
                    
                    # Validate response if schema is available and validation is requested
                    if validate_schema and hasattr(self, 'openapi_schema'):
                        self._validate_response(response, self.openapi_schema)
                    
                    # Calculate response time
                    response_time = time.time() - start_time
                    
                    # Prepare response
                    try:
                        content = response.json()
                    except (ValueError, json.JSONDecodeError):
                        content = response.text

                    api_response = APIResponse(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        content=content,
                        response_time=response_time,
                        success=response.ok,
                        error=None if response.ok else f"HTTP {response.status_code}: {response.reason}"
                    )
                    
                    # Log response
                    log_level = logging.INFO if response.ok else logging.WARNING
                    self.logger.log(log_level, 
                        f"API request completed: {method} {url} -> {response.status_code} "
                        f"(Time: {response_time:.2f}s)")
                    
                    return api_response
                    
                except requests.RequestException as e:
                    if attempt == config.retry_attempts - 1:
                        raise
                    time.sleep(config.retry_delay * (attempt + 1))
                    
        except Exception as e:
            error_message = f"API request failed: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return APIResponse(
                status_code=0,
                headers={},
                content=None,
                response_time=time.time() - start_time,
                success=False,
                error=error_message
            )

    def load_openapi_spec(self, spec_path: str) -> None:
        """
        Load OpenAPI specification for response validation
        """
        try:
            with open(spec_path, 'r') as f:
                self.openapi_schema = yaml.safe_load(f)
            self.logger.info(f"Loaded OpenAPI specification from {spec_path}")
        except Exception as e:
            self.logger.error(f"Failed to load OpenAPI spec: {str(e)}")
            raise ToolExecutionError(f"OpenAPI spec loading failed: {str(e)}")