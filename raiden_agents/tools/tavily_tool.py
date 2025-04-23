import logging
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class TavilyTool(Tool):
    """Tool for performing Tavily searches and URL content extraction"""
    
    def __init__(self, api_key: str):
        super().__init__(
            name="tavily",
            description="Perform advanced web searches and extract content from URLs using Tavily API",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": ["SEARCH", "EXTRACT"]
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for SEARCH operation",
                        "optional": True
                    },
                    "urls": {
                        "type": "string",
                        "description": "URLs to extract content from for EXTRACT operation",
                        "optional": True
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "Depth of search (basic or advanced)",
                        "enum": ["basic", "advanced"],
                        "optional": True
                    },
                    "extract_depth": {
                        "type": "string",
                        "description": "Depth of extraction (basic or advanced)",
                        "enum": ["basic", "advanced"],
                        "optional": True
                    },
                    "chunks_per_source": {
                        "type": "integer",
                        "description": "Number of text chunks per source",
                        "optional": True
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "optional": True
                    },
                    "include_answer": {
                        "type": "boolean",
                        "description": "Include AI-generated answer",
                        "optional": True
                    },
                    "include_raw_content": {
                        "type": "boolean",
                        "description": "Include raw content in results",
                        "optional": True
                    },
                    "include_images": {
                        "type": "boolean",
                        "description": "Include images in results",
                        "optional": True
                    },
                    "include_image_descriptions": {
                        "type": "boolean",
                        "description": "Include image descriptions",
                        "optional": True
                    },
                    "include_domains": {
                        "type": "array",
                        "description": "List of domains to include in search",
                        "items": {"type": "string"},
                        "optional": True
                    }
                },
                "required": ["operation"]
            }
        )
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to Tavily API with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ToolExecutionError(f"Tavily API error: {str(e)}")
        except ValueError as e:
            raise ToolExecutionError(f"Invalid JSON response: {str(e)}")

    def search(self, 
              query: str,
              search_depth: str = "advanced",
              chunks_per_source: int = 3,
              max_results: int = 3,
              include_answer: bool = True,
              include_raw_content: bool = False,
              include_images: bool = False,
              include_image_descriptions: bool = False,
              include_domains: List[str] = None) -> Dict[str, Any]:
        """
        Perform a Tavily search with specified parameters
        """
        payload = {
            "query": query,
            "search_depth": search_depth,
            "chunks_per_source": chunks_per_source,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
            
        return self._make_request("search", payload)

    def extract(self,
                urls: Union[str, List[str]],
                include_images: bool = False,
                extract_depth: str = "advanced") -> Dict[str, Any]:
        """
        Extract content from URLs using Tavily
        """
        # Handle both single URL string and list of URLs
        if isinstance(urls, str):
            urls = [urls]
            
        payload = {
            "urls": urls,
            "include_images": include_images,
            "extract_depth": extract_depth
        }
        
        return self._make_request("extract", payload)

    def _format_search_results(self, results: Dict[str, Any]) -> str:
        """Format search results for readable output"""
        output = []
        
        if "answer" in results:
            output.append(f"AI Answer: {results['answer']}\n")
            
        if "results" in results:
            output.append("Search Results:")
            for idx, result in enumerate(results["results"], 1):
                output.append(f"\n{idx}. {result.get('title', 'No title')}")
                output.append(f"   URL: {result.get('url', 'No URL')}")
                output.append(f"   Content: {result.get('content', 'No content')}\n")
                
        return "\n".join(output)

    def _format_extract_results(self, results: Dict[str, Any]) -> str:
        """Format extraction results for readable output"""
        output = ["Extracted Content:"]
        
        if isinstance(results, list):
            for idx, result in enumerate(results, 1):
                output.append(f"\nSource {idx}:")
                output.append(f"URL: {result.get('url', 'No URL')}")
                output.append(f"Title: {result.get('title', 'No title')}")
                output.append(f"Content: {result.get('content', 'No content')}\n")
        else:
            output.append(f"URL: {results.get('url', 'No URL')}")
            output.append(f"Title: {results.get('title', 'No title')}")
            output.append(f"Content: {results.get('content', 'No content')}")
            
        return "\n".join(output)

    def execute(self, **kwargs) -> str:
        """Execute the Tavily tool based on provided parameters"""
        self.validate_args(kwargs)
        operation = kwargs.get("operation")
        
        try:
            if operation == "SEARCH":
                if not kwargs.get("query"):
                    raise ToolExecutionError("query is required for SEARCH operation")
                    
                result = self.search(
                    query=kwargs["query"],
                    search_depth=kwargs.get("search_depth", "advanced"),
                    chunks_per_source=kwargs.get("chunks_per_source", 3),
                    max_results=kwargs.get("max_results", 3),
                    include_answer=kwargs.get("include_answer", True),
                    include_raw_content=kwargs.get("include_raw_content", False),
                    include_images=kwargs.get("include_images", False),
                    include_image_descriptions=kwargs.get("include_image_descriptions", False),
                    include_domains=kwargs.get("include_domains", [])
                )
                return self._format_search_results(result)
                
            elif operation == "EXTRACT":
                if not kwargs.get("urls"):
                    raise ToolExecutionError("urls is required for EXTRACT operation")
                    
                result = self.extract(
                    urls=kwargs["urls"],
                    include_images=kwargs.get("include_images", False),
                    extract_depth=kwargs.get("extract_depth", "advanced")
                )
                return self._format_extract_results(result)
                
            else:
                raise ToolExecutionError(f"Unsupported operation: {operation}")
                
        except Exception as e:
            raise ToolExecutionError(f"Error executing Tavily tool: {str(e)}")