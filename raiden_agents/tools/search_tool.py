import logging
from datetime import datetime
from duckduckgo_search import DDGS
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class SearchTool(Tool):
    def __init__(self): 
        super().__init__(
            name="perform_web_search", 
            description="General web search for facts/current info.", 
            parameters={
                "type": "object", 
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "Search query."
                    }
                }
            }, 
            required=["query"]
        )
    
    def execute(self, **kwargs):
        self.validate_args(kwargs)
        query = kwargs.get("query")
        logger.info(f"Searching: {query}")
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            
            if not results:
                return f"No results for '{query}'."
            
            formatted_results = []
            for result in results:
                text = f"Title: {result.get('title','N/A')}\nSnippet: {result.get('body','N/A')}\nURL: {result.get('href','N/A')}"
                formatted_results.append(text)
                
                # Store in vector DB if available
                try:
                    from __main__ import vector_db
                    if vector_db.is_ready():
                        vector_db.add(
                            f"Search snippet '{query}': {result.get('title', '')} - {result.get('body', '')}", 
                            {
                                "type": "search_result", 
                                "url": result.get('href'), 
                                "query": query, 
                                "time": datetime.now().isoformat()
                            }
                        )
                except ImportError:
                    pass

            return f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise ToolExecutionError(f"Search failed: {e}")
