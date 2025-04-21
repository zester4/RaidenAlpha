import logging
from dotenv import load_dotenv
import os
import requests
from datetime import datetime
from .base_tool import Tool, ToolExecutionError, VectorDBError

logger = logging.getLogger("gemini_agent")

load_dotenv()

class VectorSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="semantic_memory_search",
            description="Searches agent's long-term memory (VDB) for relevant info.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for memory."
                    },
                    "results_count": {
                        "type": "integer",
                        "description": "Num results (default: 3)."
                    }
                }
            },
            required=["query"]
        )

    def execute(self, **kwargs):
        self.validate_args(kwargs)
        query = kwargs.get("query")
        count = kwargs.get("results_count", 3)

        try:
            from __main__ import vector_db
            if not vector_db or not vector_db.is_ready():
                raise ToolExecutionError("Vector DB unavailable.")

            try:
                results = vector_db.search(query, top_k=count)
                if not results:
                    return "No relevant info found in memory."

                formatted = [
                    f"Memory {i+1} (Relevance: {r['similarity']:.2f}):\n"
                    f"Metadata: {r.get('metadata', {})}\n"
                    f"Content: {r['text']}"
                    for i, r in enumerate(results)
                ]
                
                return "Semantic Memory Search Results:\n\n" + "\n\n---\n\n".join(formatted)

            except VectorDBError as e:
                logger.error(f"VDB search failed: {e}")
                raise ToolExecutionError(f"Error searching memory: {e}")
            except Exception as e:
                logger.error(f"Unexpected VDB search error: {e}")
                raise ToolExecutionError(f"Unexpected error searching memory: {e}")

        except ImportError:
            raise ToolExecutionError("Vector DB not initialized in main application.")