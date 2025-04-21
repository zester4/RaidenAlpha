import logging
import requests
import json
from dotenv import load_dotenv
import traceback
from datetime import datetime
from firecrawl import FirecrawlApp
from .base_tool import Tool, ToolExecutionError


load_dotenv()
logger = logging.getLogger("gemini_agent")

# Get API key from environment
import os
firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")

class WebScraperTool(Tool):
    def __init__(self):
        super().__init__(
            name="scrape_website_for_llm",
            description="Fetches main content of a specific URL as Markdown.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape."
                    }
                }
            },
            required=["url"]
        )

    def execute(self, **kwargs):
        self.validate_args(kwargs)
        url = kwargs.get("url")
        logger.info(f"Scraping URL: {url}")
        
        if not firecrawl_api_key:
            raise ToolExecutionError("Firecrawl API key missing.")
            
        try:
            app = FirecrawlApp(api_key=firecrawl_api_key)
            scraped_data = app.scrape_url(url=url, params={'formats': ['markdown']})
            markdown_content = None
            
            if isinstance(scraped_data, dict):
                markdown_content = scraped_data.get('markdown')
            elif isinstance(scraped_data, str):
                markdown_content = scraped_data
                
            if markdown_content:
                logger.info(f"Scrape success: {url}")
                
                # Store in vector DB if available
                try:
                    from __main__ import vector_db
                    if vector_db.is_ready():
                        chunks = self._chunk_content(markdown_content)
                        logger.info(f"Storing {len(chunks)} chunks from {url} in VDB.")
                        for i, chunk in enumerate(chunks):
                            vector_db.add(
                                chunk,
                                {
                                    "type": "web_content",
                                    "url": url,
                                    "chunk": i+1,
                                    "total_chunks": len(chunks),
                                    "time": datetime.now().isoformat()
                                }
                            )
                except ImportError:
                    pass
                    
                return markdown_content
            else:
                error_msg = scraped_data.get('error', 'Markdown content not found or scrape failed.') if isinstance(scraped_data, dict) else "Scrape returned empty/unexpected data."
                logger.warning(f"Scrape failed for {url}: {error_msg}")
                raise ToolExecutionError(f"Scraping failed: {error_msg}")
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"Scrape HTTP error: {e}")
            msg = f"Status {e.response.status_code}"
            try:
                details = e.response.json()
                msg += f". Details: {details.get('error', details.get('message', json.dumps(details)))}"
            except json.JSONDecodeError:
                msg += f". Response: {e.response.text}"
            raise ToolExecutionError(f"Firecrawl API request failed. {msg}")
        except Exception as e:
            logger.error(f"Scrape exception: {e}")
            traceback.print_exc()
            raise ToolExecutionError(f"Unexpected scrape error: {e}")
            
    def _chunk_content(self, content, max_chars=1500, overlap=100):
        """Split content into overlapping chunks for vector storage."""
        if not isinstance(content, str) or not content:
            return []
            
        if len(content) <= max_chars:
            return [content]
            
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + max_chars, len(content))
            chunks.append(content[start:end])
            start += max_chars - overlap
            if start >= len(content):
                break
            start = max(0, start)
            
        return [c for c in chunks if c]
