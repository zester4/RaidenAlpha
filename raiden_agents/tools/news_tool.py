import os
import logging
import requests
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class NewsAPITool(Tool):
    BASE_URL = "https://newsapi.org/v2/"
    
    def __init__(self):
        super().__init__(
            name="get_news",
            description="Fetch news articles from NewsAPI",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "category": {
                        "type": "string",
                        "enum": ["business", "entertainment", "general", "health", "science", "sports", "technology"]
                    },
                    "country": {"type": "string", "pattern": "^[a-z]{2}$"},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 100}
                },
                "anyOf": [
                    {"required": ["query"]},
                    {"required": ["category"]}
                ]
            }
        )
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ToolExecutionError("NewsAPI key missing from environment variables")

    def execute(self, **kwargs):
        try:
            endpoint = "top-headlines" if "category" in kwargs else "everything"
            params = {
                "apiKey": self.api_key,
                **kwargs
            }
            
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] != 'ok':
                raise ToolExecutionError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            
            logger.info(f"Retrieved {len(data['articles'])} news articles")
            return self._format_articles(data['articles'])

        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request failed: {str(e)}")
            raise ToolExecutionError(f"News API request failed: {str(e)}")

    def _format_articles(self, articles):
        return [
            {
                "title": article['title'],
                "description": article['description'],
                "url": article['url'],
                "source": article['source']['name'],
                "published_at": article['publishedAt'],
                "content": article['content']
            }
            for article in articles
        ]