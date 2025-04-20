import logging
import requests
import json
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

# Get API key from environment
import os
openweathermap_api_key = os.environ.get("OPENWEATHERMAP_API_KEY")

class WeatherTool(Tool):
    def __init__(self): 
        super().__init__(
            name="get_current_weather", 
            description="Retrieves real-time weather conditions for a specific city.", 
            parameters={
                "type": "object", 
                "properties": { 
                    "location": {"type": "string", "description": "City name."}, 
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temp unit."}
                }
            }, 
            required=["location"]
        )
    
    def execute(self, **kwargs):
        self.validate_args(kwargs)
        location = kwargs.get("location")
        unit = kwargs.get("unit", "celsius")

        if not openweathermap_api_key:
            raise ToolExecutionError("Weather API key missing.")

        url = "http://api.openweathermap.org/data/2.5/weather"
        units = "metric" if unit == "celsius" else "imperial"
        symbol = "°C" if unit == "celsius" else "°F"
        params = {"q": location, "appid": openweathermap_api_key, "units": units}

        retries = 3
        delay = 1
        
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                
                if data.get("cod") != 200:
                    raise ToolExecutionError(f"Weather API Error: {data.get('message', 'Unknown')}")

                main = data.get("main", {})
                weather = data.get("weather", [{}])
                description = weather[0].get('description', "")
                temp = main.get('temp')
                feels_like = main.get('feels_like')
                humidity = main.get('humidity')

                res = f"Weather in {data.get('name', location)}: {description}, Temp: {temp}{symbol} (feels like {feels_like}{symbol}), Humidity: {humidity}%"

                # Store in vector DB if available
                try:
                    from __main__ import vector_db
                    if vector_db.is_ready():
                        vector_db.add(
                            f"Weather: {location}({unit}): {res}", 
                            {
                                "type": "weather", 
                                "location": location, 
                                "time": datetime.now().isoformat()
                            }
                        )
                except ImportError:
                    pass

                return res

            except requests.exceptions.Timeout:
                logger.warning(f"Weather timeout {location} (try {attempt+1}). Retrying...")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    raise ToolExecutionError(f"City '{location}' not found.")
                elif e.response.status_code == 401:
                    raise ToolExecutionError("Invalid Weather API key.")
                else:
                    logger.error(f"Weather HTTP error: {e}")
                    raise ToolExecutionError(f"HTTP error {e.response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Weather network error: {e}")
                raise ToolExecutionError(f"Network error: {e}")
            except Exception as e:
                logger.error(f"Unexpected weather error: {e}")
                raise ToolExecutionError(f"Unexpected error: {e}")

            time.sleep(delay)
            delay *= 2

        raise ToolExecutionError(f"Weather fetch failed after {retries} attempts.")
