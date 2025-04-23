import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class AlphaVantageTool(Tool):
    def __init__(self, api_key: str):
        super().__init__(
            name="alpha_vantage",
            description="Access financial market data including stocks, crypto, forex, and technical indicators",
            parameters={
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "description": "The type of data to fetch",
                        "enum": [
                            "STOCK_QUOTE",
                            "CRYPTO_PRICE",
                            "FOREX_RATE",
                            "STOCK_INTRADAY",
                            "CRYPTO_DAILY",
                            "CURRENCY_CONVERT",
                            "COMPANY_OVERVIEW",
                            "GLOBAL_MARKET_STATUS"
                        ]
                    },
                    "symbol": {
                        "type": "string",
                        "description": "The symbol to look up (stock ticker, crypto symbol, or currency pair)"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Time interval between data points",
                        "enum": ["1min", "5min", "15min", "30min", "60min", "daily"],
                        "optional": True
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "Source currency for conversion",
                        "optional": True
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Target currency for conversion",
                        "optional": True
                    }
                },
                "required": ["function", "symbol"]
            }
        )
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make request to Alpha Vantage API with error handling"""
        try:
            params['apikey'] = self.api_key
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                raise ToolExecutionError(f"Alpha Vantage API error: {data['Error Message']}")
            if "Note" in data:  # API call frequency warning
                logger.warning(f"Alpha Vantage API note: {data['Note']}")
                
            return data
        except requests.exceptions.RequestException as e:
            raise ToolExecutionError(f"Failed to fetch data from Alpha Vantage: {str(e)}")

    def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote"""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol
        }
        data = self._make_request(params)
        return data.get("Global Quote", {})

    def get_crypto_price(self, symbol: str) -> Dict[str, Any]:
        """Get current cryptocurrency price"""
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": symbol,
            "to_currency": "USD"
        }
        data = self._make_request(params)
        return data.get("Realtime Currency Exchange Rate", {})

    def get_forex_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Get current forex exchange rate"""
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency
        }
        return self._make_request(params)

    def get_intraday_data(self, symbol: str, interval: str = "5min") -> Dict[str, Any]:
        """Get intraday stock data"""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "compact"
        }
        return self._make_request(params)

    def get_crypto_daily(self, symbol: str) -> Dict[str, Any]:
        """Get daily cryptocurrency data"""
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": "USD"
        }
        return self._make_request(params)

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental data"""
        params = {
            "function": "OVERVIEW",
            "symbol": symbol
        }
        return self._make_request(params)

    def format_market_data(self, data: Dict[str, Any], data_type: str) -> str:
        """Format market data for readable output"""
        if not data:
            return "No data available"

        if data_type == "stock_quote":
            return (f"Stock Quote for {data.get('01. symbol', 'Unknown')}:\n"
                   f"Price: ${data.get('05. price', 'N/A')}\n"
                   f"Change: {data.get('09. change', 'N/A')} ({data.get('10. change percent', 'N/A')})\n"
                   f"Volume: {data.get('06. volume', 'N/A')}")

        elif data_type == "crypto":
            return (f"Cryptocurrency Rate:\n"
                   f"From: {data.get('1. From_Currency Code', 'Unknown')}\n"
                   f"Price: ${data.get('5. Exchange Rate', 'N/A')}\n"
                   f"Last Updated: {data.get('6. Last Refreshed', 'N/A')}")

        return str(data)

    def execute(self, **kwargs) -> str:
        """Execute the Alpha Vantage tool based on provided parameters"""
        self.validate_args(kwargs)
        function = kwargs.get("function")
        symbol = kwargs.get("symbol")
        
        try:
            if function == "STOCK_QUOTE":
                data = self.get_stock_quote(symbol)
                return self.format_market_data(data, "stock_quote")
                
            elif function == "CRYPTO_PRICE":
                data = self.get_crypto_price(symbol)
                return self.format_market_data(data, "crypto")
                
            elif function == "FOREX_RATE":
                from_currency = kwargs.get("from_currency")
                to_currency = kwargs.get("to_currency")
                if not (from_currency and to_currency):
                    raise ToolExecutionError("Both from_currency and to_currency are required for FOREX_RATE")
                data = self.get_forex_rate(from_currency, to_currency)
                return self.format_market_data(data, "forex")
                
            elif function == "STOCK_INTRADAY":
                interval = kwargs.get("interval", "5min")
                data = self.get_intraday_data(symbol, interval)
                return self.format_market_data(data, "intraday")
                
            elif function == "CRYPTO_DAILY":
                data = self.get_crypto_daily(symbol)
                return self.format_market_data(data, "crypto_daily")
                
            elif function == "COMPANY_OVERVIEW":
                data = self.get_company_overview(symbol)
                return self.format_market_data(data, "company")
                
            else:
                raise ToolExecutionError(f"Unsupported function: {function}")
                
        except Exception as e:
            raise ToolExecutionError(f"Error executing Alpha Vantage tool: {str(e)}")