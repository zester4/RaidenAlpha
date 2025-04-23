import logging
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class AirtableTool(Tool):
    def __init__(self, api_key: str):
        super().__init__(
            name="airtable",
            description="Interact with Airtable bases for data management and automation",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": [
                            "LIST_RECORDS",
                            "GET_RECORD",
                            "CREATE_RECORD",
                            "UPDATE_RECORD",
                            "DELETE_RECORD",
                            "BATCH_CREATE",
                            "BATCH_UPDATE",
                            "QUERY_RECORDS"
                        ]
                    },
                    "base_id": {
                        "type": "string",
                        "description": "The ID of the Airtable base"
                    },
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table"
                    },
                    "record_id": {
                        "type": "string",
                        "description": "The ID of the record (for single record operations)",
                        "optional": True
                    },
                    "fields": {
                        "type": "object",
                        "description": "The fields and values for create/update operations",
                        "optional": True
                    },
                    "filter_by_formula": {
                        "type": "string",
                        "description": "Airtable formula for filtering records",
                        "optional": True
                    },
                    "sort": {
                        "type": "array",
                        "description": "Sorting configuration",
                        "optional": True
                    },
                    "view": {
                        "type": "string",
                        "description": "The view ID or name to use",
                        "optional": True
                    }
                },
                "required": ["operation", "base_id", "table_name"]
            }
        )
        self.api_key = api_key
        self.base_url = "https://api.airtable.com/v0"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _make_request(self, method: str, url: str, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make request to Airtable API with error handling"""
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=json_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ToolExecutionError(f"Airtable API error: {str(e)}")

    def list_records(self, base_id: str, table_name: str, 
                     filter_by_formula: Optional[str] = None,
                     sort: Optional[List[Dict]] = None,
                     view: Optional[str] = None) -> Dict[str, Any]:
        """List records from a table with optional filtering and sorting"""
        url = f"{self.base_url}/{base_id}/{table_name}"
        params = {}
        
        if filter_by_formula:
            params['filterByFormula'] = filter_by_formula
        if sort:
            params['sort'] = sort
        if view:
            params['view'] = view
            
        return self._make_request("GET", url, params)

    def get_record(self, base_id: str, table_name: str, record_id: str) -> Dict[str, Any]:
        """Get a single record by ID"""
        url = f"{self.base_url}/{base_id}/{table_name}/{record_id}"
        return self._make_request("GET", url)

    def create_record(self, base_id: str, table_name: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record"""
        url = f"{self.base_url}/{base_id}/{table_name}"
        return self._make_request("POST", url, {"fields": fields})

    def update_record(self, base_id: str, table_name: str, 
                     record_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing record"""
        url = f"{self.base_url}/{base_id}/{table_name}/{record_id}"
        return self._make_request("PATCH", url, {"fields": fields})

    def delete_record(self, base_id: str, table_name: str, record_id: str) -> Dict[str, Any]:
        """Delete a record"""
        url = f"{self.base_url}/{base_id}/{table_name}/{record_id}"
        return self._make_request("DELETE", url)

    def batch_create(self, base_id: str, table_name: str, 
                    records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create multiple records in a single request"""
        url = f"{self.base_url}/{base_id}/{table_name}"
        return self._make_request("POST", url, {
            "records": [{"fields": record} for record in records]
        })

    def batch_update(self, base_id: str, table_name: str,
                    records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update multiple records in a single request"""
        url = f"{self.base_url}/{base_id}/{table_name}"
        return self._make_request("PATCH", url, {"records": records})

    def query_records(self, base_id: str, table_name: str,
                     filter_by_formula: str) -> Dict[str, Any]:
        """Query records using Airtable formula"""
        return self.list_records(base_id, table_name, filter_by_formula=filter_by_formula)

    def execute(self, **kwargs) -> str:
        """Execute the Airtable tool based on provided parameters"""
        self.validate_args(kwargs)
        operation = kwargs.get("operation")
        base_id = kwargs.get("base_id")
        table_name = kwargs.get("table_name")
        
        try:
            if operation == "LIST_RECORDS":
                result = self.list_records(
                    base_id=base_id,
                    table_name=table_name,
                    filter_by_formula=kwargs.get("filter_by_formula"),
                    sort=kwargs.get("sort"),
                    view=kwargs.get("view")
                )
                
            elif operation == "GET_RECORD":
                record_id = kwargs.get("record_id")
                if not record_id:
                    raise ToolExecutionError("record_id is required for GET_RECORD operation")
                result = self.get_record(base_id, table_name, record_id)
                
            elif operation == "CREATE_RECORD":
                fields = kwargs.get("fields")
                if not fields:
                    raise ToolExecutionError("fields are required for CREATE_RECORD operation")
                result = self.create_record(base_id, table_name, fields)
                
            elif operation == "UPDATE_RECORD":
                record_id = kwargs.get("record_id")
                fields = kwargs.get("fields")
                if not (record_id and fields):
                    raise ToolExecutionError("record_id and fields are required for UPDATE_RECORD operation")
                result = self.update_record(base_id, table_name, record_id, fields)
                
            elif operation == "DELETE_RECORD":
                record_id = kwargs.get("record_id")
                if not record_id:
                    raise ToolExecutionError("record_id is required for DELETE_RECORD operation")
                result = self.delete_record(base_id, table_name, record_id)
                
            elif operation == "BATCH_CREATE":
                records = kwargs.get("fields")
                if not records:
                    raise ToolExecutionError("fields (array of records) is required for BATCH_CREATE operation")
                result = self.batch_create(base_id, table_name, records)
                
            elif operation == "BATCH_UPDATE":
                records = kwargs.get("fields")
                if not records:
                    raise ToolExecutionError("fields (array of records) is required for BATCH_UPDATE operation")
                result = self.batch_update(base_id, table_name, records)
                
            elif operation == "QUERY_RECORDS":
                filter_by_formula = kwargs.get("filter_by_formula")
                if not filter_by_formula:
                    raise ToolExecutionError("filter_by_formula is required for QUERY_RECORDS operation")
                result = self.query_records(base_id, table_name, filter_by_formula)
                
            else:
                raise ToolExecutionError(f"Unsupported operation: {operation}")
                
            return self._format_result(result)
            
        except Exception as e:
            raise ToolExecutionError(f"Error executing Airtable tool: {str(e)}")

    def _format_result(self, result: Dict[str, Any]) -> str:
        """Format the API response for readable output"""
        if not result:
            return "No data available"
            
        if "records" in result:
            record_count = len(result["records"])
            summary = f"Found {record_count} records:\n\n"
            
            for record in result["records"][:5]:  # Show first 5 records
                summary += f"Record ID: {record.get('id')}\n"
                summary += f"Fields: {record.get('fields')}\n\n"
                
            if record_count > 5:
                summary += f"... and {record_count - 5} more records"
                
            return summary
            
        return str(result)