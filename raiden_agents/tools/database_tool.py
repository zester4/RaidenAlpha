# raiden_agents/tools/database_tool.py

import logging
from typing import Dict, List, Any, Optional, Union
import json
from pathlib import Path
import asyncio
from .base_tool import BaseTool, ToolExecutionError

class DatabaseTool(BaseTool):
    """Comprehensive database operations tool supporting PostgreSQL, MongoDB, and SQLite"""

    def __init__(self):
        super().__init__()
        self.name = "database_operations"
        self.description = "Execute database operations across PostgreSQL, MongoDB, and SQLite"
        self.logger = logging.getLogger("gemini_agent.database_tool")
        self.connections = {}
        self._initialize_drivers()

    def _initialize_drivers(self):
        """Initialize database drivers with error handling"""
        try:
            # PostgreSQL
            import psycopg2
            self.psycopg2 = psycopg2
        except ImportError:
            self.logger.warning("PostgreSQL driver not available")
            self.psycopg2 = None

        try:
            # MongoDB
            import pymongo
            self.pymongo = pymongo
        except ImportError:
            self.logger.warning("MongoDB driver not available")
            self.pymongo = None

        try:
            # SQLite
            import aiosqlite
            self.aiosqlite = aiosqlite
        except ImportError:
            self.logger.warning("SQLite driver not available")
            self.aiosqlite = None

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "connect",
                            "query",
                            "execute",
                            "backup",
                            "restore",
                            "close"
                        ],
                        "description": "Database operation to perform"
                    },
                    "db_type": {
                        "type": "string",
                        "enum": ["postgresql", "mongodb", "sqlite"],
                        "description": "Type of database to operate on"
                    },
                    "connection_params": {
                        "type": "object",
                        "description": "Database connection parameters",
                        "additionalProperties": True
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query or MongoDB command to execute"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Query parameters or additional options",
                        "additionalProperties": True
                    }
                },
                "required": ["action", "db_type"]
            }
        }

    def execute(self, action: str, db_type: str,
                connection_params: Optional[Dict] = None,
                query: Optional[str] = None,
                parameters: Optional[Dict] = None) -> Dict:
        """Execute database operations with comprehensive error handling"""
        
        try:
            # Map actions to methods
            action_map = {
                "connect": self._connect,
                "query": self._query,
                "execute": self._execute,
                "backup": self._backup,
                "restore": self._restore,
                "close": self._close
            }

            if action not in action_map:
                raise ToolExecutionError(f"Unsupported action: {action}")

            # Execute the action
            result = action_map[action](
                db_type=db_type,
                connection_params=connection_params,
                query=query,
                parameters=parameters or {}
            )

            return {
                "success": True,
                "action": action,
                "db_type": db_type,
                "result": result
            }

        except Exception as e:
            error_msg = f"Database operation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "action": action,
                "db_type": db_type,
                "error": error_msg
            }

    def _connect(self, db_type: str, connection_params: Dict, **kwargs) -> Dict:
        """Establish database connection"""
        conn_id = f"{db_type}_{connection_params.get('database', 'default')}"
        
        try:
            if db_type == "postgresql":
                if not self.psycopg2:
                    raise ToolExecutionError("PostgreSQL driver not available")
                
                conn = self.psycopg2.connect(
                    dbname=connection_params.get('database'),
                    user=connection_params.get('user'),
                    password=connection_params.get('password'),
                    host=connection_params.get('host', 'localhost'),
                    port=connection_params.get('port', 5432)
                )
                
            elif db_type == "mongodb":
                if not self.pymongo:
                    raise ToolExecutionError("MongoDB driver not available")
                
                client = self.pymongo.MongoClient(
                    host=connection_params.get('host', 'localhost'),
                    port=connection_params.get('port', 27017),
                    username=connection_params.get('user'),
                    password=connection_params.get('password')
                )
                conn = client[connection_params.get('database', 'default')]
                
            elif db_type == "sqlite":
                if not self.aiosqlite:
                    raise ToolExecutionError("SQLite driver not available")
                
                db_path = connection_params.get('database', ':memory:')
                conn = asyncio.get_event_loop().run_until_complete(
                    self.aiosqlite.connect(db_path)
                )
                
            else:
                raise ToolExecutionError(f"Unsupported database type: {db_type}")

            self.connections[conn_id] = {
                "connection": conn,
                "type": db_type,
                "params": connection_params
            }

            return {
                "message": f"Successfully connected to {db_type} database",
                "connection_id": conn_id
            }

        except Exception as e:
            raise ToolExecutionError(f"Connection failed: {str(e)}")

    def _query(self, db_type: str, query: str, 
               connection_params: Optional[Dict] = None,
               parameters: Optional[Dict] = None, **kwargs) -> Dict:
        """Execute database queries"""
        conn_id = f"{db_type}_{connection_params.get('database', 'default')}"
        
        try:
            if conn_id not in self.connections:
                self._connect(db_type, connection_params)

            conn_info = self.connections[conn_id]
            conn = conn_info["connection"]

            if db_type == "postgresql":
                with conn.cursor() as cur:
                    cur.execute(query, parameters or {})
                    if cur.description:  # Select query
                        columns = [desc[0] for desc in cur.description]
                        results = [dict(zip(columns, row)) for row in cur.fetchall()]
                    else:  # Non-select query
                        results = {"affected_rows": cur.rowcount}
                    conn.commit()

            elif db_type == "mongodb":
                collection = parameters.get("collection")
                if not collection:
                    raise ToolExecutionError("MongoDB operations require a collection name")
                
                # Parse MongoDB query from string
                mongo_query = json.loads(query)
                results = list(conn[collection].find(mongo_query))
                
                # Convert ObjectId to string for JSON serialization
                for doc in results:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])

            elif db_type == "sqlite":
                async def execute_sqlite_query():
                    async with conn.execute(query, parameters or {}) as cur:
                        if cur.description:  # Select query
                            rows = await cur.fetchall()
                            columns = [desc[0] for desc in cur.description]
                            return [dict(zip(columns, row)) for row in rows]
                        else:  # Non-select query
                            return {"affected_rows": cur.rowcount}

                results = asyncio.get_event_loop().run_until_complete(execute_sqlite_query())
                await conn.commit()

            return {
                "results": results,
                "query": query,
                "parameters": parameters
            }

        except Exception as e:
            raise ToolExecutionError(f"Query execution failed: {str(e)}")

    def _execute(self, db_type: str, query: str,
                 connection_params: Optional[Dict] = None,
                 parameters: Optional[Dict] = None, **kwargs) -> Dict:
        """Execute database modifications"""
        return self._query(
            db_type=db_type,
            query=query,
            connection_params=connection_params,
            parameters=parameters
        )

    def _backup(self, db_type: str, connection_params: Dict,
                parameters: Optional[Dict] = None, **kwargs) -> Dict:
        """Backup database"""
        try:
            backup_path = parameters.get("backup_path")
            if not backup_path:
                raise ToolExecutionError("Backup path not specified")

            if db_type == "postgresql":
                import subprocess
                
                cmd = [
                    "pg_dump",
                    "-h", connection_params.get("host", "localhost"),
                    "-p", str(connection_params.get("port", 5432)),
                    "-U", connection_params.get("user"),
                    "-d", connection_params.get("database"),
                    "-f", backup_path
                ]
                
                env = os.environ.copy()
                env["PGPASSWORD"] = connection_params.get("password")
                
                subprocess.run(cmd, env=env, check=True)

            elif db_type == "mongodb":
                cmd = [
                    "mongodump",
                    "--host", connection_params.get("host", "localhost"),
                    "--port", str(connection_params.get("port", 27017)),
                    "--db", connection_params.get("database"),
                    "--out", backup_path
                ]
                
                if connection_params.get("user"):
                    cmd.extend([
                        "--username", connection_params["user"],
                        "--password", connection_params["password"]
                    ])
                
                subprocess.run(cmd, check=True)

            elif db_type == "sqlite":
                import shutil
                source_db = connection_params.get("database")
                if source_db and source_db != ":memory:":
                    shutil.copy2(source_db, backup_path)
                else:
                    raise ToolExecutionError("Cannot backup in-memory SQLite database")

            return {
                "message": f"Successfully backed up {db_type} database",
                "backup_path": backup_path
            }

        except Exception as e:
            raise ToolExecutionError(f"Backup failed: {str(e)}")

    def _restore(self, db_type: str, connection_params: Dict,
                 parameters: Optional[Dict] = None, **kwargs) -> Dict:
        """Restore database from backup"""
        try:
            backup_path = parameters.get("backup_path")
            if not backup_path:
                raise ToolExecutionError("Backup path not specified")

            if db_type == "postgresql":
                import subprocess
                
                cmd = [
                    "psql",
                    "-h", connection_params.get("host", "localhost"),
                    "-p", str(connection_params.get("port", 5432)),
                    "-U", connection_params.get("user"),
                    "-d", connection_params.get("database"),
                    "-f", backup_path
                ]
                
                env = os.environ.copy()
                env["PGPASSWORD"] = connection_params.get("password")
                
                subprocess.run(cmd, env=env, check=True)

            elif db_type == "mongodb":
                cmd = [
                    "mongorestore",
                    "--host", connection_params.get("host", "localhost"),
                    "--port", str(connection_params.get("port", 27017)),
                    "--db", connection_params.get("database"),
                    backup_path
                ]
                
                if connection_params.get("user"):
                    cmd.extend([
                        "--username", connection_params["user"],
                        "--password", connection_params["password"]
                    ])
                
                subprocess.run(cmd, check=True)

            elif db_type == "sqlite":
                import shutil
                target_db = connection_params.get("database")
                if target_db and target_db != ":memory:":
                    shutil.copy2(backup_path, target_db)
                else:
                    raise ToolExecutionError("Cannot restore to in-memory SQLite database")

            return {
                "message": f"Successfully restored {db_type} database",
                "restore_path": backup_path
            }

        except Exception as e:
            raise ToolExecutionError(f"Restore failed: {str(e)}")

    def _close(self, db_type: str, connection_params: Dict, **kwargs) -> Dict:
        """Close database connection"""
        conn_id = f"{db_type}_{connection_params.get('database', 'default')}"
        
        try:
            if conn_id in self.connections:
                conn_info = self.connections[conn_id]
                conn = conn_info["connection"]

                if db_type == "postgresql":
                    conn.close()
                elif db_type == "mongodb":
                    conn.client.close()
                elif db_type == "sqlite":
                    asyncio.get_event_loop().run_until_complete(conn.close())

                del self.connections[conn_id]

                return {
                    "message": f"Successfully closed {db_type} connection",
                    "connection_id": conn_id
                }
            else:
                return {
                    "message": f"No active connection found for {conn_id}"
                }

        except Exception as e:
            raise ToolExecutionError(f"Connection closure failed: {str(e)}")