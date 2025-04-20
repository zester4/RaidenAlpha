import logging
import json
import ast
import io
import sys
import contextlib
import signal
import threading
import traceback
from types import ModuleType
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class CodeExecutionTool(Tool):
    def __init__(self):
        super().__init__(
            name="code_execution",
            description="Executes Python code in a sandboxed environment with safety measures",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Execution timeout in seconds (default: 10)"
                    }
                },
                "required": ["code"]
            }
        )
        # Blacklisted modules for security
        self.blacklist = {
            'os': ['system', 'popen', 'spawn', 'exec', 'execv', 'execve', 'execvp', 'execvpe'],
            'subprocess': ['*'],
            'sys': ['exit'],
            'builtins': ['exec', 'eval', '__import__'],
            'importlib': ['import_module'],
            'setuptools': ['*'],
            'pip': ['*'],
            'distutils': ['*'],
            'socket': ['*'],
            'sqlite3': ['*'],
            'multiprocessing': ['*'],
            'urllib': ['*'],
            'http': ['*'],
            'ftplib': ['*'],
            'smtplib': ['*']
        }
        
        # Safe module configurations
        self.safe_modules = {
            # Data Processing and Analysis
            'pandas': ['DataFrame', 'Series', 'read_csv', 'read_json', 'concat', 'merge'],
            'numpy': ['array', 'arange', 'linspace', 'zeros', 'ones', 'random', 'mean', 'median', 'std', 'min', 'max'],
            
            # Basic Python Utilities
            'math': '*',
            'random': '*',
            'datetime': '*',
            'json': '*',
            'collections': '*',
            're': '*',
            'itertools': '*',
            'functools': '*',
            'statistics': '*',
            'decimal': '*',
            'fractions': '*',
            'uuid': '*',
            'hashlib': ['md5', 'sha1', 'sha256', 'sha512'],
            
            # Text Processing
            'string': '*',
            'textwrap': '*',
            'difflib': '*',
            
            # Data Structures
            'heapq': '*',
            'bisect': '*',
            'array': '*',
            'enum': '*',
            'typing': '*',
            
            # Serialization
            'csv': ['reader', 'writer', 'DictReader', 'DictWriter'],
            'base64': ['b64encode', 'b64decode'],
            
            # Data Visualization
            'matplotlib.pyplot': ['plot', 'scatter', 'hist', 'bar', 'pie', 'title', 'xlabel', 'ylabel', 'show', 'savefig', 'close'],
            'seaborn': ['scatterplot', 'lineplot', 'histplot', 'boxplot', 'heatmap']
        }

    def is_safe_import(self, node):
        """Check if an import is safe"""
        if isinstance(node, ast.Import):
            return all(self._check_module_safety(name.name) for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            base_module = node.module.split('.')[0] if node.module else ''
            if base_module in self.blacklist:
                return False
            if base_module in self.safe_modules:
                module_config = self.safe_modules[base_module]
                if module_config == '*':
                    return True
                return all(n.name in module_config for n in node.names)
        return True

    def _check_module_safety(self, module_name):
        """Helper to check if a module and its specific imports are safe"""
        base_module = module_name.split('.')[0]
        if base_module in self.blacklist:
            return False
        return base_module in self.safe_modules

    def check_code_safety(self, code):
        """Analyze code for potentially unsafe operations"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Check for blacklisted imports
                if isinstance(node, (ast.Import, ast.ImportFrom)) and not self.is_safe_import(node):
                    raise ToolExecutionError("Unsafe import detected")
                
                # Check for exec/eval calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ['exec', 'eval']:
                        raise ToolExecutionError("exec/eval calls are not allowed")
                
                # Check for file operations
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['write', 'open', 'remove', 'unlink']:
                        raise ToolExecutionError("File operations are restricted")

            return True
        except SyntaxError as e:
            raise ToolExecutionError(f"Syntax error in code: {str(e)}")
        except Exception as e:
            raise ToolExecutionError(f"Code safety check failed: {str(e)}")

    def execute(self, **kwargs):
        self.validate_args(kwargs)
        code = kwargs.get("code")
        timeout = kwargs.get("timeout", 10)

        logger.info(f"Executing code (timeout: {timeout}s):\n{code}")

        # Check code safety
        self.check_code_safety(code)

        # Prepare restricted globals
        restricted_globals = {
            '__builtins__': {
                name: getattr(__builtins__, name)
                for name in dir(__builtins__)
                if name not in ['exec', 'eval', '__import__', 'open']
            }
        }

        # Add safe imports
        for module_name, allowed_items in self.safe_modules.items():
            try:
                module = __import__(module_name)
                if allowed_items == '*':
                    restricted_globals[module_name] = module
                else:
                    restricted_globals[module_name] = {item: getattr(module, item) for item in allowed_items}
            except ImportError:
                pass

        # Capture output
        output = io.StringIO()
        error_output = io.StringIO()

        # Execute with timeout
        result = None
        error = None

        def execute_with_timeout():
            nonlocal result, error
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
                    # Compile and execute in restricted environment
                    compiled_code = compile(code, '<string>', 'exec')
                    exec(compiled_code, restricted_globals)
                    
                    # If there's a value to return, it will be in locals()
                    if '_return_value' in restricted_globals:
                        result = restricted_globals['_return_value']
            except Exception as e:
                error = f"Error: {str(e)}\n{traceback.format_exc()}"

        # Run in thread with timeout
        thread = threading.Thread(target=execute_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            thread.join(0)  # Clean up the thread
            raise ToolExecutionError(f"Code execution timed out after {timeout} seconds")

        # Collect output
        stdout_content = output.getvalue()
        stderr_content = error_output.getvalue()

        # Clean up
        output.close()
        error_output.close()

        # Format response
        response_parts = []
        if stdout_content:
            response_parts.append(f"Output:\n{stdout_content}")
        if stderr_content:
            response_parts.append(f"Errors:\n{stderr_content}")
        if error:
            response_parts.append(error)
        if result is not None:
            response_parts.append(f"Return value: {result}")

        if not response_parts:
            response_parts.append("Code executed successfully with no output")

        # Store in vector DB if available
        try:
            from __main__ import vector_db
            if vector_db.is_ready():
                vector_db.add(
                    f"Code execution:\n{code}",
                    {
                        "type": "code_execution",
                        "code": code,
                        "output": stdout_content,
                        "error": stderr_content,
                        "time": datetime.now().isoformat()
                    }
                )
        except ImportError:
            pass

        return "\n\n".join(response_parts)
