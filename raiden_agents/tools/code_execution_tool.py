import logging
import json
import ast
import io
import sys
import contextlib
import signal
import threading
import traceback
import resource
import psutil
import inspect
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

@dataclass
class SecurityContext:
    memory_limit: int  # in MB
    cpu_limit: int    # in seconds
    allowed_modules: Dict[str, List[str]]
    forbidden_ast_nodes: Set[str]
    max_iterations: int
    max_recursion_depth: int

class CodeExecutionResult:
    def __init__(self):
        self.stdout: str = ""
        self.stderr: str = ""
        self.return_value: Any = None
        self.execution_time: float = 0
        self.memory_usage: int = 0
        self.error: Optional[str] = None
        self.security_violations: List[str] = []
        self.execution_trace: List[Dict] = []

class CodeAnalyzer:
    def __init__(self):
        self.violations = []
        self.complexity_score = 0
        self.used_names = set()
        
    def analyze(self, node: ast.AST) -> None:
        # Track complexity
        if isinstance(node, (ast.For, ast.While, ast.If, ast.FunctionDef)):
            self.complexity_score += 1
            
        # Track variable usage
        if isinstance(node, ast.Name):
            self.used_names.add(node.id)
            
        # Check for dangerous patterns
        if isinstance(node, ast.Call):
            self._check_call(node)
            
    def _check_call(self, node: ast.Call) -> None:
        # Add sophisticated call checking logic here
        pass

class EnhancedCodeExecutionTool(Tool):
    def __init__(self):
        super().__init__(
            name="enhanced_code_execution",
            description="Executes Python code with advanced security and monitoring",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "number", "description": "Execution timeout in seconds"},
                    "memory_limit": {"type": "number", "description": "Memory limit in MB"},
                    "security_level": {"type": "string", "enum": ["low", "medium", "high"]},
                    "enable_debugging": {"type": "boolean"},
                },
                "required": ["code"]
            }
        )
        
        self.security_contexts = {
            "high": SecurityContext(
                memory_limit=100,
                cpu_limit=5,
                allowed_modules=self._get_restricted_modules(),
                forbidden_ast_nodes={"Delete", "Await", "AsyncFor", "AsyncWith"},
                max_iterations=1000,
                max_recursion_depth=100
            ),
            # Add medium and low security contexts
        }
        
        self.execution_history = []
        self._setup_monitoring()

    def _setup_monitoring(self):
        self.execution_stats = {
            'total_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0,
            'peak_memory_usage': 0
        }

    def _get_restricted_modules(self) -> Dict[str, List[str]]:
        # Enhanced version of your safe_modules with more granular control
        base_modules = {
            # Data Processing and Analysis
            'pandas': {
                'allowed': ['DataFrame', 'Series', 'read_csv', 'read_json'],
                'version_requirements': '>=1.0.0,<2.0.0',
                'security_level': 'medium'
            },
            # Add more modules with detailed configurations
        }
        return base_modules

    def _check_memory_usage(self) -> int:
        process = psutil.Process()
        return process.memory_info().rss // (1024 * 1024)  # Convert to MB

    def _setup_resource_limits(self, security_context: SecurityContext):
        def limit_memory():
            resource.setrlimit(resource.RLIMIT_AS, 
                             (security_context.memory_limit * 1024 * 1024, 
                              security_context.memory_limit * 1024 * 1024))
        
        resource.setrlimit(resource.RLIMIT_CPU, 
                          (security_context.cpu_limit, security_context.cpu_limit))

    def analyze_code(self, code: str, security_context: SecurityContext) -> List[str]:
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            
            for node in ast.walk(tree):
                # Check for forbidden AST nodes
                if node.__class__.__name__ in security_context.forbidden_ast_nodes:
                    raise ToolExecutionError(f"Forbidden operation: {node.__class__.__name__}")
                
                # Analyze node
                analyzer.analyze(node)
                
                # Check complexity
                if analyzer.complexity_score > 100:
                    raise ToolExecutionError("Code too complex")
                
            return analyzer.violations
            
        except Exception as e:
            raise ToolExecutionError(f"Code analysis failed: {str(e)}")

    def create_sandbox(self, security_context: SecurityContext) -> Dict[str, Any]:
        sandbox = {
            '__builtins__': self._create_restricted_builtins(),
        }
        
        # Add allowed modules with proper isolation
        for module_name, config in security_context.allowed_modules.items():
            try:
                module = self._load_module_safely(module_name, config)
                sandbox[module_name] = module
            except ImportError as e:
                logger.warning(f"Failed to load module {module_name}: {e}")
                
        return sandbox

    def _create_restricted_builtins(self) -> Dict[str, Any]:
        safe_builtins = {}
        for name in dir(__builtins__):
            if name not in ['exec', 'eval', '__import__', 'open']:
                safe_builtins[name] = getattr(__builtins__, name)
        return safe_builtins

    def execute(self, **kwargs) -> str:
        self.validate_args(kwargs)
        code = kwargs.get("code")
        security_level = kwargs.get("security_level", "high")
        enable_debugging = kwargs.get("enable_debugging", False)
        
        security_context = self.security_contexts[security_level]
        result = CodeExecutionResult()
        
        # Analyze code before execution
        violations = self.analyze_code(code, security_context)
        if violations:
            raise ToolExecutionError(f"Security violations found: {', '.join(violations)}")

        # Setup execution environment
        sandbox = self.create_sandbox(security_context)
        
        def execute_with_monitoring():
            start_time = datetime.now()
            
            try:
                with contextlib.redirect_stdout(io.StringIO()) as stdout, \
                     contextlib.redirect_stderr(io.StringIO()) as stderr:
                    
                    # Set up resource limits
                    self._setup_resource_limits(security_context)
                    
                    # Execute code
                    exec(compile(code, '<string>', 'exec'), sandbox)
                    
                    result.stdout = stdout.getvalue()
                    result.stderr = stderr.getvalue()
                    
            except Exception as e:
                result.error = f"Error: {str(e)}\n{traceback.format_exc()}"
            finally:
                result.execution_time = (datetime.now() - start_time).total_seconds()
                result.memory_usage = self._check_memory_usage()

        # Execute with timeout and monitoring
        thread = threading.Thread(target=execute_with_monitoring)
        thread.daemon = True
        thread.start()
        thread.join(kwargs.get("timeout", 10))

        if thread.is_alive():
            raise ToolExecutionError(f"Execution timed out")

        # Update execution statistics
        self._update_execution_stats(result)
        
        # Log execution
        self._log_execution(code, result)

        return self._format_result(result)

    def _update_execution_stats(self, result: CodeExecutionResult):
        self.execution_stats['total_executions'] += 1
        if result.error:
            self.execution_stats['failed_executions'] += 1
        
        # Update running averages
        prev_avg = self.execution_stats['average_execution_time']
        n = self.execution_stats['total_executions']
        self.execution_stats['average_execution_time'] = \
            (prev_avg * (n-1) + result.execution_time) / n
        
        # Update peak memory usage
        self.execution_stats['peak_memory_usage'] = max(
            self.execution_stats['peak_memory_usage'],
            result.memory_usage
        )

    def _log_execution(self, code: str, result: CodeExecutionResult):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'code': code,
            'execution_time': result.execution_time,
            'memory_usage': result.memory_usage,
            'error': result.error,
            'success': result.error is None
        }
        self.execution_history.append(log_entry)
        
        # Trim history if too long
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def _format_result(self, result: CodeExecutionResult) -> str:
        parts = []
        
        if result.stdout:
            parts.append(f"Output:\n{result.stdout}")
        if result.stderr:
            parts.append(f"Errors:\n{result.stderr}")
        if result.error:
            parts.append(result.error)
            
        parts.append(f"\nExecution Stats:")
        parts.append(f"Time: {result.execution_time:.2f}s")
        parts.append(f"Memory: {result.memory_usage}MB")
        
        return "\n\n".join(parts)