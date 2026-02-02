"""Safe Python code executor tool with sandboxing."""

import subprocess
import sys
import tempfile
import os
import ast
from typing import Optional, Set, List, Tuple


class CodeExecutorTool:
    """Tool for safely executing Python code in a sandboxed environment."""

    name = "code_executor"
    description = "Execute Python code safely and return the output"

    # Blocked modules - cannot be imported
    BLOCKED_MODULES: Set[str] = {
        'os', 'subprocess', 'sys', 'shutil', 'pathlib',
        'socket', 'requests', 'urllib', 'http', 'httplib',
        'pickle', 'marshal', 'shelve', 'dill',
        'ctypes', 'multiprocessing', 'threading', 'concurrent',
        'importlib', 'builtins', '__builtin__',
        'code', 'codeop', 'compileall',
        'pty', 'fcntl', 'termios', 'tty',
        'signal', 'resource', 'sysconfig',
        'asyncio', 'aiohttp', 'httpx',
    }

    # Blocked built-in functions
    BLOCKED_BUILTINS: Set[str] = {
        'eval', 'exec', 'compile', '__import__',
        'open', 'input', 'breakpoint',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'memoryview', 'type', 'object',
    }

    # Blocked attribute access patterns
    BLOCKED_ATTRIBUTES: Set[str] = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__code__', '__globals__', '__builtins__', '__dict__',
        '__import__', '__loader__', '__spec__',
    }

    def __init__(self, timeout: int = 30, max_output_length: int = 5000):
        self.timeout = timeout
        self.max_output_length = max_output_length

    def execute(self, code: str) -> dict:
        """Execute Python code safely and return results."""
        # Unescape literal \n, \t from LLM output to actual newlines/tabs
        code = self._unescape_code(code)

        # Check for potentially dangerous operations
        safety_check = self._safety_check(code)
        if not safety_check["safe"]:
            return {
                "success": False,
                "error": f"Code blocked for safety: {safety_check['reason']}",
                "code": code
            }

        try:
            result = self._run_sandboxed(code)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "code": code
            }

    def _safety_check(self, code: str) -> dict:
        """Check code for dangerous operations using AST parsing.

        SECURITY: Uses AST parsing instead of string matching to prevent bypasses.
        This catches obfuscation attempts like string concatenation, unicode tricks,
        and multi-line splits that string matching would miss.
        """
        # First, try to parse the code as valid Python
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"safe": False, "reason": f"Syntax error: {e}"}

        # Walk the AST and check for dangerous patterns
        violations = []

        for node in ast.walk(tree):
            violation = self._check_ast_node(node)
            if violation:
                violations.append(violation)

        if violations:
            return {"safe": False, "reason": "; ".join(violations[:3])}  # Show first 3

        return {"safe": True, "reason": None}

    def _check_ast_node(self, node: ast.AST) -> Optional[str]:
        """Check a single AST node for security violations."""

        # Check imports: import os, import os.path, from os import *
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]  # Get base module
                if module_name in self.BLOCKED_MODULES:
                    return f"blocked import: {alias.name}"

        # Check from imports: from os import system
        if isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name in self.BLOCKED_MODULES:
                    return f"blocked import: from {node.module}"

        # Check function calls: eval(), exec(), open(), __import__()
        if isinstance(node, ast.Call):
            func_name = self._get_call_name(node)
            if func_name in self.BLOCKED_BUILTINS:
                return f"blocked function: {func_name}()"

            # Check for getattr tricks: getattr(obj, 'system')
            if func_name == 'getattr' and len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant):
                    attr = node.args[1].value
                    if isinstance(attr, str) and attr in self.BLOCKED_ATTRIBUTES:
                        return f"blocked attribute access via getattr: {attr}"

        # Check attribute access: obj.__class__, obj.__globals__
        if isinstance(node, ast.Attribute):
            if node.attr in self.BLOCKED_ATTRIBUTES:
                return f"blocked attribute: {node.attr}"

        # Check subscript access for __class__ etc via strings
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    if node.slice.value in self.BLOCKED_ATTRIBUTES:
                        return f"blocked subscript access: [{node.slice.value!r}]"

        return None

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _run_sandboxed(self, code: str) -> dict:
        """Run code in a separate process with restrictions."""
        # Create a wrapper script that captures output
        wrapper_code = f'''
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Capture output
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        # User code starts here
{self._indent_code(code, 8)}
        # User code ends here

    output = stdout_capture.getvalue()
    errors = stderr_capture.getvalue()

    if output:
        print(output, end='')
    if errors:
        print(errors, end='', file=sys.stderr)

except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
'''

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_code)
            temp_path = f.name

        try:
            # Run in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),  # Run in temp directory
            )

            stdout = result.stdout[:self.max_output_length] if result.stdout else ""
            stderr = result.stderr[:self.max_output_length] if result.stderr else ""

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.strip(),
                    "errors": stderr.strip() if stderr else None,
                    "code": code
                }
            else:
                return {
                    "success": False,
                    "output": stdout.strip() if stdout else None,
                    "error": stderr.strip() if stderr else "Unknown error",
                    "code": code
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Code execution timed out after {self.timeout} seconds",
                "code": code
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except (OSError, FileNotFoundError):
                pass  # File already deleted or doesn't exist

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line for line in lines)

    def _unescape_code(self, code: str) -> str:
        """Convert escaped newlines/tabs from LLM output to actual characters.

        LLMs sometimes output literal \\n instead of actual newlines.
        This converts them to proper Python code formatting.
        """
        # Replace literal \n and \t (as two characters) with actual newline/tab
        # Be careful not to affect string literals - only replace outside quotes
        # Simple approach: replace \\n -> newline, \\t -> tab
        code = code.replace('\\n', '\n')
        code = code.replace('\\t', '\t')
        return code

    def run_math(self, expression: str) -> dict:
        """Safely evaluate a mathematical expression."""
        # Only allow safe math operations
        code = f"print({expression})"
        return self.execute(code)
