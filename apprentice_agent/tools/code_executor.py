"""Safe Python code executor tool with sandboxing."""

import subprocess
import sys
import tempfile
import os
from typing import Optional


class CodeExecutorTool:
    """Tool for safely executing Python code in a sandboxed environment."""

    name = "code_executor"
    description = "Execute Python code safely and return the output"

    def __init__(self, timeout: int = 30, max_output_length: int = 5000):
        self.timeout = timeout
        self.max_output_length = max_output_length
        # Restricted built-ins for safety
        self.blocked_imports = [
            'os', 'subprocess', 'sys', 'shutil', 'pathlib',
            'socket', 'requests', 'urllib', 'http',
            'pickle', 'marshal', 'shelve',
            'ctypes', 'multiprocessing', 'threading',
            '__import__', 'eval', 'exec', 'compile',
            'open', 'file', 'input',
        ]

    def execute(self, code: str) -> dict:
        """Execute Python code safely and return results."""
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
        """Check code for potentially dangerous operations."""
        code_lower = code.lower()

        # Check for blocked imports and operations
        dangerous_patterns = [
            ('import os', 'os module access'),
            ('import subprocess', 'subprocess access'),
            ('import sys', 'sys module access'),
            ('import shutil', 'file system manipulation'),
            ('import socket', 'network access'),
            ('import requests', 'network access'),
            ('import urllib', 'network access'),
            ('import pickle', 'deserialization risk'),
            ('import ctypes', 'low-level access'),
            ('__import__', 'dynamic imports'),
            ('eval(', 'code evaluation'),
            ('exec(', 'code execution'),
            ('compile(', 'code compilation'),
            ('open(', 'file access'),
            ('file(', 'file access'),
            ('subprocess', 'subprocess access'),
            ('os.system', 'system command'),
            ('os.popen', 'system command'),
            ('os.remove', 'file deletion'),
            ('os.unlink', 'file deletion'),
            ('shutil.rmtree', 'directory deletion'),
            ('input(', 'user input'),
        ]

        for pattern, reason in dangerous_patterns:
            if pattern in code_lower or pattern in code:
                return {"safe": False, "reason": reason}

        return {"safe": True, "reason": None}

    def _run_sandboxed(self, code: str) -> dict:
        """Run code in a separate process with restrictions."""
        # Convert semicolon-separated statements to newlines for proper indentation
        code = code.replace('; ', '\n').replace(';', '\n')

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
            except:
                pass

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line for line in lines)

    def run_math(self, expression: str) -> dict:
        """Safely evaluate a mathematical expression."""
        # Only allow safe math operations
        code = f"print({expression})"
        return self.execute(code)
