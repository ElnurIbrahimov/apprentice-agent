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
        'importlib', 'builtins', '__builtin__', '__builtins__',
        'code', 'codeop', 'compileall',
        'pty', 'fcntl', 'termios', 'tty',
        'signal', 'resource', 'sysconfig',
        'asyncio', 'aiohttp', 'httpx',
        'hashlib', 'base64', 'binascii', 'codecs',
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
        '__aura_stdout_cap__', '__aura_stderr_cap__',
    }

    def __init__(self, timeout: int = 30, max_output_length: int = 5000):
        self.timeout = timeout
        self.max_output_length = max_output_length

    def _execute_monty(self, code: str) -> Optional[dict]:
        """Tier 1: Monty safe sandbox for pure computation (no IO, instant).

        Returns dict if Monty is available and code is suitable, None otherwise.
        """
        try:
            from monty import evaluate  # type: ignore
            result = evaluate(code)
            return {
                "success": True,
                "output": str(result),
                "errors": "",
                "sandbox": "monty",
                "code": code,
            }
        except ImportError:
            return None  # Monty not installed — fall through to next tier
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "sandbox": "monty",
                "code": code,
            }

    def _is_pure_computation(self, code: str) -> bool:
        """Heuristic: code is safe for Monty if it has no imports and no IO calls."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False
            if isinstance(node, ast.Call):
                name = self._get_call_name(node)
                if name in {"print", "input", "open", "exec", "eval"}:
                    return False
        return True

    def _execute_e2b(self, code: str) -> Optional[dict]:
        """Tier 2: E2B cloud VM sandbox via SandboxExecutor — real Python + packages, isolated.

        Returns dict if E2B_API_KEY is set, None otherwise (fall through to subprocess).
        """
        try:
            from aura.sandbox import SandboxExecutor
        except ImportError:
            return self._execute_e2b_direct(code)

        if not os.environ.get("E2B_API_KEY", ""):
            return None  # No key — skip this tier

        try:
            if not hasattr(self, '_sandbox') or self._sandbox is None:
                self._sandbox = SandboxExecutor(timeout=self.timeout)

            result = self._sandbox.run_python(code)
            if result.sandbox == "none":
                return None  # SandboxExecutor couldn't run it

            return {
                "success": result.success,
                "output": result.stdout,
                "errors": result.stderr,
                "sandbox": result.sandbox,
                "code": code,
                **({"error": result.error} if result.error else {}),
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "sandbox": "e2b",
                "code": code,
            }

    def _execute_e2b_direct(self, code: str) -> Optional[dict]:
        """Direct E2B execution (fallback if sandbox module unavailable)."""
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore
        except ImportError:
            return None

        api_key = os.environ.get("E2B_API_KEY", "")
        if not api_key:
            return None

        try:
            with Sandbox(api_key=api_key) as sbx:
                execution = sbx.run_code(code)
                output = "\n".join(str(r) for r in execution.results) if execution.results else ""
                error_msg = execution.error.value if execution.error else ""
                return {
                    "success": not bool(execution.error),
                    "output": output,
                    "errors": error_msg,
                    "sandbox": "e2b",
                    "code": code,
                }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "sandbox": "e2b",
                "code": code,
            }

    def execute(self, code: str) -> dict:
        """Execute Python code safely using a three-tier sandbox.

        Tier 1: Monty (pure computation, instant, no IO)
        Tier 2: E2B cloud VM (real Python + packages, requires E2B_API_KEY)
        Tier 3: Subprocess sandbox (AST-checked, offline fallback)
        """
        # Unescape literal \n, \t from LLM output to actual newlines/tabs
        code = self._unescape_code(code)

        # Tier 1: Monty for pure computation
        if self._is_pure_computation(code):
            monty_result = self._execute_monty(code)
            if monty_result is not None:
                return monty_result

        # Tier 2: E2B cloud VM for general code
        e2b_result = self._execute_e2b(code)
        if e2b_result is not None:
            return e2b_result

        # Tier 3: Subprocess sandbox (offline fallback)
        # Check for potentially dangerous operations before subprocess
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
        # SECURITY: sys is NOT imported — user code must not access sys.modules.
        # We capture stdout/stderr refs before exec to print results without
        # exposing the sys module to user code.
        # SECURITY: Check that user code doesn't try to access wrapper internals
        if '__aura_stdout_cap__' in code or '__aura_stderr_cap__' in code:
            return {
                "success": False,
                "error": "Code blocked for safety: references to internal capture variables are not allowed",
                "code": code,
            }

        wrapper_code = f'''
import io as _io
from contextlib import redirect_stdout as _redirect_stdout, redirect_stderr as _redirect_stderr

# Capture references to real stdout/stderr before user code runs
__aura_stderr_cap__ = __import__('sys').stderr
__aura_stdout_cap__ = __import__('sys').stdout

# Capture output
_stdout_capture = _io.StringIO()
_stderr_capture = _io.StringIO()

try:
    with _redirect_stdout(_stdout_capture), _redirect_stderr(_stderr_capture):
        # User code starts here
{self._indent_code(code, 8)}
        # User code ends here

    _output = _stdout_capture.getvalue()
    _errors = _stderr_capture.getvalue()

    if _output:
        __aura_stdout_cap__.write(_output)
    if _errors:
        __aura_stderr_cap__.write(_errors)

except Exception as _e:
    __aura_stderr_cap__.write(f"Error: {{type(_e).__name__}}: {{_e}}\\n")
'''

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_code)
            temp_path = f.name

        proc = None
        try:
            # Run in subprocess with timeout
            # Sanitize environment to avoid leaking API keys/tokens
            safe_env = {k: v for k, v in os.environ.items()
                        if k in ("PATH", "HOME", "USERPROFILE", "TEMP", "TMP",
                                 "SYSTEMROOT", "WINDIR", "COMSPEC", "PYTHONPATH")}
            proc = subprocess.Popen(
                [sys.executable, temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir(),  # Run in temp directory
                env=safe_env,
            )
            try:
                stdout_data, stderr_data = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()  # Drain pipes to avoid zombie
                return {
                    "success": False,
                    "error": f"Code execution timed out after {self.timeout} seconds",
                    "code": code
                }

            stdout = stdout_data[:self.max_output_length] if stdout_data else ""
            stderr = stderr_data[:self.max_output_length] if stderr_data else ""

            if proc.returncode == 0:
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

        Only applies when the code is a single line (suggesting the LLM
        serialized it with literal \\n instead of real newlines). If the code
        already contains real newlines it is left untouched, because blind
        replacement corrupts string literals like "C:\\new_folder" or "col\\ten".
        """
        if '\n' not in code:
            # Single-line input — likely serialized; safe to unescape
            code = code.replace('\\n', '\n')
            code = code.replace('\\t', '\t')
        return code

    def run_math(self, expression: str) -> dict:
        import ast as _ast
        import math as _math

        if not expression or not expression.strip():
            return {"success": False, "output": "", "error": "Empty expression"}

        # Validate the expression is a pure math expression
        try:
            tree = _ast.parse(expression.strip(), mode='eval')
        except SyntaxError as e:
            return {"success": False, "output": "", "error": f"Invalid expression syntax: {e}"}

        ALLOWED_NODES = (
            _ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.BoolOp,
            _ast.Constant,
            _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.Mod, _ast.Pow,
            _ast.FloorDiv, _ast.BitAnd, _ast.BitOr, _ast.BitXor,
            _ast.LShift, _ast.RShift, _ast.Invert, _ast.Not, _ast.UAdd, _ast.USub,
            _ast.Compare, _ast.Eq, _ast.NotEq, _ast.Lt, _ast.LtE, _ast.Gt, _ast.GtE,
            _ast.Name,  # needed for function names and 'math' prefix
            _ast.Attribute,  # needed for math.sqrt etc.
            _ast.Call,
        )
        MATH_FUNCS = {"abs", "round", "min", "max", "sum", "pow", "int", "float"}
        for node in _ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return {"success": False, "output": "", "error": f"Expression contains disallowed construct: {type(node).__name__}"}
            # Block dunder attribute access (e.g. math.__class__.__bases__)
            if isinstance(node, _ast.Attribute):
                if node.attr.startswith("__"):
                    return {"success": False, "output": "", "error": f"Attribute '{node.attr}' not allowed in math expressions"}
            if isinstance(node, _ast.Call):
                if isinstance(node.func, _ast.Name):
                    if node.func.id not in MATH_FUNCS:
                        return {"success": False, "output": "", "error": f"Function '{node.func.id}' not allowed in math expressions"}
                elif isinstance(node.func, _ast.Attribute):
                    if not (isinstance(node.func.value, _ast.Name) and node.func.value.id == "math"):
                        return {"success": False, "output": "", "error": "Only math.* functions allowed"}

        # Evaluate directly in-process with a restricted namespace — avoids spawning a
        # subprocess just for math and bypasses the blocked-'math'-import safety check.
        safe_globals = {"__builtins__": {}, "math": _math}
        safe_locals = {f: getattr(__builtins__, f, None) or getattr(_math, f, None)
                       for f in MATH_FUNCS}
        try:
            result = eval(compile(tree, "<math>", "eval"), safe_globals, safe_locals)  # noqa: S307
            return {"success": True, "output": str(result), "errors": None, "code": expression}
        except ZeroDivisionError:
            return {"success": False, "output": "", "error": "Division by zero"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
