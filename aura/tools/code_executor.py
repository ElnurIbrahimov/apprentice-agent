"""Safe Python code executor tool with sandboxing."""

import ast
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


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

    def __init__(self, timeout: int = 30, max_output_length: int = 5000,
                 allowed_modules: Optional[Set[str]] = None,
                 persist_traces: bool = False):
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.persist_traces = persist_traces
        # When allowed_modules is provided, create a per-instance copy with those removed.
        # The default (None) keeps the class-level blocklist intact for the agent path.
        if allowed_modules:
            self.BLOCKED_MODULES = self.__class__.BLOCKED_MODULES - allowed_modules

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

    def execute(self, code: str, seed: Optional[int] = None) -> dict:
        """Execute Python code safely using a three-tier sandbox.

        Tier 1: Monty (pure computation, instant, no IO)
        Tier 2: E2B cloud VM (real Python + packages, requires E2B_API_KEY)
        Tier 3: Subprocess sandbox (AST-checked, offline fallback)

        Args:
            code: Python source to execute.
            seed: Optional integer — if provided and execution lands on Tier 3
                (the only tier where Aura controls the runtime), the wrapper
                seeds ``random.seed(seed)`` before user code runs. Useful for
                deterministic replay of flaky runs.

        The Tier 3 path additionally captures an execution trace and returns
        it under ``result["trace"]`` with per-line stdout/stderr timestamps,
        a Python-line-event count, and a structured exception block on failure.
        """
        # Unescape literal \n, \t from LLM output to actual newlines/tabs
        code = self._unescape_code(code)

        # SECURITY: AST validation runs BEFORE any execution tier.
        # Previously this only guarded Tier 3 (subprocess), allowing Tier 1
        # (Monty) and Tier 2 (E2B) to execute unchecked LLM-generated code.
        safety_check = self._safety_check(code)
        if not safety_check["safe"]:
            return {
                "success": False,
                "error": f"Code blocked for safety: {safety_check['reason']}",
                "code": code,
            }

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
        # AST safety check already passed at the top of execute().
        try:
            result = self._run_sandboxed(code, seed=seed)
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

    def _run_subprocess(self, script: str, code: str,
                        seed: Optional[int] = None) -> dict:
        """Run a complete Python script in a subprocess with timeout and env sanitization.

        Args:
            script: The full Python script to execute (written to a temp file).
            code:   The original user code (returned in the result dict for reference).
            seed:   Optional deterministic seed, threaded to the wrapper via
                    the ``AURA_EXEC_SEED`` env var.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            temp_path = f.name

        # Known trace path passed via env var so parent and child agree.
        trace_uuid = uuid.uuid4().hex
        trace_path = os.path.join(tempfile.gettempdir(), f"aura_exec_{trace_uuid}.json")

        proc = None
        try:
            # Sanitize environment to avoid leaking API keys/tokens
            safe_env = {k: v for k, v in os.environ.items()
                        if k in ("PATH", "HOME", "USERPROFILE", "TEMP", "TMP",
                                 "SYSTEMROOT", "WINDIR", "COMSPEC")}
            safe_env["AURA_EXEC_TRACE_PATH"] = trace_path
            if seed is not None:
                safe_env["AURA_EXEC_SEED"] = str(seed)

            proc = subprocess.Popen(
                [sys.executable, temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir(),
                env=safe_env,
            )
            try:
                stdout_data, stderr_data = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()  # Drain pipes to avoid zombie
                trace_data = self._read_trace(trace_path)
                return {
                    "success": False,
                    "error": f"Code execution timed out after {self.timeout} seconds",
                    "code": code,
                    **({"trace": trace_data} if trace_data else {}),
                }

            stdout = stdout_data[:self.max_output_length] if stdout_data else ""
            stderr = stderr_data[:self.max_output_length] if stderr_data else ""

            trace_data = self._read_trace(trace_path)

            if proc.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.strip(),
                    "errors": stderr.strip() if stderr else None,
                    "code": code,
                    **({"trace": trace_data} if trace_data else {}),
                }
            else:
                return {
                    "success": False,
                    "output": stdout.strip() if stdout else None,
                    "error": stderr.strip() if stderr else "Unknown error",
                    "code": code,
                    **({"trace": trace_data} if trace_data else {}),
                }

        finally:
            try:
                os.unlink(temp_path)
            except (OSError, FileNotFoundError):
                pass

    def _read_trace(self, trace_path: str) -> Optional[Dict[str, Any]]:
        """Load an execution trace JSON from disk. Deletes it on read unless
        ``self.persist_traces`` is True, in which case the trace is moved under
        ``data/code_traces/`` and its final path is returned in the trace dict."""
        if not os.path.exists(trace_path):
            return None
        try:
            with open(trace_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.debug(f"[CodeExecutor] trace read failed: {e}")
            try:
                os.unlink(trace_path)
            except OSError:
                pass
            return None

        if self.persist_traces:
            try:
                persist_dir = Path("data") / "code_traces"
                persist_dir.mkdir(parents=True, exist_ok=True)
                final_path = persist_dir / f"exec_{int(time.time() * 1000)}.json"
                shutil.move(trace_path, final_path)
                data["log_path"] = str(final_path)
            except Exception as e:
                logger.debug(f"[CodeExecutor] trace persist failed: {e}")
                try:
                    os.unlink(trace_path)
                except OSError:
                    pass
        else:
            try:
                os.unlink(trace_path)
            except OSError:
                pass
        return data

    def _run_sandboxed(self, code: str, seed: Optional[int] = None) -> dict:
        """Run code in a separate process with restrictions.

        Injects a trace-capture preamble before user code so the subprocess
        writes an execution-trace JSON (per-line timestamps on stdout/stderr,
        Python line-event count, structured exception) to ``AURA_EXEC_TRACE_PATH``.
        The parent reads it back in ``_run_subprocess``.
        """
        # SECURITY: sys is NOT imported — user code must not access sys.modules.
        # We capture stdout/stderr refs before exec to print results without
        # exposing the sys module to user code.
        # SECURITY: Check that user code doesn't try to access wrapper internals
        banned_markers = (
            "__aura_stdout_cap__", "__aura_stderr_cap__",
            "_aura_trace", "_aura_sys", "_aura_tb",
            "AURA_EXEC_TRACE_PATH", "AURA_EXEC_SEED",
        )
        for marker in banned_markers:
            if marker in code:
                return {
                    "success": False,
                    "error": f"Code blocked for safety: references to internal variable {marker!r} are not allowed",
                    "code": code,
                }

        wrapper_code = f'''
import io as _io
import json as _aura_json
import os as _aura_os
import sys as _aura_sys
import time as _aura_time
import atexit as _aura_atexit
import tempfile as _aura_tempfile
from contextlib import redirect_stdout as _redirect_stdout, redirect_stderr as _redirect_stderr

# Trace file path: parent passes it via env var so both sides agree.
_aura_trace_path = _aura_os.environ.get("AURA_EXEC_TRACE_PATH") or _aura_os.path.join(
    _aura_tempfile.gettempdir(), f"aura_exec_{{_aura_os.getpid()}}.json"
)
_aura_trace = {{
    "stdout_lines": [],
    "stderr_lines": [],
    "line_count": 0,
    "exception": None,
    "seed": None,
    "start_time": _aura_time.time(),
}}

_aura_line_counter = [0]
def _aura_trace_fn(frame, event, arg):
    if event == "line":
        _aura_line_counter[0] += 1
    return _aura_trace_fn

# Deterministic seed from env (set by parent when execute(..., seed=N) is called)
_aura_seed_env = _aura_os.environ.get("AURA_EXEC_SEED")
if _aura_seed_env:
    try:
        _aura_seed_int = int(_aura_seed_env)
        import random as _aura_rnd
        _aura_rnd.seed(_aura_seed_int)
        _aura_trace["seed"] = _aura_seed_int
    except ValueError:
        pass

def _aura_write_trace():
    _aura_trace["line_count"] = _aura_line_counter[0]
    _aura_trace["duration_s"] = round(_aura_time.time() - _aura_trace["start_time"], 4)
    # Cap collected lines so pathological loops don't balloon the JSON.
    if len(_aura_trace["stdout_lines"]) > 2000:
        _aura_trace["stdout_lines"] = _aura_trace["stdout_lines"][:2000]
    if len(_aura_trace["stderr_lines"]) > 2000:
        _aura_trace["stderr_lines"] = _aura_trace["stderr_lines"][:2000]
    try:
        with open(_aura_trace_path, "w") as _f:
            _aura_json.dump(_aura_trace, _f)
    except Exception:
        pass

_aura_atexit.register(_aura_write_trace)

# Capture references to real stdout/stderr before user code runs
__aura_stderr_cap__ = _aura_sys.stderr
__aura_stdout_cap__ = _aura_sys.stdout

class _AuraCaptureIO(_io.StringIO):
    def __init__(self, bucket, start_ref):
        super().__init__()
        self._bucket = bucket
        self._start = start_ref
    def write(self, s):
        if s:
            self._bucket.append({{"t": round(_aura_time.time() - self._start, 4), "text": s}})
        return super().write(s)

_stdout_capture = _AuraCaptureIO(_aura_trace["stdout_lines"], _aura_trace["start_time"])
_stderr_capture = _AuraCaptureIO(_aura_trace["stderr_lines"], _aura_trace["start_time"])

# Pre-bind traceback module — survives the del inside `with` so the except
# block below can format the user traceback without re-importing.
_aura_tb = __import__("traceback")

_aura_sys.settrace(_aura_trace_fn)
try:
    with _redirect_stdout(_stdout_capture), _redirect_stderr(_stderr_capture):
        # SECURITY: Remove wrapper internals from namespace before user code runs
        # to prevent escalation via __aura_stdout_cap__ -> sys -> os
        del _io, _redirect_stdout, _redirect_stderr
        # SECURITY: Restrict builtins to prevent getattr/eval/exec sandbox escapes
        # e.g. getattr(__builtins__, 'op'+'en') bypasses AST checks
        import builtins as _b
        _safe_builtins = {{k: v for k, v in vars(_b).items() if k not in {{
            'eval', 'exec', 'compile', '__import__', 'open', 'input',
            'breakpoint', 'getattr', 'setattr', 'delattr', 'hasattr',
            'globals', 'locals', 'vars', 'dir', 'memoryview', 'type',
            'object', 'exit', 'quit',
        }}}}
        _safe_builtins['__build_class__'] = _b.__build_class__
        __builtins__ = _safe_builtins
        del _b, _safe_builtins
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
    _aura_trace["exception"] = {{
        "type": type(_e).__name__,
        "message": str(_e),
        "traceback": _aura_tb.format_exc()[-2000:],
    }}
    __aura_stderr_cap__.write(f"Error: {{type(_e).__name__}}: {{_e}}\\n")
finally:
    try:
        _aura_sys.settrace(None)
    except Exception:
        pass
'''

        return self._run_subprocess(wrapper_code, code, seed=seed)

    def _execute_raw(self, full_script: str, user_code: str = "") -> dict:
        """Execute a pre-built script directly in a subprocess (PRIVATE).

        This is for internal callers (like the CodePanel API) that have already:
        1. Validated the user code with their own AST safety check
        2. Wrapped the user code with their own preamble/epilogue

        The full_script is run as-is in a subprocess — no additional safety
        check or wrapper is applied.  The caller is responsible for ensuring
        the user code portion was validated before calling this method.

        Args:
            full_script: Complete Python script ready to execute.
            user_code:   Original user code (for the result dict).
        """
        return self._run_subprocess(full_script, user_code)

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
        # Threads can't be force-killed, so block math.* calls that can burn CPU
        # or memory in pure-Python loops before the watchdog would trip.
        DANGEROUS_MATH_ATTRS = {"factorial", "perm", "comb"}
        for node in _ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return {"success": False, "output": "", "error": f"Expression contains disallowed construct: {type(node).__name__}"}
            # Block dunder attribute access (e.g. math.__class__.__bases__)
            if isinstance(node, _ast.Attribute):
                if node.attr.startswith("__"):
                    return {"success": False, "output": "", "error": f"Attribute '{node.attr}' not allowed in math expressions"}
                if node.attr in DANGEROUS_MATH_ATTRS:
                    return {"success": False, "output": "", "error": f"math.{node.attr} is not allowed (unbounded work)"}
            # Block huge exponents like 10**10**10 which cause CPU hang
            if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Pow):
                if isinstance(node.right, _ast.Constant) and isinstance(node.right.value, (int, float)):
                    if abs(node.right.value) > 10000:
                        return {"success": False, "output": "", "error": "Exponent too large (max 10000)"}
            if isinstance(node, _ast.Call):
                if isinstance(node.func, _ast.Name):
                    if node.func.id not in MATH_FUNCS:
                        return {"success": False, "output": "", "error": f"Function '{node.func.id}' not allowed in math expressions"}
                elif isinstance(node.func, _ast.Attribute):
                    if not (isinstance(node.func.value, _ast.Name) and node.func.value.id == "math"):
                        return {"success": False, "output": "", "error": "Only math.* functions allowed"}

        # Evaluate in-process with a restricted namespace. A subprocess would add
        # ~200ms on Windows per call; math eval is a hot agent path, so we keep it
        # in-process and enforce a wall-clock watchdog. On timeout the caller
        # returns immediately but the worker thread leaks until the expression
        # finishes on its own — DANGEROUS_MATH_ATTRS and the exponent guard are
        # the actual CPU bounds that prevent runaway leaks in practice.
        # Do NOT use `with ThreadPoolExecutor(...)` — its __exit__ blocks until
        # running futures complete, which defeats the timeout.
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as _FuturesTimeout
        safe_globals = {"__builtins__": {}, "math": _math}
        safe_locals = {f: getattr(__builtins__, f, None) or getattr(_math, f, None)
                       for f in MATH_FUNCS}
        compiled = compile(tree, "<math>", "eval")
        def _eval_math():
            return eval(compiled, safe_globals, safe_locals)
        _ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="math-watchdog")
        try:
            result = _ex.submit(_eval_math).result(timeout=self.timeout)
            _ex.shutdown(wait=False)
            return {"success": True, "output": str(result), "errors": None, "code": expression}
        except _FuturesTimeout:
            _ex.shutdown(wait=False, cancel_futures=True)
            return {"success": False, "output": "", "error": f"Math evaluation timed out after {self.timeout}s"}
        except ZeroDivisionError:
            _ex.shutdown(wait=False)
            return {"success": False, "output": "", "error": "Division by zero"}
        except Exception as e:
            _ex.shutdown(wait=False)
            return {"success": False, "output": "", "error": str(e)}
