"""Security validation for custom tool code and LLM-generated scripts.

Prevents arbitrary code execution via malicious custom tools or
LLM-generated code blocks.  All checks are deterministic (no LLM call).

Extracted from aura.agent (2026-03-23).
"""

import ast
import re
from typing import Tuple

# Allowed imports for custom tools
ALLOWED_TOOL_IMPORTS = {
    "typing", "dataclasses", "json", "re", "datetime",
    "pathlib", "collections", "enum", "abc", "math",
    "itertools", "functools", "operator", "string",
    "urllib.parse",  # urllib.parse is safe (URL encoding/decoding)
}

# Forbidden patterns that indicate potentially malicious code
FORBIDDEN_PATTERNS = [
    "os.system", "subprocess", "eval(", "exec(", "__import__",
    "shutil.rmtree", "shutil.move", "socket", "requests.get",
    "urllib.request", "urllib.urlopen", "importlib", "ctypes", "pickle", "marshal",
    "compile(", "globals(", "locals(", "vars(", "open(",
    "__builtins__", "__code__", "__class__",
    "type(", "dir(",
    "__getattribute__", "__subclasses__",
]


def _normalize_code_for_check(code: str) -> str:
    """Normalize code before forbidden-pattern checks.

    Collapses whitespace, strips comments, and joins string concatenations
    so that tricks like '"sub" + "process"' are caught.
    """
    # Remove single-line comments
    lines = []
    for line in code.splitlines():
        stripped = line.split("#", 1)[0]
        lines.append(stripped)
    code = " ".join(lines)
    # Collapse whitespace
    code = re.sub(r"\s+", " ", code)
    # Join adjacent string literals: "sub" + "process" -> "subprocess"
    # Also handles mixed quotes: "sub" + 'process', 'sub' + "process"
    code = re.sub(r"""["']\s*\+\s*["']""", '', code)
    return code


def _extract_string_constants(node: ast.AST) -> list:
    """Recursively extract all string constants from an AST expression.

    Handles: Constant("str"), IfExp branches, BoolOp values.
    Returns a list of possible string values the expression could produce.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        # f"sub{'process' if x else ''}" — check both branches
        return _extract_string_constants(node.body) + _extract_string_constants(node.orelse)
    if isinstance(node, ast.BoolOp):
        # 'a' or 'b' — check all values
        result = []
        for v in node.values:
            result.extend(_extract_string_constants(v))
        return result
    return []


def _check_fstring_evasion(tree: ast.AST, forbidden: list, source: str) -> Tuple[bool, str]:
    """Check f-strings for forbidden pattern evasion.

    Walks the AST for JoinedStr (f-string) nodes and reconstructs the
    constant parts to detect smuggled forbidden strings like f"sub{'process'}".
    Also handles conditional expressions: f"sub{'process' if x else ''}".
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        # Collect constant string parts and substitute a placeholder for
        # FormattedValue nodes, then check each possible expansion.
        # Strategy: concatenate all Constant children (ignoring FormattedValue)
        # and also try concatenating Constant + the string literal inside
        # FormattedValue if the latter is itself a Constant string or conditional.
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append([value.value])
            elif isinstance(value, ast.FormattedValue):
                extracted = _extract_string_constants(value.value)
                if extracted:
                    parts.append(extracted)
                else:
                    # Unknown expression — insert separator
                    parts.append(["\x00"])
            else:
                parts.append(["\x00"])

        # Build all possible reconstructions (cartesian product of branches)
        # Limit combinations to prevent explosion (max 32 paths)
        from itertools import product
        combos = list(product(*parts))[:32]
        for combo in combos:
            reconstructed = "".join(combo)
            for pattern in forbidden:
                if pattern in reconstructed:
                    return False, f"Forbidden pattern '{pattern}' found in f-string in {source}"
    return True, ""


def validate_custom_tool_code(code: str, tool_path: str) -> Tuple[bool, str]:
    """
    Validate custom tool code before dynamic import.

    SECURITY: Prevents arbitrary code execution via malicious custom tools.
    Checks:
    - No forbidden imports (os.system, subprocess, etc.)
    - No forbidden patterns (eval, exec, __import__)
    - No f-string evasion (f"sub{'process'}")
    - Has required Tool class with execute method
    - Valid Python syntax

    Args:
        code: The tool source code
        tool_path: Path for error messages

    Returns:
        (is_valid, error_message_or_ok)
    """
    # 1. Normalize code before pattern checks (catches string-concat evasion)
    normalized = _normalize_code_for_check(code)
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in normalized:
            return False, f"Forbidden pattern '{pattern}' found in {tool_path}"

    # 2. Parse as AST to validate structure
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in {tool_path}: {e}"

    # 2b. Check f-string evasion (e.g. f"sub{'process'}")
    ok, err = _check_fstring_evasion(tree, FORBIDDEN_PATTERNS, tool_path)
    if not ok:
        return False, err

    # 3. Check imports and dangerous AST patterns
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_base = alias.name.split('.')[0]
                if module_base not in ALLOWED_TOOL_IMPORTS:
                    return False, f"Forbidden import '{alias.name}' in {tool_path}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_base = node.module.split('.')[0]
                if module_base not in ALLOWED_TOOL_IMPORTS:
                    return False, f"Forbidden import 'from {node.module}' in {tool_path}"

        # Block dynamic import calls: importlib.import_module(), __import__(), etc.
        elif isinstance(node, ast.Call):
            func = node.func
            # Check for direct calls: __import__("os"), eval("code"), exec("code")
            if isinstance(func, ast.Name) and func.id in ("__import__", "eval", "exec", "compile", "getattr", "delattr"):
                return False, f"Forbidden call '{func.id}()' in {tool_path}"
            # Check for attribute calls: importlib.import_module(), builtins.__import__()
            if isinstance(func, ast.Attribute) and func.attr in ("import_module", "__import__", "system", "popen", "call", "run", "Popen"):
                return False, f"Forbidden call '*.{func.attr}()' in {tool_path}"

        # Block access to dunder attributes that enable sandbox escape
        elif isinstance(node, ast.Attribute):
            if node.attr in ("__subclasses__", "__bases__", "__mro__", "__globals__",
                             "__code__", "__builtins__", "__getattribute__",
                             "__class__", "__dict__"):
                return False, f"Forbidden attribute access '.{node.attr}' in {tool_path}"

    # 4. Check for required class structure OR module-level execute() function
    has_tool_class = False
    has_execute = False
    has_module_execute = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Look for Tool class or any class ending with 'Tool'
            if node.name == "Tool" or node.name.endswith("Tool"):
                has_tool_class = True
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in ("execute", "run", "__call__"):
                            has_execute = True
        elif isinstance(node, ast.FunctionDef) and node.name == "execute":
            # Synthesized tools may use a module-level execute() function
            has_module_execute = True

    if has_tool_class and has_execute:
        return True, "Valid"

    if has_module_execute:
        return True, "Valid (module-level execute)"

    if not has_tool_class:
        return False, f"No Tool class found in {tool_path}"

    return False, f"Tool class missing execute/run method in {tool_path}"


def validate_script_code(code: str, source: str) -> tuple:
    """Validate a raw script (not a Tool class) before execution.

    Runs AST checks + forbidden pattern matching but does NOT require
    a Tool class structure. Used for code-agent mode code blocks.

    Returns:
        (is_valid, error_message_or_ok)
    """
    # 1. Normalize and check for forbidden patterns (catches string-concat evasion)
    normalized = _normalize_code_for_check(code)
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in normalized:
            return False, f"Forbidden pattern '{pattern}' found in {source}"

    # 2. Parse as AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in {source}: {e}"

    # 2b. Check f-string evasion (e.g. f"sub{'process'}")
    ok, err = _check_fstring_evasion(tree, FORBIDDEN_PATTERNS, source)
    if not ok:
        return False, err

    # 3. Check imports and dangerous AST patterns (same rules as validate_custom_tool_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_base = alias.name.split('.')[0]
                if module_base not in ALLOWED_TOOL_IMPORTS:
                    return False, f"Forbidden import '{alias.name}' in {source}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_base = node.module.split('.')[0]
                if module_base not in ALLOWED_TOOL_IMPORTS:
                    return False, f"Forbidden import 'from {node.module}' in {source}"

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in (
                "__import__", "eval", "exec", "compile", "getattr", "delattr",
                "type", "vars", "dir",
            ):
                return False, f"Forbidden call '{func.id}()' in {source}"
            if isinstance(func, ast.Attribute) and func.attr in (
                "import_module", "__import__", "system", "popen",
                "call", "run", "Popen",
            ):
                return False, f"Forbidden call '*.{func.attr}()' in {source}"

        elif isinstance(node, ast.Attribute):
            if node.attr in (
                "__subclasses__", "__bases__", "__mro__", "__globals__",
                "__code__", "__builtins__", "__getattribute__",
                "__class__", "__dict__",
            ):
                return False, f"Forbidden attribute access '.{node.attr}' in {source}"

    return True, "Valid"
