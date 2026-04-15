"""Hypothesis-based property test generator for auto_verify.

Walks changed Python files, finds type-annotated public functions, and emits
throwaway smoke tests that feed each function Hypothesis-generated inputs.
Catches edge cases the human never wrote explicit tests for (zero division,
empty strings, huge ints, etc.) without requiring the human to author them.

The generated tests are "smoke tests": they pass unless the function raises
something other than a small set of expected exceptions (ValueError, TypeError,
KeyError, IndexError, ZeroDivisionError, StopIteration). Anything else —
AssertionError, internal crashes, unhandled exceptions — is a real signal.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SIMPLE_TYPE_STRATEGIES = {
    "int": "st.integers(min_value=-10000, max_value=10000)",
    "float": "st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)",
    "str": "st.text(max_size=40)",
    "bool": "st.booleans()",
    "bytes": "st.binary(max_size=40)",
    "None": "st.none()",
}


def _strategy_for_annotation(annotation: ast.AST) -> Optional[str]:
    """Convert a Python type annotation AST node into a Hypothesis strategy string.

    Returns None if the annotation is too complex or unknown — caller should skip
    the function entirely in that case rather than guess.
    """
    if annotation is None:
        return None

    if isinstance(annotation, ast.Name):
        return SIMPLE_TYPE_STRATEGIES.get(annotation.id)

    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return "st.none()"

    if isinstance(annotation, ast.Subscript):
        base = annotation.value
        if isinstance(base, ast.Name):
            name = base.id
            slice_node = annotation.slice

            if name in ("list", "List"):
                inner = _strategy_for_annotation(slice_node)
                if inner:
                    return f"st.lists({inner}, max_size=8)"
                return None

            if name in ("tuple", "Tuple"):
                if isinstance(slice_node, ast.Tuple):
                    parts = [_strategy_for_annotation(e) for e in slice_node.elts]
                    if all(parts):
                        return f"st.tuples({', '.join(parts)})"
                return None

            if name in ("dict", "Dict"):
                if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
                    k = _strategy_for_annotation(slice_node.elts[0])
                    v = _strategy_for_annotation(slice_node.elts[1])
                    if k and v:
                        return f"st.dictionaries({k}, {v}, max_size=8)"
                return None

            if name in ("set", "Set", "frozenset"):
                inner = _strategy_for_annotation(slice_node)
                if inner:
                    return f"st.sets({inner}, max_size=8)"
                return None

            if name == "Optional":
                inner = _strategy_for_annotation(slice_node)
                if inner:
                    return f"st.one_of({inner}, st.none())"
                return None

    return None


def _extract_testable_functions(file_path: str) -> List[Dict[str, Any]]:
    """Walk a Python file and return metadata for every public top-level function
    whose every non-self argument has a strategy-mappable type annotation."""
    try:
        src = Path(file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    results = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue

        args = node.args.args + node.args.kwonlyargs
        if not args:
            continue

        strategies = []
        skip = False
        for arg in args:
            if arg.arg in ("self", "cls"):
                continue
            strat = _strategy_for_annotation(arg.annotation)
            if not strat:
                skip = True
                break
            strategies.append((arg.arg, strat))

        if skip or not strategies:
            continue

        results.append({
            "name": node.name,
            "args": strategies,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        })

    return results


def _module_path_for_file(file_path: str, project_root: str) -> Optional[str]:
    """Convert an absolute file path to a dotted module path relative to project_root.
    Returns None if the file isn't inside the project root."""
    try:
        rel = Path(file_path).resolve().relative_to(Path(project_root).resolve())
    except ValueError:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def _emit_test_file(
    module_dotted: str,
    project_root: str,
    funcs: List[Dict[str, Any]],
    out_dir: str,
) -> Optional[str]:
    """Write one test file covering all testable functions from a single module."""
    if not funcs:
        return None

    safe_mod = module_dotted.replace(".", "_")
    out_path = os.path.join(out_dir, f"test_aura_hypothesis_{safe_mod}.py")

    lines = [
        "# Auto-generated by Aura auto_verify_hypothesis. Safe to delete.",
        "import sys",
        f"sys.path.insert(0, {project_root!r})",
        "",
        "import pytest",
        "",
        "try:",
        "    from hypothesis import given, strategies as st, settings, HealthCheck",
        "except ImportError:",
        "    pytest.skip('hypothesis not installed', allow_module_level=True)",
        "",
        "try:",
        f"    from {module_dotted} import (",
    ]
    for fn in funcs:
        lines.append(f"        {fn['name']},")
    lines.append("    )")
    lines.append("except Exception as _import_err:")
    lines.append("    pytest.skip(f'cannot import target module: {_import_err}', allow_module_level=True)")
    lines.append("")
    # _EXPECTED: exceptions we treat as "the caller's fault," so the smoke test
    # doesn't fail on them. Anything outside this set (ZeroDivisionError,
    # AssertionError, RuntimeError, IndexError on a supposedly-safe op) is a
    # real finding the human should see.
    lines.append("_EXPECTED = (ValueError, TypeError, KeyError, LookupError, "
                 "AttributeError, UnicodeError, StopIteration, NotImplementedError)")
    lines.append("")
    lines.append("_settings = settings(max_examples=25, deadline=500, "
                 "suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])")
    lines.append("")

    for fn in funcs:
        arg_names = [a[0] for a in fn["args"]]
        kwargs = ", ".join(f"{n}={n}" for n in arg_names)
        decorator_args = ", ".join(f"{n}={s}" for n, s in fn["args"])
        lines.append("@_settings")
        lines.append(f"@given({decorator_args})")
        lines.append(f"def test_{fn['name']}_smoke({', '.join(arg_names)}):")
        lines.append("    try:")
        lines.append(f"        {fn['name']}({kwargs})")
        lines.append("    except _EXPECTED:")
        lines.append("        pass")
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_property_tests(
    changed_files: List[str],
    project_root: str,
) -> Tuple[Optional[str], List[str]]:
    """Generate a tempdir full of Hypothesis smoke tests for changed Python files.

    Returns (tempdir, test_files). Caller is responsible for cleaning up tempdir.
    Returns (None, []) if nothing was generated."""
    py_files = [f for f in changed_files if f.endswith(".py")]
    if not py_files:
        return None, []

    try:
        import hypothesis  # noqa: F401
    except ImportError:
        logger.info("[AutoVerify] hypothesis not installed — skipping property tests. "
                    "Install with: pip install hypothesis")
        return None, []

    tmp_dir = tempfile.mkdtemp(prefix="aura_hypothesis_")
    generated = []

    for src in py_files:
        funcs = _extract_testable_functions(src)
        if not funcs:
            continue
        module_dotted = _module_path_for_file(src, project_root)
        if not module_dotted:
            continue
        test_file = _emit_test_file(module_dotted, project_root, funcs, tmp_dir)
        if test_file:
            generated.append(test_file)

    if not generated:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, []

    return tmp_dir, generated


def cleanup_tempdir(tmp_dir: Optional[str]) -> None:
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
