"""Auto-Verification Pipeline — detect test runner and run tests after code edits.

Automatically detects the project's test framework (pytest, unittest, vitest,
jest, mocha, cargo test, go test) and runs tests after code edits to catch
regressions immediately.

Features:
- Multi-framework detection: Python (pytest/unittest), Node (vitest/jest/mocha),
  Rust (cargo test), Go (go test)
- Smart test targeting: match changed files to their test files
- Test output parsing: extract pass/fail counts from test runner output
- Changed-function matching: map modified functions to test names
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .test_detection import (
    FRAMEWORK_MARKERS,
    detect_framework as _detect_framework,
    parse_individual_tests as _parse_individual_tests,
    parse_test_output as _parse_test_output,
)

logger = logging.getLogger(__name__)

# Maps project type to default test commands
TEST_RUNNER_MAP = {
    "python": ["pytest", "--tb=short", "-q"],
    "node": ["npm", "test"],
    "rust": ["cargo", "test"],
    "go": ["go", "test", "./..."],
}


def _build_test_command(framework: str, project_root: str,
                        changed_files: Optional[List[str]] = None) -> List[str]:
    """Build the test command for a given framework, optionally targeting changed files.

    Args:
        framework: Detected framework name
        project_root: Project root directory
        changed_files: Optional list of changed source file paths

    Returns:
        Command list suitable for subprocess
    """
    # If we have changed files, try to find and run only their tests
    targeted_tests = []
    if changed_files:
        targeted_tests = _find_tests_for_files(changed_files, project_root)

    if framework == "pytest":
        cmd = ["python", "-m", "pytest", "--tb=short", "-q", "-x"]
        if targeted_tests:
            cmd.extend(targeted_tests)
        return cmd

    if framework == "unittest":
        cmd = ["python", "-m", "pytest", "--tb=short", "-q", "-x"]
        if targeted_tests:
            cmd.extend(targeted_tests)
        else:
            cmd.append("tests/")
        return cmd

    if framework == "jest":
        cmd = ["npx", "jest", "--verbose", "--bail"]
        if targeted_tests:
            # Jest accepts test file patterns
            cmd.extend(targeted_tests)
        return cmd

    if framework == "vitest":
        cmd = ["npx", "vitest", "run", "--reporter=verbose", "--bail", "1"]
        if targeted_tests:
            cmd.extend(targeted_tests)
        return cmd

    if framework == "mocha":
        cmd = ["npx", "mocha", "--reporter", "spec", "--bail"]
        if targeted_tests:
            cmd.extend(targeted_tests)
        return cmd

    if framework == "cargo":
        return ["cargo", "test"]

    if framework == "go":
        if changed_files:
            # Run tests only in packages containing changed files
            packages = set()
            for f in changed_files:
                pkg_dir = os.path.dirname(f)
                if pkg_dir:
                    rel = os.path.relpath(pkg_dir, project_root)
                    packages.add(f"./{rel}/...")
            if packages:
                return ["go", "test"] + list(packages)
        return ["go", "test", "./..."]

    if framework == "node":
        return ["npm", "test"]

    return []


def _find_tests_for_files(changed_files: List[str],
                           project_root: str) -> List[str]:
    """Find test files corresponding to a list of changed source files.

    Searches common patterns:
    - test_X.py / X_test.py (same dir)
    - tests/test_X.py (sibling tests/ dir)
    - __tests__/X.test.js (JS convention)
    - X.test.ts / X.spec.ts (TS convention)
    """
    test_files = []

    for src_file in changed_files:
        base = os.path.basename(src_file)
        name, ext = os.path.splitext(base)
        src_dir = os.path.dirname(src_file)

        candidates = []

        if ext in (".py",):
            candidates = [
                os.path.join(src_dir, f"test_{name}.py"),
                os.path.join(src_dir, f"{name}_test.py"),
                os.path.join(src_dir, "tests", f"test_{name}.py"),
                os.path.join(os.path.dirname(src_dir), "tests", f"test_{name}.py"),
                os.path.join(project_root, "tests", f"test_{name}.py"),
            ]
        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            candidates = [
                os.path.join(src_dir, f"{name}.test{ext}"),
                os.path.join(src_dir, f"{name}.spec{ext}"),
                os.path.join(src_dir, "__tests__", f"{name}.test{ext}"),
                os.path.join(src_dir, "__tests__", f"{name}.spec{ext}"),
                os.path.join(os.path.dirname(src_dir), "__tests__", f"{name}.test{ext}"),
            ]
        elif ext in (".go",):
            # Go tests live next to the source file
            candidates = [
                os.path.join(src_dir, f"{name}_test.go"),
            ]
        elif ext in (".rs",):
            # Rust tests are usually in the same file or tests/ dir
            candidates = [
                os.path.join(project_root, "tests", f"{name}.rs"),
            ]

        for candidate in candidates:
            if os.path.exists(candidate):
                test_files.append(candidate)
                break  # One test file per source file

    return test_files


# _parse_test_output moved to aura/tools/test_detection.py — re-imported above
# so existing call sites keep working unchanged.


def _find_affected_test_names(changed_files: List[str]) -> List[str]:
    """Extract function/method names from changed files to match against test names.

    Reads the changed files and extracts def/function names, which can then
    be matched against test names (e.g., def foo -> test_foo).
    """
    import ast as _ast

    function_names = []

    for file_path in changed_files:
        if not file_path.endswith(".py"):
            continue
        try:
            with open(file_path, encoding="utf-8") as f:
                tree = _ast.parse(f.read())
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        function_names.append(node.name)
        except (SyntaxError, OSError):
            pass

    return function_names


# _parse_individual_tests moved to aura/tools/test_detection.py — re-imported above.


def _git_file_at_head(project_path: str, rel_file: str) -> Optional[str]:
    """Return the HEAD version of `rel_file` via `git show HEAD:<path>`, or None."""
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{rel_file}"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _run_baseline_tests(
    project_path: str,
    changed_files: List[str],
    framework: str,
    shell_tool,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """Re-run the current test command against the git HEAD version of changed files.

    Strategy: copy the project into a shadow dir, overwrite the changed files
    with their HEAD content, run tests there. Returns (per-test results, raw output)
    or (None, reason_string) if the baseline couldn't be established.
    """
    if not changed_files:
        return None, "no changed files"

    try:
        in_git = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if in_git.returncode != 0:
            return None, "not a git repo"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, "git not available"

    # Resolve changed files to their HEAD content BEFORE cloning
    baseline_content: Dict[str, str] = {}
    for f in changed_files:
        abs_f = os.path.abspath(f)
        try:
            rel = os.path.relpath(abs_f, project_path).replace(os.sep, "/")
        except ValueError:
            continue
        content = _git_file_at_head(project_path, rel)
        if content is None:
            continue  # file is new — no baseline version
        baseline_content[rel] = content

    if not baseline_content:
        return None, "no changed files exist at HEAD"

    shadow_dir = tempfile.mkdtemp(prefix="aura_verify_baseline_")
    try:
        # Copy project (skipping heavy dirs). Use copytree with ignore.
        def _ignore(_src, names):
            return {
                n for n in names
                if n in {".git", "node_modules", ".venv", "venv", "__pycache__",
                         ".next", "dist", "build", ".mypy_cache", ".pytest_cache",
                         "target", ".gradle", ".idea", ".vscode"}
            }
        shadow_proj = os.path.join(shadow_dir, "proj")
        shutil.copytree(project_path, shadow_proj, ignore=_ignore, symlinks=False)

        # Overwrite changed files with baseline content
        for rel, content in baseline_content.items():
            dest = os.path.join(shadow_proj, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            Path(dest).write_text(content, encoding="utf-8")

        # Build a baseline test command. For pytest we drop -x so we see every
        # test; the per-test diff depends on seeing the full set, not bailing early.
        cmd_list = _build_test_command(framework, shadow_proj, list(baseline_content.keys()))
        if framework == "pytest" and "-x" in cmd_list:
            cmd_list = [c for c in cmd_list if c != "-x"]
            if "-q" in cmd_list:
                cmd_list = [c if c != "-q" else "-v" for c in cmd_list]
            else:
                cmd_list.append("-v")

        cmd_str = " ".join(cmd_list)
        logger.info(f"[AutoVerify][baseline] {cmd_str} in {shadow_proj}")
        try:
            result = shell_tool.run(command=cmd_str, cwd=shadow_proj, timeout=180)
        except Exception as e:
            return None, f"baseline shell error: {e}"

        output = (result.get("stdout", "") or result.get("output", "") or "")
        stderr = result.get("stderr", "")
        if stderr:
            output = f"{output}\n{stderr}".strip()

        per_test = _parse_individual_tests(framework, output)
        return per_test, output[:5000]
    finally:
        shutil.rmtree(shadow_dir, ignore_errors=True)


def _diff_test_results(
    baseline: List[Dict[str, str]],
    current: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    """Compare two per-test result lists, return regressions / fixes / new / removed."""
    base_by_name = {t["name"]: t for t in baseline}
    cur_by_name = {t["name"]: t for t in current}

    regressions = []
    fixes = []
    new_tests = []
    removed = []

    for name, cur in cur_by_name.items():
        if name not in base_by_name:
            new_tests.append(cur)
            continue
        base = base_by_name[name]
        if base["status"] == "passed" and cur["status"] == "failed":
            regressions.append({**cur, "was": "passed"})
        elif base["status"] == "failed" and cur["status"] == "passed":
            fixes.append({**cur, "was": "failed"})

    for name, base in base_by_name.items():
        if name not in cur_by_name:
            removed.append(base)

    return {
        "regressions": regressions,
        "fixes": fixes,
        "new_tests": new_tests,
        "removed": removed,
    }


def _run_property_tests(
    project_path: str,
    changed_files: List[str],
    shell_tool,
) -> Optional[Dict]:
    """Generate and run Hypothesis property tests for changed Python files."""
    py_files = [f for f in changed_files if f.endswith(".py")]
    if not py_files:
        return None

    try:
        from .auto_verify_hypothesis import generate_property_tests, cleanup_tempdir
    except ImportError as e:
        logger.info(f"[AutoVerify] hypothesis generator unavailable: {e}")
        return None

    tmp_dir, test_files = generate_property_tests(py_files, project_path)
    if not tmp_dir:
        return None

    try:
        cmd_str = f"python -m pytest -v --tb=short -p no:cacheprovider {tmp_dir}"
        logger.info(f"[AutoVerify][property] {cmd_str}")
        try:
            result = shell_tool.run(command=cmd_str, cwd=project_path, timeout=180)
        except Exception as e:
            return {"skipped": True, "reason": f"shell error: {e}", "generated": len(test_files)}

        output = (result.get("stdout", "") or result.get("output", "") or "")
        stderr = result.get("stderr", "")
        if stderr:
            output = f"{output}\n{stderr}".strip()
        exit_code = result.get("exit_code", 1)

        parsed = _parse_test_output("pytest", output)
        per_test = _parse_individual_tests("pytest", output)

        return {
            "generated": len(test_files),
            "exit_code": exit_code,
            "success": exit_code == 0,
            "summary": parsed.get("summary", ""),
            "passed": parsed.get("passed", 0),
            "failed": parsed.get("failed", 0),
            "failures": [t for t in per_test if t["status"] == "failed"][:10],
            "output": output[:3000],
        }
    finally:
        cleanup_tempdir(tmp_dir)


def detect_test_command(project_path: str) -> Optional[List[str]]:
    """Detect the appropriate test command for a project.

    Returns a command list (for subprocess) or None if no test runner found.
    """
    framework = _detect_framework(project_path)
    if framework:
        return _build_test_command(framework, project_path)

    # Legacy fallback: try code_search detection
    try:
        from .code_search import CodeSearchTool
        searcher = CodeSearchTool()
        detected = searcher.detect_project_type(project_path)
        project_type = detected.get("project_type", "unknown") if detected.get("success") else "unknown"
    except Exception:
        project_type = "unknown"

    if project_type in TEST_RUNNER_MAP:
        return TEST_RUNNER_MAP[project_type]

    return None


def detect_and_run_tests(project_path: str, shell_tool,
                         changed_files: Optional[List[str]] = None,
                         differential: bool = True,
                         property_based: bool = True) -> dict:
    """Detect test framework and run relevant tests.

    Smarter than auto_verify: targets tests for changed files when possible,
    parses output for pass/fail counts, reports affected functions, and
    optionally runs a differential pass against the git HEAD baseline plus
    Hypothesis-generated property tests.

    Args:
        project_path: Root directory of the project
        shell_tool: ShellExecutorTool instance with .run() method
        changed_files: Optional list of recently changed file paths
        differential: If True and git is available, re-run tests against the
            HEAD version of changed files and diff the results. Regressions
            and fixes are reported in the return dict.
        property_based: If True, generate Hypothesis smoke tests for changed
            Python files with type-annotated public functions.

    Returns:
        dict with: success, framework, tests_found, test_command, parsed_results,
                   per_test, regressions, fixes, property_results,
                   affected_functions, exit_code, output
    """
    framework = _detect_framework(project_path)
    if not framework:
        return {"success": True, "framework": None, "tests_found": False,
                "skipped": True, "reason": "No test framework detected"}

    if shell_tool is None:
        return {"success": True, "framework": framework, "tests_found": True,
                "skipped": True, "reason": "No shell tool available"}

    # For pytest we want full verbosity in the differential pass so the diff
    # engine has individual test names. Don't bail on first failure.
    cmd_list = _build_test_command(framework, project_path, changed_files)
    if not cmd_list:
        return {"success": True, "framework": framework, "tests_found": False,
                "skipped": True, "reason": "Could not build test command"}

    if differential and framework == "pytest":
        cmd_list = [c for c in cmd_list if c != "-x"]
        if "-q" in cmd_list:
            cmd_list = [c if c != "-q" else "-v" for c in cmd_list]
        elif "-v" not in cmd_list:
            cmd_list.append("-v")

    cmd_str = " ".join(cmd_list)
    logger.info(f"[AutoVerify] Running: {cmd_str} in {project_path}")

    affected_functions = []
    if changed_files:
        affected_functions = _find_affected_test_names(changed_files)

    try:
        result = shell_tool.run(command=cmd_str, cwd=project_path, timeout=120)
        output = result.get("stdout", "") or result.get("output", "")
        stderr = result.get("stderr", "")
        if stderr:
            output = f"{output}\n{stderr}".strip()
        exit_code = result.get("exit_code", 1)

        parsed = _parse_test_output(framework, output)
        per_test = _parse_individual_tests(framework, output)

        response = {
            "success": exit_code == 0,
            "framework": framework,
            "tests_found": True,
            "exit_code": exit_code,
            "output": output[:5000],
            "test_command": cmd_str,
            "parsed_results": parsed,
            "per_test": per_test,
            "affected_functions": affected_functions[:20],
        }

        # Differential: re-run against git HEAD baseline
        if differential and changed_files:
            baseline, baseline_info = _run_baseline_tests(
                project_path, changed_files, framework, shell_tool,
            )
            if baseline is not None:
                diff = _diff_test_results(baseline, per_test)
                response["baseline_per_test"] = baseline
                response["regressions"] = diff["regressions"]
                response["fixes"] = diff["fixes"]
                response["new_tests"] = diff["new_tests"]
                response["removed_tests"] = diff["removed"]
                if diff["regressions"]:
                    response["success"] = False
            else:
                response["baseline_skipped"] = baseline_info

        # Property-based: generate + run Hypothesis smoke tests
        if property_based and changed_files:
            prop = _run_property_tests(project_path, changed_files, shell_tool)
            if prop is not None:
                response["property_results"] = prop
                if prop.get("failed", 0) > 0:
                    response["success"] = False

        return response

    except Exception as e:
        logger.warning(f"[AutoVerify] Test execution failed: {e}")
        return {
            "success": True,
            "framework": framework,
            "tests_found": True,
            "skipped": True,
            "reason": f"Test execution error: {e}",
        }


def _find_project_root(file_path: str) -> Optional[str]:
    """Walk up from file_path looking for project root markers."""
    markers = {"package.json", "pyproject.toml", "Cargo.toml", "go.mod", ".git"}
    current = Path(file_path).resolve()
    if current.is_file():
        current = current.parent

    for _ in range(10):
        for marker in markers:
            if (current / marker).exists():
                return str(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def run_changed_tests(
    project_root: str,
    changed_files: List[str],
    shell_tool=None,
    timeout: int = 60,
    override_cmd: Optional[str] = None,
) -> dict:
    """Run only the tests that cover the given changed files.

    Uses the existing source→test mapping (`_find_tests_for_files`) plus
    framework-specific targeted flags (pytest filename, vitest related,
    jest --findRelatedTests, go test <pkg>, cargo test). If no tests map
    to the changes, returns success=True with skipped=True — "no tests to
    run" is not the same as "tests failed."

    *override_cmd* lets AURA.md specify a custom command. The placeholder
    `{files}` (if present) is replaced with the test file list; otherwise
    the files are appended.

    Returns: {success, skipped, output, duration_s, failures, framework, reason}.
    """
    import time as _t
    import shlex as _shlex

    start = _t.monotonic()

    if not changed_files:
        return {
            "success": True, "skipped": True, "reason": "no changed files",
            "duration_s": 0.0, "failures": [], "output": "", "framework": "",
        }

    framework = _detect_framework(project_root)
    if not framework:
        return {
            "success": True, "skipped": True, "reason": "no framework detected",
            "duration_s": 0.0, "failures": [], "output": "", "framework": "",
        }

    # Build command: override wins, otherwise fall back to targeted default.
    cmd_list: List[str]
    if override_cmd and override_cmd.strip():
        targeted = _find_tests_for_files(changed_files, project_root)
        tokens = _shlex.split(override_cmd.strip())
        if "{files}" in tokens:
            idx = tokens.index("{files}")
            cmd_list = tokens[:idx] + targeted + tokens[idx + 1:]
        else:
            cmd_list = tokens + targeted
    else:
        cmd_list = _build_test_command(framework, project_root, changed_files)

    if not cmd_list:
        return {
            "success": True, "skipped": True, "reason": "no runnable command",
            "duration_s": _t.monotonic() - start, "failures": [],
            "output": "", "framework": framework,
        }

    if shell_tool is None:
        return {
            "success": True, "skipped": True, "reason": "no shell tool available",
            "duration_s": _t.monotonic() - start, "failures": [],
            "output": "", "framework": framework,
        }

    cmd_str = " ".join(cmd_list)
    logger.info(f"[ChangedTests] Running: {cmd_str}")

    try:
        r = shell_tool.run(command=cmd_str, cwd=project_root, timeout=timeout)
        out = r.get("stdout", "") or r.get("output", "") or ""
        err = r.get("stderr", "") or ""
        full = (out + "\n" + err).strip()
        exit_code = r.get("exit_code", 1)

        # Leverage existing parsers: aggregate counts + per-test failure detail.
        parsed = _parse_test_output(framework, full)
        individual = _parse_individual_tests(framework, full)
        failures = [
            {"file": "", "line": 0, "message": f"{t.get('name', '?')} — {t.get('error', '')}".strip(" —")}
            for t in individual if t.get("status") == "failed"
        ]

        return {
            "success": exit_code == 0,
            "skipped": False,
            "exit_code": exit_code,
            "output": full[:5000],
            "duration_s": _t.monotonic() - start,
            "failures": failures,
            "passed": parsed.get("passed", 0),
            "failed": parsed.get("failed", 0),
            "framework": framework,
            "command": cmd_str,
        }
    except Exception as e:
        logger.warning(f"[ChangedTests] execution failed: {e}")
        return {
            "success": False, "skipped": False,
            "reason": f"execution error: {e}",
            "duration_s": _t.monotonic() - start,
            "failures": [], "output": "", "framework": framework,
        }


def auto_verify(project_path: str, shell_tool, test_cmd_override: Optional[str] = None) -> dict:
    """Run project tests and return results.

    Backward-compatible entry point. For richer results, use detect_and_run_tests().

    Args:
        project_path: Root directory of the project
        shell_tool: ShellExecutorTool instance with .run() method
        test_cmd_override: If set (e.g. from AURA.md `test_cmd:`), skip framework
            detection and run this command string verbatim.

    Returns:
        dict with keys: success, exit_code, output, test_command, skipped, reason
    """
    if test_cmd_override:
        cmd_list = test_cmd_override.strip().split() if isinstance(test_cmd_override, str) else list(test_cmd_override)
    else:
        cmd_list = detect_test_command(project_path)

    if not cmd_list:
        return {"success": True, "skipped": True, "reason": "No test runner detected"}

    if shell_tool is None:
        return {"success": True, "skipped": True, "reason": "No shell tool available"}

    cmd_str = " ".join(cmd_list)
    logger.info(f"[AutoVerify] Running: {cmd_str} in {project_path}")

    try:
        result = shell_tool.run(command=cmd_str, cwd=project_path, timeout=120)
        output = result.get("stdout", "") or result.get("output", "")
        stderr = result.get("stderr", "")
        if stderr:
            output = f"{output}\n{stderr}".strip()
        exit_code = result.get("exit_code", 1)

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "output": output[:5000],  # Cap output size
            "test_command": cmd_str,
        }
    except Exception as e:
        logger.warning(f"[AutoVerify] Test execution failed: {e}")
        return {
            "success": True,
            "skipped": True,
            "reason": f"Test execution error: {e}",
        }


def _has_tool(name: str) -> bool:
    """Return True if a CLI tool is on PATH."""
    import shutil
    return shutil.which(name) is not None


def _pyproject_has_section(project_path: str, section: str) -> bool:
    p = Path(project_path) / "pyproject.toml"
    if not p.exists():
        return False
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        return f"[tool.{section}]" in text or f"[tool.{section}." in text
    except Exception:
        return False


def lint_and_typecheck(project_path: str, shell_tool,
                       lint_cmd_override: Optional[str] = None,
                       typecheck_cmd_override: Optional[str] = None) -> dict:
    """Run ruff + mypy if configured or detectable. Returns a uniform result.

    Args:
        project_path: Root of the project
        shell_tool: ShellExecutorTool instance with .run()
        lint_cmd_override: Explicit lint command (from AURA.md `lint_cmd:`). If set
            and non-empty, runs verbatim instead of auto-detection.
        typecheck_cmd_override: Explicit typecheck command (from AURA.md
            `typecheck_cmd:`).

    Returns:
        dict with: success, skipped, lint (optional sub-result), typecheck
        (optional sub-result), reason (if skipped).

        Each sub-result has: success, exit_code, output, command.
    """
    if shell_tool is None:
        return {"success": True, "skipped": True, "reason": "No shell tool available"}

    results: dict = {"success": True, "skipped": False}

    # Lint
    lint_cmd = None
    if lint_cmd_override and lint_cmd_override.strip():
        lint_cmd = lint_cmd_override.strip()
    elif _has_tool("ruff") and (_pyproject_has_section(project_path, "ruff")
                                or (Path(project_path) / ".ruff.toml").exists()
                                or any((Path(project_path) / f).exists() for f in ["pyproject.toml", "ruff.toml"])):
        lint_cmd = "ruff check ."

    if lint_cmd:
        try:
            r = shell_tool.run(command=lint_cmd, cwd=project_path, timeout=60)
            out = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
            code = r.get("exit_code", 1)
            results["lint"] = {"success": code == 0, "exit_code": code,
                               "output": out.strip()[:3000], "command": lint_cmd}
            if code != 0:
                results["success"] = False
        except Exception as e:
            logger.debug(f"[LintCheck] Lint failed: {e}")

    # Typecheck
    typecheck_cmd = None
    if typecheck_cmd_override and typecheck_cmd_override.strip():
        typecheck_cmd = typecheck_cmd_override.strip()
    elif _has_tool("mypy") and _pyproject_has_section(project_path, "mypy"):
        typecheck_cmd = "mypy ."

    if typecheck_cmd:
        try:
            r = shell_tool.run(command=typecheck_cmd, cwd=project_path, timeout=120)
            out = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
            code = r.get("exit_code", 1)
            results["typecheck"] = {"success": code == 0, "exit_code": code,
                                    "output": out.strip()[:3000], "command": typecheck_cmd}
            if code != 0:
                results["success"] = False
        except Exception as e:
            logger.debug(f"[TypeCheck] mypy failed: {e}")

    if "lint" not in results and "typecheck" not in results:
        return {"success": True, "skipped": True, "reason": "No lint/typecheck tool configured or detected"}

    return results
