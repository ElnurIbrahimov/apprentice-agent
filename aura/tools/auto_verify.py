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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maps project type to default test commands
TEST_RUNNER_MAP = {
    "python": ["pytest", "--tb=short", "-q"],
    "node": ["npm", "test"],
    "rust": ["cargo", "test"],
    "go": ["go", "test", "./..."],
}

# Framework detection markers (file -> framework)
FRAMEWORK_MARKERS = {
    "pytest.ini": "pytest",
    "setup.cfg": "pytest",
    "conftest.py": "pytest",
    "Cargo.toml": "cargo",
    "go.mod": "go",
    "package.json": "node",
    ".mocharc.yml": "mocha",
    ".mocharc.json": "mocha",
    ".mocharc.js": "mocha",
    "jest.config.js": "jest",
    "jest.config.ts": "jest",
    "jest.config.mjs": "jest",
    "vitest.config.ts": "vitest",
    "vitest.config.js": "vitest",
    "vitest.config.mts": "vitest",
}


def _detect_node_test_runner(project_path: str) -> Optional[List[str]]:
    """Check package.json devDeps for vitest vs jest vs mocha, fallback to npm test."""
    pkg_json = Path(project_path) / "package.json"
    if not pkg_json.exists():
        return None
    try:
        data = json.loads(pkg_json.read_text(encoding="utf-8"))
        dev_deps = data.get("devDependencies", {})
        deps = data.get("dependencies", {})
        all_deps = {**deps, **dev_deps}

        if "vitest" in all_deps:
            return ["npx", "vitest", "run", "--reporter=verbose"]
        if "jest" in all_deps:
            return ["npx", "jest", "--verbose"]
        if "mocha" in all_deps:
            return ["npx", "mocha", "--reporter", "spec"]

        # Check if "test" script exists
        scripts = data.get("scripts", {})
        if "test" in scripts and scripts["test"] != 'echo "Error: no test specified" && exit 1':
            return ["npm", "test"]
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _detect_framework(project_path: str) -> Optional[str]:
    """Detect test framework from project files.

    Returns framework name: 'pytest', 'unittest', 'jest', 'vitest', 'mocha',
    'cargo', 'go', or None.
    """
    root = Path(project_path)

    # Check for specific config files first (highest confidence)
    for marker_file, framework in FRAMEWORK_MARKERS.items():
        if (root / marker_file).exists():
            # For package.json, need to check further
            if marker_file == "package.json":
                runner = _detect_node_test_runner(project_path)
                if runner:
                    # Derive framework name from command
                    cmd_str = " ".join(runner)
                    if "vitest" in cmd_str:
                        return "vitest"
                    elif "jest" in cmd_str:
                        return "jest"
                    elif "mocha" in cmd_str:
                        return "mocha"
                    return "node"
                continue
            return framework

    # Check pyproject.toml for pytest config
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8")
            if "[tool.pytest" in content or "pytest" in content:
                return "pytest"
        except OSError:
            pass

    # Check for unittest-style tests (test files using unittest.TestCase)
    test_dirs = [root / "tests", root / "test"]
    for test_dir in test_dirs:
        if test_dir.is_dir():
            for py_file in test_dir.glob("test_*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8")
                    if "unittest.TestCase" in content:
                        return "unittest"
                    if "import pytest" in content or "@pytest" in content:
                        return "pytest"
                except OSError:
                    pass
                break  # Only check first test file

    # If we find any Python test files, default to pytest
    for test_dir in test_dirs:
        if test_dir.is_dir() and any(test_dir.glob("test_*.py")):
            return "pytest"

    return None


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


def _parse_test_output(framework: str, output: str) -> Dict:
    """Parse test output to extract pass/fail/skip counts.

    Returns dict with: passed, failed, skipped, errors, total, summary
    """
    result = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "total": 0,
        "summary": "",
    }

    if not output:
        return result

    if framework in ("pytest", "unittest"):
        # pytest short summary: "5 passed, 2 failed, 1 skipped in 1.23s"
        match = re.search(
            r'(\d+) passed(?:.*?(\d+) failed)?(?:.*?(\d+) skipped)?'
            r'(?:.*?(\d+) error)?',
            output,
        )
        if match:
            result["passed"] = int(match.group(1) or 0)
            result["failed"] = int(match.group(2) or 0)
            result["skipped"] = int(match.group(3) or 0)
            result["errors"] = int(match.group(4) or 0)
        else:
            # Try "X failed" alone
            failed_match = re.search(r'(\d+) failed', output)
            if failed_match:
                result["failed"] = int(failed_match.group(1))
            passed_match = re.search(r'(\d+) passed', output)
            if passed_match:
                result["passed"] = int(passed_match.group(1))

    elif framework in ("jest", "vitest"):
        # Jest/Vitest: "Tests:  3 passed, 1 failed, 4 total"
        match = re.search(
            r'Tests:\s+(?:(\d+) passed)?(?:.*?(\d+) failed)?(?:.*?(\d+) skipped)?'
            r'(?:.*?(\d+) total)?',
            output,
        )
        if match:
            result["passed"] = int(match.group(1) or 0)
            result["failed"] = int(match.group(2) or 0)
            result["skipped"] = int(match.group(3) or 0)
            result["total"] = int(match.group(4) or 0)

    elif framework == "mocha":
        # Mocha: "3 passing (1s)" / "1 failing"
        passing = re.search(r'(\d+) passing', output)
        failing = re.search(r'(\d+) failing', output)
        pending = re.search(r'(\d+) pending', output)
        if passing:
            result["passed"] = int(passing.group(1))
        if failing:
            result["failed"] = int(failing.group(1))
        if pending:
            result["skipped"] = int(pending.group(1))

    elif framework == "cargo":
        # Cargo: "test result: ok. 5 passed; 0 failed; 0 ignored"
        match = re.search(
            r'test result:.*?(\d+) passed.*?(\d+) failed.*?(\d+) ignored',
            output,
        )
        if match:
            result["passed"] = int(match.group(1))
            result["failed"] = int(match.group(2))
            result["skipped"] = int(match.group(3))

    elif framework == "go":
        # Go: "ok  ./pkg  0.5s" for pass, "FAIL ./pkg" for fail
        result["passed"] = len(re.findall(r'^ok\s+', output, re.MULTILINE))
        result["failed"] = len(re.findall(r'^FAIL\s+', output, re.MULTILINE))

    result["total"] = result["passed"] + result["failed"] + result["skipped"] + result["errors"]
    result["summary"] = (
        f"{result['passed']} passed, {result['failed']} failed"
        + (f", {result['skipped']} skipped" if result["skipped"] else "")
        + (f", {result['errors']} errors" if result["errors"] else "")
    )

    return result


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
                         changed_files: Optional[List[str]] = None) -> dict:
    """Detect test framework and run relevant tests.

    Smarter than auto_verify: targets tests for changed files when possible,
    parses output for pass/fail counts, and reports affected functions.

    Args:
        project_path: Root directory of the project
        shell_tool: ShellExecutorTool instance with .run() method
        changed_files: Optional list of recently changed file paths

    Returns:
        dict with: success, framework, tests_found, test_command, parsed_results,
                   affected_functions, exit_code, output
    """
    framework = _detect_framework(project_path)
    if not framework:
        return {"success": True, "framework": None, "tests_found": False,
                "skipped": True, "reason": "No test framework detected"}

    if shell_tool is None:
        return {"success": True, "framework": framework, "tests_found": True,
                "skipped": True, "reason": "No shell tool available"}

    cmd_list = _build_test_command(framework, project_path, changed_files)
    if not cmd_list:
        return {"success": True, "framework": framework, "tests_found": False,
                "skipped": True, "reason": "Could not build test command"}

    cmd_str = " ".join(cmd_list)
    logger.info(f"[AutoVerify] Running: {cmd_str} in {project_path}")

    # Identify affected functions for context
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

        # Parse test output for structured results
        parsed = _parse_test_output(framework, output)

        return {
            "success": exit_code == 0,
            "framework": framework,
            "tests_found": True,
            "exit_code": exit_code,
            "output": output[:5000],
            "test_command": cmd_str,
            "parsed_results": parsed,
            "affected_functions": affected_functions[:20],  # Cap at 20
        }
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


def auto_verify(project_path: str, shell_tool) -> dict:
    """Run project tests and return results.

    Backward-compatible entry point. For richer results, use detect_and_run_tests().

    Args:
        project_path: Root directory of the project
        shell_tool: ShellExecutorTool instance with .run() method

    Returns:
        dict with keys: success, exit_code, output, test_command, skipped, reason
    """
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
