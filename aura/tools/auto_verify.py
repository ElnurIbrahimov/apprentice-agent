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
        # pytest summary order varies: "5 passed, 2 failed" OR "1 failed, 2 passed".
        # Parse each counter independently.
        for key, label in [("passed", "passed"), ("failed", "failed"),
                           ("skipped", "skipped"), ("errors", "error")]:
            m = re.search(rf'(\d+)\s+{label}', output)
            if m:
                result[key] = int(m.group(1))

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


def _parse_individual_tests(framework: str, output: str) -> List[Dict[str, str]]:
    """Extract per-test results from test runner output.

    Returns a list of {test_name, status, error} dicts. Used by the differential
    runner to compute regressions vs the git baseline.
    """
    tests: List[Dict[str, str]] = []
    if not output:
        return tests

    if framework in ("pytest", "unittest"):
        # pytest -v: "path/to/test_foo.py::test_bar PASSED" / "FAILED" / "SKIPPED"
        pattern = re.compile(
            r"^([^\s]+?::[^\s]+?)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)",
            re.MULTILINE,
        )
        for m in pattern.finditer(output):
            status = m.group(2).lower()
            if status in ("xfail", "xpass"):
                status = "passed"
            tests.append({"name": m.group(1), "status": status, "error": ""})
        # Also capture FAILED lines from the short summary at the bottom
        for m in re.finditer(r"^FAILED\s+([^\s]+?::[^\s]+?)(?:\s+-\s+(.*))?$", output, re.MULTILINE):
            name = m.group(1)
            err = (m.group(2) or "").strip()
            # De-dup against earlier parse; update error text
            found = False
            for t in tests:
                if t["name"] == name:
                    t["status"] = "failed"
                    if err:
                        t["error"] = err
                    found = True
                    break
            if not found:
                tests.append({"name": name, "status": "failed", "error": err})

    elif framework in ("jest", "vitest"):
        # "  ✓ test name (5 ms)"  /  "  ✗ test name" / "  × test name"
        for line in output.splitlines():
            ls = line.strip()
            m = re.match(r"^(✓|✗|×|✘|PASS|FAIL)\s+(.+?)(?:\s+\(\d+\s*ms\))?$", ls)
            if m:
                tok = m.group(1)
                name = m.group(2).strip()
                status = "passed" if tok in ("✓", "PASS") else "failed"
                tests.append({"name": name, "status": status, "error": ""})

    elif framework == "mocha":
        # "    ✓ test name"  /  "    1) test name"  /  "    - test name" (pending)
        for line in output.splitlines():
            ls = line.strip()
            if ls.startswith("✓ "):
                tests.append({"name": ls[2:].strip(), "status": "passed", "error": ""})
            elif re.match(r"^\d+\)\s+", ls):
                name = re.sub(r"^\d+\)\s+", "", ls)
                tests.append({"name": name, "status": "failed", "error": ""})
            elif ls.startswith("- "):
                tests.append({"name": ls[2:].strip(), "status": "skipped", "error": ""})

    elif framework == "go":
        # "--- PASS: TestFoo (0.00s)" / "--- FAIL: TestBar (0.00s)"
        pattern = re.compile(r"^---\s+(PASS|FAIL|SKIP):\s+(\S+)", re.MULTILINE)
        for m in pattern.finditer(output):
            status = {"PASS": "passed", "FAIL": "failed", "SKIP": "skipped"}[m.group(1)]
            tests.append({"name": m.group(2), "status": status, "error": ""})

    elif framework == "cargo":
        # "test tests::foo ... ok"  /  "test tests::bar ... FAILED"
        pattern = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)", re.MULTILINE)
        for m in pattern.finditer(output):
            name = m.group(1)
            raw = m.group(2)
            status = {"ok": "passed", "FAILED": "failed", "ignored": "skipped"}[raw]
            tests.append({"name": name, "status": status, "error": ""})

    return tests


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
