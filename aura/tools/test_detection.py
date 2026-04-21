"""Shared test-framework detection and output parsing.

Both `aura/cli/test_runner.py` (the interactive `/test` command) and
`aura/tools/auto_verify.py` (the POST_EDIT verification pipeline) need to
detect which test runner a project uses and parse that runner's output.
Before this module existed, the detection regexes and the "X passed, Y failed"
parser were copy-pasted across both files and drifted over time. This module
is the single source of truth — both callers import from here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


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


def _detect_node_test_runner(project_path: str) -> Optional[str]:
    """Return the Node test framework name (vitest/jest/mocha/node) from package.json."""
    pkg_json = Path(project_path) / "package.json"
    if not pkg_json.exists():
        return None
    try:
        data = json.loads(pkg_json.read_text(encoding="utf-8"))
        all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        if "vitest" in all_deps:
            return "vitest"
        if "jest" in all_deps:
            return "jest"
        if "mocha" in all_deps:
            return "mocha"
        scripts = data.get("scripts", {})
        if "test" in scripts and scripts["test"] != 'echo "Error: no test specified" && exit 1':
            return "node"
    except (json.JSONDecodeError, OSError):
        pass
    return None


def detect_framework(project_path: str) -> Optional[str]:
    """Detect which test framework a project uses.

    Returns one of: 'pytest', 'unittest', 'jest', 'vitest', 'mocha', 'cargo',
    'go', 'node', or None if no runner can be inferred.
    """
    root = Path(project_path)

    for marker_file, framework in FRAMEWORK_MARKERS.items():
        if (root / marker_file).exists():
            if marker_file == "package.json":
                resolved = _detect_node_test_runner(project_path)
                if resolved:
                    return resolved
                continue
            return framework

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8")
            if "[tool.pytest" in content or "pytest" in content:
                return "pytest"
        except OSError:
            pass

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
                break
    for test_dir in test_dirs:
        if test_dir.is_dir() and any(test_dir.glob("test_*.py")):
            return "pytest"

    return None


def parse_test_output(framework: Optional[str], output: str) -> Dict:
    """Parse per-framework test runner output into structured counts.

    Returns a dict with keys: passed, failed, skipped, errors, total, summary.
    `framework=None` falls back to a generic regex sweep that handles pytest
    and jest together — used by callers that don't care about the source
    runner (interactive `/test` command).
    """
    result: Dict = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "total": 0,
        "summary": "",
    }

    if not output:
        return result

    if framework in ("pytest", "unittest", None):
        for key, label in [("passed", "passed"), ("failed", "failed"),
                           ("skipped", "skipped"), ("errors", "error")]:
            m = re.search(rf'(\d+)\s+{label}', output)
            if m:
                result[key] = int(m.group(1))

    if framework in ("jest", "vitest", None):
        match = re.search(
            r'Tests:\s+(?:(\d+) passed)?(?:.*?(\d+) failed)?(?:.*?(\d+) skipped)?'
            r'(?:.*?(\d+) total)?',
            output,
        )
        if match and (framework in ("jest", "vitest") or result["passed"] == 0):
            result["passed"] = int(match.group(1) or 0)
            result["failed"] = int(match.group(2) or 0)
            result["skipped"] = int(match.group(3) or 0)
            if match.group(4):
                result["total"] = int(match.group(4))

    if framework == "mocha":
        passing = re.search(r'(\d+) passing', output)
        failing = re.search(r'(\d+) failing', output)
        pending = re.search(r'(\d+) pending', output)
        if passing: result["passed"] = int(passing.group(1))
        if failing: result["failed"] = int(failing.group(1))
        if pending: result["skipped"] = int(pending.group(1))

    elif framework == "cargo":
        match = re.search(
            r'test result:.*?(\d+) passed.*?(\d+) failed.*?(\d+) ignored',
            output,
        )
        if match:
            result["passed"] = int(match.group(1))
            result["failed"] = int(match.group(2))
            result["skipped"] = int(match.group(3))

    elif framework == "go":
        result["passed"] = len(re.findall(r'^ok\s+', output, re.MULTILINE))
        result["failed"] = len(re.findall(r'^FAIL\s+', output, re.MULTILINE))

    if not result["total"]:
        result["total"] = result["passed"] + result["failed"] + result["skipped"] + result["errors"]

    result["summary"] = (
        f"{result['passed']} passed, {result['failed']} failed"
        + (f", {result['skipped']} skipped" if result["skipped"] else "")
        + (f", {result['errors']} errors" if result["errors"] else "")
    )

    return result


def parse_individual_tests(framework: Optional[str], output: str) -> List[Dict[str, str]]:
    """Extract per-test {name, status, error} records from runner output.

    Used by the differential runner in auto_verify to compute regressions vs
    the git HEAD baseline. Interactive /test uses only the aggregate counts.
    """
    tests: List[Dict[str, str]] = []
    if not output:
        return tests

    if framework in ("pytest", "unittest"):
        pattern = re.compile(
            r"^([^\s]+?::[^\s]+?)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)",
            re.MULTILINE,
        )
        for m in pattern.finditer(output):
            status = m.group(2).lower()
            if status in ("xfail", "xpass"):
                status = "passed"
            tests.append({"name": m.group(1), "status": status, "error": ""})
        for m in re.finditer(r"^FAILED\s+([^\s]+?::[^\s]+?)(?:\s+-\s+(.*))?$", output, re.MULTILINE):
            name = m.group(1)
            err = (m.group(2) or "").strip()
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
        for line in output.splitlines():
            ls = line.strip()
            m = re.match(r"^(\u2713|\u2717|\u00d7|\u2718|PASS|FAIL)\s+(.+?)(?:\s+\(\d+\s*ms\))?$", ls)
            if m:
                tok = m.group(1)
                name = m.group(2).strip()
                status = "passed" if tok in ("\u2713", "PASS") else "failed"
                tests.append({"name": name, "status": status, "error": ""})

    elif framework == "mocha":
        for line in output.splitlines():
            ls = line.strip()
            if ls.startswith("\u2713 "):
                tests.append({"name": ls[2:].strip(), "status": "passed", "error": ""})
            elif re.match(r"^\d+\)\s+", ls):
                name = re.sub(r"^\d+\)\s+", "", ls)
                tests.append({"name": name, "status": "failed", "error": ""})
            elif ls.startswith("- "):
                tests.append({"name": ls[2:].strip(), "status": "skipped", "error": ""})

    elif framework == "go":
        pattern = re.compile(r"^---\s+(PASS|FAIL|SKIP):\s+(\S+)", re.MULTILINE)
        for m in pattern.finditer(output):
            status = {"PASS": "passed", "FAIL": "failed", "SKIP": "skipped"}[m.group(1)]
            tests.append({"name": m.group(2), "status": status, "error": ""})

    elif framework == "cargo":
        pattern = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)", re.MULTILINE)
        for m in pattern.finditer(output):
            name = m.group(1)
            raw = m.group(2)
            status = {"ok": "passed", "FAILED": "failed", "ignored": "skipped"}[raw]
            tests.append({"name": name, "status": status, "error": ""})

    return tests
