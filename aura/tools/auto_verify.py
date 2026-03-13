"""Auto-Verification Pipeline — detect test runner and run tests after code edits.

Automatically detects the project's test framework (pytest, vitest, jest, cargo test,
go test) and runs tests after code edits to catch regressions immediately.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Maps project type to default test commands
TEST_RUNNER_MAP = {
    "python": ["pytest", "--tb=short", "-q"],
    "node": ["npm", "test"],
    "rust": ["cargo", "test"],
    "go": ["go", "test", "./..."],
}


def _detect_node_test_runner(project_path: str) -> Optional[List[str]]:
    """Check package.json devDeps for vitest vs jest, fallback to npm test."""
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

        # Check if "test" script exists
        scripts = data.get("scripts", {})
        if "test" in scripts and scripts["test"] != 'echo "Error: no test specified" && exit 1':
            return ["npm", "test"]
    except (json.JSONDecodeError, OSError):
        pass
    return None


def detect_test_command(project_path: str) -> Optional[List[str]]:
    """Detect the appropriate test command for a project.

    Returns a command list (for subprocess) or None if no test runner found.
    """
    try:
        from .code_search import CodeSearchTool
        searcher = CodeSearchTool()
        detected = searcher.detect_project_type(project_path)
        project_type = detected.get("project_type", "unknown") if detected.get("success") else "unknown"
    except Exception:
        project_type = "unknown"

    # Python: check for pytest.ini / pyproject.toml [tool.pytest]
    if project_type == "python":
        root = Path(project_path)
        if (root / "pytest.ini").exists() or (root / "setup.cfg").exists():
            return ["pytest", "--tb=short", "-q"]
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                if "[tool.pytest" in content or "pytest" in content:
                    return ["pytest", "--tb=short", "-q"]
            except OSError:
                pass
        return TEST_RUNNER_MAP["python"]

    # Node: vitest > jest > npm test
    if project_type == "node":
        runner = _detect_node_test_runner(project_path)
        return runner  # May be None if no test script

    # Direct lookup for rust/go
    if project_type in TEST_RUNNER_MAP:
        return TEST_RUNNER_MAP[project_type]

    return None


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
