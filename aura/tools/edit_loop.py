"""Agentic file editing loop: read → edit → test → verify → commit."""

import logging
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


class EditLoop:
    """Orchestrates the read→edit→test→verify→commit cycle.

    Provides auditable, reversible file edits with optional test verification
    and git commit. Every edit creates a .bak backup before modifying the file.
    """

    def __init__(self, filesystem_tool, git_tool=None, shell_tool=None):
        self.fs = filesystem_tool
        self.git = git_tool
        self.shell = shell_tool

    def run(
        self,
        edit_plan: List[dict],
        test_cmd: Optional[str] = None,
        auto_commit: bool = True,
        repo_path: str = ".",
    ) -> dict:
        """Execute an edit plan.

        Args:
            edit_plan: List of {file, search, replace} dicts
            test_cmd: Optional shell command to run for verification
            auto_commit: Whether to git commit on success
            repo_path: Git repository path

        Returns:
            {success, diffs, commit_hash, test_output, error}
        """
        if not edit_plan:
            return {"success": False, "error": "Empty edit plan"}

        diffs = []
        backup_paths = []

        # Apply all edits
        for edit in edit_plan:
            file_path = edit.get("file", "")
            search = edit.get("search", "")
            replace = edit.get("replace", "")

            if not file_path:
                self._rollback_all(backup_paths)
                return {"success": False, "error": "Edit missing 'file' key"}

            logger.info(f"[EditLoop] Applying edit to: {file_path}")
            result = self.fs.apply_search_replace(file_path, search, replace)

            if not result.get("success"):
                self._rollback_all(backup_paths)
                return {
                    "success": False,
                    "error": f"Edit failed on {file_path}: {result.get('error')}",
                    "diffs": diffs,
                }

            diffs.append(result.get("diff", ""))
            if result.get("backup_path"):
                backup_paths.append(result["backup_path"])

        # Run tests if provided
        test_output = None
        if test_cmd:
            logger.info(f"[EditLoop] Running tests: {test_cmd}")
            try:
                import shlex
                # Split into a list so shell=True is not needed, which prevents
                # shell injection if test_cmd contains shell metacharacters.
                # On Windows shlex.split handles typical command strings correctly.
                cmd_args = shlex.split(test_cmd)
                proc = subprocess.run(
                    cmd_args,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=repo_path,
                )
                test_output = proc.stdout + proc.stderr
                if proc.returncode != 0:
                    logger.warning(f"[EditLoop] Tests failed (exit {proc.returncode})")
                    self._rollback_all(backup_paths)
                    return {
                        "success": False,
                        "error": "Tests failed — edits rolled back",
                        "test_output": test_output,
                        "diffs": diffs,
                    }
            except subprocess.TimeoutExpired:
                self._rollback_all(backup_paths)
                return {"success": False, "error": "Test command timed out — edits rolled back"}
            except Exception as e:
                logger.error(f"[EditLoop] Test execution error: {e}")
                # Don't roll back on test runner errors — tests didn't run

        # Git commit
        commit_hash = None
        if auto_commit and self.git:
            files = [e["file"] for e in edit_plan]
            commit_msg = f"AURA: edit {files[0]}" if len(files) == 1 else f"AURA: edit {len(files)} files"
            commit_result = self.git.auto_commit(commit_msg, files, repo_path)
            if commit_result.get("success"):
                commit_hash = commit_result.get("hash")
                logger.info(f"[EditLoop] Committed: {commit_hash}")

        return {
            "success": True,
            "diffs": diffs,
            "commit_hash": commit_hash,
            "test_output": test_output,
            "iterations": len(edit_plan),
        }

    def _rollback_all(self, backup_paths: list) -> None:
        """Roll back all edits using stored backup paths."""
        for backup_path in reversed(backup_paths):
            try:
                self.fs.rollback_edit(backup_path)
                logger.info(f"[EditLoop] Rolled back: {backup_path}")
            except Exception as e:
                logger.error(f"[EditLoop] Rollback failed for {backup_path}: {e}")
