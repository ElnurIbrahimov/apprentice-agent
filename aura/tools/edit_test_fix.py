"""Edit-Test-Fix Loop — auto-correct failing tests after code edits.

When auto-verify detects test failures, this loop asks the LLM to propose
a fix, applies it via CodeEditTool, and re-runs tests — up to max_attempts.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class EditTestFixLoop:
    """Iterative fix loop: read test output → ask LLM for fix → apply → re-test."""

    def __init__(self, brain, code_edit_tool, shell_tool, max_attempts: int = 3):
        self.brain = brain
        self.editor = code_edit_tool
        self.shell = shell_tool
        self.max_attempts = max_attempts

    def run(self, file_path: str, edit_diff: str, test_output: str, project_root: str) -> dict:
        """Attempt to fix failing tests.

        Args:
            file_path: The file that was edited
            edit_diff: The diff/description of the original edit
            test_output: Output from the failing test run
            project_root: Root directory of the project

        Returns:
            dict with: success, attempts, final_output
        """
        from .auto_verify import auto_verify

        for attempt in range(self.max_attempts):
            logger.info(f"[EditTestFix] Attempt {attempt + 1}/{self.max_attempts}")

            # 1. Ask LLM for a fix
            fix_prompt = (
                f"Tests failed after editing {file_path}.\n"
                f"Test output (last 2000 chars):\n{test_output[-2000:]}\n\n"
                f"Previous edit diff:\n{edit_diff[:1000]}\n\n"
                f"Propose a fix. Respond with ONLY a JSON object:\n"
                f'{{"path": "file/to/edit.py", "old_string": "exact text to replace", "new_string": "replacement text"}}'
            )

            try:
                fix_response = self.brain.think(fix_prompt, use_history=False)
            except Exception as e:
                logger.warning(f"[EditTestFix] LLM error: {e}")
                continue

            # 2. Parse JSON fix
            fix = self._parse_fix(fix_response)
            if not fix:
                logger.debug(f"[EditTestFix] Could not parse fix from LLM response")
                continue

            # 3. Apply the fix
            try:
                edit_result = self.editor.edit(
                    path=fix["path"],
                    old_string=fix["old_string"],
                    new_string=fix["new_string"],
                )
            except Exception as e:
                logger.debug(f"[EditTestFix] Edit failed: {e}")
                continue

            if not edit_result.get("success"):
                logger.debug(f"[EditTestFix] Edit unsuccessful: {edit_result.get('error')}")
                continue

            # 4. Re-run tests
            verify = auto_verify(project_root, self.shell)
            if verify.get("success"):
                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "final_output": verify.get("output", ""),
                    "fix_applied": fix,
                }

            # Update for next iteration
            test_output = verify.get("output", "")
            edit_diff = edit_result.get("diff", str(fix))

        return {
            "success": False,
            "attempts": self.max_attempts,
            "final_output": test_output[-1000:],
        }

    def _parse_fix(self, response: str) -> Optional[dict]:
        """Extract a JSON fix object from LLM response."""
        # Try to find JSON in the response
        # Look for { ... } blocks
        for match in re.finditer(r'\{[^{}]*\}', response, re.DOTALL):
            try:
                obj = json.loads(match.group())
                if all(k in obj for k in ("path", "old_string", "new_string")):
                    return obj
            except json.JSONDecodeError:
                continue

        # Try the whole response as JSON
        try:
            obj = json.loads(response.strip())
            if all(k in obj for k in ("path", "old_string", "new_string")):
                return obj
        except json.JSONDecodeError:
            pass

        return None
