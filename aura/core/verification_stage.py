"""First-class verification stage for AgenticLoop.

Runs typecheck and/or tests after mutating tool calls, emits LoopEvents so
the CLI (or any surface) can display progress, and injects structured
failures back into the conversation so the agent self-corrects on the next
turn instead of declaring success.

This replaces the ad-hoc `_run_auto_test` callback pattern with a real
loop stage that:
  - knows its mode (typecheck / tests / both / none) from AURA.md config
  - scopes work to changed files when possible (fast)
  - emits verification_start / verification_passed / verification_failed
  - falls closed: if the stage itself crashes, the agent still gets a
    failure record so it doesn't unknowingly proceed
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────

@dataclass
class VerificationOutcome:
    """Structured result of a verification run.

    Either stage/runner pair succeeds or fails; the agent sees this via
    `to_conversation_message()` which is what gets appended to history on
    failure.
    """
    mode: str                              # "typecheck" / "tests" / "both" / "none"
    success: bool
    duration_s: float
    stages: list[dict] = field(default_factory=list)
    # Each stage dict: {name, runner, success, duration_s, failures: [{file, line, message}], stdout, stderr}
    changed_files: list[str] = field(default_factory=list)
    skipped_reason: str = ""

    def to_conversation_message(self) -> str:
        """Format the failure for injection into conversation history.

        Only called on failure. Kept terse so it doesn't overwhelm context.
        """
        if self.success:
            return ""
        lines: list[str] = ["[Verification failed]"]
        for st in self.stages:
            if st.get("success"):
                continue
            name = st.get("name", "?")
            runner = st.get("runner", "?")
            lines.append(f"  {name} ({runner}):")
            for f in st.get("failures", [])[:10]:  # cap to 10 per stage
                file = f.get("file", "?")
                line = f.get("line", "?")
                msg = f.get("message", "?")
                lines.append(f"    {file}:{line}  {msg}")
            remaining = len(st.get("failures", [])) - 10
            if remaining > 0:
                lines.append(f"    ... and {remaining} more")
            # If there were no structured failures but the stage still
            # failed, surface the raw stderr so the agent has something
            # to work with.
            if not st.get("failures"):
                err = (st.get("stderr") or st.get("stdout") or "").strip()
                if err:
                    snip = err[:500] + ("\n..." if len(err) > 500 else "")
                    lines.append(f"    {snip}")
        lines.append("Fix these before continuing.")
        return "\n".join(lines)


# ── Stage ─────────────────────────────────────────────────────────────────

_VALID_MODES = {"typecheck", "tests", "both", "none"}


class VerificationStage:
    """Run typecheck and/or tests on changed files after an edit iteration."""

    def __init__(
        self,
        project_root: str,
        aura_config: Optional[dict] = None,
        shell_tool: Any = None,
    ) -> None:
        cfg = (aura_config or {}).get("verification", {})
        # Back-compat: legacy top-level auto_test: true → verification.on_edit: both
        if (aura_config or {}).get("auto_test") and not cfg.get("on_edit"):
            logger.info(
                "[VerificationStage] Legacy auto_test:true interpreted as verification.on_edit:both"
            )
            cfg = {**cfg, "on_edit": "both"}

        mode = str(cfg.get("on_edit", "typecheck")).lower()
        if mode not in _VALID_MODES:
            logger.warning(
                f"[VerificationStage] Unknown mode '{mode}'; defaulting to 'typecheck'"
            )
            mode = "typecheck"

        self.mode: str = mode
        self.project_root: str = project_root
        self.shell_tool = shell_tool
        self.tests_cmd: str = str(cfg.get("tests_cmd", "") or "")
        self.typecheck_cmd: str = str(cfg.get("typecheck_cmd", "") or "")
        self.timeout_s: int = int(cfg.get("timeout_s", 30))

    def should_run(self, tool_name: str) -> bool:
        """True if the stage should run after this tool."""
        if self.mode == "none":
            return False
        return tool_name in _EDIT_TOOL_NAMES

    def run(
        self,
        changed_files: list[str],
        emitter: Optional[Any] = None,
        session_id: str = "",
    ) -> VerificationOutcome:
        """Dispatch verification by mode.

        *emitter* is the LoopEventEmitter (optional — runs headless if None).
        *session_id* is attached to event-log entries for later `/why` lookup.
        """
        if self.mode == "none" or not changed_files:
            return VerificationOutcome(
                mode=self.mode, success=True, duration_s=0.0,
                changed_files=list(changed_files),
                skipped_reason="mode=none" if self.mode == "none" else "no changed files",
            )

        if emitter is not None:
            try:
                emitter.emit(
                    "verification_start",
                    mode=self.mode,
                    changed_files=list(changed_files),
                )
            except Exception:
                logger.debug("verification_start emit failed", exc_info=True)

        start = time.monotonic()
        stages: list[dict] = []
        overall_success = True

        if self.mode in ("typecheck", "both"):
            stage_result = self._run_typecheck(changed_files)
            stages.append(stage_result)
            if not stage_result["success"]:
                overall_success = False

        # "both": skip tests if typecheck already failed (fail fast).
        should_run_tests = (
            self.mode == "tests"
            or (self.mode == "both" and overall_success)
        )
        if should_run_tests:
            stage_result = self._run_tests(changed_files)
            stages.append(stage_result)
            if not stage_result["success"]:
                overall_success = False

        outcome = VerificationOutcome(
            mode=self.mode,
            success=overall_success,
            duration_s=time.monotonic() - start,
            stages=stages,
            changed_files=list(changed_files),
        )

        # Log to JSONL event log (best-effort, non-blocking).
        try:
            from aura.core.event_log import log_verification
            for st in stages:
                log_verification(
                    session_id=session_id,
                    stage=st.get("name", "?"),
                    runner=st.get("runner", "?"),
                    status="passed" if st.get("success") else "failed",
                    duration_s=st.get("duration_s", 0.0),
                    failures=st.get("failures", []),
                )
        except Exception:
            logger.debug("verification event-log write failed", exc_info=True)

        if emitter is not None:
            try:
                if outcome.success:
                    emitter.emit(
                        "verification_passed",
                        mode=outcome.mode,
                        duration_s=outcome.duration_s,
                        stages=[
                            {"name": s["name"], "runner": s.get("runner", "")}
                            for s in stages
                        ],
                    )
                else:
                    emitter.emit(
                        "verification_failed",
                        mode=outcome.mode,
                        duration_s=outcome.duration_s,
                        stages=stages,
                    )
            except Exception:
                logger.debug("verification result emit failed", exc_info=True)

        return outcome

    # ── Private runners ──────────────────────────────────────────────────

    def _run_typecheck(self, changed_files: list[str]) -> dict:
        from aura.tools.typecheck import typecheck_changed_files
        try:
            res = typecheck_changed_files(
                self.project_root,
                changed_files,
                timeout=self.timeout_s,
                override_cmd=self.typecheck_cmd or None,
            )
            failures = [
                {
                    "file": d.file,
                    "line": d.line,
                    "col": d.col,
                    "code": d.code,
                    "severity": d.severity,
                    "message": d.message,
                }
                for d in res.diagnostics
                if d.severity == "error"
            ]
            return {
                "name": "typecheck",
                "runner": res.runner,
                "success": res.success,
                "duration_s": res.duration_s,
                "failures": failures,
                "stdout": res.stdout[:2000],
                "stderr": res.stderr[:2000],
                "skipped_reason": res.skipped_reason,
            }
        except Exception as e:
            # Fail closed: report stage failure so the agent investigates
            # rather than silently skipping.
            logger.exception("VerificationStage._run_typecheck crashed")
            return {
                "name": "typecheck",
                "runner": "error",
                "success": False,
                "duration_s": 0.0,
                "failures": [],
                "stdout": "",
                "stderr": f"typecheck stage crashed: {e}",
            }

    def _run_tests(self, changed_files: list[str]) -> dict:
        from aura.tools.auto_verify import run_changed_tests
        try:
            res = run_changed_tests(
                self.project_root,
                changed_files,
                shell_tool=self.shell_tool,
                timeout=max(60, self.timeout_s * 2),  # tests need more time
                override_cmd=self.tests_cmd or None,
            )
            return {
                "name": "tests",
                "runner": res.get("framework", "unknown"),
                "success": res.get("success", True),
                "duration_s": res.get("duration_s", 0.0),
                "failures": res.get("failures", []),
                "stdout": (res.get("output") or "")[:2000],
                "stderr": "",
                "skipped_reason": res.get("reason", "") if res.get("skipped") else "",
            }
        except Exception as e:
            logger.exception("VerificationStage._run_tests crashed")
            return {
                "name": "tests",
                "runner": "error",
                "success": False,
                "duration_s": 0.0,
                "failures": [],
                "stdout": "",
                "stderr": f"tests stage crashed: {e}",
            }


# Must match aura/cli/chat_session_execution.py._EDIT_TOOL_NAMES.
_EDIT_TOOL_NAMES = {
    "edit_file", "write_file", "patch_file", "apply_diff", "str_replace_editor",
}
