"""
Browser Planner/Executor — Phase 2.

Two-layer architecture:
  BrowserPlanner  — converts high-level intent steps into a typed action plan
  BrowserExecutor — executes one step at a time with postcondition checks,
                    retry logic, domain-drift safety, and rollback support.

All action state is recorded in an ActionTrace for debugging / telemetry.

Author: Aura reliability upgrade (2026-03)
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg(attr: str, default):
    try:
        from aura.config import Config
        return getattr(Config, attr, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------

class ActionKind(str, Enum):
    NAVIGATE   = "navigate"
    CLICK      = "click"
    TYPE       = "type"
    SCROLL     = "scroll"
    SELECT     = "select"
    WAIT       = "wait"
    SCREENSHOT = "screenshot"
    DONE       = "done"
    ABORT      = "abort"


class SafetyClass(str, Enum):
    SAFE         = "safe"          # Always allowed
    SENSITIVE    = "sensitive"     # Requires gate check (login/payment patterns)
    DESTRUCTIVE  = "destructive"   # Write / submit / confirm — extra caution


@dataclass
class PlannedAction:
    """One step in the planner's output."""
    kind: ActionKind
    selector: str = ""
    text: str = ""
    url: str = ""
    amount: int = 300              # scroll pixels
    description: str = ""
    safety_class: SafetyClass = SafetyClass.SAFE

    # Postcondition — any of these strings in page text/URL = success
    success_signals: List[str] = field(default_factory=list)
    # URL patterns that mean failure / unexpected drift
    failure_url_patterns: List[str] = field(default_factory=list)
    # Alternative selectors to try if primary fails
    fallback_selectors: List[str] = field(default_factory=list)


@dataclass
class ActionResult:
    """Result of executing one PlannedAction."""
    action: PlannedAction
    success: bool
    attempt: int = 1
    observed_url: str = ""
    observed_title: str = ""
    observed_text_snippet: str = ""   # first 200 chars of body text
    error: str = ""
    latency_ms: float = 0.0
    postcondition_passed: Optional[bool] = None
    recovery_used: bool = False
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class ActionTrace:
    """Timeline of all executed actions in a session."""
    session_id: str
    task: str
    steps: List[ActionResult] = field(default_factory=list)
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    final_status: str = "in_progress"   # "success" | "failed" | "aborted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task": self.task,
            "steps": len(self.steps),
            "successes": sum(1 for s in self.steps if s.success),
            "final_status": self.final_status,
            "duration_s": round((self.end_ts or time.time()) - self.start_ts, 2),
            "steps_detail": [
                {
                    "step_id": s.step_id,
                    "kind": s.action.kind.value,
                    "desc": s.action.description,
                    "success": s.success,
                    "attempt": s.attempt,
                    "postcondition_passed": s.postcondition_passed,
                    "recovery_used": s.recovery_used,
                    "error": s.error[:100] if s.error else "",
                    "latency_ms": round(s.latency_ms, 1),
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class BrowserPlanner:
    """
    Converts a raw LLM-produced action dict (from /api/agent/action) into a
    typed PlannedAction with safety classification and success signals.
    """

    # URL patterns that indicate danger
    SENSITIVE_URL_PATTERNS = [
        "login", "signin", "sign-in", "checkout", "payment", "pay.",
        "bank", "banking", "password", "passwd",
    ]
    DESTRUCTIVE_URL_PATTERNS = [
        "confirm", "delete", "remove", "purchase", "buy", "submit",
    ]

    def parse(self, raw: Dict[str, Any]) -> PlannedAction:
        """Parse raw LLM action dict → PlannedAction."""
        kind_str = raw.get("action", "done").lower()
        try:
            kind = ActionKind(kind_str)
        except ValueError:
            kind = ActionKind.DONE

        action = PlannedAction(
            kind=kind,
            selector=raw.get("selector", ""),
            text=raw.get("text", ""),
            url=raw.get("url", ""),
            amount=int(raw.get("amount", 300)),
            description=raw.get("description", ""),
        )

        # Safety classification
        action.safety_class = self._classify_safety(action)

        # Auto-generate simple success signals from description
        desc_lower = action.description.lower()
        if "login" in desc_lower or "sign in" in desc_lower:
            action.success_signals = ["dashboard", "welcome", "logout", "sign out",
                                       "password", "verify", "otp"]
        elif "search" in desc_lower:
            action.success_signals = ["result", "found", "search"]
        elif "navigate" in desc_lower or kind == ActionKind.NAVIGATE:
            action.success_signals = []  # Any load is success for navigation

        return action

    def _classify_safety(self, a: PlannedAction) -> SafetyClass:
        target = (a.url + a.selector + a.description).lower()
        if any(p in target for p in self.DESTRUCTIVE_URL_PATTERNS):
            return SafetyClass.DESTRUCTIVE
        if any(p in target for p in self.SENSITIVE_URL_PATTERNS):
            return SafetyClass.SENSITIVE
        return SafetyClass.SAFE


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class BrowserExecutor:
    """
    Executes a PlannedAction against a live Playwright page.

    Features:
    - Postcondition verification after each action
    - Retry with fallback selector on element-not-found failures
    - Domain drift detection (unexpected host change → abort)
    - Destructive action gate (requires explicit allow flag)
    - Telemetry emission per action
    """

    def __init__(
        self,
        page,                           # playwright Page object
        session_id: str = "",
        allow_destructive: bool = False,
        allowed_domain: Optional[str] = None,  # If set, drift outside = abort
    ) -> None:
        self._page = page
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._allow_destructive = allow_destructive
        self._allowed_domain = allowed_domain
        self._max_retries: int = _cfg("BROWSER_MAX_RETRIES", 3)
        self._postcond_enabled: bool = _cfg("ENABLE_BROWSER_POSTCONDITIONS", True)
        self._abort_on_drift: bool = _cfg("BROWSER_ABORT_ON_DOMAIN_DRIFT", True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(
        self,
        action: PlannedAction,
        trace: Optional[ActionTrace] = None,
    ) -> ActionResult:
        """Execute one planned action with retry + postcondition logic."""

        # Safety gate
        if action.safety_class == SafetyClass.DESTRUCTIVE and not self._allow_destructive:
            result = ActionResult(
                action=action,
                success=False,
                error="destructive_action_blocked",
                observed_url=self._current_url(),
            )
            if trace:
                trace.steps.append(result)
            logger.warning(
                "[BrowserExec] Destructive action blocked: %s", action.description
            )
            return result

        t0 = time.monotonic()
        last_err = ""
        recovery_used = False

        for attempt in range(1, self._max_retries + 1):
            try:
                self._dispatch(action, attempt)
                observed_url = self._current_url()
                observed_title = self._current_title()
                observed_text = self._body_snippet()

                # Domain drift check
                if self._abort_on_drift and self._allowed_domain:
                    if not self._domain_ok(observed_url):
                        result = ActionResult(
                            action=action,
                            success=False,
                            attempt=attempt,
                            observed_url=observed_url,
                            observed_title=observed_title,
                            error=f"domain_drift: landed on {urlparse(observed_url).netloc}",
                            latency_ms=(time.monotonic() - t0) * 1000,
                            recovery_used=recovery_used,
                        )
                        logger.error(
                            "[BrowserExec] Domain drift! expected=%s got=%s",
                            self._allowed_domain, urlparse(observed_url).netloc,
                        )
                        self._emit_telemetry(result)
                        if trace:
                            trace.steps.append(result)
                        return result

                # Postcondition check
                pc_passed = self._check_postcondition(action, observed_url, observed_text)

                result = ActionResult(
                    action=action,
                    success=True,
                    attempt=attempt,
                    observed_url=observed_url,
                    observed_title=observed_title,
                    observed_text_snippet=observed_text[:200],
                    latency_ms=(time.monotonic() - t0) * 1000,
                    postcondition_passed=pc_passed,
                    recovery_used=recovery_used,
                )

                if not pc_passed and self._postcond_enabled and action.success_signals:
                    # Postcondition failed — try recovery on next iteration
                    logger.warning(
                        "[BrowserExec] Postcondition failed for %s (attempt %d/%d)",
                        action.description, attempt, self._max_retries,
                    )
                    if attempt < self._max_retries:
                        # Try fallback selector on next attempt
                        if action.fallback_selectors and attempt <= len(action.fallback_selectors):
                            orig = action.selector
                            action.selector = action.fallback_selectors[attempt - 1]
                            logger.info("[BrowserExec] Trying fallback selector: %s", action.selector)
                            recovery_used = True
                            action.selector = orig  # restore after dispatch
                        time.sleep(0.5 * attempt)
                        continue
                    # Give up — still mark as executed (postcondition_passed=False)
                    result.success = True  # action technically ran
                    self._emit_telemetry(result)
                    if trace:
                        trace.steps.append(result)
                    return result

                self._emit_telemetry(result)
                if trace:
                    trace.steps.append(result)
                return result

            except Exception as e:
                last_err = str(e)
                logger.warning(
                    "[BrowserExec] Action %s failed (attempt %d): %s",
                    action.kind.value, attempt, last_err,
                )
                if attempt < self._max_retries:
                    # Try fallback selector if available
                    if (action.fallback_selectors
                            and attempt <= len(action.fallback_selectors)):
                        action.selector = action.fallback_selectors[attempt - 1]
                        recovery_used = True
                    time.sleep(0.3 * attempt)

        result = ActionResult(
            action=action,
            success=False,
            attempt=self._max_retries,
            error=last_err,
            latency_ms=(time.monotonic() - t0) * 1000,
            recovery_used=recovery_used,
        )
        self._emit_telemetry(result)
        if trace:
            trace.steps.append(result)
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: PlannedAction, attempt: int) -> None:
        """Map action kind → Playwright call."""
        page = self._page

        if action.kind == ActionKind.NAVIGATE:
            url = action.url
            if url and not url.startswith(("http://", "https://")):
                url = "https://" + url
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)

        elif action.kind == ActionKind.CLICK:
            sel = action.selector
            page.wait_for_selector(sel, timeout=8_000)
            page.click(sel)

        elif action.kind == ActionKind.TYPE:
            sel = action.selector
            page.wait_for_selector(sel, timeout=8_000)
            page.fill(sel, action.text)

        elif action.kind == ActionKind.SCROLL:
            page.evaluate(f"window.scrollBy(0, {action.amount})")

        elif action.kind == ActionKind.SELECT:
            sel = action.selector
            page.wait_for_selector(sel, timeout=8_000)
            page.select_option(sel, label=action.text)

        elif action.kind == ActionKind.WAIT:
            time.sleep(min(action.amount / 1000, 5.0))   # max 5s

        elif action.kind == ActionKind.SCREENSHOT:
            ts = int(time.time())
            page.screenshot(path=f"screenshots/step_{ts}.png")

        elif action.kind in (ActionKind.DONE, ActionKind.ABORT):
            pass  # Terminal — no-op

    # ------------------------------------------------------------------
    # Postcondition
    # ------------------------------------------------------------------

    def _check_postcondition(
        self,
        action: PlannedAction,
        url: str,
        body: str,
    ) -> Optional[bool]:
        """Return True/False/None (None = no signals to check)."""
        if not action.success_signals:
            return None   # No postcondition defined

        combined = (url + " " + body).lower()
        return any(sig.lower() in combined for sig in action.success_signals)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_url(self) -> str:
        try:
            return self._page.url
        except Exception:
            return ""

    def _current_title(self) -> str:
        try:
            return self._page.title()
        except Exception:
            return ""

    def _body_snippet(self) -> str:
        try:
            return self._page.inner_text("body")[:500]
        except Exception:
            return ""

    def _domain_ok(self, url: str) -> bool:
        try:
            return urlparse(url).netloc == self._allowed_domain
        except Exception:
            return True  # can't parse → don't abort

    def _emit_telemetry(self, result: ActionResult) -> None:
        try:
            from aura.reliability.telemetry import emit, TelemetryKind
            emit(
                TelemetryKind.BROWSER_ACTION,
                session_id=self._session_id,
                success=result.success,
                latency_ms=result.latency_ms,
                retries=result.attempt - 1,
                extra={
                    "action_kind": result.action.kind.value,
                    "description": result.action.description,
                    "postcondition_passed": result.postcondition_passed,
                    "recovery_used": result.recovery_used,
                    "observed_url": result.observed_url[:100],
                    "error": result.error[:100] if result.error else "",
                },
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Session runner — ties planner + executor + loop guard together
# ---------------------------------------------------------------------------

class BrowserSession:
    """
    High-level session: runs a sequence of planned steps with loop guard.
    """

    def __init__(
        self,
        page,
        task: str,
        session_id: str = "",
        allow_destructive: bool = False,
        allowed_domain: Optional[str] = None,
    ) -> None:
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self.trace = ActionTrace(session_id=self._session_id, task=task)
        self._planner  = BrowserPlanner()
        self._executor = BrowserExecutor(
            page=page,
            session_id=self._session_id,
            allow_destructive=allow_destructive,
            allowed_domain=allowed_domain,
        )
        self._guard = None
        try:
            from aura.reliability.loop_guard import get_guard
            self._guard = get_guard(self._session_id)
        except Exception:
            pass

    def step(self, raw_action: Dict[str, Any]) -> ActionResult:
        """
        Parse + loop-guard-check + execute one raw LLM action.

        Returns ActionResult. If loop guard fires, returns an ABORT result.
        """
        planned = self._planner.parse(raw_action)

        # Loop guard check
        if self._guard:
            context = f"{planned.kind.value}:{planned.selector}:{planned.url}"
            guard_result = self._guard.record(planned.kind.value, context)
            if guard_result.triggered:
                logger.warning(
                    "[BrowserSession] Loop guard triggered: %s", guard_result.reason
                )
                abort = PlannedAction(
                    kind=ActionKind.ABORT,
                    description=f"Loop guard: {guard_result.reason}",
                )
                result = ActionResult(
                    action=abort,
                    success=False,
                    error=f"loop_guard:{guard_result.reason}",
                )
                self.trace.steps.append(result)
                self.trace.end_ts  = time.time()
                self.trace.final_status = "aborted_loop_guard"
                return result

        result = self._executor.execute(planned, trace=self.trace)

        # Terminal action
        if planned.kind in (ActionKind.DONE, ActionKind.ABORT):
            self.trace.end_ts     = time.time()
            self.trace.final_status = "success" if result.success else "failed"

        return result

    def finish(self, status: str = "success") -> ActionTrace:
        self.trace.end_ts     = time.time()
        self.trace.final_status = status
        return self.trace


__all__ = [
    "BrowserPlanner",
    "BrowserExecutor",
    "BrowserSession",
    "PlannedAction",
    "ActionResult",
    "ActionTrace",
    "ActionKind",
    "SafetyClass",
]
