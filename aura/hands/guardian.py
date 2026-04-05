"""Guardian Hand — self-monitoring and security watchdog.

Runs periodically to:
- Verify audit chain integrity
- Check for anomalous behavior patterns
- Scan recent memory writes for leaked secrets
- Monitor resource usage (tokens, cost)
- Flag stale or contradictory knowledge

This is Aura watching itself — enabled by the consciousness stack.
"""

import logging
import time
from typing import Any

from aura.hands.base import Hand, HandManifest, HandResult

logger = logging.getLogger(__name__)


class GuardianHand(Hand):
    """Security and integrity watchdog Hand."""

    def get_manifest(self) -> HandManifest:
        return HandManifest(
            name="guardian",
            version="0.1.0",
            description="Self-monitoring: audit chain integrity, secret leak detection, anomaly flagging",
            interval_minutes=120,       # Every 2 hours
            idle_only=False,            # Security checks run regardless
            min_idle_seconds=0,
            max_tokens=10000,
            max_cost_usd=0.10,
            max_duration_seconds=300,   # 5 min max (mostly local checks)
            model_preference="fast",
            require_approval_for=[],    # Read-only — no approvals needed
            extra_blocked_tools=["code_executor", "write_file", "edit_file"],
            max_iterations=5,
            trigger_on_drive="coherence",
            trigger_drive_threshold=0.8,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are Aura's guardian — a security and integrity watchdog. "
            "Your job is to monitor Aura's own systems for anomalies, "
            "leaked secrets, audit chain tampering, and resource abuse. "
            "Report findings clearly. Never modify anything — only observe and report."
        )

    async def execute(self, brain: Any, tools: dict, context: dict) -> HandResult:
        """Run guardian checks."""
        start = time.time()
        findings = []
        issues = []
        iterations = 0
        step_cb = context.get("step_callback")

        # Check 1: Audit chain integrity
        if step_cb:
            await step_cb(1, "Verifying audit chain integrity...")
        try:
            from aura.security.audit_chain import get_audit_chain
            chain = get_audit_chain()
            is_valid, entries_checked, error = chain.verify(limit=1000)
            iterations += 1

            if is_valid:
                findings.append(f"Audit chain: OK ({entries_checked} entries verified)")
            else:
                issues.append(f"AUDIT CHAIN TAMPERED: {error}")
                logger.error(f"[Guardian] Audit chain integrity failure: {error}")
        except Exception as e:
            findings.append(f"Audit chain: could not verify ({e})")

        # Check 2: Scan recent memory for leaked secrets
        if step_cb:
            await step_cb(2, "Scanning for leaked secrets...")
        try:
            from aura.security.taint_tracker import scan_for_secrets
            # Check monologue logs for leaked secrets
            import glob
            import os
            data_dir = os.environ.get("AURA_DATA_DIR", "data")
            log_dir = os.path.join(data_dir, "inner_monologue", "sessions")
            if os.path.isdir(log_dir):
                recent_logs = sorted(glob.glob(os.path.join(log_dir, "*.jsonl")))[-5:]
                leaked_count = 0
                for log_file in recent_logs:
                    try:
                        with open(log_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        matches = scan_for_secrets(content, check_pii=False)
                        if matches:
                            leaked_count += len(matches)
                            issues.append(
                                f"SECRET IN LOG: {os.path.basename(log_file)} "
                                f"contains {len(matches)} secret(s): "
                                f"{', '.join(m.description for m in matches[:3])}"
                            )
                    except Exception as e:
                        logger.warning(f"[Guardian] Failed to scan log {log_file}: {e}")
                iterations += 1

                if leaked_count == 0:
                    findings.append("Monologue logs: no secrets detected")
                else:
                    findings.append(f"Monologue logs: {leaked_count} secret(s) found!")
        except Exception as e:
            findings.append(f"Secret scan: could not run ({e})")

        # Check 3: Hand health (resource usage trends)
        if step_cb:
            await step_cb(3, "Checking hand health...")
        try:
            from aura.hands.manager import get_hand_manager
            manager = get_hand_manager()
            hand_stats = manager.list_hands()
            total_cost = sum(h.get("total_cost", 0) for h in hand_stats)
            high_failure = [
                h["name"] for h in hand_stats
                if h.get("consecutive_failures", 0) >= 3
            ]
            iterations += 1

            findings.append(f"Total Hand cost: ${total_cost:.4f}")
            if high_failure:
                issues.append(f"HANDS WITH REPEATED FAILURES: {', '.join(high_failure)}")
            else:
                findings.append("All Hands: healthy (no repeated failures)")
        except Exception as e:
            findings.append(f"Hand health: could not check ({e})")

        # Check 3.5: Taint tracker stats
        try:
            from aura.security.taint_tracker import get_tracker
            stats = get_tracker().get_stats()
            iterations += 1
            if stats["total_detections"] > 0:
                findings.append(
                    f"Taint tracker: {stats['total_detections']} detections, "
                    f"{stats['total_redactions']} redactions across {stats['active_sessions']} sessions"
                )
            else:
                findings.append("Taint tracker: no secrets detected in current session")
        except Exception as e:
            findings.append(f"Taint stats: could not check ({e})")

        # Check 4: Knowledge graph contradictions
        if step_cb:
            await step_cb(4, "Reviewing knowledge graph contradictions...")
        try:
            from aura.memory.kg_contradiction import get_contradictions
            contradictions = get_contradictions(limit=10)
            iterations += 1

            if contradictions:
                issues.append(f"KG CONTRADICTIONS: {len(contradictions)} unresolved")
                findings.append(f"Knowledge graph: {len(contradictions)} contradictions found")
            else:
                findings.append("Knowledge graph: no contradictions")
        except Exception as e:
            findings.append(f"KG check: could not run ({e})")

        # Build summary
        status = "ISSUES FOUND" if issues else "ALL CLEAR"
        summary_lines = [f"Guardian report: {status}"]
        if issues:
            summary_lines.append("Issues:")
            summary_lines.extend(f"  - {i}" for i in issues)
        summary_lines.append("Checks:")
        summary_lines.extend(f"  + {f}" for f in findings)
        summary = "\n".join(summary_lines)

        # Log issues at warning level
        for issue in issues:
            logger.warning(f"[Guardian] {issue}")

        return HandResult(
            hand_name="guardian",
            success=len(issues) == 0,
            summary=summary,
            iterations=iterations,
            artifacts=[{"type": "guardian_report", "issues": issues, "findings": findings}],
            duration_seconds=time.time() - start,
        )
