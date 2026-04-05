"""Collector Hand — autonomous OSINT monitoring with change detection.

Runs on schedule (or when curiosity drive is high), monitors:
- URLs from world model watch targets
- KG nodes tagged as "monitoring"
- Topics from intrinsic motivation curiosity targets

Compares current content with previous snapshots stored in UnifiedMemory.
Reports changes as diffs via LLM summarization.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any

from aura.hands.base import Hand, HandManifest, HandResult

logger = logging.getLogger(__name__)


class CollectorHand(Hand):
    """Autonomous OSINT monitoring Hand — watches URLs/topics for changes."""

    def get_manifest(self) -> HandManifest:
        return HandManifest(
            name="collector",
            version="0.1.0",
            description="OSINT monitoring: watches URLs and topics for changes, reports diffs",
            interval_minutes=480,       # Every 8 hours
            idle_only=True,
            min_idle_seconds=600,       # 10 min idle
            max_tokens=15000,
            max_cost_usd=0.15,
            max_duration_seconds=600,   # 10 min max
            model_preference="fast",
            require_approval_for=["send_message", "write_file", "publish"],
            extra_blocked_tools=["code_executor", "shell"],
            max_iterations=6,
            trigger_on_drive="curiosity",
            trigger_drive_threshold=0.8,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are Aura's collector hand — an OSINT monitoring agent. "
            "Your job is to check watched URLs and topics for changes, "
            "compare with previous snapshots, and report meaningful differences. "
            "Be concise — only report real changes, not formatting noise.\n\n"
            "Rules:\n"
            "1. Check at most 3 targets per run — breadth over depth.\n"
            "2. Ignore minor formatting or timestamp changes.\n"
            "3. Flag significant content additions, removals, or shifts.\n"
            "4. Store updated snapshots for future comparison.\n"
            "5. Never publish or send messages without approval.\n"
        )

    async def execute(self, brain: Any, tools: dict, context: dict) -> HandResult:
        """Run OSINT monitoring cycle."""
        start = time.time()
        iterations = 0
        artifacts = []
        changes_found = []
        step_cb = context.get("step_callback")

        # Capture token state at start
        start_tokens = (
            getattr(brain, '_session_input_tokens', 0) +
            getattr(brain, '_session_output_tokens', 0)
        )
        start_cost = getattr(brain, '_session_cost_usd', 0.0)

        try:
            # Step 1: Find watch targets
            if step_cb:
                await step_cb(1, "Finding watch targets...")
            targets = self._get_watch_targets(context)
            if not targets:
                return HandResult(
                    hand_name="collector",
                    success=True,
                    summary="No watch targets configured. Add targets via KG or world model.",
                    iterations=0,
                    tokens_used=0,
                    cost_usd=0.0,
                    duration_seconds=time.time() - start,
                )

            logger.info(f"[Collector] Monitoring {len(targets)} targets")
            search_tool = tools.get("web_search") or tools.get("brave_search")

            # Step 2: Fetch each target (max 3)
            for target in targets[:3]:
                target_url = target.get("url") or target.get("topic", "")
                if not target_url:
                    continue

                if step_cb:
                    await step_cb(2, f"Fetching target: {target_url}...")

                iterations += 1

                # Fetch current content
                current_content = ""
                if search_tool and not target_url.startswith("http"):
                    # Topic-based: search for it
                    try:
                        results = search_tool.execute(query=target_url, num_results=3) if hasattr(search_tool, 'execute') else ""
                        current_content = str(results)[:3000] if results else ""
                    except Exception as e:
                        logger.debug(f"[Collector] Search failed for {target_url}: {e}")
                elif target_url.startswith("http"):
                    # URL-based: fetch directly
                    try:
                        browser_tool = tools.get("browser")
                        if browser_tool and hasattr(browser_tool, 'fetch'):
                            result = browser_tool.fetch(target_url)
                            current_content = result.get("text", "")[:3000] if isinstance(result, dict) else str(result)[:3000]
                        elif search_tool:
                            results = search_tool.execute(query=f"site:{target_url}", num_results=1) if hasattr(search_tool, 'execute') else ""
                            current_content = str(results)[:3000] if results else ""
                    except Exception as e:
                        logger.debug(f"[Collector] Fetch failed for {target_url}: {e}")

                if not current_content:
                    continue

                # Step 3: Compare with previous snapshot
                if step_cb:
                    await step_cb(3, "Comparing snapshots...")
                current_hash = hashlib.sha256(current_content.encode()).hexdigest()[:16]
                prev_snapshot = self._get_snapshot(target_url)

                if prev_snapshot and prev_snapshot.get("hash") == current_hash:
                    # No change
                    continue

                # Content changed — generate diff summary
                if step_cb:
                    await step_cb(4, "Generating diff summaries...")
                if prev_snapshot and brain:
                    diff_prompt = (
                        f"Compare these two versions of content from '{target_url}':\n\n"
                        f"PREVIOUS:\n{prev_snapshot.get('content', '')[:1500]}\n\n"
                        f"CURRENT:\n{current_content[:1500]}\n\n"
                        "Summarize the key changes in 2-3 sentences. Ignore formatting noise."
                    )
                    try:
                        response = await asyncio.to_thread(
                            lambda: brain.think(diff_prompt, system_prompt=self.get_system_prompt())
                        )
                        diff_summary = str(response)[:500] if response else "Content changed (no summary available)"
                        iterations += 1
                    except Exception:
                        diff_summary = "Content changed (LLM summary failed)"
                else:
                    diff_summary = "First snapshot captured — no previous version to compare."

                changes_found.append({
                    "target": target_url,
                    "summary": diff_summary,
                    "hash": current_hash,
                })

                artifacts.append({
                    "type": "change_detected",
                    "target": target_url,
                    "summary": diff_summary,
                    "new_hash": current_hash,
                    "prev_hash": prev_snapshot.get("hash") if prev_snapshot else None,
                })

                # Store updated snapshot
                self._store_snapshot(target_url, current_content, current_hash)

            # Build summary
            if changes_found:
                change_lines = [f"- {c['target']}: {c['summary']}" for c in changes_found]
                summary = f"Monitored {len(targets[:3])} targets, {len(changes_found)} changed:\n" + "\n".join(change_lines)
            else:
                summary = f"Monitored {len(targets[:3])} targets — no changes detected."

            # Compute tokens and cost delta
            end_tokens = (
                getattr(brain, '_session_input_tokens', 0) +
                getattr(brain, '_session_output_tokens', 0)
            )
            end_cost = getattr(brain, '_session_cost_usd', 0.0)
            tokens_used = end_tokens - start_tokens
            cost_used = end_cost - start_cost

            return HandResult(
                hand_name="collector",
                success=True,
                summary=summary,
                iterations=iterations,
                artifacts=artifacts,
                duration_seconds=time.time() - start,
                tokens_used=tokens_used,
                cost_usd=cost_used,
            )

        except Exception as e:
            # Compute tokens and cost delta even on error
            end_tokens = (
                getattr(brain, '_session_input_tokens', 0) +
                getattr(brain, '_session_output_tokens', 0)
            )
            end_cost = getattr(brain, '_session_cost_usd', 0.0)
            tokens_used = end_tokens - start_tokens
            cost_used = end_cost - start_cost

            return HandResult(
                hand_name="collector",
                success=False,
                summary=f"Collector failed: {e}",
                iterations=iterations,
                error=str(e),
                duration_seconds=time.time() - start,
                tokens_used=tokens_used,
                cost_usd=cost_used,
            )

    def _get_watch_targets(self, context: dict) -> list[dict]:
        """Get URLs/topics to monitor from world model and KG."""
        targets = []

        # Source 1: World model watch targets
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            if wm and hasattr(wm, 'get_watch_targets'):
                wm_targets = wm.get_watch_targets()
                if wm_targets:
                    targets.extend(wm_targets)
        except Exception:
            pass

        # Source 2: KG nodes tagged as "monitoring"
        try:
            from aura.memory.unified_memory import get_unified_memory
            umem = get_unified_memory()
            results = umem.query("monitoring watch target URL", k=5)
            if results:
                for r in results:
                    content = r.content if hasattr(r, 'content') else str(r)
                    # Extract URLs from content
                    import re
                    urls = re.findall(r'https?://[^\s<>"]+', content)
                    for url in urls[:2]:
                        if not any(t.get("url") == url for t in targets):
                            targets.append({"url": url, "source": "memory"})
        except Exception:
            pass

        # Source 3: Context-provided targets
        ctx_targets = context.get("watch_targets", [])
        if ctx_targets:
            targets.extend(ctx_targets)

        return targets

    def _get_snapshot(self, target: str) -> dict | None:
        """Get previous snapshot for a target from memory."""
        try:
            from aura.memory.unified_memory import get_unified_memory
            umem = get_unified_memory()
            results = umem.query(f"collector_snapshot:{target}", k=1)
            if results:
                content = results[0].content if hasattr(results[0], 'content') else ""
                if content and "collector_hash:" in content:
                    # Parse stored format: "collector_hash:<hash>\n<content>"
                    lines = content.split("\n", 1)
                    hash_val = lines[0].replace("collector_hash:", "").strip()
                    stored_content = lines[1] if len(lines) > 1 else ""
                    return {"hash": hash_val, "content": stored_content}
        except Exception:
            pass
        return None

    def _store_snapshot(self, target: str, content: str, content_hash: str):
        """Store a snapshot in memory for future comparison."""
        try:
            from aura.memory.unified_memory import get_unified_memory
            umem = get_unified_memory()
            snapshot_text = f"collector_hash:{content_hash}\n{content[:2000]}"
            umem.add(
                content=snapshot_text,
                metadata={
                    "source": "collector_hand",
                    "target": target,
                    "hash": content_hash,
                    "type": "collector_snapshot",
                },
            )
        except Exception as e:
            logger.debug(f"[Collector] Failed to store snapshot for {target}: {e}")
