"""Researcher Hand — autonomous deep research on topics of interest.

Runs on schedule (or when curiosity drive is high), picks topics from:
- Knowledge graph gaps (orphan nodes, low-confidence facts)
- World model's active projects (stale ones that need updates)
- Intrinsic motivation's curiosity targets

Produces: memory entries, KG updates, research summaries.
"""

import asyncio
import logging
import time
from typing import Any

from aura.hands.base import Hand, HandManifest, HandResult

logger = logging.getLogger(__name__)


class ResearcherHand(Hand):
    """Autonomous research Hand — explores knowledge gaps without prompting."""

    def get_manifest(self) -> HandManifest:
        return HandManifest(
            name="researcher",
            version="0.1.0",
            description="Autonomous deep research on topics from KG gaps and curiosity drive",
            interval_minutes=240,       # Every 4 hours
            idle_only=True,
            min_idle_seconds=600,       # 10 min idle
            max_tokens=40000,
            max_cost_usd=0.40,
            max_duration_seconds=1200,  # 20 min max
            model_preference="reasoning",
            require_approval_for=["send_message", "write_file", "publish"],
            extra_blocked_tools=["code_executor"],
            max_iterations=8,
            trigger_on_drive="curiosity",
            trigger_drive_threshold=0.7,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are Aura's autonomous research hand. Your job is to explore "
            "knowledge gaps, verify uncertain facts, and discover new connections. "
            "You operate independently — no human is watching. Be thorough but efficient.\n\n"
            "Rules:\n"
            "1. Pick ONE topic per run — depth over breadth.\n"
            "2. Cross-reference at least 2 sources before updating knowledge.\n"
            "3. If you find contradictions with existing knowledge, flag them explicitly.\n"
            "4. Store findings in memory with clear source attribution.\n"
            "5. Never publish, send messages, or modify files without approval.\n"
            "6. If a topic turns out to be a dead end, record why and move on.\n"
        )

    async def execute(self, brain: Any, tools: dict, context: dict) -> HandResult:
        """Run autonomous research cycle."""
        start = time.time()
        iterations = 0
        artifacts = []
        topic = None
        step_cb = context.get("step_callback")

        # Capture token state at start
        start_tokens = (
            getattr(brain, '_session_input_tokens', 0) +
            getattr(brain, '_session_output_tokens', 0)
        )
        start_cost = getattr(brain, '_session_cost_usd', 0.0)

        try:
            # Step 1: Pick a research topic
            if step_cb:
                await step_cb(1, "Selecting research topic...")
            topic = await self._pick_topic(brain, context)
            if not topic:
                return HandResult(
                    hand_name="researcher",
                    success=True,
                    summary="No research topics found — knowledge is up to date.",
                    iterations=0,
                )

            logger.info(f"[Researcher] Researching: {topic}")

            # Step 2: Query memory for prior knowledge
            if step_cb:
                await step_cb(2, "Querying memory for prior knowledge...")
            findings = []

            # Check existing knowledge via UnifiedMemory
            # (memory_retriever was consolidated into UnifiedMemory — use it directly)
            try:
                from aura.memory.unified_memory import get_unified_memory
                umem = get_unified_memory()
                results = umem.query(topic, k=5)
                if results:
                    existing = "\n".join(r.content for r in results if r.content)
                    if existing:
                        findings.append({"source": "memory", "content": existing[:2000]})
                        iterations += 1
            except Exception as e:
                logger.debug(f"[Researcher] Memory lookup failed: {e}")

            # Step 3: Search the web for new information
            if step_cb:
                await step_cb(3, "Searching the web...")
            try:
                from aura.tools.search_fallback import web_search_with_fallback
                search_result = web_search_with_fallback(query=topic, max_results=5)
                if search_result.get("results"):
                    web_text = "\n".join(
                        f"- {r.get('title', '')}: {r.get('snippet', '')[:200]} ({r.get('url', '')})"
                        for r in search_result["results"][:5]
                    )
                    findings.append({"source": "web", "content": web_text})
                    iterations += 1
            except Exception as e:
                logger.debug(f"[Researcher] Web search failed: {e}")

            if not findings:
                return HandResult(
                    hand_name="researcher",
                    success=False,
                    summary=f"Could not find information on: {topic}",
                    iterations=iterations,
                    error="No search tools available or all searches failed",
                    tokens_used=0,
                    cost_usd=0.0,
                )

            # Step 4: Synthesize findings via LLM
            if step_cb:
                await step_cb(4, "Synthesizing findings...")
            synthesis_prompt = (
                f"Research topic: {topic}\n\n"
                f"Findings:\n" + "\n---\n".join(
                    f"[{f['source']}]: {f['content']}" for f in findings
                ) + "\n\n"
                "Synthesize these findings into a concise knowledge update. "
                "Note any contradictions with what was already known. "
                "Format: one paragraph summary + key facts as bullet points."
            )

            response = await asyncio.to_thread(
                lambda: brain.think(
                    synthesis_prompt,
                    system_prompt=self.get_system_prompt(),
                    model_override=None,
                )
            )
            iterations += 1

            if response:
                summary = str(response)[:2000]
                artifacts.append({
                    "type": "research_finding",
                    "topic": topic,
                    "summary": summary[:500],
                    "sources": [f["source"] for f in findings],
                })

                # Store in memory
                try:
                    from aura.memory.unified_memory import get_unified_memory
                    um = get_unified_memory()
                    um.add(
                        content=f"Research finding on '{topic}': {summary}",
                        metadata={"type": "research_finding", "topic": topic, "hand": "researcher"},
                    )
                except Exception as e:
                    logger.debug(f"[ResearcherHand] Failed to save finding to memory: {e}")

                logger.info(f"[Researcher] Completed research on: {topic}")

                # Compute tokens and cost delta
                end_tokens = (
                    getattr(brain, '_session_input_tokens', 0) +
                    getattr(brain, '_session_output_tokens', 0)
                )
                end_cost = getattr(brain, '_session_cost_usd', 0.0)
                tokens_used = end_tokens - start_tokens
                cost_used = end_cost - start_cost

                return HandResult(
                    hand_name="researcher",
                    success=True,
                    summary=f"Researched '{topic}': {summary[:200]}...",
                    iterations=iterations,
                    artifacts=artifacts,
                    duration_seconds=time.time() - start,
                    tokens_used=tokens_used,
                    cost_usd=cost_used,
                )

            # Compute tokens and cost delta
            end_tokens = (
                getattr(brain, '_session_input_tokens', 0) +
                getattr(brain, '_session_output_tokens', 0)
            )
            end_cost = getattr(brain, '_session_cost_usd', 0.0)
            tokens_used = end_tokens - start_tokens
            cost_used = end_cost - start_cost

            return HandResult(
                hand_name="researcher",
                success=False,
                summary=f"LLM synthesis failed for topic: {topic}",
                iterations=iterations,
                error="Brain returned empty response",
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
                hand_name="researcher",
                success=False,
                summary=f"Research failed for topic: {topic or 'unknown'}",
                iterations=iterations,
                error=str(e),
                duration_seconds=time.time() - start,
                tokens_used=tokens_used,
                cost_usd=cost_used,
            )

    async def _pick_topic(self, brain: Any, context: dict) -> str | None:
        """Pick the most valuable research topic."""
        candidates = []

        # Source 1: Curiosity scanner targets
        try:
            context.get("drive_urgencies", {})
            curiosity_targets = context.get("curiosity_targets", [])
            if curiosity_targets:
                candidates.extend(curiosity_targets[:3])
        except Exception:
            pass

        # Source 2: KG gaps (orphan nodes, low-confidence)
        try:
            from aura.consciousness.intrinsic_motivation import get_motivation_engine
            engine = get_motivation_engine()
            if engine:
                state = engine.get_drive_state("curiosity")
                if state and state.triggers:
                    candidates.extend(state.triggers[:3])
        except Exception:
            pass

        # Source 3: World model stale projects
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            if wm:
                projects = wm.get_stale_projects(days=7)
                if projects:
                    candidates.extend([f"updates on {p['name']}" for p in projects[:2]])
        except Exception:
            pass

        if not candidates:
            return None

        # Pick the first viable candidate (could use LLM to rank later)
        return candidates[0] if isinstance(candidates[0], str) else str(candidates[0])
