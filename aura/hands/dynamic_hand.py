"""DynamicHand — config-driven Hand class for user-defined autonomous tasks.

Lets users create custom Hands at runtime by passing a config dict instead of
subclassing. The HandManager treats DynamicHand instances identically to
built-in Hands.
"""

import asyncio
import logging
import time
from typing import Any

from .base import Hand, HandManifest, HandResult

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are Aura's autonomous task hand. Your job is to complete the assigned "
    "goal efficiently and store useful findings in memory for future reference. "
    "Be concise, factual, and always cite your sources."
)


class DynamicHand(Hand):
    """Config-driven Hand — no subclassing required.

    Config schema
    -------------
    name              str          Required. Unique identifier.
    description       str          Required. Human-readable description.
    goal              str          Required. What this hand is trying to accomplish.
    search_queries    list[str]    Queries to run. Defaults to [goal].
    system_prompt     str          LLM system prompt. Defaults to generic.
    interval_minutes  int          How often to run. Default 240.
    idle_only         bool         Only run when user is idle. Default True.
    trigger_on_drive  str|None     Drive name to trigger on (e.g. "curiosity").
    trigger_drive_threshold float  Minimum drive urgency. Default 0.7.
    model_preference  str          "fast" | "reasoning" | "code". Default "fast".
    max_tokens        int          Token budget per run. Default 20000.
    max_cost_usd      float        Cost cap per run. Default 0.20.
    is_custom         bool         Always True for DynamicHand instances.
    """

    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        # Validate required fields early so errors surface at construction time
        for required in ("name", "description", "goal"):
            if not config.get(required):
                raise ValueError(f"DynamicHand config missing required field: '{required}'")

    # ------------------------------------------------------------------
    # Hand interface
    # ------------------------------------------------------------------

    def get_manifest(self) -> HandManifest:
        cfg = self._config
        return HandManifest(
            name=cfg["name"],
            description=cfg["description"],
            interval_minutes=int(cfg.get("interval_minutes", 240)),
            idle_only=bool(cfg.get("idle_only", True)),
            max_tokens=int(cfg.get("max_tokens", 20000)),
            max_cost_usd=float(cfg.get("max_cost_usd", 0.20)),
            model_preference=str(cfg.get("model_preference", "fast")),
            trigger_on_drive=cfg.get("trigger_on_drive") or None,
            trigger_drive_threshold=float(cfg.get("trigger_drive_threshold", 0.7)),
        )

    def get_system_prompt(self) -> str:
        return self._config.get("system_prompt") or _DEFAULT_SYSTEM_PROMPT

    async def execute(self, brain: Any, tools: dict, context: dict) -> HandResult:
        """Run the dynamic hand: search → synthesize → store."""
        start = time.time()
        cfg = self._config
        hand_name: str = cfg["name"]
        goal: str = cfg["goal"]
        search_queries: list[str] = cfg.get("search_queries") or [goal]
        step_cb = context.get("step_callback")

        # Capture token baseline
        start_tokens = (
            getattr(brain, "_session_input_tokens", 0)
            + getattr(brain, "_session_output_tokens", 0)
        )

        raw_findings: list[dict] = []
        iterations = 0

        try:
            # ----------------------------------------------------------------
            # Step 1: Search
            # ----------------------------------------------------------------
            if step_cb:
                await step_cb(1, f"Searching for: {', '.join(search_queries[:2])}...")

            from aura.tools.search_fallback import web_search_with_fallback
            for query in search_queries:
                try:
                    result = web_search_with_fallback(query=query, max_results=5)
                    if result.get("results"):
                        web_text = "\n".join(
                            f"- {r.get('title', '')}: {r.get('snippet', '')[:200]} ({r.get('url', '')})"
                            for r in result["results"][:5]
                        )
                        raw_findings.append({"query": query, "content": web_text})
                        iterations += 1
                except Exception as exc:
                    logger.debug(f"[DynamicHand:{hand_name}] Search failed for '{query}': {exc}")

            if not raw_findings:
                logger.debug(f"[DynamicHand:{hand_name}] No search results — proceeding with synthesis from goal only")

            # ----------------------------------------------------------------
            # Step 2: Synthesize
            # ----------------------------------------------------------------
            if step_cb:
                await step_cb(2, "Synthesizing findings...")

            findings_text = "\n\n---\n\n".join(
                f"Query: {f['query']}\nResults:\n{f['content']}" for f in raw_findings
            ) if raw_findings else "(No search results available — synthesize from prior knowledge.)"

            synthesis_prompt = (
                f"Goal: {goal}\n\n"
                f"Search findings:\n{findings_text}\n\n"
                "Synthesize the above into a concise, factual summary. "
                "Include key facts, any notable developments, and direct relevance to the goal. "
                "Format: short summary paragraph + bullet-point key takeaways."
            )

            response = await asyncio.to_thread(
                lambda: brain.think(
                    synthesis_prompt,
                    system_prompt=self.get_system_prompt(),
                    use_history=False,
                )
            )
            iterations += 1

            summary = str(response).strip() if response else ""

            # ----------------------------------------------------------------
            # Step 3: Store finding to UnifiedMemory
            # ----------------------------------------------------------------
            if step_cb:
                await step_cb(3, "Storing finding to memory...")

            if summary:
                try:
                    from aura.memory.unified_memory import get_unified_memory
                    um = get_unified_memory()
                    um.store(
                        content=f"[{hand_name}] {goal}: {summary}",
                        source="dynamic_hand",
                        importance=0.6,
                        tags=[hand_name, "dynamic_hand_finding"],
                        episode_type="dynamic_hand_finding",
                    )
                except Exception as exc:
                    logger.debug(f"[DynamicHand:{hand_name}] Memory store failed: {exc}")

            # ----------------------------------------------------------------
            # Compute token delta and return
            # ----------------------------------------------------------------
            end_tokens = (
                getattr(brain, "_session_input_tokens", 0)
                + getattr(brain, "_session_output_tokens", 0)
            )
            tokens_used = end_tokens - start_tokens
            end_cost = getattr(brain, "_session_cost_usd", 0.0)
            start_cost = getattr(brain, "_session_cost_usd", 0.0)  # approximation; delta may be 0 before first LLM call

            if not summary:
                return HandResult(
                    hand_name=hand_name,
                    success=False,
                    summary=f"LLM returned empty response for goal: {goal}",
                    iterations=iterations,
                    tokens_used=tokens_used,
                    cost_usd=0.0,
                    duration_seconds=time.time() - start,
                    error="Brain returned empty response",
                )

            return HandResult(
                hand_name=hand_name,
                success=True,
                summary=f"[{hand_name}] {summary[:300]}",
                iterations=iterations,
                tokens_used=tokens_used,
                cost_usd=0.0,
                duration_seconds=time.time() - start,
                artifacts=[
                    {
                        "type": "dynamic_hand_finding",
                        "hand_name": hand_name,
                        "goal": goal,
                        "summary": summary[:500],
                        "queries": search_queries,
                    }
                ],
            )

        except Exception as exc:
            end_tokens = (
                getattr(brain, "_session_input_tokens", 0)
                + getattr(brain, "_session_output_tokens", 0)
            )
            tokens_used = end_tokens - start_tokens
            return HandResult(
                hand_name=hand_name,
                success=False,
                summary=f"DynamicHand '{hand_name}' failed: {exc}",
                iterations=iterations,
                tokens_used=tokens_used,
                cost_usd=0.0,
                duration_seconds=time.time() - start,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Stats override
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["is_custom"] = True
        stats["goal"] = self._config.get("goal", "")
        return stats
