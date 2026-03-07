# D:/Aura/aura/parliament.py
"""
ParliamentConductor — orchestrates AURA's intelligence modules.

Three tiers:
  SIMPLE   → FastPath (~50ms, 0 extra LLM calls)
  STANDARD → Single brain.think() + async MirrorMind score
  COMPLEX  → Parallel: Proposer + CognitiveTheater, then optional synthesis
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

logger = logging.getLogger(__name__)

try:
    from aura.tools.cognitive_theater import is_decision_question as _is_decision_question
except ImportError:
    _is_decision_question = None

_CONCISE_SYSTEM_HINT = (
    "Reply conversationally and concisely. "
    "Do NOT use structured analysis formats, multi-perspective breakdowns, "
    "Pro/Con sections, or Confidence ratings unless explicitly asked."
)


class QueryTier(Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


@dataclass
class ParliamentResult:
    tier: QueryTier
    response: str
    proposer: str = ""
    synthesis_used: bool = False
    latency_ms: float = 0.0


class ParliamentConductor:
    """
    Routes queries to the right level of intelligence.
    Instantiated once in ApprenticeAgent.__init__().

    Usage:
        response = parliament.handle(query, context_addon="")
    """

    def __init__(self, agent):
        self.agent = agent
        self._executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="parliament")

    def classify(self, query: str) -> QueryTier:
        """Classify query tier — no LLM calls."""
        # Use IntrospectionCircuit if available
        if getattr(self.agent, 'introspection', None):
            try:
                qtype = self.agent.introspection._classify_query(query)
                qval = qtype.value if hasattr(qtype, 'value') else str(qtype)
                if qval in ("conversational", "greeting"):
                    return QueryTier.SIMPLE
                if qval in ("analytical", "decision", "multi_step", "research", "procedural"):
                    return QueryTier.COMPLEX
            except Exception:
                pass

        # Fallback heuristics
        try:
            if _is_decision_question and _is_decision_question(query):
                return QueryTier.COMPLEX
        except Exception:
            pass

        if len(query.split()) < 8:
            return QueryTier.SIMPLE
        return QueryTier.STANDARD

    def handle(self, query: str, context_addon: str = "") -> str:
        """Main entry point. Returns response text."""
        t0 = time.time()

        # Loop guard: detect repeated / low-novelty reasoning cycles
        try:
            from aura.reliability.loop_guard import get_guard
            _session_id = getattr(self.agent, '_session_id', 'parliament-default')
            _guard = get_guard(_session_id)
            _guard_result = _guard.record(f"parliament:{query[:80]}", context=query)
            if _guard_result.triggered:
                logger.warning(
                    "[PARLIAMENT] Loop guard triggered: %s (tier=%s)",
                    _guard_result.reason, _guard_result.actions_taken
                )
                return _guard_result.fallback_message
        except Exception:
            pass

        tier = self.classify(query)

        if tier == QueryTier.SIMPLE:
            response = self._simple_response(query, context_addon)
        elif tier == QueryTier.STANDARD:
            response = self._standard_response(query, context_addon)
            self._async_mirrormind_score(query, response)
        else:
            response = self._parliament_response(query, context_addon)

        latency = (time.time() - t0) * 1000
        logger.debug(f"[PARLIAMENT] {tier.value} tier, {latency:.0f}ms")
        return response

    def _simple_response(self, query: str, context_addon: str) -> str:
        concise_hint = _CONCISE_SYSTEM_HINT
        system = f"{context_addon}\n\n{concise_hint}" if context_addon else concise_hint
        return self.agent.brain.think(query, system_prompt=system)

    def _standard_response(self, query: str, context_addon: str) -> str:
        return self.agent.brain.think(query, system_prompt=context_addon or None)

    def _parliament_response(self, query: str, context_addon: str) -> str:
        """Parallel deliberation: Proposer + CognitiveTheater simultaneously."""
        futures = {}

        futures["proposer"] = self._executor.submit(
            self.agent.brain.think, query,
            system_prompt=context_addon or None
        )

        if getattr(self.agent, 'theater', None):
            futures["critic"] = self._executor.submit(
                self._get_theater_perspectives, query
            )

        results = {}
        for key, fut in futures.items():
            try:
                results[key] = fut.result(timeout=30)
            except (TimeoutError, Exception) as e:
                logger.debug(f"[PARLIAMENT] {key} failed: {e}")
                results[key] = None

        proposer = results.get("proposer") or ""
        critic = results.get("critic")

        # Only synthesize if Theater adds meaningful perspective
        if critic and proposer and self._worth_synthesizing(critic):
            try:
                synthesis_prompt = (
                    f"Synthesize into one clear, direct response.\n\n"
                    f"PRIMARY RESPONSE: {proposer[:1200]}\n\n"
                    f"ADDITIONAL PERSPECTIVES:\n{str(critic)[:800]}\n\n"
                    f"Original question: {query}\n\n"
                    f"Produce a single unified answer."
                )
                synthesized = self.agent.brain.think(synthesis_prompt, use_history=False)
                if synthesized:
                    return synthesized
            except Exception as e:
                logger.debug(f"[PARLIAMENT] Synthesis failed: {e}")

        return proposer or self.agent.brain.think(query, system_prompt=context_addon or None)

    def _get_theater_perspectives(self, query: str) -> Optional[str]:
        try:
            result = self.agent.theater.deliberate(query)
            if isinstance(result, dict):
                parts = []
                for role in ("critic", "analyst", "integrator"):
                    if result.get(role):
                        parts.append(f"{role.upper()}: {result[role][:200]}")
                return "\n".join(parts) if parts else None
            return str(result)[:600] if result else None
        except Exception as e:
            logger.debug(f"[PARLIAMENT] Theater failed: {e}")
            return None

    def _worth_synthesizing(self, critic_output: Any) -> bool:
        """Only synthesize if Theater produced substantive content."""
        if not critic_output:
            return False
        text = critic_output.strip() if isinstance(critic_output, str) else str(critic_output).strip()
        return len(text) > 80 and any(c.isalpha() for c in text)

    def handle_stream(self, query: str, context_addon: str = ""):
        """Streaming entry point — yields text chunks as they arrive.

        SIMPLE and STANDARD tiers stream directly via brain.think_stream().
        COMPLEX tier falls back to the blocking _parliament_response() and
        yields the complete result as one chunk (parallel synthesis can't stream).
        """
        # Loop guard check (same guard as handle(), shared per session)
        try:
            from aura.reliability.loop_guard import get_guard
            _session_id = getattr(self.agent, '_session_id', 'parliament-default')
            _guard = get_guard(_session_id)
            _guard_result = _guard.record(f"parliament:{query[:80]}", context=query)
            if _guard_result.triggered:
                logger.warning(
                    "[PARLIAMENT] Loop guard triggered (stream): %s", _guard_result.reason
                )
                yield _guard_result.fallback_message
                return
        except Exception:
            pass

        tier = self.classify(query)

        if tier in (QueryTier.SIMPLE, QueryTier.STANDARD):
            if tier == QueryTier.SIMPLE:
                stream_system = f"{context_addon}\n\n{_CONCISE_SYSTEM_HINT}" if context_addon else _CONCISE_SYSTEM_HINT
            else:
                stream_system = context_addon or None
            full_response = ""
            for chunk in self.agent.brain.think_stream(
                query,
                system_prompt=stream_system,
            ):
                full_response += chunk
                yield chunk
            if tier == QueryTier.STANDARD:
                self._async_mirrormind_score(query, full_response)
        else:
            # COMPLEX: parallel deliberation — can't stream mid-flight, yield whole response
            response = self._parliament_response(query, context_addon)
            if response:
                yield response

    def _async_mirrormind_score(self, query: str, response: str):
        """Fire MirrorMind scoring in background — never blocks response."""
        if not getattr(self.agent, 'mirrormind', None):
            return

        def _score():
            try:
                score = self.agent.mirrormind.quick_score(query, response)
                logger.debug(f"[PARLIAMENT] MirrorMind score: {score}")
            except Exception:
                pass

        threading.Thread(target=_score, daemon=True).start()
