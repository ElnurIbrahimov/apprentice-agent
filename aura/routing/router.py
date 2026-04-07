"""Main router entry point – orchestrates all three layers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aura.routing.profiles import ProfileStore, DIMENSIONS
from aura.routing.classifier import extract_features, score_task
from aura.routing.matcher import match, PREFERENCE_WEIGHTS
from aura.routing.conversation import ConversationTracker


@dataclass
class RoutingResult:
    model: str
    reason: str
    task_dimensions: Dict[str, float]
    alternatives: List[str] = field(default_factory=list)
    conversation_turn: int = 0
    latency_ms: float = 0.0


class Router:
    """Orchestrator that ties all three routing layers together."""

    def __init__(self, profiles_path: str = "data/model_profiles.json") -> None:
        self._profiles = ProfileStore(profiles_path)
        self._conversations = ConversationTracker()

    @property
    def profiles(self) -> ProfileStore:
        return self._profiles

    @property
    def conversations(self) -> ConversationTracker:
        return self._conversations

    def route(
        self,
        prompt: str,
        model: Optional[str] = None,
        preference: str = "balanced",
        feature: Optional[str] = None,
        conversation_id: Optional[str] = None,
        has_attachment: bool = False,
    ) -> RoutingResult:
        t0 = time.perf_counter()

        # 1. Explicit override
        if model is not None:
            return RoutingResult(
                model=model,
                reason="explicit_override",
                task_dimensions={},
                latency_ms=0.0,
            )

        # 2. Conversation context
        conv_tokens = 0
        regen_count = 0
        conv_turn = 0
        if conversation_id:
            cp = self._conversations.get_profile(conversation_id)
            conv_tokens = cp.total_tokens
            regen_count = cp.regen_count
            conv_turn = cp.turn_count

        # 3. Layer 1: score task
        task_needs = score_task(
            prompt,
            has_attachment=has_attachment,
            conversation_tokens=conv_tokens,
            recent_regen_count=regen_count,
        )

        # 4. Layer 3: conversation adjustment
        if conversation_id:
            task_needs = self._conversations.adjust(conversation_id, task_needs)

        # 5. Layer 2: match best model
        best = match(task_needs, preference, self._profiles)

        # 6. Stickiness check
        reason_parts: List[str] = []
        if conversation_id:
            last_model = self._conversations.get_last_model(conversation_id)
            if last_model and last_model != best:
                sticky_score = self._dot_score(last_model, task_needs)
                best_score = self._dot_score(best, task_needs)
                if best_score <= sticky_score * 1.15:
                    best = last_model
                    reason_parts.append("sticky")

        # 7. Build reason
        top_dim = max(task_needs, key=task_needs.get)  # type: ignore[arg-type]
        reason_parts.insert(0, f"top_dim={top_dim}")
        reason_parts.insert(1, f"pref={preference}")
        reason = " ".join(reason_parts)

        # 8. Alternatives: score all models, top 3 excluding winner
        all_profiles = self._profiles.all_profiles()
        scored: List[tuple] = []
        for m in all_profiles:
            if m == best:
                continue
            scored.append((m, self._dot_score(m, task_needs)))
        scored.sort(key=lambda x: x[1], reverse=True)
        alternatives = [m for m, _ in scored[:3]]

        # 9. Update conversation
        if conversation_id:
            feats = extract_features(prompt, has_attachment=has_attachment)
            complexity = max(task_needs.get("reason", 0.0), task_needs.get("code", 0.0))
            tokens_estimate = len(prompt.split()) * 2
            self._conversations.update(
                conversation_id,
                code_ratio=feats["code_ratio"],
                complexity=complexity,
                model_used=best,
                tokens=tokens_estimate,
            )

        # 10. Return result with timing
        latency = (time.perf_counter() - t0) * 1000.0
        return RoutingResult(
            model=best,
            reason=reason,
            task_dimensions=task_needs,
            alternatives=alternatives,
            conversation_turn=conv_turn,
            latency_ms=latency,
        )

    def _dot_score(self, model: str, task_needs: Dict[str, float]) -> float:
        """Compute dot product of model profile against task needs."""
        profile = self._profiles.get(model)
        return sum(
            task_needs.get(dim, 0.0) * profile.get(dim, 0.0)
            for dim in DIMENSIONS
        )


# ── module-level singleton ──────────────────────────────────────────

_router_instance: Optional[Router] = None
_router_lock = threading.Lock()


def get_router() -> Router:
    global _router_instance
    if _router_instance is None:
        with _router_lock:
            if _router_instance is None:
                _router_instance = Router()
    return _router_instance
