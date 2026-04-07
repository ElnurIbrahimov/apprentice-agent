"""Per-conversation context tracker for the neural router."""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ConversationProfile:
    topic_embedding: Optional[List[float]] = None
    topic_drift_score: float = 0.0
    complexity_history: List[float] = field(default_factory=list)
    complexity_trend: float = 0.0
    code_ratio_history: List[float] = field(default_factory=list)
    in_code_mode: bool = False
    models_used: List[str] = field(default_factory=list)
    last_model: Optional[str] = None
    regen_count: int = 0
    model_switches: int = 0
    total_tokens: int = 0
    turn_count: int = 0
    thumbs: List[int] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class ConversationTracker:
    """Thread-safe per-conversation profile tracker with LRU eviction."""

    def __init__(self, max_conversations: int = 200):
        self._max = max_conversations
        self._profiles: OrderedDict[str, ConversationProfile] = OrderedDict()
        self._lock = threading.Lock()

    def get_profile(self, conversation_id: str) -> ConversationProfile:
        with self._lock:
            if conversation_id in self._profiles:
                return self._profiles[conversation_id]
            return ConversationProfile()

    def update(
        self,
        conversation_id: str,
        code_ratio: float,
        complexity: float,
        model_used: str,
        tokens: int,
    ) -> None:
        with self._lock:
            if conversation_id not in self._profiles:
                self._profiles[conversation_id] = ConversationProfile()
            p = self._profiles[conversation_id]

            p.turn_count += 1
            p.total_tokens += tokens
            p.last_model = model_used
            if model_used not in p.models_used:
                p.models_used.append(model_used)
            p.last_active = time.time()

            # Sliding window of 10
            p.complexity_history.append(complexity)
            if len(p.complexity_history) > 10:
                p.complexity_history = p.complexity_history[-10:]

            p.code_ratio_history.append(code_ratio)
            if len(p.code_ratio_history) > 10:
                p.code_ratio_history = p.code_ratio_history[-10:]

            # Complexity trend from last 3 entries
            recent = p.complexity_history[-3:]
            if len(recent) >= 2:
                p.complexity_trend = recent[-1] - recent[0]
            else:
                p.complexity_trend = 0.0

            # Code mode: last 3 code_ratios all > 0.2
            last3 = p.code_ratio_history[-3:]
            p.in_code_mode = len(last3) >= 3 and all(r > 0.2 for r in last3)

            # LRU: move to end
            self._profiles.move_to_end(conversation_id)

            # Evict oldest if over limit
            while len(self._profiles) > self._max:
                self._profiles.popitem(last=False)

    def record_feedback(
        self, conversation_id: str, signal: str, **kwargs
    ) -> None:
        with self._lock:
            if conversation_id not in self._profiles:
                self._profiles[conversation_id] = ConversationProfile()
            p = self._profiles[conversation_id]

            if signal == "regeneration":
                p.regen_count += 1
            elif signal == "model_switch":
                p.model_switches += 1
            elif signal == "thumbs_up":
                p.thumbs.append(1)
            elif signal == "thumbs_down":
                p.thumbs.append(-1)
            elif signal == "abort":
                p.regen_count += 1

    def adjust(self, conversation_id: str, task_needs: Dict[str, float]) -> Dict[str, float]:
        with self._lock:
            known = conversation_id in self._profiles
        if not known:
            return task_needs

        profile = self.get_profile(conversation_id)

        adjusted = dict(task_needs)

        if profile.in_code_mode:
            adjusted["code"] = min(adjusted.get("code", 0.0) + 0.3, 1.0)

        if profile.complexity_trend > 0.3:
            adjusted["reason"] = min(adjusted.get("reason", 0.0) + 0.2, 1.0)
            adjusted["speed"] = max(adjusted.get("speed", 0.0) - 0.2, 0.0)
        elif profile.complexity_trend < -0.3:
            adjusted["speed"] = min(adjusted.get("speed", 0.0) + 0.2, 1.0)

        if profile.regen_count > 0:
            adjusted["quality"] = min(
                adjusted.get("quality", 0.0) + 0.15 * profile.regen_count, 1.0
            )

        if profile.total_tokens > 100_000:
            adjusted["context"] = max(adjusted.get("context", 0.0), 0.7)
        elif profile.total_tokens > 50_000:
            adjusted["context"] = max(adjusted.get("context", 0.0), 0.5)

        return adjusted

    def get_last_model(self, conversation_id: str) -> Optional[str]:
        return self.get_profile(conversation_id).last_model
