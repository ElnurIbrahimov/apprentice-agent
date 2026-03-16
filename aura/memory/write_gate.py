"""
Memory Write Gate — Phase 1.

Sits between any "I want to write something to memory" call and the actual
fan-out into A-MEM / Episodic / KG.  Scores candidates and decides:
  DISCARD          — skip entirely
  STORE_NEW        — write as a fresh memory
  MERGE_INTO       — update an existing memory in-place
  SUPERSEDE        — mark the old memory as archived, write the new one
  ARCHIVE_CANDIDATE — defer: write but tag as candidate for Dream consolidation

Author: Aura reliability upgrade (2026-03)
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryLifecycleState(str, Enum):
    CANDIDATE  = "candidate"   # newly written, not yet confirmed stable
    STABLE     = "stable"      # confirmed, long-term value
    SUMMARY    = "summary"     # produced by Dream consolidation
    ARCHIVED   = "archived"    # superseded / stale, kept for history
    FORGOTTEN  = "forgotten"   # soft-deleted (not retrieved; may be pruned later)


class MemoryDecisionKind(str, Enum):
    DISCARD           = "discard"
    STORE_NEW         = "store_new"
    MERGE_INTO        = "merge_into"
    SUPERSEDE         = "supersede"
    ARCHIVE_CANDIDATE = "archive_candidate"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryCandidate:
    """Input to the write gate."""
    content: str
    source: str                          # e.g. "conversation", "task_execution"
    user_id: str = "default_user"
    importance: float = 0.5             # 0–1 caller-supplied hint
    emotional_salience: float = 0.0     # 0–1 caller-supplied ALMA hint
    tags: List[str] = field(default_factory=list)
    explicit_save: bool = False          # user said "remember this"
    confidence: float = 1.0             # source quality / certainty
    extra: Dict[str, Any] = field(default_factory=dict)

    # Computed at gate time
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.md5(
            self.content[:300].lower().strip().encode()
        ).hexdigest()[:16]


@dataclass
class MemoryDecision:
    """Decision produced by the write gate."""
    kind: MemoryDecisionKind
    candidate: MemoryCandidate
    score: float                         # Overall write-worthiness (0–1)
    novelty: float = 0.0
    future_utility: float = 0.0
    emotional_salience: float = 0.0
    user_specificity: float = 0.0
    confidence_score: float = 0.0

    # For MERGE / SUPERSEDE
    target_id: Optional[str] = None
    target_source: Optional[str] = None  # "amem" | "episodic"
    superseded_ids: List[str] = field(default_factory=list)

    lifecycle_state: MemoryLifecycleState = MemoryLifecycleState.CANDIDATE
    decision_ts: float = field(default_factory=time.time)
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    reason: str = ""

    def as_log_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "user_id": self.candidate.user_id,
            "source": self.candidate.source,
            "kind": self.kind.value,
            "score": round(self.score, 3),
            "novelty": round(self.novelty, 3),
            "future_utility": round(self.future_utility, 3),
            "emotional_salience": round(self.emotional_salience, 3),
            "user_specificity": round(self.user_specificity, 3),
            "confidence": round(self.confidence_score, 3),
            "explicit_save": self.candidate.explicit_save,
            "target_id": self.target_id,
            "superseded_ids": self.superseded_ids,
            "lifecycle_state": self.lifecycle_state.value,
            "reason": self.reason,
            "content_preview": self.candidate.content[:80].replace("\n", " "),
        }


# ---------------------------------------------------------------------------
# Noise patterns — content that is almost never worth persisting
# ---------------------------------------------------------------------------

_NOISE_PATTERNS = [
    r"^(ok|okay|sure|thanks?|got it|understood|no problem|np|i see|noted)[.!]*$",
    r"^(yes|no|maybe|alright|great|cool|nice)[.!?]*$",
    r"^\.{1,3}$",
    r"^\s*$",
]
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]

def _is_noise(text: str) -> bool:
    t = text.strip()
    if len(t) < 4:
        return True
    return any(r.match(t) for r in _NOISE_RE)


# ---------------------------------------------------------------------------
# Write Gate
# ---------------------------------------------------------------------------

class MemoryWriteGate:
    """
    Score memory candidates and decide what to do with them.

    Config knobs (all can be overridden via Config or env):
      ENABLE_MEMORY_WRITE_GATE     — master switch (default True)
      MEMORY_WRITE_THRESHOLD       — min score to store anything (default 0.35)
      MEMORY_MERGE_THRESHOLD       — similarity above which we merge (default 0.88)
      MEMORY_SUPERSEDE_THRESHOLD   — cosine sim to trigger supersession (default 0.80)
    """

    # Score weights
    W_NOVELTY      = 0.35
    W_UTILITY      = 0.25
    W_EMOTION      = 0.15
    W_SPECIFICITY  = 0.15
    W_CONFIDENCE   = 0.10

    # Default thresholds (can be overridden by Config)
    DEFAULT_WRITE_THRESHOLD      = 0.35
    DEFAULT_MERGE_THRESHOLD      = 0.88
    DEFAULT_SUPERSEDE_THRESHOLD  = 0.80

    def __init__(self) -> None:
        try:
            from aura.config import Config
            self._enabled      = getattr(Config, "ENABLE_MEMORY_WRITE_GATE", True)
            self._write_thr    = getattr(Config, "MEMORY_WRITE_THRESHOLD",      self.DEFAULT_WRITE_THRESHOLD)
            self._merge_thr    = getattr(Config, "MEMORY_MERGE_THRESHOLD",      self.DEFAULT_MERGE_THRESHOLD)
            self._supersede_thr= getattr(Config, "MEMORY_SUPERSEDE_THRESHOLD",  self.DEFAULT_SUPERSEDE_THRESHOLD)
        except Exception:
            self._enabled       = True
            self._write_thr     = self.DEFAULT_WRITE_THRESHOLD
            self._merge_thr     = self.DEFAULT_MERGE_THRESHOLD
            self._supersede_thr = self.DEFAULT_SUPERSEDE_THRESHOLD

        # Recent candidate hashes for ultra-fast exact-dup suppression
        self._recent_hashes: Dict[str, float] = {}   # hash → epoch
        self._hash_lock = threading.Lock()  # Protect _recent_hashes across threads
        self._RECENT_TTL = 300.0  # 5 min

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidate: MemoryCandidate,
        nearby: Optional[List[Dict[str, Any]]] = None,
    ) -> MemoryDecision:
        """
        Evaluate a candidate and return a MemoryDecision.

        Args:
            candidate: The memory candidate to evaluate.
            nearby:    List of nearby memories already retrieved (each dict must
                       have at least 'content', 'source_id', 'score' (similarity)
                       and optionally 'source').
        """
        if not self._enabled:
            return MemoryDecision(
                kind=MemoryDecisionKind.STORE_NEW,
                candidate=candidate,
                score=1.0,
                reason="gate_disabled",
                lifecycle_state=MemoryLifecycleState.CANDIDATE,
            )

        # 0. Hard noise rejection
        if _is_noise(candidate.content):
            return self._decide(MemoryDecisionKind.DISCARD, candidate, 0.0,
                                reason="noise_content")

        # 1. Exact-duplicate suppression (recent memory)
        if self._is_recent_exact_dup(candidate):
            return self._decide(MemoryDecisionKind.DISCARD, candidate, 0.0,
                                reason="exact_duplicate_recent")

        # 2. Compute component scores
        novelty      = self._score_novelty(candidate, nearby or [])
        utility      = self._score_utility(candidate)
        emotion      = self._score_emotion(candidate)
        specificity  = self._score_specificity(candidate)
        confidence   = self._score_confidence(candidate)

        total = (
            self.W_NOVELTY     * novelty
            + self.W_UTILITY   * utility
            + self.W_EMOTION   * emotion
            + self.W_SPECIFICITY * specificity
            + self.W_CONFIDENCE  * confidence
        )

        # 3. Explicit-save bias — always store, but still dedupe
        if candidate.explicit_save:
            total = max(total, self._write_thr + 0.15)

        # 4. Check for merge / supersede opportunities
        if nearby:
            merge_target   = self._find_merge_target(candidate, nearby)
            supersede_tgt  = self._find_supersede_target(candidate, nearby)

            if merge_target and total >= self._write_thr:
                d = self._decide(MemoryDecisionKind.MERGE_INTO, candidate, total,
                                 novelty, utility, emotion, specificity, confidence,
                                 reason="semantic_near_duplicate")
                d.target_id     = merge_target.get("source_id", "")
                d.target_source = merge_target.get("source", "")
                d.lifecycle_state = MemoryLifecycleState.STABLE
                self._record_hash(candidate)
                self._log(d)
                return d

            if supersede_tgt and total >= self._write_thr:
                d = self._decide(MemoryDecisionKind.SUPERSEDE, candidate, total,
                                 novelty, utility, emotion, specificity, confidence,
                                 reason="fact_correction_detected")
                d.target_id       = supersede_tgt.get("source_id", "")
                d.target_source   = supersede_tgt.get("source", "")
                d.superseded_ids  = [supersede_tgt.get("source_id", "")]
                d.lifecycle_state = MemoryLifecycleState.STABLE
                self._record_hash(candidate)
                self._log(d)
                return d

        # 5. Below threshold → discard
        if total < self._write_thr:
            return self._decide(MemoryDecisionKind.DISCARD, candidate, total,
                                novelty, utility, emotion, specificity, confidence,
                                reason=f"score_below_threshold({total:.2f}<{self._write_thr})")

        # 6. Low-importance → archive candidate (Dream will decide later)
        if total < self._write_thr + 0.15 and not candidate.explicit_save:
            d = self._decide(MemoryDecisionKind.ARCHIVE_CANDIDATE, candidate, total,
                             novelty, utility, emotion, specificity, confidence,
                             reason="low_priority_deferred_to_dream")
            d.lifecycle_state = MemoryLifecycleState.CANDIDATE
            self._record_hash(candidate)
            self._log(d)
            return d

        # 7. Normal store
        d = self._decide(MemoryDecisionKind.STORE_NEW, candidate, total,
                         novelty, utility, emotion, specificity, confidence,
                         reason="scored_above_threshold")
        d.lifecycle_state = (
            MemoryLifecycleState.STABLE if total > 0.65
            else MemoryLifecycleState.CANDIDATE
        )
        self._record_hash(candidate)
        self._log(d)
        return d

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_novelty(self, c: MemoryCandidate, nearby: List[Dict]) -> float:
        """How different is this from what we already know?"""
        if not nearby:
            return 0.75  # No nearby → assume novel
        best_sim = max((n.get("score", 0.0) for n in nearby), default=0.0)
        # sim=1.0 → novelty=0; sim=0 → novelty=1
        return max(0.0, 1.0 - best_sim)

    def _score_utility(self, c: MemoryCandidate) -> float:
        """Estimate future usefulness."""
        score = c.importance  # Caller already provided a hint
        # Boosts
        if any(t in ("preference", "fact", "goal", "decision") for t in c.tags):
            score = min(1.0, score + 0.2)
        if c.source in ("explicit_clip", "task_outcome", "learning"):
            score = min(1.0, score + 0.15)
        # Short ephemeral content
        if len(c.content) < 30:
            score = max(0.0, score - 0.2)
        return max(0.0, min(1.0, score))

    def _score_emotion(self, c: MemoryCandidate) -> float:
        """Emotionally salient memories are more worth keeping."""
        base = c.emotional_salience
        # Explicit emotional markers
        emotional_words = {"love", "hate", "fear", "excited", "angry", "sad",
                           "happy", "worried", "frustrated", "thrilled"}
        lower = c.content.lower()
        if any(w in lower for w in emotional_words):
            base = min(1.0, base + 0.3)
        return max(0.0, min(1.0, base))

    def _score_specificity(self, c: MemoryCandidate) -> float:
        """Is this about the user specifically, or generic knowledge?"""
        user_markers = {
            "i prefer", "i like", "i want", "i need", "i use",
            "my project", "my setup", "my name", "i am", "i work",
            "remember that i", "elnur", "my ",
        }
        lower = c.content.lower()
        hits = sum(1 for m in user_markers if m in lower)
        return min(1.0, hits * 0.25)

    def _score_confidence(self, c: MemoryCandidate) -> float:
        return max(0.0, min(1.0, c.confidence))

    # ------------------------------------------------------------------
    # Merge / supersede detection
    # ------------------------------------------------------------------

    def _find_merge_target(
        self, c: MemoryCandidate, nearby: List[Dict]
    ) -> Optional[Dict]:
        """Find the best near-duplicate suitable for merging (highest similarity above threshold)."""
        best: Optional[Dict] = None
        best_score = 0.0
        for n in nearby:
            sim = n.get("score", 0.0)
            if sim >= self._merge_thr and sim > best_score:
                best = n
                best_score = sim
        return best

    def _find_supersede_target(
        self, c: MemoryCandidate, nearby: List[Dict]
    ) -> Optional[Dict]:
        """
        Find a memory that the candidate likely corrects / supersedes.
        Heuristic: high semantic similarity + negation / correction language.
        Selects the highest-similarity candidate above the threshold.
        """
        correction_phrases = [
            "actually", "correction:", "I was wrong", "update:", "changed to",
            "now I", "no longer", "instead of", "replaced by", "migrated to",
        ]
        lower = c.content.lower()
        has_correction = any(ph.lower() in lower for ph in correction_phrases)
        if not has_correction:
            return None

        best: Optional[Dict] = None
        best_score = 0.0
        for n in nearby:
            sim = n.get("score", 0.0)
            if sim >= self._supersede_thr and sim > best_score:
                best = n
                best_score = sim
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_recent_exact_dup(self, c: MemoryCandidate) -> bool:
        now = time.time()
        with self._hash_lock:
            # Purge stale entries
            stale = [h for h, ts in self._recent_hashes.items() if now - ts > self._RECENT_TTL]
            for h in stale:
                del self._recent_hashes[h]
            return c.content_hash in self._recent_hashes

    def _record_hash(self, c: MemoryCandidate) -> None:
        with self._hash_lock:
            self._recent_hashes[c.content_hash] = time.time()
            # Hard cap to prevent unbounded growth
            if len(self._recent_hashes) > 10000:
                sorted_items = sorted(self._recent_hashes.items(), key=lambda x: x[1])
                for h, _ in sorted_items[:5000]:
                    del self._recent_hashes[h]

    def _decide(
        self,
        kind: MemoryDecisionKind,
        candidate: MemoryCandidate,
        score: float,
        novelty: float = 0.0,
        utility: float = 0.0,
        emotion: float = 0.0,
        specificity: float = 0.0,
        confidence: float = 0.0,
        reason: str = "",
    ) -> MemoryDecision:
        return MemoryDecision(
            kind=kind,
            candidate=candidate,
            score=score,
            novelty=novelty,
            future_utility=utility,
            emotional_salience=emotion,
            user_specificity=specificity,
            confidence_score=confidence,
            reason=reason,
        )

    def _log(self, d: MemoryDecision) -> None:
        log = d.as_log_dict()
        logger.info("[MemoryGate] decision=%s score=%.2f reason=%s user=%s source=%s preview='%s'",
                    log["kind"], log["score"], log["reason"],
                    log["user_id"], log["source"], log["content_preview"])


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_gate_instance: Optional[MemoryWriteGate] = None
_gate_lock = threading.Lock()

def get_write_gate() -> MemoryWriteGate:
    global _gate_instance
    if _gate_instance is None:
        with _gate_lock:
            if _gate_instance is None:
                _gate_instance = MemoryWriteGate()
    return _gate_instance


__all__ = [
    "MemoryWriteGate",
    "MemoryCandidate",
    "MemoryDecision",
    "MemoryLifecycleState",
    "MemoryDecisionKind",
    "get_write_gate",
]
