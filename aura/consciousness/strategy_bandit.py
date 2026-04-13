"""
Strategy Bandit — Thompson Sampling over reasoning strategies.

Adaptively selects the best reasoning strategy per problem category using
Beta-distributed Thompson Sampling with composite reward signals.

Part of AURA's meta-cognitive self-improvement system.
"""

import json
import logging
import math
import os
import random
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ProblemCategory(Enum):
    """Problem categories for strategy routing."""
    MATH = "math"
    CODE = "code"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    PLANNING = "planning"
    DEBUG = "debug"


class ReasoningStrategy(Enum):
    """Available reasoning strategies mapped to AURA tools."""
    CHAIN_OF_THOUGHT = "chain_of_thought"       # brain.think() — 1 LLM call
    MCTS = "mcts"                                # ReasoningTreeTool.execute() — 10-30 calls


# Strategy availability per category (not all strategies suit all problems)
CATEGORY_STRATEGIES: Dict[ProblemCategory, List[ReasoningStrategy]] = {
    ProblemCategory.MATH: [
        ReasoningStrategy.CHAIN_OF_THOUGHT,
        ReasoningStrategy.MCTS,
    ],
    ProblemCategory.CODE: [
        ReasoningStrategy.CHAIN_OF_THOUGHT,
        ReasoningStrategy.MCTS,
    ],
    ProblemCategory.ANALYSIS: [
        ReasoningStrategy.CHAIN_OF_THOUGHT,
    ],
    ProblemCategory.CREATIVE: [
        ReasoningStrategy.CHAIN_OF_THOUGHT,
    ],
    ProblemCategory.PLANNING: [
        ReasoningStrategy.CHAIN_OF_THOUGHT,
        ReasoningStrategy.MCTS,
    ],
    ProblemCategory.DEBUG: [
        ReasoningStrategy.CHAIN_OF_THOUGHT,
    ],
}

# Estimated LLM call costs per strategy (for logging/awareness)
STRATEGY_COST: Dict[ReasoningStrategy, int] = {
    ReasoningStrategy.CHAIN_OF_THOUGHT: 1,
    ReasoningStrategy.MCTS: 15,
}


# ============================================================================
# Problem Classifier
# ============================================================================

class ProblemClassifier:
    """Keyword-based query classification into problem categories.

    Consolidates patterns from brain.py's _select_model() and
    self_improvement.py's _infer_domain().
    """

    KEYWORD_MAP: Dict[ProblemCategory, List[str]] = {
        ProblemCategory.MATH: [
            "calculate", "compute", "factorial", "fibonacci", "prime",
            "equation", "integral", "derivative", "algebra", "geometry",
            "trigonometry", "statistics", "probability", "matrix", "solve",
            "sum", "product", "proof", "theorem", "math",
        ],
        ProblemCategory.CODE: [
            "code", "python", "function", "script", "api", "class",
            "variable", "compile", "syntax", "algorithm", "database",
            "sql", "html", "css", "javascript", "implement", "refactor",
            "write a script", "program", "library", "framework",
            "typescript", "rust", "golang", "java",
        ],
        ProblemCategory.ANALYSIS: [
            "analyze", "data", "chart", "compare", "evaluate", "assess",
            "measure", "trend", "report", "statistics", "research",
            "investigate", "study", "paper", "pros and cons", "tradeoff",
            "trade-off", "review", "benchmark",
        ],
        ProblemCategory.CREATIVE: [
            "creative", "imagine", "brainstorm", "idea", "generate",
            "story", "poem", "design", "invent", "novel", "write",
            "essay", "draft", "compose", "narrative", "fiction",
        ],
        ProblemCategory.PLANNING: [
            "plan", "schedule", "roadmap", "strategy", "organize",
            "prioritize", "timeline", "milestone", "project", "task",
            "workflow", "architecture", "design system", "steps to",
            "how should i", "what order",
        ],
        ProblemCategory.DEBUG: [
            "debug", "fix", "error", "exception", "traceback", "bug",
            "crash", "issue", "broken", "failing", "not working",
            "wrong output", "unexpected", "diagnose", "troubleshoot",
        ],
    }

    def classify(self, query: str) -> ProblemCategory:
        """Classify a query into a problem category using keyword matching."""
        query_lower = query.lower()
        best_category = ProblemCategory.ANALYSIS  # default if keywords match
        best_score = 0

        for category, keywords in self.KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_category = category

        # Zero keyword matches = conversational/greeting — use general-purpose category
        if best_score == 0:
            best_category = ProblemCategory.ANALYSIS

        return best_category


# ============================================================================
# Quality Heuristics (cheap, no LLM calls)
# ============================================================================

# Phrases that indicate refusal or inability
_REFUSAL_PHRASES = [
    "i can't", "i cannot", "i'm unable", "i am unable",
    "i don't have the ability", "i'm not able", "i am not able",
    "i apologize, but i", "i'm sorry, but i can't",
    "as an ai", "as a language model",
]

# Sentence-ending punctuation pattern
_SENTENCE_END_RE = re.compile(r'[.!?]\s')


def compute_quality_metrics(
    query: str,
    response: str,
) -> Dict[str, float]:
    """Compute cheap quality heuristics for a query-response pair.

    Returns dict with:
        coherence_score: 1.0 well-formed, 0.5 truncated/partial, 0.0 error/empty
        judge_score: 0.0-1.0 based on relevance, structure, and refusal detection

    No LLM calls — pure string heuristics. ~0.1ms.
    """
    if not response or not response.strip():
        return {"coherence_score": 0.0, "judge_score": 0.0}

    resp_lower = response.lower().strip()
    resp_stripped = response.strip()

    # --- coherence_score ---
    # Check for well-formedness: ends with punctuation, has complete sentences
    ends_cleanly = resp_stripped[-1] in '.!?"\')' or resp_stripped.endswith('```')
    has_sentences = bool(_SENTENCE_END_RE.search(response))
    long_enough = len(resp_stripped) > 40

    if ends_cleanly and has_sentences and long_enough:
        coherence = 1.0
    elif long_enough and (ends_cleanly or has_sentences):
        coherence = 0.75
    elif long_enough:
        coherence = 0.5  # long but no sentence structure — truncated or code dump
    else:
        coherence = 0.3  # very short, likely partial

    # --- judge_score ---
    score = 0.5  # neutral baseline

    # 1) Keyword overlap: do query keywords appear in the response?
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    # Remove stop words
    _stops = {"the", "and", "for", "are", "but", "not", "you", "all",
              "can", "her", "was", "one", "our", "out", "has", "have",
              "this", "that", "with", "what", "how", "why", "who",
              "from", "they", "been", "will", "would", "could", "should",
              "about", "which", "when", "make", "like", "just", "into",
              "some", "than", "them", "very", "does", "also"}
    query_keywords = query_words - _stops
    if query_keywords:
        resp_words = set(re.findall(r'\b[a-z]{3,}\b', resp_lower))
        overlap = len(query_keywords & resp_words) / len(query_keywords)
        score += 0.2 * overlap  # up to +0.2

    # 2) Penalize refusals
    refusal_count = sum(1 for p in _REFUSAL_PHRASES if p in resp_lower)
    if refusal_count > 0:
        score -= 0.3 * min(refusal_count, 2)  # up to -0.6

    # 3) Reward structured responses (code blocks, lists, sections)
    has_code_block = '```' in response
    has_bullet_list = bool(re.search(r'^\s*[-*•]\s', response, re.MULTILINE))
    has_numbered_list = bool(re.search(r'^\s*\d+[.)]\s', response, re.MULTILINE))
    has_headers = bool(re.search(r'^#{1,3}\s', response, re.MULTILINE))
    structure_signals = sum([has_code_block, has_bullet_list, has_numbered_list, has_headers])
    score += 0.05 * structure_signals  # up to +0.2

    # 4) Response length relative to query length (not just raw length)
    if len(query) > 100 and len(response) < 80:
        score -= 0.15  # complex query, tiny response
    elif len(response) > 200:
        score += 0.05  # substantive response

    # Clamp to [0, 1]
    judge = max(0.0, min(1.0, score))

    return {"coherence_score": coherence, "judge_score": judge}


# ============================================================================
# Composite Reward Computer
# ============================================================================

class CompositeRewardComputer:
    """Weighted combination of metrics into a single [0, 1] reward signal.

    Weights:
      - latency_score: 0.2 — faster is better (normalized)
      - self_consistency: 0.2 — would the model give the same answer?
      - judge_score: 0.3 — binary PASS/FAIL from LLM judge
      - coherence_score: 0.2 — stepwise logical coherence
      - user_feedback: 0.1 — explicit user thumbs up/down (upweighted when present)

    When user_feedback is provided, its weight doubles (0.2) and others rescale.
    """

    DEFAULT_WEIGHTS = {
        "latency_score": 0.2,
        "self_consistency": 0.2,
        "judge_score": 0.3,
        "coherence_score": 0.2,
        "user_feedback": 0.1,
    }

    # When user feedback is present, upweight it
    USER_FEEDBACK_WEIGHTS = {
        "latency_score": 0.15,
        "self_consistency": 0.15,
        "judge_score": 0.25,
        "coherence_score": 0.15,
        "user_feedback": 0.3,
    }

    def compute(self, metrics: Dict[str, Optional[float]]) -> float:
        """Compute composite reward from available metrics.

        Args:
            metrics: Dict with keys matching weight names. Values are
                     floats in [0, 1] or None if unavailable.

        Returns:
            Composite reward in [0, 1]. Missing metrics are excluded and
            remaining weights are renormalized.
        """
        has_user_feedback = (
            metrics.get("user_feedback") is not None
        )
        weights = (
            self.USER_FEEDBACK_WEIGHTS if has_user_feedback
            else self.DEFAULT_WEIGHTS
        )

        total_weight = 0.0
        weighted_sum = 0.0

        for key, weight in weights.items():
            value = metrics.get(key)
            if value is not None:
                # Clamp to [0, 1]
                clamped = max(0.0, min(1.0, float(value)))
                weighted_sum += weight * clamped
                total_weight += weight

        if total_weight == 0:
            return 0.5  # No data → neutral prior

        return weighted_sum / total_weight


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ArmState:
    """State of a single bandit arm (strategy + category pair)."""
    strategy: str
    category: str
    alpha: float = 1.0  # Beta distribution success parameter
    beta: float = 1.0   # Beta distribution failure parameter
    total_pulls: int = 0
    total_reward: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def mean_reward(self) -> float:
        """Expected reward = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Thompson Sampling: draw from Beta(alpha, beta)."""
        return random.betavariate(self.alpha, self.beta)


@dataclass
class StrategyOutcome:
    """Outcome of executing a strategy."""
    request_id: str
    strategy: str
    category: str
    latency_ms: float
    response_length: int
    composite_reward: Optional[float] = None
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class BanditSelection:
    """Result of strategy selection by the bandit."""
    request_id: str
    strategy: ReasoningStrategy
    category: ProblemCategory
    exploration: bool  # True if epsilon-greedy exploration, False if Thompson Sampling
    sampled_values: Dict[str, float] = field(default_factory=dict)
    arm_state: Optional[ArmState] = None


# ============================================================================
# Strategy Bandit
# ============================================================================

class StrategyBandit:
    """Thompson Sampling bandit for adaptive reasoning strategy selection.

    Uses Beta distributions to model reward for each (category, strategy) pair.
    SQLite-backed for persistence across sessions.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        epsilon: float = 0.1,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.epsilon = epsilon
        self._lock = threading.Lock()
        self._db_lock = threading.Lock()  # Separate lock for DB I/O to avoid stalling non-DB ops
        self._classifier = ProblemClassifier()
        self._reward_computer = CompositeRewardComputer()

        # Resolve DB path
        if db_path is None:
            data_dir = Path(os.getenv("AURA_DATA_DIR", "data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(data_dir / "aura_meta.db")
        else:
            self._db_path = db_path

        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema and seed arms."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS strategy_arms (
                    strategy TEXT NOT NULL,
                    category TEXT NOT NULL,
                    alpha REAL NOT NULL DEFAULT 1.0,
                    beta REAL NOT NULL DEFAULT 1.0,
                    total_pulls INTEGER NOT NULL DEFAULT 0,
                    total_reward REAL NOT NULL DEFAULT 0.0,
                    last_updated REAL NOT NULL,
                    PRIMARY KEY (strategy, category)
                );

                CREATE TABLE IF NOT EXISTS strategy_outcomes (
                    request_id TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    category TEXT NOT NULL,
                    latency_ms REAL,
                    response_length INTEGER,
                    composite_reward REAL,
                    metrics_json TEXT,
                    user_feedback REAL,
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT,
                    timestamp REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_outcomes_category
                    ON strategy_outcomes(category);
                CREATE INDEX IF NOT EXISTS idx_outcomes_strategy
                    ON strategy_outcomes(strategy, category);
                CREATE INDEX IF NOT EXISTS idx_perf_logs_request
                    ON performance_logs(request_id);
            """)

            # Seed arms for all valid (category, strategy) pairs
            now = time.time()
            for category, strategies in CATEGORY_STRATEGIES.items():
                for strategy in strategies:
                    conn.execute(
                        """INSERT OR IGNORE INTO strategy_arms
                           (strategy, category, alpha, beta, total_pulls, total_reward, last_updated)
                           VALUES (?, ?, 1.0, 1.0, 0, 0.0, ?)""",
                        (strategy.value, category.value, now),
                    )

            conn.commit()
        finally:
            conn.close()
        logger.info(f"[StrategyBandit] DB initialized at {self._db_path}")

    def _get_arms(self, category: str) -> List[ArmState]:
        """Load arm states for a category from DB."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA busy_timeout = 3000")
        try:
            rows = conn.execute(
                "SELECT strategy, category, alpha, beta, total_pulls, total_reward, last_updated "
                "FROM strategy_arms WHERE category = ?",
                (category,),
            ).fetchall()
        finally:
            conn.close()

        return [
            ArmState(
                strategy=r[0], category=r[1],
                alpha=r[2], beta=r[3],
                total_pulls=r[4], total_reward=r[5],
                last_updated=r[6],
            )
            for r in rows
        ]

    def select_strategy(
        self,
        query: str,
        category: Optional[ProblemCategory] = None,
        available_strategies: Optional[List[ReasoningStrategy]] = None,
    ) -> BanditSelection:
        """Select the best reasoning strategy for a query.

        Args:
            query: The user query text.
            category: Override auto-classification if provided.
            available_strategies: Override available strategies for this category.

        Returns:
            BanditSelection with chosen strategy and metadata.
        """
        request_id = str(uuid.uuid4())[:12]

        if not self.enabled:
            cat = category or self._classifier.classify(query)
            return BanditSelection(
                request_id=request_id,
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                category=cat,
                exploration=False,
            )

        # Classify problem
        cat = category or self._classifier.classify(query)
        strategies = available_strategies or CATEGORY_STRATEGIES.get(
            cat, [ReasoningStrategy.CHAIN_OF_THOUGHT]
        )

        if not strategies:
            strategies = [ReasoningStrategy.CHAIN_OF_THOUGHT]

        # Load arms outside lock to avoid holding lock during DB I/O
        arms = self._get_arms(cat.value)

        with self._lock:
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                chosen = random.choice(strategies)
                logger.info(
                    f"[StrategyBandit] EXPLORE: {chosen.value} for {cat.value} "
                    f"(request={request_id})"
                )
                try:
                    from aura.activity_logger import record_activity
                    record_activity(
                        "strategy", chosen.value,
                        f"Strategy: {chosen.value} for {cat.value}",
                        {"category": cat.value, "explore": True},
                    )
                except Exception as e:
                    logger.debug(f"[StrategyBandit] activity log failed: {e}")
                return BanditSelection(
                    request_id=request_id,
                    strategy=chosen,
                    category=cat,
                    exploration=True,
                )

            # Thompson Sampling
            arm_map = {a.strategy: a for a in arms}

            sampled_values = {}
            best_strategy = strategies[0]
            best_sample = -1.0

            for strategy in strategies:
                arm = arm_map.get(strategy.value)
                if arm:
                    sample = arm.sample()
                else:
                    # Unknown arm → sample from prior Beta(1,1) = Uniform
                    sample = random.betavariate(1.0, 1.0)
                sampled_values[strategy.value] = sample
                if sample > best_sample:
                    best_sample = sample
                    best_strategy = strategy

            arm_state = arm_map.get(best_strategy.value)

            logger.info(
                f"[StrategyBandit] selected: {best_strategy.value} for {cat.value} "
                f"(sample={best_sample:.3f}, pulls={arm_state.total_pulls if arm_state else 0}, "
                f"request={request_id})"
            )

            try:
                from aura.activity_logger import record_activity
                record_activity(
                    "strategy", best_strategy.value,
                    f"Strategy: {best_strategy.value} for {cat.value}",
                    {"category": cat.value, "explore": False},
                )
            except Exception as e:
                logger.debug(f"[StrategyBandit] activity log failed: {e}")

            return BanditSelection(
                request_id=request_id,
                strategy=best_strategy,
                category=cat,
                exploration=False,
                sampled_values=sampled_values,
                arm_state=arm_state,
            )

    def record_outcome(
        self,
        request_id: str,
        strategy: ReasoningStrategy,
        category: ProblemCategory,
        latency_ms: float,
        response_length: int = 0,
        metrics: Optional[Dict[str, Optional[float]]] = None,
    ) -> float:
        """Record the outcome of a strategy execution.

        Updates the Beta distribution parameters for the arm.

        Args:
            request_id: Unique request identifier.
            strategy: Strategy that was executed.
            category: Problem category.
            latency_ms: Execution time in milliseconds.
            response_length: Length of generated response.
            metrics: Optional dict of evaluation metrics.

        Returns:
            Composite reward value in [0, 1].
        """
        if not self.enabled:
            return 0.5

        all_metrics = dict(metrics or {})

        # Compute latency score: lower is better, normalized with sigmoid
        # Target: 2000ms → 0.5, <500ms → ~0.9, >10000ms → ~0.1
        latency_score = 1.0 / (1.0 + math.exp((latency_ms - 2000) / 1500))
        all_metrics["latency_score"] = latency_score

        composite_reward = self._reward_computer.compute(all_metrics)

        # Use dedicated DB lock (not self._lock) to avoid blocking non-DB operations.
        # SQLite busy_timeout handles inter-process contention; this lock handles
        # intra-process serialization of writes.
        with self._db_lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA busy_timeout = 5000")
            try:
                # Store outcome
                conn.execute(
                    """INSERT OR REPLACE INTO strategy_outcomes
                       (request_id, strategy, category, latency_ms, response_length,
                        composite_reward, metrics_json, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        request_id, strategy.value, category.value,
                        latency_ms, response_length, composite_reward,
                        json.dumps(all_metrics, default=str), time.time(),
                    ),
                )

                # Update arm: reward → increase alpha, (1-reward) → increase beta
                conn.execute(
                    """UPDATE strategy_arms
                       SET alpha = alpha + ?,
                           beta = beta + ?,
                           total_pulls = total_pulls + 1,
                           total_reward = total_reward + ?,
                           last_updated = ?
                       WHERE strategy = ? AND category = ?""",
                    (
                        composite_reward,
                        1.0 - composite_reward,
                        composite_reward,
                        time.time(),
                        strategy.value,
                        category.value,
                    ),
                )

                # Performance log
                conn.execute(
                    """INSERT INTO performance_logs
                       (request_id, event_type, payload_json, timestamp)
                       VALUES (?, 'outcome', ?, ?)""",
                    (
                        request_id,
                        json.dumps({
                            "strategy": strategy.value,
                            "category": category.value,
                            "composite_reward": composite_reward,
                            "latency_ms": latency_ms,
                        }),
                        time.time(),
                    ),
                )

                conn.commit()
            finally:
                conn.close()

        logger.info(
            f"[StrategyBandit] outcome: {strategy.value}/{category.value} "
            f"reward={composite_reward:.3f} latency={latency_ms:.0f}ms "
            f"(request={request_id})"
        )

        # Fire-and-forget push so the Mini App Strategies tab updates live.
        try:
            from api.services.websocket_hub import push_bandit_pull
            push_bandit_pull(
                arm=f"{category.value}:{strategy.value}",
                reward=composite_reward,
                totals={
                    "category": category.value,
                    "strategy": strategy.value,
                    "latency_ms": latency_ms,
                },
            )
        except Exception as _push_exc:
            logger.debug("[StrategyBandit] push_bandit_pull skipped: %s", _push_exc)

        return composite_reward

    def record_user_feedback(
        self,
        request_id: str,
        feedback: float,
    ):
        """Record delayed user feedback for a previous interaction.

        Args:
            request_id: The request to update.
            feedback: User feedback score in [0, 1] (0=bad, 1=good).
        """
        if not self.enabled:
            return

        feedback = max(0.0, min(1.0, float(feedback)))

        with self._db_lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA busy_timeout = 5000")
            try:
                # Get existing outcome
                row = conn.execute(
                    "SELECT strategy, category, metrics_json, composite_reward "
                    "FROM strategy_outcomes WHERE request_id = ?",
                    (request_id,),
                ).fetchone()

                if row is None:
                    logger.warning(
                        f"[StrategyBandit] No outcome found for request={request_id}"
                    )
                    return

                strategy, category, metrics_json, old_reward = row
                metrics = json.loads(metrics_json) if metrics_json else {}

                # Recompute with user feedback
                metrics["user_feedback"] = feedback
                new_reward = self._reward_computer.compute(metrics)
                reward_delta = new_reward - (old_reward or 0.5)

                # Update outcome
                conn.execute(
                    """UPDATE strategy_outcomes
                       SET user_feedback = ?, metrics_json = ?,
                           composite_reward = ?
                       WHERE request_id = ?""",
                    (feedback, json.dumps(metrics, default=str), new_reward, request_id),
                )

                # Adjust arm parameters by delta (floor alpha/beta at 1.0)
                conn.execute(
                    """UPDATE strategy_arms
                       SET alpha = MAX(1.0, alpha + ?),
                           beta = MAX(1.0, beta + ?),
                           total_reward = MAX(0.0, total_reward + ?),
                           last_updated = ?
                       WHERE strategy = ? AND category = ?""",
                    (
                        max(0, reward_delta),
                        max(0, -reward_delta),
                        reward_delta,
                        time.time(),
                        strategy,
                        category,
                    ),
                )

                conn.commit()

                logger.info(
                    f"[StrategyBandit] user feedback: {feedback:.2f} for "
                    f"{strategy}/{category} reward {old_reward:.3f}→{new_reward:.3f} "
                    f"(request={request_id})"
                )
            finally:
                conn.close()

    def decay_arms(self, half_life_days: float = 30.0):
        """Apply temporal decay to arm parameters, moving toward priors.

        This prevents stale data from dominating. Arms decay toward
        Beta(1,1) with exponential decay based on time since last update.
        """
        if not self.enabled:
            return

        now = time.time()
        decay_constant = math.log(2) / (half_life_days * 86400)

        with self._db_lock:
            conn = sqlite3.connect(self._db_path)
            try:
                rows = conn.execute(
                    "SELECT strategy, category, alpha, beta, last_updated "
                    "FROM strategy_arms"
                ).fetchall()

                for strategy, category, alpha, beta, last_updated in rows:
                    age_seconds = now - last_updated
                    decay_factor = math.exp(-decay_constant * age_seconds)

                    # Decay toward prior Beta(1,1)
                    new_alpha = 1.0 + (alpha - 1.0) * decay_factor
                    new_beta = 1.0 + (beta - 1.0) * decay_factor

                    conn.execute(
                        """UPDATE strategy_arms
                           SET alpha = ?, beta = ?, last_updated = ?
                           WHERE strategy = ? AND category = ?""",
                        (new_alpha, new_beta, now, strategy, category),
                    )

                conn.commit()
                logger.info(
                    f"[StrategyBandit] decay applied to {len(rows)} arms "
                    f"(half_life={half_life_days}d)"
                )
            finally:
                conn.close()

    def get_arm_stats(self) -> Dict[str, List[Dict]]:
        """Get current arm statistics grouped by category.

        Returns:
            Dict mapping category names to lists of arm stat dicts.
        """
        with self._db_lock:
            conn = sqlite3.connect(self._db_path)
            try:
                rows = conn.execute(
                    "SELECT strategy, category, alpha, beta, total_pulls, total_reward, last_updated "
                    "FROM strategy_arms ORDER BY category, strategy"
                ).fetchall()
            finally:
                conn.close()

        stats: Dict[str, List[Dict]] = {}
        for strategy, category, alpha, beta, pulls, reward, updated in rows:
            if category not in stats:
                stats[category] = []
            stats[category].append({
                "strategy": strategy,
                "alpha": round(alpha, 3),
                "beta": round(beta, 3),
                "mean_reward": round(alpha / (alpha + beta), 3),
                "total_pulls": pulls,
                "total_reward": round(reward, 3),
                "last_updated": updated,
            })
        return stats

    def get_best_strategy(self, category: ProblemCategory) -> Optional[str]:
        """Get the strategy with the highest mean reward for a category."""
        with self._db_lock:
            conn = sqlite3.connect(self._db_path)
            try:
                row = conn.execute(
                    """SELECT strategy, alpha / (alpha + beta) as mean_reward
                       FROM strategy_arms
                       WHERE category = ?
                       ORDER BY mean_reward DESC
                       LIMIT 1""",
                    (category.value,),
                ).fetchone()
            finally:
                conn.close()
        return row[0] if row else None

    def get_stats_summary(self) -> Dict:
        """Get a summary of bandit state for monitoring/debugging."""
        with self._db_lock:
            conn = sqlite3.connect(self._db_path)
            try:
                total_outcomes = conn.execute(
                    "SELECT COUNT(*) FROM strategy_outcomes"
                ).fetchone()[0]

                total_arms = conn.execute(
                    "SELECT COUNT(*) FROM strategy_arms"
                ).fetchone()[0]

                # Category breakdown
                category_counts = conn.execute(
                    "SELECT category, COUNT(*) FROM strategy_outcomes GROUP BY category"
                ).fetchall()

                # Best per category
                best_per_category = {}
                for cat in ProblemCategory:
                    row = conn.execute(
                        """SELECT strategy, alpha / (alpha + beta) as mean_reward, total_pulls
                           FROM strategy_arms
                           WHERE category = ? AND total_pulls > 0
                           ORDER BY mean_reward DESC
                           LIMIT 1""",
                        (cat.value,),
                    ).fetchone()
                    if row:
                        best_per_category[cat.value] = {
                            "strategy": row[0],
                            "mean_reward": round(row[1], 3),
                            "pulls": row[2],
                        }

                return {
                    "enabled": self.enabled,
                    "epsilon": self.epsilon,
                    "total_outcomes": total_outcomes,
                    "total_arms": total_arms,
                    "category_counts": dict(category_counts),
                    "best_per_category": best_per_category,
                }
            finally:
                conn.close()


# ============================================================================
# Singleton
# ============================================================================

_strategy_bandit: Optional[StrategyBandit] = None
_singleton_lock = threading.Lock()


def get_strategy_bandit() -> StrategyBandit:
    """Get or create the singleton StrategyBandit."""
    global _strategy_bandit
    if _strategy_bandit is None:
        with _singleton_lock:
            if _strategy_bandit is None:
                from aura.config import Config
                _strategy_bandit = StrategyBandit(
                    epsilon=getattr(Config, "STRATEGY_BANDIT_EPSILON", 0.1),
                    enabled=getattr(Config, "STRATEGY_BANDIT_ENABLED", True),
                )
    return _strategy_bandit
