"""Prompt Evolution Engine — Phase 4 of ADV-01 meta-cognitive self-improvement.

A three-stage pipeline (critique -> revise -> evaluate) that self-modifies
system prompts based on accumulated performance data.

Disabled by default (PROMPT_EVOLUTION_ENABLED=false). Only the "reasoner"
module is wired up initially; schema supports planner, critic, tool_selector
for future work.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default seed prompt for the reasoner module
DEFAULT_REASONER_PROMPT = (
    "You are a careful, step-by-step reasoner. "
    "Break complex problems into parts. "
    "Show your reasoning process. "
    "Verify your conclusions before presenting them. "
    "If uncertain, say so explicitly."
)


class PromptEvolutionEngine:
    """Self-modifying prompt engine that evolves system prompts based on
    accumulated performance data.

    Three-stage pipeline:
    1. Critique — analyze current prompt weaknesses from performance data
    2. Revise — generate candidate improved prompts
    3. Evaluate — score candidates on held-out examples, accept if improved
    """

    # --- Constants ---
    EVOLVE_EVERY_N = 50          # check evolution every N invocations
    MIN_HELD_OUT = 20            # minimum held-out examples for evaluation
    NUM_CANDIDATES = 3           # candidates per revision round
    IMPROVEMENT_THRESHOLD = 1.0  # mean must improve by >1 std
    REGRESSION_LIMIT = 0.20      # no failure category may regress >20%
    RATE_LIMIT_DAYS = 7          # max 1 change per module per week
    MAX_VERSIONS_KEPT = 5        # keep last 5 for rollback
    EVOLVABLE_MODULES = ("reasoner", "planner", "critic", "tool_selector")

    def __init__(
        self,
        db_path: Optional[str] = None,
        enabled: bool = False,
        evolve_interval: int = 50,
    ):
        self.enabled = enabled
        self._lock = threading.Lock()
        self.EVOLVE_EVERY_N = evolve_interval

        if db_path is None:
            data_dir = Path(os.getenv("AURA_DATA_DIR", "data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(data_dir / "aura_meta.db")
        else:
            self._db_path = db_path

        self._init_db()

    # ------------------------------------------------------------------
    # Database initialization
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and indices if they don't exist."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    prompt_text TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 0,
                    total_invocations INTEGER DEFAULT 0,
                    avg_composite_reward REAL DEFAULT 0.0,
                    avg_user_satisfaction REAL DEFAULT 0.0,
                    failure_counts TEXT DEFAULT '{}',
                    UNIQUE(module, version)
                );

                CREATE TABLE IF NOT EXISTS prompt_evolution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module TEXT NOT NULL,
                    old_version INTEGER,
                    new_version INTEGER,
                    change_type TEXT NOT NULL,
                    change_reason TEXT,
                    critique_text TEXT,
                    candidates_json TEXT,
                    eval_scores_json TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_pv_module_active
                    ON prompt_versions(module, is_active);

                CREATE INDEX IF NOT EXISTS idx_pel_module_ts
                    ON prompt_evolution_log(module, timestamp);
            """)
            conn.commit()
        finally:
            conn.close()
        logger.info("[PromptEvolution] DB initialized")

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # seed_prompt — insert version 1 if none exists
    # ------------------------------------------------------------------

    def seed_prompt(self, module: str, prompt_text: str) -> int:
        """Insert version 1 of a prompt if none exists for this module.

        Returns the version number (1 if newly seeded, or existing version).
        """
        if module not in self.EVOLVABLE_MODULES:
            raise ValueError(f"Module '{module}' not in EVOLVABLE_MODULES")

        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                # Check if any version already exists
                row = conn.execute(
                    "SELECT version FROM prompt_versions WHERE module = ? "
                    "ORDER BY version DESC LIMIT 1",
                    (module,),
                ).fetchone()

                if row is not None:
                    return row["version"]

                # Insert version 1 as active
                conn.execute(
                    "INSERT INTO prompt_versions "
                    "(module, version, prompt_text, prompt_hash, created_at, is_active) "
                    "VALUES (?, 1, ?, ?, ?, 1)",
                    (module, prompt_text, prompt_hash, now),
                )

                # Log seed event
                conn.execute(
                    "INSERT INTO prompt_evolution_log "
                    "(module, old_version, new_version, change_type, change_reason, timestamp) "
                    "VALUES (?, NULL, 1, 'seed', 'Initial seed prompt', ?)",
                    (module, now),
                )

                conn.commit()
                logger.info(f"[PromptEvolution] Seeded {module} v1 (hash={prompt_hash})")
                return 1
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # get_active_prompt — fast indexed read (hot path)
    # ------------------------------------------------------------------

    def get_active_prompt(self, module: str) -> Optional[str]:
        """Return the currently active prompt text for a module, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT prompt_text FROM prompt_versions "
                "WHERE module = ? AND is_active = 1",
                (module,),
            ).fetchone()
            return row["prompt_text"] if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # record_invocation — update running avg + failure counts
    # ------------------------------------------------------------------

    def record_invocation(
        self, module: str, reward: float, failure_type: Optional[str] = None
    ) -> None:
        """Record an invocation result and potentially trigger evolution."""
        if not self.enabled:
            return

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, total_invocations, avg_composite_reward, failure_counts "
                    "FROM prompt_versions WHERE module = ? AND is_active = 1",
                    (module,),
                ).fetchone()

                if row is None:
                    return

                n = row["total_invocations"]
                old_avg = row["avg_composite_reward"]
                new_n = n + 1
                # Running average update
                new_avg = old_avg + (reward - old_avg) / new_n

                # Update failure counts
                failure_counts = json.loads(row["failure_counts"] or "{}")
                if failure_type:
                    failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

                conn.execute(
                    "UPDATE prompt_versions SET "
                    "total_invocations = ?, avg_composite_reward = ?, failure_counts = ? "
                    "WHERE id = ?",
                    (new_n, new_avg, json.dumps(failure_counts), row["id"]),
                )
                conn.commit()
            finally:
                conn.close()

        # Check if evolution should be triggered (outside lock)
        if new_n > 0 and new_n % self.EVOLVE_EVERY_N == 0:
            self.maybe_evolve(module)

    # ------------------------------------------------------------------
    # maybe_evolve — gate checks + spawn background thread
    # ------------------------------------------------------------------

    def maybe_evolve(self, module: str) -> bool:
        """Check if evolution should run, and spawn background thread if so.

        Returns True if evolution was triggered.
        """
        if not self.enabled:
            return False

        # Rate limit check
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT timestamp FROM prompt_evolution_log "
                "WHERE module = ? AND change_type = 'promote' "
                "ORDER BY timestamp DESC LIMIT 1",
                (module,),
            ).fetchone()

            if row:
                last_change = datetime.fromisoformat(row["timestamp"])
                cutoff = datetime.now(timezone.utc) - timedelta(days=self.RATE_LIMIT_DAYS)
                if last_change > cutoff:
                    logger.info(f"[PromptEvolution] Rate limited for {module}")
                    return False

            # Data sufficiency check — need MIN_HELD_OUT traces
            try:
                trace_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM reasoning_traces "
                    "WHERE composite_reward IS NOT NULL",
                ).fetchone()["cnt"]
            except sqlite3.OperationalError:
                # reasoning_traces table may not exist yet
                logger.info(f"[PromptEvolution] reasoning_traces table not found, skipping")
                return False

            if trace_count < self.MIN_HELD_OUT:
                logger.info(
                    f"[PromptEvolution] Insufficient data for {module}: "
                    f"{trace_count} < {self.MIN_HELD_OUT}"
                )
                return False
        finally:
            conn.close()

        # Spawn background thread
        thread = threading.Thread(
            target=self._run_evolution,
            args=(module,),
            daemon=True,
            name=f"prompt-evolution-{module}",
        )
        thread.start()
        logger.info(f"[PromptEvolution] Evolution thread started for {module}")
        return True

    # ------------------------------------------------------------------
    # _run_evolution — orchestrate: critique -> revise -> evaluate -> promote
    # ------------------------------------------------------------------

    def _run_evolution(self, module: str) -> None:
        """Run the full evolution pipeline in a background thread."""
        try:
            logger.info(f"[PromptEvolution] Running evolution for {module}")

            # Stage 1: Critique
            critique = self._critique(module)
            if not critique:
                logger.info(f"[PromptEvolution] No critique generated for {module}")
                return

            # Stage 2: Revise
            candidates = self._revise(module, critique)
            if not candidates:
                logger.info(f"[PromptEvolution] No candidates generated for {module}")
                return

            # Stage 3: Evaluate
            best = self._evaluate_candidates(module, candidates)
            if best is None:
                logger.info(f"[PromptEvolution] No candidate passed evaluation for {module}")
                return

            # Stage 4: Promote
            new_version = self._promote(module, best, critique, "Evolution pipeline")
            logger.info(f"[PromptEvolution] Promoted {module} to v{new_version}")

        except Exception as e:
            logger.error(f"[PromptEvolution] Evolution failed for {module}: {e}")

    # ------------------------------------------------------------------
    # _critique — analyze current prompt weaknesses
    # ------------------------------------------------------------------

    def _critique(self, module: str) -> str:
        """Query performance data and ask LLM to critique the current prompt."""
        conn = self._connect()
        try:
            # Get active prompt and its stats
            row = conn.execute(
                "SELECT prompt_text, total_invocations, avg_composite_reward, failure_counts "
                "FROM prompt_versions WHERE module = ? AND is_active = 1",
                (module,),
            ).fetchone()

            if row is None:
                return ""

            prompt_text = row["prompt_text"]
            stats = {
                "total_invocations": row["total_invocations"],
                "avg_composite_reward": row["avg_composite_reward"],
                "failure_counts": json.loads(row["failure_counts"] or "{}"),
            }

            # Get recent low-reward traces
            low_traces = conn.execute(
                "SELECT problem, composite_reward FROM reasoning_traces "
                "WHERE composite_reward < 0.5 "
                "ORDER BY created_at DESC LIMIT 5",
            ).fetchall()

            low_examples = "\n".join(
                f"- Problem: {t['problem'][:200]} (reward: {t['composite_reward']:.2f})"
                for t in low_traces
            )
        finally:
            conn.close()

        critique_prompt = (
            f"You are analyzing a system prompt used for the '{module}' module.\n\n"
            f"Current prompt:\n\"\"\"\n{prompt_text}\n\"\"\"\n\n"
            f"Performance statistics:\n"
            f"- Total invocations: {stats['total_invocations']}\n"
            f"- Average composite reward: {stats['avg_composite_reward']:.3f}\n"
            f"- Failure counts: {json.dumps(stats['failure_counts'])}\n\n"
            f"Recent low-reward examples:\n{low_examples}\n\n"
            f"Provide a brief, specific critique of this prompt. "
            f"What weaknesses could be improved? Focus on actionable issues."
        )

        try:
            from aura.brain import OllamaBrain
            brain = OllamaBrain(warmup=False)
            critique = brain.think(critique_prompt, task_type=None)
            return critique or ""
        except Exception as e:
            logger.error(f"[PromptEvolution] Critique LLM call failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # _revise — generate candidate prompts
    # ------------------------------------------------------------------

    def _revise(self, module: str, critique: str) -> List[str]:
        """Ask LLM for NUM_CANDIDATES candidate prompts based on critique."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT prompt_text FROM prompt_versions "
                "WHERE module = ? AND is_active = 1",
                (module,),
            ).fetchone()
            if row is None:
                return []
            current_prompt = row["prompt_text"]
        finally:
            conn.close()

        revise_prompt = (
            f"You are improving a system prompt for the '{module}' module.\n\n"
            f"Current prompt:\n\"\"\"\n{current_prompt}\n\"\"\"\n\n"
            f"Critique:\n{critique}\n\n"
            f"Generate exactly {self.NUM_CANDIDATES} improved versions of this prompt. "
            f"Each should address the critique while preserving the core intent.\n"
            f"Separate each candidate with the delimiter: ===CANDIDATE===\n"
            f"Do not include the delimiter before the first candidate."
        )

        try:
            from aura.brain import OllamaBrain
            brain = OllamaBrain(warmup=False)
            raw = brain.think(revise_prompt, task_type=None)
            if not raw:
                return []

            # Parse candidates by delimiter
            parts = raw.split("===CANDIDATE===")
            candidates = [p.strip() for p in parts if p.strip()]
            return candidates[:self.NUM_CANDIDATES]
        except Exception as e:
            logger.error(f"[PromptEvolution] Revise LLM call failed: {e}")
            return []

    # ------------------------------------------------------------------
    # _evaluate_candidates — score on held-out data
    # ------------------------------------------------------------------

    def _evaluate_candidates(
        self, module: str, candidates: List[str]
    ) -> Optional[str]:
        """Re-run Brain.think on held-out problems, score with JudgeEvaluator.

        Accept if mean improves by >1 std and no failure category regresses >20%.
        Returns the best candidate prompt text, or None.
        """
        conn = self._connect()
        try:
            # Get held-out examples from reasoning_traces
            held_out = conn.execute(
                "SELECT problem, composite_reward FROM reasoning_traces "
                "WHERE composite_reward IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (self.MIN_HELD_OUT,),
            ).fetchall()

            if len(held_out) < self.MIN_HELD_OUT:
                return None

            # Baseline scores (existing rewards)
            baseline_scores = [row["composite_reward"] for row in held_out]
        finally:
            conn.close()

        try:
            from aura.brain import OllamaBrain
            from aura.consciousness.reward_signals import JudgeEvaluator

            brain = OllamaBrain(warmup=False)
            judge = JudgeEvaluator()
            llm_call = lambda prompt: brain.think(prompt, task_type=None)

            best_candidate = None
            best_mean = -1.0

            import statistics
            baseline_mean = statistics.mean(baseline_scores)
            baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0.1

            for candidate in candidates:
                scores = []
                failure_counts = {}

                for example in held_out:
                    problem = example["problem"]
                    # Re-run with candidate prompt as system context
                    response = brain.think(
                        problem, task_type=None, system_prompt=candidate
                    )
                    # Score with judge
                    result = judge.evaluate(problem, response or "", llm_call)
                    scores.append(result.score)
                    if result.score < 0.5:
                        cat = "low_quality"
                        failure_counts[cat] = failure_counts.get(cat, 0) + 1

                if not scores:
                    continue

                candidate_mean = statistics.mean(scores)

                # Check improvement threshold: >1 std above baseline
                if candidate_mean < baseline_mean + self.IMPROVEMENT_THRESHOLD * baseline_std:
                    continue

                # Check regression limit
                baseline_failure_rate = sum(
                    1 for s in baseline_scores if s < 0.5
                ) / len(baseline_scores)
                candidate_failure_rate = sum(
                    1 for s in scores if s < 0.5
                ) / len(scores)

                if (
                    candidate_failure_rate
                    > baseline_failure_rate + self.REGRESSION_LIMIT
                ):
                    continue

                if candidate_mean > best_mean:
                    best_mean = candidate_mean
                    best_candidate = candidate

            return best_candidate

        except Exception as e:
            logger.error(f"[PromptEvolution] Evaluation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # _promote — activate new prompt version
    # ------------------------------------------------------------------

    def _promote(
        self,
        module: str,
        new_prompt: str,
        critique: str,
        reason: str,
    ) -> int:
        """Deactivate old prompt, insert new version, log, prune old versions."""
        prompt_hash = hashlib.sha256(new_prompt.encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                # Get current version
                row = conn.execute(
                    "SELECT version FROM prompt_versions "
                    "WHERE module = ? AND is_active = 1",
                    (module,),
                ).fetchone()
                old_version = row["version"] if row else 0
                new_version = old_version + 1

                # Deactivate all versions for this module
                conn.execute(
                    "UPDATE prompt_versions SET is_active = 0 WHERE module = ?",
                    (module,),
                )

                # Insert new version
                conn.execute(
                    "INSERT INTO prompt_versions "
                    "(module, version, prompt_text, prompt_hash, created_at, is_active) "
                    "VALUES (?, ?, ?, ?, ?, 1)",
                    (module, new_version, new_prompt, prompt_hash, now),
                )

                # Log promote event
                conn.execute(
                    "INSERT INTO prompt_evolution_log "
                    "(module, old_version, new_version, change_type, change_reason, "
                    "critique_text, timestamp) "
                    "VALUES (?, ?, ?, 'promote', ?, ?, ?)",
                    (module, old_version, new_version, reason, critique, now),
                )

                # Prune old versions (keep MAX_VERSIONS_KEPT most recent)
                conn.execute(
                    "DELETE FROM prompt_versions WHERE module = ? AND id NOT IN ("
                    "  SELECT id FROM prompt_versions WHERE module = ? "
                    "  ORDER BY version DESC LIMIT ?"
                    ")",
                    (module, module, self.MAX_VERSIONS_KEPT),
                )

                conn.commit()
                logger.info(
                    f"[PromptEvolution] Promoted {module} v{old_version} -> v{new_version}"
                )
                return new_version
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # rollback — reactivate previous version
    # ------------------------------------------------------------------

    def rollback(self, module: str) -> bool:
        """Roll back to the previous prompt version.

        Returns True if rollback succeeded, False if no previous version.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                # Get current active version
                current = conn.execute(
                    "SELECT version FROM prompt_versions "
                    "WHERE module = ? AND is_active = 1",
                    (module,),
                ).fetchone()

                if current is None:
                    return False

                current_version = current["version"]

                # Find previous version
                previous = conn.execute(
                    "SELECT version FROM prompt_versions "
                    "WHERE module = ? AND version < ? "
                    "ORDER BY version DESC LIMIT 1",
                    (module, current_version),
                ).fetchone()

                if previous is None:
                    return False

                prev_version = previous["version"]

                # Deactivate all, activate previous
                conn.execute(
                    "UPDATE prompt_versions SET is_active = 0 WHERE module = ?",
                    (module,),
                )
                conn.execute(
                    "UPDATE prompt_versions SET is_active = 1, "
                    "total_invocations = 0, avg_composite_reward = 0.0, "
                    "failure_counts = '{}' "
                    "WHERE module = ? AND version = ?",
                    (module, prev_version),
                )

                # Log rollback
                conn.execute(
                    "INSERT INTO prompt_evolution_log "
                    "(module, old_version, new_version, change_type, change_reason, timestamp) "
                    "VALUES (?, ?, ?, 'rollback', 'Manual rollback', ?)",
                    (module, current_version, prev_version, now),
                )

                conn.commit()
                logger.info(
                    f"[PromptEvolution] Rolled back {module} v{current_version} -> v{prev_version}"
                )
                return True
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # get_stats — monitoring summary
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a monitoring summary for all modules."""
        conn = self._connect()
        try:
            modules = {}
            rows = conn.execute(
                "SELECT module, version, is_active, total_invocations, "
                "avg_composite_reward, failure_counts "
                "FROM prompt_versions ORDER BY module, version"
            ).fetchall()

            for row in rows:
                mod = row["module"]
                if mod not in modules:
                    modules[mod] = {
                        "versions": [],
                        "active_version": None,
                        "total_invocations": 0,
                    }
                entry = {
                    "version": row["version"],
                    "is_active": bool(row["is_active"]),
                    "total_invocations": row["total_invocations"],
                    "avg_composite_reward": row["avg_composite_reward"],
                    "failure_counts": json.loads(row["failure_counts"] or "{}"),
                }
                modules[mod]["versions"].append(entry)
                if row["is_active"]:
                    modules[mod]["active_version"] = row["version"]
                    modules[mod]["total_invocations"] = row["total_invocations"]

            # Evolution event counts
            events = conn.execute(
                "SELECT module, change_type, COUNT(*) as cnt "
                "FROM prompt_evolution_log GROUP BY module, change_type"
            ).fetchall()

            for ev in events:
                mod = ev["module"]
                if mod in modules:
                    key = f"{ev['change_type']}_count"
                    modules[mod][key] = ev["cnt"]

            return {
                "enabled": self.enabled,
                "evolve_every_n": self.EVOLVE_EVERY_N,
                "modules": modules,
            }
        finally:
            conn.close()


# ======================================================================
# Singleton
# ======================================================================

_prompt_evolution_engine: Optional[PromptEvolutionEngine] = None
_singleton_lock = threading.Lock()


def get_prompt_evolution_engine() -> PromptEvolutionEngine:
    """Get or create the singleton PromptEvolutionEngine instance."""
    global _prompt_evolution_engine
    if _prompt_evolution_engine is None:
        with _singleton_lock:
            if _prompt_evolution_engine is None:
                from aura.config import Config
                _prompt_evolution_engine = PromptEvolutionEngine(
                    enabled=getattr(Config, "PROMPT_EVOLUTION_ENABLED", False),
                    evolve_interval=getattr(Config, "PROMPT_EVOLUTION_INTERVAL", 50),
                )
    return _prompt_evolution_engine
