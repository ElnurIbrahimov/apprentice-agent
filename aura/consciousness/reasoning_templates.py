"""
Reasoning Template Library — Collect high-reward traces, abstract into reusable templates.

Phase 3 of ADV-01: AURA learns *how* to reason by collecting successful reasoning
traces, abstracting them into templates via LLM, and injecting template guidance
into future reasoning via system prompt.

Part of AURA's meta-cognitive self-improvement system.
"""

import json
import logging
import math
import os
import sqlite3
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Rich Trace Builders (module-level — strategy-aware trace construction)
# ============================================================================

def build_trace_from_mcts(mcts_result: dict) -> str:
    """Build a structured trace string from an MCTS result dict.

    Extracts reasoning_steps (type/content/confidence/value per node) and
    metadata (iterations, nodes_explored, time_taken, reflections_count).
    Truncates step content to 500 chars.

    Returns JSON string; falls back to simple fallback on any error.
    """
    try:
        steps = []
        for step in mcts_result.get("reasoning_steps", []):
            steps.append({
                "type": step.get("type", "unknown"),
                "content": str(step.get("content", ""))[:500],
                "confidence": step.get("confidence"),
                "value": step.get("value"),
            })

        metadata = {}
        raw_meta = mcts_result.get("metadata", {})
        for key in ("iterations", "nodes_explored", "time_taken", "reflections_count"):
            if key in raw_meta:
                metadata[key] = raw_meta[key]

        trace = {
            "strategy": "mcts",
            "steps": steps,
            "metadata": metadata,
            "confidence": mcts_result.get("confidence"),
            "success": mcts_result.get("success"),
        }
        return json.dumps(trace)
    except Exception as e:
        logger.warning(f"[TemplateLib] build_trace_from_mcts fallback: {e}")
        return json.dumps([{"step": "mcts_fallback", "content": str(mcts_result)[:500]}])


def build_trace_from_reflexion(reflexion_result) -> str:
    """Build a structured trace string from a Reflexion result.

    Uses getattr() for graceful degradation — result may be a string or
    a ReflexionResult dataclass.

    Returns JSON string; falls back to simple fallback on any error.
    """
    try:
        attempts = getattr(reflexion_result, "attempts", None)
        reflections_used = getattr(reflexion_result, "reflections_used", [])
        new_reflection = getattr(reflexion_result, "new_reflection", None)
        success = getattr(reflexion_result, "success", None)
        final_output = str(getattr(reflexion_result, "final_output", ""))[:500]

        trace = {
            "strategy": "reflexion",
            "attempts": attempts,
            "reflections_used": list(reflections_used) if reflections_used else [],
            "new_reflection": str(new_reflection)[:500] if new_reflection else None,
            "success": success,
            "final_output_preview": final_output,
        }
        return json.dumps(trace)
    except Exception as e:
        logger.warning(f"[TemplateLib] build_trace_from_reflexion fallback: {e}")
        return json.dumps([{"step": "reflexion_fallback", "content": str(reflexion_result)[:500]}])


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ReasoningTrace:
    """A recorded high-reward reasoning trace."""
    trace_id: str
    problem: str
    category: str
    strategy: str
    full_trace: str          # JSON array of step dicts
    composite_reward: float
    problem_embedding: Optional[bytes] = None
    user_feedback: Optional[str] = None
    created_at: str = ""


@dataclass
class ReasoningTemplate:
    """An abstracted reasoning pattern derived from multiple traces."""
    template_id: str
    name: str
    description: str
    abstract_steps: str      # JSON array
    applicable_categories: str  # JSON array
    source_trace_ids: str    # JSON array
    embedding: Optional[bytes] = None
    granularity: str = "pattern"  # atomic | pattern | meta
    times_used: int = 0
    avg_reward_when_used: float = 0.0
    avg_reward_baseline: float = 0.0
    status: str = "active"   # active | deprecated | archived
    created_at: str = ""
    last_used: Optional[str] = None


@dataclass
class TemplateMatch:
    """Result of template retrieval — a matched template with scoring."""
    template: ReasoningTemplate
    similarity_score: float
    guidance_text: str


# ============================================================================
# Embedding Helper (self-contained to avoid circular imports)
# ============================================================================

class _EmbeddingHelper:
    """Lazy-loaded sentence-transformers wrapper for embedding operations.

    Uses all-MiniLM-L6-v2 (same model as episodic memory, skill library).
    384-dimensional float32 embeddings stored as packed bytes in SQLite BLOBs.
    """

    EMBEDDING_DIM = 384
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._available = None

    def _load_model(self):
        """Lazy load the embedding model on first use."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from aura.tools._shared_models import get_sentence_transformer
                self._model = get_sentence_transformer()
                self._available = True
                logger.info(f"[TemplateLib] Loaded embedding model: {self.MODEL_NAME}")
            except Exception as e:
                self._available = False
                logger.warning(f"[TemplateLib] Embedding model unavailable: {e}")

    @property
    def available(self) -> bool:
        if self._available is None:
            self._load_model()
        return self._available

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed text into a 384-dim vector. Returns None if model unavailable."""
        self._load_model()
        if not self._available or self._model is None:
            return None
        try:
            vec = self._model.encode(text, show_progress_bar=False)
            return vec.tolist()
        except Exception as e:
            logger.warning(f"[TemplateLib] Embedding error: {e}")
            return None

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def serialize_embedding(vec: List[float]) -> bytes:
        """Pack a float list into bytes for SQLite BLOB storage."""
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def deserialize_embedding(blob: bytes) -> List[float]:
        """Unpack bytes back into a float list."""
        count = len(blob) // 4  # 4 bytes per float32
        return list(struct.unpack(f"{count}f", blob))


# ============================================================================
# Reasoning Template Library
# ============================================================================

class ReasoningTemplateLibrary:
    """Collects high-reward reasoning traces, abstracts them into reusable
    templates, and retrieves relevant templates at inference time.

    SQLite-backed, thread-safe, singleton pattern (same as StrategyBandit).
    """

    # Thresholds
    REWARD_THRESHOLD = 0.8        # Minimum reward to collect trace
    BATCH_SIZE = 100              # Traces before triggering abstraction
    SIMILARITY_THRESHOLD = 0.5    # Minimum similarity for template match
    DEDUP_THRESHOLD = 0.85        # Embedding similarity for deduplication
    DEPRECATION_MIN_USES = 20     # Minimum uses before deprecation check
    DEPRECATION_REWARD_GAP = 0.0  # Deprecated if avg_reward <= baseline

    def __init__(
        self,
        db_path: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._embedder = _EmbeddingHelper()
        self._trace_count_since_abstraction = 0

        # Resolve DB path (shared aura_meta.db)
        if db_path is None:
            data_dir = Path(os.getenv("AURA_DATA_DIR", "data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(data_dir / "aura_meta.db")
        else:
            self._db_path = db_path

        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema for traces and templates."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS reasoning_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT UNIQUE NOT NULL,
                problem TEXT NOT NULL,
                problem_category TEXT,
                strategy_used TEXT,
                full_trace TEXT NOT NULL,
                composite_reward REAL NOT NULL,
                user_feedback TEXT,
                problem_embedding BLOB,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reasoning_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                abstract_steps TEXT NOT NULL,
                applicable_categories TEXT,
                source_trace_ids TEXT,
                embedding BLOB,
                times_used INTEGER DEFAULT 0,
                avg_reward_when_used REAL DEFAULT 0.0,
                avg_reward_baseline REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                last_used TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_traces_category
                ON reasoning_traces(problem_category);
            CREATE INDEX IF NOT EXISTS idx_traces_reward
                ON reasoning_traces(composite_reward);
            CREATE INDEX IF NOT EXISTS idx_templates_status
                ON reasoning_templates(status);
        """)

        # Schema migration: add granularity column if missing
        try:
            conn.execute("ALTER TABLE reasoning_templates ADD COLUMN granularity TEXT DEFAULT 'pattern'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        conn.commit()
        conn.close()
        logger.info(f"[TemplateLib] DB initialized at {self._db_path}")

    # ----------------------------------------------------------------
    # Trace Collection
    # ----------------------------------------------------------------

    def collect_trace(
        self,
        request_id: str,
        problem: str,
        category: str,
        strategy: str,
        full_trace: str,
        reward: float,
        user_feedback: Optional[str] = None,
    ) -> bool:
        """Store a high-reward reasoning trace.

        Args:
            request_id: Unique request identifier.
            problem: The problem/query text.
            category: Problem category (e.g. 'code', 'math').
            strategy: Strategy used (e.g. 'chain_of_thought').
            full_trace: JSON string of reasoning steps.
            reward: Composite reward value.
            user_feedback: Optional user feedback text.

        Returns:
            True if trace was stored, False if skipped.
        """
        if not self.enabled:
            return False

        if reward < self.REWARD_THRESHOLD:
            return False

        if not problem or not problem.strip():
            return False

        trace_id = f"trace_{request_id}_{uuid.uuid4().hex[:8]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Embed the problem text
        embedding_blob = None
        vec = self._embedder.embed(problem)
        if vec is not None:
            embedding_blob = self._embedder.serialize_embedding(vec)

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO reasoning_traces
                       (trace_id, problem, problem_category, strategy_used,
                        full_trace, composite_reward, user_feedback,
                        problem_embedding, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        trace_id, problem, category, strategy,
                        full_trace, reward, user_feedback,
                        embedding_blob, now,
                    ),
                )
                conn.commit()
                self._trace_count_since_abstraction += 1
                stored = True
                logger.info(
                    f"[TemplateLib] Collected trace {trace_id} "
                    f"(reward={reward:.3f}, category={category})"
                )
            except Exception as e:
                logger.error(f"[TemplateLib] Trace collection error: {e}")
                stored = False
            finally:
                conn.close()

        # Check if we should run batch abstraction (outside lock to avoid deadlock)
        if stored:
            self._maybe_run_abstraction()

        return stored

    # ----------------------------------------------------------------
    # Template Retrieval
    # ----------------------------------------------------------------

    def retrieve_templates(
        self,
        problem: str,
        category: Optional[str] = None,
        granularity: Optional[str] = None,
        top_k: int = 3,
    ) -> List[TemplateMatch]:
        """Retrieve the top-K matching templates for a problem.

        Args:
            problem: The problem/query text.
            category: Optional problem category for bonus scoring.
            granularity: Optional filter — 'atomic', 'pattern', or 'meta'.
            top_k: Maximum number of matches to return.

        Returns:
            List of TemplateMatch objects, sorted by score descending.
        """
        if not self.enabled:
            return []

        # Embed the problem
        problem_vec = self._embedder.embed(problem)
        if problem_vec is None:
            return []

        # Load active templates
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                query = """SELECT template_id, name, description, abstract_steps,
                                  applicable_categories, source_trace_ids, embedding,
                                  times_used, avg_reward_when_used, avg_reward_baseline,
                                  status, created_at, last_used
                           FROM reasoning_templates
                           WHERE status = 'active' AND embedding IS NOT NULL"""
                params = []
                if granularity is not None:
                    query += " AND granularity = ?"
                    params.append(granularity)
                rows = conn.execute(query, params).fetchall()
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"[TemplateLib] Template retrieval DB error: {e}")
            return []

        if not rows:
            return []

        scored: List[tuple] = []  # (total_score, similarity, template)

        for row in rows:
            template = ReasoningTemplate(
                template_id=row[0], name=row[1], description=row[2],
                abstract_steps=row[3], applicable_categories=row[4],
                source_trace_ids=row[5], embedding=row[6],
                times_used=row[7], avg_reward_when_used=row[8],
                avg_reward_baseline=row[9], status=row[10],
                created_at=row[11], last_used=row[12],
            )

            # Compute cosine similarity
            template_vec = self._embedder.deserialize_embedding(template.embedding)
            similarity = self._embedder.cosine_similarity(problem_vec, template_vec)

            # Category bonus: +0.1 if category matches
            category_bonus = 0.0
            if category:
                try:
                    categories = json.loads(template.applicable_categories or "[]")
                    if category in categories:
                        category_bonus = 0.1
                except (json.JSONDecodeError, TypeError):
                    pass

            # Performance bonus: scale by how much better than baseline
            performance_bonus = 0.0
            if template.times_used > 5 and template.avg_reward_when_used > template.avg_reward_baseline:
                performance_bonus = min(0.1, (template.avg_reward_when_used - template.avg_reward_baseline) * 0.5)

            total_score = similarity + category_bonus + performance_bonus
            scored.append((total_score, similarity, template))

        # Sort descending by total_score, filter by similarity threshold, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for total_score, similarity, template in scored:
            if similarity < self.SIMILARITY_THRESHOLD:
                continue
            guidance_text = self._format_guidance(template)
            results.append(TemplateMatch(
                template=template,
                similarity_score=similarity,
                guidance_text=guidance_text,
            ))
            if len(results) >= top_k:
                break

        if results:
            logger.info(
                f"[TemplateLib] Retrieved {len(results)} template(s), "
                f"best: {results[0].template.name} (similarity={results[0].similarity_score:.3f})"
            )

        return results

    def retrieve_template(
        self,
        problem: str,
        category: Optional[str] = None,
        granularity: Optional[str] = None,
    ) -> Optional[TemplateMatch]:
        """Retrieve the best matching template for a problem.

        Delegates to retrieve_templates(top_k=1) for backward compatibility.

        Args:
            problem: The problem/query text.
            category: Optional problem category for bonus scoring.
            granularity: Optional filter — 'atomic', 'pattern', or 'meta'.

        Returns:
            TemplateMatch if a good match is found, None otherwise.
        """
        matches = self.retrieve_templates(problem, category=category, granularity=granularity, top_k=1)
        return matches[0] if matches else None

    # ----------------------------------------------------------------
    # Usage Recording
    # ----------------------------------------------------------------

    def record_template_usage(
        self,
        template_id: str,
        composite_reward: float,
    ):
        """Record that a template was used and its outcome reward.

        Updates running average and checks deprecation.

        Args:
            template_id: The template that was used.
            composite_reward: The composite reward of the interaction.
        """
        if not self.enabled:
            return

        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                # Get current stats
                row = conn.execute(
                    "SELECT times_used, avg_reward_when_used FROM reasoning_templates "
                    "WHERE template_id = ?",
                    (template_id,),
                ).fetchone()

                if row is None:
                    return

                old_count, old_avg = row
                new_count = old_count + 1
                # Running average update
                new_avg = old_avg + (composite_reward - old_avg) / new_count

                conn.execute(
                    """UPDATE reasoning_templates
                       SET times_used = ?, avg_reward_when_used = ?, last_used = ?
                       WHERE template_id = ?""",
                    (new_count, new_avg, now, template_id),
                )
                conn.commit()

                logger.debug(
                    f"[TemplateLib] Template {template_id} used "
                    f"(count={new_count}, avg_reward={new_avg:.3f})"
                )

                # Check deprecation
                self._check_deprecation(template_id)

            except Exception as e:
                logger.error(f"[TemplateLib] Usage recording error: {e}")
            finally:
                conn.close()

    # ----------------------------------------------------------------
    # Batch Abstraction
    # ----------------------------------------------------------------

    def _maybe_run_abstraction(self):
        """Trigger batch abstraction if trace count >= BATCH_SIZE."""
        if self._trace_count_since_abstraction >= self.BATCH_SIZE:
            self._trace_count_since_abstraction = 0
            thread = threading.Thread(
                target=self._run_abstraction_batch,
                daemon=True,
                name="TemplateAbstraction",
            )
            thread.start()
            logger.info("[TemplateLib] Triggered batch abstraction")

    def _run_abstraction_batch(self):
        """Group traces by category and abstract patterns into templates via LLM."""
        try:
            conn = sqlite3.connect(self._db_path)

            # Get unprocessed high-reward traces
            rows = conn.execute(
                """SELECT trace_id, problem, problem_category, strategy_used,
                          full_trace, composite_reward
                   FROM reasoning_traces
                   WHERE composite_reward >= ?
                   ORDER BY composite_reward DESC
                   LIMIT 200""",
                (self.REWARD_THRESHOLD,),
            ).fetchall()
            conn.close()

            if not rows:
                return

            # Group by category
            by_category: Dict[str, list] = {}
            for row in rows:
                cat = row[2] or "general"
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append({
                    "trace_id": row[0],
                    "problem": row[1],
                    "category": row[2],
                    "strategy": row[3],
                    "full_trace": row[4],
                    "reward": row[5],
                })

            # Abstract each category group
            for category, traces in by_category.items():
                if len(traces) < 3:
                    continue  # Need at least 3 traces to find a pattern
                self._abstract_category(category, traces)

        except Exception as e:
            logger.error(f"[TemplateLib] Abstraction batch error: {e}")

    # Cached brain instance to avoid per-batch instantiation overhead
    _shared_brain = None
    _shared_brain_lock = threading.Lock()

    @classmethod
    def _get_shared_brain(cls):
        """Get or create a shared OllamaBrain instance for template abstraction."""
        if cls._shared_brain is None:
            with cls._shared_brain_lock:
                if cls._shared_brain is None:
                    from aura.brain import OllamaBrain
                    cls._shared_brain = OllamaBrain(warmup=False)
        return cls._shared_brain

    def _abstract_category(self, category: str, traces: list):
        """Use LLM to abstract reasoning patterns from traces in a category."""
        try:

            # Build prompt with trace examples (limit to top 10)
            sample = traces[:10]
            examples = []
            trace_ids = []
            for t in sample:
                trace_ids.append(t["trace_id"])
                examples.append(
                    f"Problem: {t['problem'][:200]}\n"
                    f"Strategy: {t['strategy']}\n"
                    f"Reward: {t['reward']:.2f}\n"
                    f"Trace: {t['full_trace'][:500]}"
                )

            prompt = (
                "Analyze these successful reasoning traces and extract a reusable "
                "reasoning template pattern.\n\n"
                + "\n---\n".join(examples) +
                "\n\nRespond with ONLY a JSON object (no markdown, no explanation):\n"
                '{"name": "short template name",'
                ' "description": "when to use this pattern",'
                ' "abstract_steps": ["step 1 description", "step 2 description", ...],'
                ' "applicable_categories": ["category1", "category2"]}'
            )

            brain = self._get_shared_brain()
            result = brain.think(prompt, task_type=None)

            # Parse JSON from LLM response
            template_data = self._parse_template_json(result)
            if template_data is None:
                return

            # Validate required fields
            name = template_data.get("name", "")
            description = template_data.get("description", "")
            abstract_steps = template_data.get("abstract_steps", [])
            applicable_categories = template_data.get("applicable_categories", [category])

            if not name or not abstract_steps:
                logger.warning("[TemplateLib] Abstraction produced incomplete template")
                return

            # Embed the template description for later similarity search
            embed_text = f"{name}: {description}"
            vec = self._embedder.embed(embed_text)
            embedding_blob = None
            if vec is not None:
                embedding_blob = self._embedder.serialize_embedding(vec)

                # Deduplication: check similarity against existing templates
                if self._is_duplicate(vec):
                    logger.info(f"[TemplateLib] Skipping duplicate template: {name}")
                    return

            # Compute baseline reward from source traces
            avg_baseline = sum(t["reward"] for t in sample) / len(sample)

            template_id = f"tmpl_{uuid.uuid4().hex[:12]}"
            now = time.strftime("%Y-%m-%dT%H:%M:%S")

            with self._lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO reasoning_templates
                           (template_id, name, description, abstract_steps,
                            applicable_categories, source_trace_ids, embedding,
                            avg_reward_baseline, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            template_id, name, description,
                            json.dumps(abstract_steps),
                            json.dumps(applicable_categories),
                            json.dumps(trace_ids),
                            embedding_blob,
                            avg_baseline, now,
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

            logger.info(
                f"[TemplateLib] Created template: {name} ({template_id}) "
                f"from {len(trace_ids)} traces in category={category}"
            )

        except ImportError:
            logger.warning("[TemplateLib] Brain not available for abstraction")
        except Exception as e:
            logger.error(f"[TemplateLib] Category abstraction error: {e}")

    def _parse_template_json(self, text: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        from aura.core.json_utils import parse_llm_json

        result = parse_llm_json(text)
        if result is None:
            logger.warning("[TemplateLib] Failed to parse template JSON from LLM response")
        return result if isinstance(result, dict) else None

    def _is_duplicate(self, new_vec: List[float]) -> bool:
        """Check if a template with similar embedding already exists."""
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT embedding FROM reasoning_templates "
                "WHERE status = 'active' AND embedding IS NOT NULL"
            ).fetchall()
            conn.close()

            for (blob,) in rows:
                existing_vec = self._embedder.deserialize_embedding(blob)
                sim = self._embedder.cosine_similarity(new_vec, existing_vec)
                if sim > self.DEDUP_THRESHOLD:
                    return True
            return False
        except Exception as e:
            logger.debug(f"[TemplateLib] dedup check failed: {e}")
            return False

    # ----------------------------------------------------------------
    # Deprecation
    # ----------------------------------------------------------------

    def _check_deprecation(self, template_id: str):
        """Deprecate a template if its performance falls below baseline."""
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                row = conn.execute(
                    "SELECT times_used, avg_reward_when_used, avg_reward_baseline "
                    "FROM reasoning_templates WHERE template_id = ?",
                    (template_id,),
                ).fetchone()

                if row is None:
                    return

                times_used, avg_reward, baseline = row

                if times_used >= self.DEPRECATION_MIN_USES:
                    if avg_reward <= baseline + self.DEPRECATION_REWARD_GAP:
                        conn.execute(
                            "UPDATE reasoning_templates SET status = 'deprecated' "
                            "WHERE template_id = ?",
                            (template_id,),
                        )
                        conn.commit()
                        logger.info(
                            f"[TemplateLib] Deprecated template {template_id} "
                            f"(avg_reward={avg_reward:.3f} <= baseline={baseline:.3f})"
                        )
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"[TemplateLib] Deprecation check error: {e}")

    # ----------------------------------------------------------------
    # Guidance Formatting
    # ----------------------------------------------------------------

    @staticmethod
    def _format_guidance_multi(matches: List[TemplateMatch]) -> str:
        """Format multiple template matches into combined guidance text.

        - 0 matches: returns ""
        - 1 match: returns its guidance_text directly (no ranking header)
        - 2+ matches: wraps each with ranking markers
        """
        if not matches:
            return ""
        if len(matches) == 1:
            return matches[0].guidance_text
        parts = []
        for i, m in enumerate(matches, 1):
            parts.append(
                f"--- Template #{i} (similarity={m.similarity_score:.2f}) ---\n"
                f"{m.guidance_text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_guidance(template: ReasoningTemplate) -> str:
        """Format a template into guidance text for system prompt injection."""
        try:
            steps = json.loads(template.abstract_steps)
        except (json.JSONDecodeError, TypeError):
            steps = []

        steps_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(steps))

        return (
            f"[Reasoning Template: {template.name}]\n"
            f"{template.description}\n"
            f"Recommended approach:\n"
            f"{steps_text}"
        )

    # ----------------------------------------------------------------
    # Stats / Monitoring
    # ----------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Get monitoring/debugging summary."""
        try:
            conn = sqlite3.connect(self._db_path)

            trace_count = conn.execute(
                "SELECT COUNT(*) FROM reasoning_traces"
            ).fetchone()[0]

            template_count = conn.execute(
                "SELECT COUNT(*) FROM reasoning_templates"
            ).fetchone()[0]

            active_templates = conn.execute(
                "SELECT COUNT(*) FROM reasoning_templates WHERE status = 'active'"
            ).fetchone()[0]

            deprecated_templates = conn.execute(
                "SELECT COUNT(*) FROM reasoning_templates WHERE status = 'deprecated'"
            ).fetchone()[0]

            total_template_uses = conn.execute(
                "SELECT COALESCE(SUM(times_used), 0) FROM reasoning_templates"
            ).fetchone()[0]

            avg_reward = conn.execute(
                "SELECT COALESCE(AVG(composite_reward), 0) FROM reasoning_traces"
            ).fetchone()[0]

            # Category breakdown of traces
            category_counts = conn.execute(
                "SELECT problem_category, COUNT(*) FROM reasoning_traces "
                "GROUP BY problem_category"
            ).fetchall()

            conn.close()

            return {
                "enabled": self.enabled,
                "trace_count": trace_count,
                "template_count": template_count,
                "active_templates": active_templates,
                "deprecated_templates": deprecated_templates,
                "total_template_uses": total_template_uses,
                "avg_trace_reward": round(avg_reward, 3),
                "traces_since_abstraction": self._trace_count_since_abstraction,
                "category_trace_counts": dict(category_counts),
            }
        except Exception as e:
            logger.error(f"[TemplateLib] Stats error: {e}")
            return {"enabled": self.enabled, "error": str(e)}


# ============================================================================
# Singleton
# ============================================================================

_template_library: Optional[ReasoningTemplateLibrary] = None
_singleton_lock = threading.Lock()


def get_template_library() -> ReasoningTemplateLibrary:
    """Get or create the singleton ReasoningTemplateLibrary."""
    global _template_library
    if _template_library is None:
        with _singleton_lock:
            if _template_library is None:
                from aura.config import Config
                _template_library = ReasoningTemplateLibrary(
                    enabled=getattr(Config, "REASONING_TEMPLATES_ENABLED", True),
                )
    return _template_library
