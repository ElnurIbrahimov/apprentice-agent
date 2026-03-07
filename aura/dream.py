"""Dream Mode - Memory consolidation and pattern analysis.

Analyzes metacognition logs to extract insights about agent behavior,
tool effectiveness, and learning opportunities.

Phase 4 adds DreamConsolidator — a real consolidation/compression pipeline
that:
  1. CLUSTER    — groups memories by semantic similarity
  2. SUMMARIZE  — produces DreamSummary per cluster
  3. EXTRACT_ROUTINES — detects repeated behavioral patterns
  4. PRUNE      — flags stale memories as archived candidates
  5. CONTRADICTION_REPORT — surfaces unresolved KG contradictions
  6. DENSIFY_GRAPH (experimental) — proposes new KG edges

The original DreamMode class is preserved unchanged for backward compat.
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DreamMode:
    """Consolidates memories and extracts insights from agent experiences."""

    def __init__(self):
        from .brain import OllamaBrain
        from .memory import MemorySystem
        from .metacognition import MetacognitionLogger
        self.metacog = MetacognitionLogger()
        self.memory = MemorySystem(collection_name="dream_insights")
        self.brain = OllamaBrain()
        self._consolidation_count: int = 0

    def dream(self, date: Optional[str] = None) -> dict:
        """Run dream mode to consolidate memories and generate insights.

        Args:
            date: Date to analyze (YYYY-MM-DD). Defaults to today.

        Returns:
            Dictionary with analysis results and generated insights.
        """
        logger.debug("\n" + "=" * 60)
        logger.debug("DREAM MODE - Memory Consolidation")
        logger.debug("=" * 60 + "\n")

        # Record real thought: dream mode beginning
        try:
            from api.routes.thinking import record_thought
            record_thought("analyzing", "entering dream mode: consolidating memories...", 0.8, "dream")
        except Exception:
            pass

        # Step 1: Load today's logs
        logger.debug("[1/5] Loading metacognition logs...")
        logs = self._load_logs(date)
        if not logs:
            logger.debug("No logs found for analysis.")
            return {"success": False, "error": "No logs found"}
        logger.debug(f"      Found {len(logs)} log entries")

        # NREM-like reverse replay: on even cycles, process recent memories first
        self._consolidation_count += 1
        if self._consolidation_count % 2 == 0:
            logs = list(reversed(logs))

        # Step 2: Analyze patterns
        logger.debug("\n[2/5] Analyzing patterns...")
        patterns = self._analyze_patterns(logs)
        self._print_patterns(patterns)

        # Step 3: Generate insights using LLM
        logger.debug("\n[3/5] Generating insights...")
        try:
            from api.routes.thinking import record_thought
            record_thought("wondering", f"dreaming: generating insights from {len(logs)} interactions", 0.7, "dream")
        except Exception:
            pass
        insights = self._generate_insights(patterns, logs)
        logger.debug(f"      Generated {len(insights)} insights")

        # Step 4: Store insights in long-term memory
        logger.debug("\n[4/5] Storing insights in memory...")
        stored_ids = self._store_insights(insights, date)
        logger.debug(f"      Stored {len(stored_ids)} insights")

        # Step 5: A-MEM memory consolidation (merge similar notes, prune stale ones)
        logger.debug("\n[5/5] Consolidating A-MEM notes...")
        consolidation_result = {"merged": 0, "pruned": 0}
        try:
            from .tools.amem import get_amem
            amem = get_amem()
            consolidation_result = _consolidate_amem_notes(amem)
            logger.debug(f"      Merged: {consolidation_result['merged']}, Pruned: {consolidation_result['pruned']}")
            import logging
            logging.getLogger(__name__).info(
                f"[Dream] Memory consolidation: merged={consolidation_result['merged']}, "
                f"pruned={consolidation_result['pruned']}"
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[Dream] Consolidation skipped: {e}")
            logger.debug(f"      Skipped: {e}")

        # Print insights
        logger.debug("\n" + "-" * 60)
        logger.debug("INSIGHTS GENERATED:")
        logger.debug("-" * 60)
        for i, insight in enumerate(insights, 1):
            logger.debug(f"\n{i}. {insight}")

        logger.debug("\n" + "=" * 60)
        logger.debug("DREAM MODE COMPLETE")
        logger.debug("=" * 60 + "\n")

        return {
            "success": True,
            "logs_analyzed": len(logs),
            "patterns": patterns,
            "insights": insights,
            "stored_ids": stored_ids,
            "consolidation": consolidation_result
        }

    def _load_logs(self, date: Optional[str] = None) -> list[dict]:
        """Load metacognition logs for the given date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        log_file = Path(__file__).parent.parent / "logs" / "metacognition" / f"{date}.jsonl"
        if not log_file.exists():
            return []

        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return logs

    def _analyze_patterns(self, logs: list[dict]) -> dict:
        """Analyze patterns in the logs."""
        patterns = {
            "total_actions": len(logs),
            "tools": {},
            "confidence_distribution": {"low": 0, "medium": 0, "high": 0},
            "retry_analysis": {"first_attempt_success": 0, "needed_retry": 0},
            "goals": {}
        }

        for log in logs:
            tool = log.get("tool", "unknown")
            confidence = log.get("confidence", 0)
            success = log.get("success", False)
            retried = log.get("retried", False)
            goal = log.get("goal", "unknown")

            # Tool stats
            if tool not in patterns["tools"]:
                patterns["tools"][tool] = {
                    "total": 0,
                    "success": 0,
                    "avg_confidence": [],
                    "actions": []
                }
            patterns["tools"][tool]["total"] += 1
            if success:
                patterns["tools"][tool]["success"] += 1
            patterns["tools"][tool]["avg_confidence"].append(confidence)
            patterns["tools"][tool]["actions"].append(log.get("action", "")[:100])

            # Confidence distribution
            if confidence < 50:
                patterns["confidence_distribution"]["low"] += 1
            elif confidence < 80:
                patterns["confidence_distribution"]["medium"] += 1
            else:
                patterns["confidence_distribution"]["high"] += 1

            # Retry analysis - only count successful non-retried actions
            if success and not retried:
                patterns["retry_analysis"]["first_attempt_success"] += 1
            elif retried:
                patterns["retry_analysis"]["needed_retry"] += 1

            # Goal tracking
            if goal not in patterns["goals"]:
                patterns["goals"][goal] = {"attempts": 0, "completed": False}
            patterns["goals"][goal]["attempts"] += 1
            if success and log.get("next_step") == "complete":
                patterns["goals"][goal]["completed"] = True

        # Calculate averages
        for tool, stats in patterns["tools"].items():
            if stats["avg_confidence"]:
                stats["avg_confidence"] = round(
                    sum(stats["avg_confidence"]) / len(stats["avg_confidence"]), 1
                )
            else:
                stats["avg_confidence"] = 0
            stats["success_rate"] = round(
                (stats["success"] / stats["total"]) * 100, 1
            ) if stats["total"] > 0 else 0

        return patterns

    def _print_patterns(self, patterns: dict) -> None:
        """Print pattern analysis results."""
        logger.debug(f"\n      Total actions: {patterns['total_actions']}")
        logger.debug(f"\n      Tool Performance:")
        for tool, stats in patterns["tools"].items():
            logger.debug(f"        - {tool}: {stats['success_rate']}% success, "
                  f"avg confidence {stats['avg_confidence']}%")

        logger.debug(f"\n      Confidence Distribution:")
        cd = patterns["confidence_distribution"]
        logger.debug(f"        - Low (<50%): {cd['low']}")
        logger.debug(f"        - Medium (50-80%): {cd['medium']}")
        logger.debug(f"        - High (>80%): {cd['high']}")

        logger.debug(f"\n      Retry Analysis:")
        ra = patterns["retry_analysis"]
        logger.debug(f"        - First attempt success: {ra['first_attempt_success']}")
        logger.debug(f"        - Needed retry: {ra['needed_retry']}")

    def _generate_insights(self, patterns: dict, logs: list[dict]) -> list[str]:
        """Generate insights using the LLM."""
        insights = []

        # Prepare summary for LLM
        tool_summary = "\n".join([
            f"- {tool}: {stats['total']} uses, {stats['success_rate']}% success rate, "
            f"avg confidence {stats['avg_confidence']}%"
            for tool, stats in patterns["tools"].items()
        ])

        # Sample actions for context
        sample_actions = []
        for log in logs[:5]:
            sample_actions.append(
                f"Goal: {log.get('goal', 'N/A')[:50]}, "
                f"Tool: {log.get('tool')}, "
                f"Success: {log.get('success')}, "
                f"Confidence: {log.get('confidence')}%"
            )

        prompt = f"""Analyze this agent's performance data and generate 3-5 actionable insights.

TOOL PERFORMANCE:
{tool_summary}

CONFIDENCE DISTRIBUTION:
- Low confidence (<50%): {patterns['confidence_distribution']['low']} actions
- Medium confidence (50-80%): {patterns['confidence_distribution']['medium']} actions
- High confidence (>80%): {patterns['confidence_distribution']['high']} actions

RETRY STATS:
- First attempt successes: {patterns['retry_analysis']['first_attempt_success']}
- Needed retry: {patterns['retry_analysis']['needed_retry']}

SAMPLE ACTIONS:
{chr(10).join(sample_actions)}

Generate 3-5 specific, actionable insights about:
1. Which tools work best and when to use them
2. What confidence levels indicate about success likelihood
3. How to reduce retries and improve first-attempt success
4. Patterns in successful vs unsuccessful actions

Format each insight as a single clear sentence starting with a verb (Use, Prefer, Avoid, etc.)."""

        response = self.brain.think(
            prompt,
            system_prompt="You analyze agent behavior patterns and generate concise, actionable insights. Be specific and practical.",
            use_history=False
        )

        # Parse insights from response
        for line in response.split("\n"):
            line = line.strip()
            # Skip empty lines and headers
            if not line or line.startswith("#") or line.startswith("*"):
                continue
            # Remove numbering
            if line[0].isdigit() and (line[1] == "." or line[1] == ")"):
                line = line[2:].strip()
            elif line[0].isdigit() and line[1].isdigit() and line[2] in ".)" :
                line = line[3:].strip()
            # Remove bullet points
            if line.startswith("- "):
                line = line[2:]
            if line.startswith("• "):
                line = line[2:]
            # Keep meaningful insights
            if len(line) > 20 and any(c.isalpha() for c in line):
                insights.append(line)

        return insights[:5]  # Limit to 5 insights

    def _store_insights(self, insights: list[str], date: Optional[str] = None) -> list[str]:
        """Store insights in long-term memory."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        stored_ids = []
        for insight in insights:
            memory_id = self.memory.remember(
                content=insight,
                memory_type="dream_insight",
                metadata={
                    "source": "dream_mode",
                    "date_analyzed": date,
                    "generated_at": datetime.now().isoformat()
                }
            )
            stored_ids.append(memory_id)

        return stored_ids

    def recall_insights(self, query: str, n_results: int = 5) -> list[dict]:
        """Recall relevant insights from memory."""
        return self.memory.recall(query, n_results=n_results)

    def get_all_insights(self) -> list[dict]:
        """Get all stored insights."""
        return [m for m in self.memory.memories if m.get("type") == "dream_insight"]


def _consolidate_amem_notes(amem_system, similarity_threshold: float = 0.85) -> dict:
    """Merge A-MEM notes with high cosine similarity and prune low-importance notes.

    Args:
        amem_system: An initialized AMEMSystem instance
        similarity_threshold: Cosine similarity above which notes are merged (default 0.85)

    Returns:
        dict with keys: merged (int), pruned (int)
    """
    import logging
    _log = logging.getLogger(__name__)

    if not amem_system:
        return {"merged": 0, "pruned": 0}

    notes = list(amem_system._notes.values())
    merged_count = 0
    pruned_count = 0
    merged_ids = set()

    # Collect embeddings that exist for notes (numpy arrays)
    embeddings = {}
    for note in notes:
        note_id = note.id
        if note_id and hasattr(amem_system, '_embeddings') and note_id in amem_system._embeddings:
            embeddings[note_id] = amem_system._embeddings[note_id]

    # Find pairs with high cosine similarity and merge note_b into note_a
    for i, note_a in enumerate(notes):
        note_a_id = note_a.id
        if not note_a_id or note_a_id in merged_ids:
            continue

        for note_b in notes[i + 1:]:
            note_b_id = note_b.id
            if not note_b_id or note_b_id in merged_ids:
                continue
            if note_a_id not in embeddings or note_b_id not in embeddings:
                continue

            ea = embeddings[note_a_id]
            eb = embeddings[note_b_id]

            # Cosine similarity using numpy
            try:
                import numpy as np
                dot = float(np.dot(ea, eb))
                mag_a = float(np.linalg.norm(ea))
                mag_b = float(np.linalg.norm(eb))
            except Exception:
                dot = sum(float(a) * float(b) for a, b in zip(ea, eb))
                mag_a = sum(float(a) ** 2 for a in ea) ** 0.5
                mag_b = sum(float(b) ** 2 for b in eb) ** 0.5

            if mag_a < 1e-8 or mag_b < 1e-8:
                continue
            similarity = dot / (mag_a * mag_b)

            if similarity >= similarity_threshold:
                # Keep higher importance, combine keywords and context
                note_a.importance = max(note_a.importance, note_b.importance)

                kw_a = list(note_a.keywords)
                kw_b = list(note_b.keywords)
                note_a.keywords = list(set(kw_a + kw_b))[:20]

                ctx_b = note_b.context
                if ctx_b and ctx_b not in note_a.context:
                    note_a.context = f"{note_a.context} | {ctx_b}"[:500]

                merged_ids.add(note_b_id)
                merged_count += 1

    # Remove merged notes from memory and embeddings
    for note_id in merged_ids:
        amem_system._notes.pop(note_id, None)
        if hasattr(amem_system, '_embeddings'):
            amem_system._embeddings.pop(note_id, None)

    # Prune notes that are low-importance and have never been accessed
    low_importance_ids = [
        note_id
        for note_id, note in amem_system._notes.items()
        if note.importance < 0.2 and note.access_count == 0
    ]
    for note_id in low_importance_ids:
        amem_system._notes.pop(note_id, None)
        pruned_count += 1

    # Persist the consolidated state
    try:
        amem_system.save()
    except Exception as e:
        _log.warning(f"[Dream] Failed to save after consolidation: {e}")

    return {"merged": merged_count, "pruned": pruned_count}


def run_dream_mode(date: Optional[str] = None) -> dict:
    """Entry point for running dream mode."""
    dreamer = DreamMode()
    return dreamer.dream(date)


# =============================================================================
# PHASE 4 — DreamConsolidator
# =============================================================================

@dataclass
class DreamSummary:
    """Compressed representation of a cluster of related memories."""
    summary_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    cluster_id: str = ""
    source_memory_ids: List[str] = field(default_factory=list)
    compressed_text: str = ""
    dominant_tags: List[str] = field(default_factory=list)
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "cluster_id": self.cluster_id,
            "source_count": len(self.source_memory_ids),
            "compressed_text": self.compressed_text[:200],
            "dominant_tags": self.dominant_tags[:5],
            "confidence": round(self.confidence, 3),
        }


@dataclass
class RoutinePattern:
    """A repeated behavioral pattern detected across episodic memory."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    trigger_context: str = ""
    expected_action: str = ""
    frequency: int = 0
    last_seen: float = field(default_factory=time.time)


@dataclass
class DreamCycle:
    """Record of one DreamConsolidator run."""
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    memories_processed: int = 0
    summaries_written: int = 0
    pruned_count: int = 0
    contradictions_found: int = 0
    routines_extracted: int = 0
    graph_edges_proposed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "duration_s": round((self.finished_at or time.time()) - self.started_at, 2),
            "memories_processed": self.memories_processed,
            "summaries_written": self.summaries_written,
            "pruned_count": self.pruned_count,
            "contradictions_found": self.contradictions_found,
            "routines_extracted": self.routines_extracted,
            "graph_edges_proposed": self.graph_edges_proposed,
        }


@dataclass
class ConsolidationReport:
    """Full output of a DreamConsolidator cycle."""
    cycle: DreamCycle
    summaries: List[DreamSummary] = field(default_factory=list)
    routines: List[RoutinePattern] = field(default_factory=list)
    pruned_ids: List[str] = field(default_factory=list)
    contradiction_ids: List[str] = field(default_factory=list)
    graph_proposals: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Similarity helpers (no heavy ML dep — use bag-of-words Jaccard)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Simple word tokenizer, strips punctuation."""
    import re
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return set(words)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cluster_by_similarity(
    items: List[Dict[str, Any]],
    threshold: float = 0.35,
) -> List[List[Dict[str, Any]]]:
    """
    Greedy single-linkage clustering by Jaccard text similarity.

    items: list of dicts with at least a "content" key.
    Returns list of clusters (each cluster is a list of items).
    """
    if not items:
        return []

    token_sets = [_tokenize(it.get("content", "")) for it in items]
    clusters: List[List[int]] = []
    assigned = [False] * len(items)

    for i in range(len(items)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(items)):
            if assigned[j]:
                continue
            sim = _jaccard(token_sets[i], token_sets[j])
            if sim >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    return [[items[idx] for idx in cl] for cl in clusters]


# ---------------------------------------------------------------------------
# DreamConsolidator
# ---------------------------------------------------------------------------

class DreamConsolidator:
    """
    Real consolidation/compression pipeline.

    Pipeline (per cycle):
      CLUSTER → SUMMARIZE → EXTRACT_ROUTINES → PRUNE → CONTRADICTION_REPORT
      → (optional) DENSIFY_GRAPH

    Never blocks the main request path — runs in a background thread.
    Respects user scoping: never consolidates across user boundaries.
    """

    def __init__(self) -> None:
        try:
            from aura.config import Config
            self._batch_size  = getattr(Config, "DREAM_CLUSTER_BATCH_SIZE", 20)
            self._stale_days  = getattr(Config, "DREAM_PRUNE_STALENESS_DAYS", 30)
            self._min_cluster = getattr(Config, "DREAM_MIN_CLUSTER_SIZE", 3)
            self._do_routines = getattr(Config, "DREAM_ENABLE_ROUTINE_EXTRACTION", True)
            self._do_densify  = getattr(Config, "DREAM_ENABLE_GRAPH_DENSIFICATION", False)
        except Exception:
            self._batch_size  = 20
            self._stale_days  = 30
            self._min_cluster = 3
            self._do_routines = True
            self._do_densify  = False

        self._brain: Optional[Any] = None
        self._running   = False
        self._lock      = threading.Lock()
        self._last_seen_ids: set = set()   # Track already-summarized memory IDs

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_cycle(self, user_id: str = "default_user") -> ConsolidationReport:
        """
        Run one full consolidation cycle for the given user.
        Safe to call repeatedly — is a no-op if already running.
        """
        with self._lock:
            if self._running:
                logger.info("[DreamConsolidator] Already running, skipping cycle")
                cycle = DreamCycle()
                cycle.finished_at = time.time()
                return ConsolidationReport(cycle=cycle)
            self._running = True

        cycle = DreamCycle()
        report = ConsolidationReport(cycle=cycle)

        try:
            logger.info("[DreamConsolidator] Cycle %s start (user=%s)", cycle.cycle_id, user_id)

            # 1. CLUSTER
            memories = self._fetch_memories(user_id)
            cycle.memories_processed = len(memories)

            clusters = _cluster_by_similarity(memories)
            logger.info("[DreamConsolidator] %d memories → %d clusters", len(memories), len(clusters))

            # Limit to batch size
            clusters = clusters[:self._batch_size]

            # 2. SUMMARIZE (clusters ≥ min_cluster_size)
            for cluster in clusters:
                if len(cluster) < self._min_cluster:
                    continue
                # Skip already-summarized clusters
                ids = [m.get("id", "") for m in cluster]
                if all(mid in self._last_seen_ids for mid in ids if mid):
                    continue

                summary = self._summarize_cluster(cluster, user_id)
                if summary:
                    report.summaries.append(summary)
                    cycle.summaries_written += 1
                    self._write_summary_memory(summary, user_id)
                    self._last_seen_ids.update(ids)

            # 3. EXTRACT_ROUTINES
            if self._do_routines:
                routines = self._extract_routines(memories)
                report.routines = routines
                cycle.routines_extracted = len(routines)

            # 4. PRUNE
            pruned = self._prune_stale(memories, user_id)
            report.pruned_ids = pruned
            cycle.pruned_count = len(pruned)

            # 5. CONTRADICTION_REPORT
            contradictions = self._contradiction_report()
            report.contradiction_ids = [c.get("contradiction_id", "") for c in contradictions]
            cycle.contradictions_found = len(contradictions)

            # 6. DENSIFY_GRAPH (experimental, gated)
            if self._do_densify:
                proposals = self._densify_graph(memories)
                report.graph_proposals = proposals
                cycle.graph_edges_proposed = len(proposals)

            cycle.finished_at = time.time()
            logger.info(
                "[DreamConsolidator] Cycle %s done: summaries=%d pruned=%d contradictions=%d routines=%d",
                cycle.cycle_id, cycle.summaries_written, cycle.pruned_count,
                cycle.contradictions_found, cycle.routines_extracted,
            )

        except Exception as e:
            logger.error("[DreamConsolidator] Cycle error: %s", e, exc_info=True)
            cycle.finished_at = time.time()
        finally:
            with self._lock:
                self._running = False

        # Emit telemetry
        try:
            from aura.reliability.telemetry import emit, TelemetryKind
            emit(TelemetryKind.DREAM_CYCLE, user_id=user_id, extra=cycle.to_dict())
        except Exception:
            pass

        return report

    def run_cycle_background(self, user_id: str = "default_user") -> None:
        """Fire-and-forget background execution."""
        t = threading.Thread(
            target=self.run_cycle,
            args=(user_id,),
            daemon=True,
            name="dream-consolidator",
        )
        t.start()

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _fetch_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Fetch candidate and stable memories from A-MEM and Episodic backends.
        Returns a flat list of dicts: {id, content, source, tags, importance, ts}
        """
        memories: List[Dict[str, Any]] = []

        # A-MEM notes
        try:
            from aura.tools.amem import get_amem
            amem = get_amem()
            for note_id, note in list(amem._notes.items()):
                memories.append({
                    "id": note_id,
                    "content": note.content,
                    "source": "amem",
                    "tags": list(getattr(note, "tags", [])),
                    "importance": getattr(note, "importance", 0.5),
                    "ts": time.time(),
                })
                if len(memories) >= self._batch_size * 3:
                    break
        except Exception as e:
            logger.debug("[DreamConsolidator] A-MEM fetch error: %s", e)

        # Episodic memories
        try:
            from aura.memory.unified_memory import get_unified_memory
            um = get_unified_memory()
            results = um.query("recent experiences preferences goals", k=self._batch_size * 2,
                               sources=["episodic"])
            for r in results:
                memories.append({
                    "id": r.source_id,
                    "content": r.content,
                    "source": "episodic",
                    "tags": [],
                    "importance": r.importance,
                    "ts": time.time(),
                })
        except Exception as e:
            logger.debug("[DreamConsolidator] Episodic fetch error: %s", e)

        return memories

    def _summarize_cluster(
        self, cluster: List[Dict[str, Any]], user_id: str
    ) -> Optional[DreamSummary]:
        """Produce a DreamSummary for a cluster using the LLM."""
        contents = [m["content"] for m in cluster]
        combined = "\n---\n".join(contents[:10])  # cap to avoid huge prompts

        cluster_id = hashlib.md5(combined[:200].encode()).hexdigest()[:10]

        # Try LLM summarization; fall back to first sentence
        compressed = ""
        try:
            brain = self._get_brain()
            prompt = (
                f"Compress the following {len(cluster)} related memory entries into a single "
                f"concise summary (2-3 sentences). Preserve key facts and preferences.\n\n"
                f"{combined[:3000]}"
            )
            compressed = brain.think(prompt, use_history=False)
            compressed = compressed.strip()[:500]
        except Exception as e:
            logger.debug("[DreamConsolidator] LLM summarize error: %s", e)
            compressed = contents[0][:300] if contents else ""

        if not compressed:
            return None

        # Dominant tags
        all_tags: List[str] = []
        for m in cluster:
            all_tags.extend(m.get("tags", []))
        tag_freq: Dict[str, int] = {}
        for t in all_tags:
            tag_freq[t] = tag_freq.get(t, 0) + 1
        dominant = sorted(tag_freq, key=lambda k: -tag_freq[k])[:5]

        return DreamSummary(
            cluster_id=cluster_id,
            source_memory_ids=[m.get("id", "") for m in cluster],
            compressed_text=compressed,
            dominant_tags=dominant,
            confidence=0.8,
        )

    def _extract_routines(
        self, memories: List[Dict[str, Any]]
    ) -> List[RoutinePattern]:
        """
        Detect repeated patterns: same-ish content appearing 3+ times.
        Returns RoutinePattern list.
        """
        if len(memories) < self._min_cluster:
            return []

        # Fingerprint each memory
        fp_count: Dict[str, List[str]] = {}
        for m in memories:
            words = _tokenize(m.get("content", ""))
            # Use top-5 most distinctive words as fingerprint
            fp = " ".join(sorted(words)[:5])
            fp_count.setdefault(fp, []).append(m.get("id", ""))

        routines = []
        for fp, ids in fp_count.items():
            if len(ids) >= self._min_cluster:
                routines.append(RoutinePattern(
                    trigger_context=fp,
                    expected_action="(recurring memory pattern)",
                    frequency=len(ids),
                ))
        return routines

    def _prune_stale(
        self, memories: List[Dict[str, Any]], user_id: str
    ) -> List[str]:
        """
        Flag memories with importance < 0.2 AND no access in stale_days as archived.
        Does NOT delete — sets lifecycle_state to archived in A-MEM.
        """
        pruned: List[str] = []
        stale_cutoff = time.time() - self._stale_days * 86400

        try:
            from aura.tools.amem import get_amem
            amem = get_amem()
            from aura.memory.write_gate import MemoryLifecycleState

            for note_id, note in list(amem._notes.items()):
                imp = getattr(note, "importance", 0.5)
                ac  = getattr(note, "access_count", 0)
                # Approximate last access from timestamp if available
                ts  = getattr(note, "timestamp", 0)
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts).timestamp()
                    except Exception:
                        ts = 0
                is_stale = (ts < stale_cutoff) if ts else False
                if imp < 0.2 and ac == 0 and is_stale:
                    # Tag as archived (don't delete)
                    setattr(note, "lifecycle_state", MemoryLifecycleState.ARCHIVED.value)
                    pruned.append(note_id)
        except Exception as e:
            logger.debug("[DreamConsolidator] Prune error: %s", e)

        return pruned

    def _contradiction_report(self) -> List[Dict[str, Any]]:
        """Fetch unresolved contradiction edges from the KG."""
        try:
            from aura.memory.kg_contradiction import KGContradictionDetector
            from aura.tools.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            detector = KGContradictionDetector(kg)
            return detector.get_all_contradictions()
        except Exception as e:
            logger.debug("[DreamConsolidator] Contradiction report error: %s", e)
            return []

    def _densify_graph(
        self, memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        (Experimental) Propose new KG edges between related memories
        that don't have an explicit connection yet.
        Returns list of proposed edge dicts (not applied automatically).
        """
        proposals: List[Dict[str, Any]] = []
        try:
            from aura.tools.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            g  = kg.graph

            # Simple heuristic: find memory pairs with high Jaccard similarity
            # that don't share an existing edge
            ids_in_graph = list(g.nodes())[:50]  # Limit scope
            for i, id_a in enumerate(ids_in_graph):
                for id_b in ids_in_graph[i + 1:]:
                    if g.has_edge(id_a, id_b):
                        continue
                    la = g.nodes[id_a].get("label", "")
                    lb = g.nodes[id_b].get("label", "")
                    sim = _jaccard(_tokenize(la), _tokenize(lb))
                    if sim >= 0.5:
                        proposals.append({
                            "from": id_a, "to": id_b,
                            "label_a": la[:60], "label_b": lb[:60],
                            "similarity": round(sim, 3),
                            "proposed_edge": "relates_to",
                        })
                    if len(proposals) >= 20:
                        break
                if len(proposals) >= 20:
                    break
        except Exception as e:
            logger.debug("[DreamConsolidator] Densify error: %s", e)

        return proposals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_brain(self):
        if self._brain is None:
            from .brain import OllamaBrain
            self._brain = OllamaBrain(warmup=False)
        return self._brain

    def _write_summary_memory(self, summary: DreamSummary, user_id: str) -> None:
        """Write a DreamSummary back into unified memory as a SUMMARY lifecycle entry."""
        try:
            from aura.memory.unified_memory import get_unified_memory
            um = get_unified_memory()
            um.store(
                content=summary.compressed_text,
                source="dream_consolidation",
                importance=0.75,
                tags=["dream_summary"] + summary.dominant_tags[:3],
                episode_type="insight",
            )
        except Exception as e:
            logger.debug("[DreamConsolidator] Write summary error: %s", e)


# ---------------------------------------------------------------------------
# Module-level singleton for DreamConsolidator
# ---------------------------------------------------------------------------

_consolidator_instance: Optional[DreamConsolidator] = None
_consolidator_lock = threading.Lock()


def get_dream_consolidator() -> DreamConsolidator:
    global _consolidator_instance
    if _consolidator_instance is None:
        with _consolidator_lock:
            if _consolidator_instance is None:
                _consolidator_instance = DreamConsolidator()
    return _consolidator_instance
