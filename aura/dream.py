"""Dream Mode - Memory consolidation and pattern analysis.

Analyzes metacognition logs to extract insights about agent behavior,
tool effectiveness, and learning opportunities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .brain import OllamaBrain
from .memory import MemorySystem
from .metacognition import MetacognitionLogger

logger = logging.getLogger(__name__)

class DreamMode:
    """Consolidates memories and extracts insights from agent experiences."""

    def __init__(self):
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
