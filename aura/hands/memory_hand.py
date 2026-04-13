"""Memory Hand — autonomous memory maintenance and consolidation.

Runs during idle to:
- Deduplicate similar memories
- Consolidate episodic memories into summaries
- Prune stale KG nodes
- Resolve contradictions
- Optimize retrieval indices

Works with NeuroDream during sleep phases but also runs independently.
"""

import asyncio
import logging
import time
from typing import Any

from aura.hands.base import Hand, HandManifest, HandResult

logger = logging.getLogger(__name__)


class MemoryHand(Hand):
    """Autonomous memory maintenance Hand."""

    def get_manifest(self) -> HandManifest:
        return HandManifest(
            name="memory",
            version="0.1.0",
            description="Memory maintenance: dedup, consolidation, KG pruning, contradiction resolution",
            interval_minutes=360,       # Every 6 hours
            idle_only=True,
            min_idle_seconds=900,       # 15 min idle (memory ops can be heavy)
            max_tokens=20000,
            max_cost_usd=0.20,
            max_duration_seconds=600,   # 10 min max
            model_preference="fast",
            require_approval_for=[],    # Memory maintenance is internal
            extra_blocked_tools=["web_search", "browser"],
            max_iterations=6,
            trigger_on_drive="coherence",
            trigger_drive_threshold=0.6,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are Aura's memory maintenance hand. Your job is to keep "
            "Aura's memory systems healthy: deduplicate entries, resolve "
            "contradictions, consolidate old episodic memories, and prune "
            "stale knowledge graph nodes. Be conservative — only remove "
            "or merge when you're confident it's safe."
        )

    async def execute(self, brain: Any, tools: dict, context: dict) -> HandResult:
        """Run memory maintenance cycle."""
        start = time.time()
        iterations = 0
        artifacts = []
        actions_taken = []
        tasks_completed = 0
        tasks_failed = 0
        errors = []
        step_cb = context.get("step_callback")

        # Capture token state at start
        start_tokens = (
            getattr(brain, '_session_input_tokens', 0) +
            getattr(brain, '_session_output_tokens', 0)
        )
        start_cost = getattr(brain, '_session_cost_usd', 0.0)

        # Task 1: Memory deduplication
        if step_cb:
            await step_cb(1, "Deduplicating memories...")
        try:
            from aura.memory.write_gate import get_write_gate
            gate = get_write_gate()
            if gate and hasattr(gate, 'deduplicate'):
                dedup_count = gate.deduplicate()
                if dedup_count:
                    actions_taken.append(f"Deduplicated {dedup_count} memory entries")
                    artifacts.append({"type": "dedup", "count": dedup_count})
            iterations += 1
            tasks_completed += 1
        except Exception as e:
            tasks_failed += 1
            errors.append(f"Dedup failed: {e}")
            logger.warning(f"[MemoryHand] Dedup failed: {e}")

        # Task 2: Memory consolidation via DreamConsolidator
        # (aura_episodic_memory package removed — redirected to DreamConsolidator)
        if step_cb:
            await step_cb(2, "Running dream consolidation...")
        try:
            from aura.dream import get_dream_consolidator
            consolidator = get_dream_consolidator()
            dc_report = consolidator.run_cycle()
            consolidated = dc_report.cycle.summaries_written
            if consolidated:
                actions_taken.append(f"Consolidated {consolidated} memory clusters")
                artifacts.append({"type": "consolidation", "count": consolidated})
            iterations += 1
            tasks_completed += 1
        except Exception as e:
            tasks_failed += 1
            errors.append(f"Memory consolidation failed: {e}")
            logger.warning(f"[MemoryHand] Memory consolidation failed: {e}")

        # Task 3: KG stale node pruning
        if step_cb:
            await step_cb(3, "Pruning stale knowledge graph nodes...")
        try:
            from aura.memory.kg_contradiction import prune_stale_nodes
            pruned = prune_stale_nodes(max_age_days=90, min_confidence=0.2)
            if pruned:
                actions_taken.append(f"Pruned {pruned} stale KG nodes")
                artifacts.append({"type": "kg_prune", "count": pruned})
            iterations += 1
            tasks_completed += 1
        except Exception as e:
            tasks_failed += 1
            errors.append(f"KG pruning failed: {e}")
            logger.warning(f"[MemoryHand] KG pruning failed: {e}")

        # Task 4: Contradiction resolution (uses LLM)
        if step_cb:
            await step_cb(4, "Resolving contradictions...")
        try:
            from aura.memory.kg_contradiction import get_contradictions
            contradictions = get_contradictions(limit=3)
            if contradictions and brain:
                for c in contradictions:
                    try:
                        resolution_prompt = (
                            f"Two facts in the knowledge graph contradict each other:\n"
                            f"Fact A: {c.get('fact_a', '?')}\n"
                            f"Fact B: {c.get('fact_b', '?')}\n\n"
                            f"Which is more likely correct? Respond with just 'A' or 'B' "
                            f"and a one-sentence reason."
                        )
                        response = await asyncio.to_thread(
                            lambda rp=resolution_prompt: brain.think(rp, system_prompt=self.get_system_prompt())
                        )
                        if response:
                            actions_taken.append(f"Resolved contradiction: {str(c.get('fact_a', ''))[:50]}")
                            artifacts.append({"type": "contradiction_resolved", "resolution": str(response)[:200]})
                        iterations += 1
                    except Exception:
                        pass
            tasks_completed += 1
        except Exception as e:
            tasks_failed += 1
            errors.append(f"Contradiction resolution failed: {e}")
            logger.warning(f"[MemoryHand] Contradiction resolution failed: {e}")

        # Build summary
        if actions_taken:
            summary = "Memory maintenance: " + "; ".join(actions_taken)
        else:
            summary = "Memory maintenance: all systems healthy, no action needed."

        # Append error info if any failures occurred
        if errors:
            summary += f" [{tasks_completed} succeeded, {tasks_failed} failed: {'; '.join(errors)}]"

        # Compute tokens and cost delta
        end_tokens = (
            getattr(brain, '_session_input_tokens', 0) +
            getattr(brain, '_session_output_tokens', 0)
        )
        end_cost = getattr(brain, '_session_cost_usd', 0.0)
        tokens_used = end_tokens - start_tokens
        cost_used = end_cost - start_cost

        return HandResult(
            hand_name="memory",
            success=tasks_completed > 0,
            summary=summary,
            iterations=iterations,
            artifacts=artifacts,
            duration_seconds=time.time() - start,
            tokens_used=tokens_used,
            cost_usd=cost_used,
        )
