"""
GEPA Proposers — Reflective mutation and genealogy-aware merge.

The core insight: instead of blind mutation, read execution traces
to understand WHY things fail, then propose targeted fixes.
"""

import json
import logging
import random
import re
from typing import Callable, Dict, List, Optional, Tuple

from .types import (
    Candidate,
    CandidateSelectionStrategy,
    ComponentSelectionMode,
    GEPAConfig,
    ReflectionItem,
    Trajectory,
)
from .pareto import ParetoFrontier

logger = logging.getLogger(__name__)


class ReflectiveMutationProposer:
    """
    Reads failure traces, reflects on what went wrong,
    and proposes improved skill text. This is the heart of GEPA.
    """

    def __init__(
        self,
        config: GEPAConfig,
        llm_func: Callable[[str, str], str],  # (system, user) -> response
    ):
        self.config = config
        self.llm_func = llm_func
        self._component_index = 0  # For round-robin

    def select_components(
        self,
        candidate: Candidate,
        iteration: int,
    ) -> List[str]:
        """Select which skill components to evolve this iteration."""
        component_ids = list(candidate.components.keys())

        if self.config.component_mode == ComponentSelectionMode.ALL:
            return component_ids

        # Round-robin: one component per iteration
        if not component_ids:
            return []
        idx = iteration % len(component_ids)
        return [component_ids[idx]]

    def build_reflection_dataset(
        self,
        candidate: Candidate,
        trajectories: List[Trajectory],
        components_to_update: List[str],
    ) -> List[ReflectionItem]:
        """
        From execution traces, extract failure cases formatted
        for LLM reflection. This is the "Actionable Side Information" (ASI).
        """
        items = []

        # Focus on failures and low-scoring examples
        failed = [t for t in trajectories if not t.success or t.score < 0.7]
        if not failed:
            # Even if no failures, include lowest-scoring for improvement
            failed = sorted(trajectories, key=lambda t: t.score)[:2]

        for trajectory in failed:
            for comp_id in components_to_update:
                if comp_id not in candidate.components:
                    continue

                items.append(ReflectionItem(
                    component_id=comp_id,
                    current_text=candidate.components[comp_id],
                    task_input=trajectory.task_input,
                    task_output=trajectory.task_output,
                    score=trajectory.score,
                    error=trajectory.error,
                    feedback=trajectory.evaluator_feedback,
                    trajectory_summary=trajectory.summary(),
                ))

        return items

    def propose(
        self,
        candidate: Candidate,
        reflection_data: List[ReflectionItem],
        components_to_update: List[str],
    ) -> Dict[str, str]:
        """
        Use LLM to generate improved component texts based on failure analysis.

        Returns:
            Dict of component_id -> new text
        """
        new_texts = {}

        for comp_id in components_to_update:
            # Gather all reflection items for this component
            comp_items = [r for r in reflection_data if r.component_id == comp_id]
            if not comp_items:
                continue

            # Build reflection prompt
            failures_text = ""
            for i, item in enumerate(comp_items[:5], 1):
                failures_text += f"""
--- Failure {i} (score: {item.score:.2f}) ---
Task: {item.task_input[:300]}
Output produced: {item.task_output[:300]}
{f'Error: {item.error}' if item.error else ''}
{f'Evaluator says: {item.feedback}' if item.feedback else ''}
Trace: {item.trajectory_summary[:500]}
"""

            system_prompt = """You are an expert at improving AI agent skill procedures.
You analyze failure cases and produce improved procedure text that prevents those failures.
You must preserve the skill's core purpose while making targeted improvements."""

            user_prompt = f"""The following skill procedure is underperforming. Analyze the failures and write an improved version.

## Current Procedure
{comp_items[0].current_text}

## Failure Cases
{failures_text}

## Instructions
1. Identify what's going wrong — look for patterns across failures
2. Write an improved procedure that addresses these failure modes
3. Keep the same structure and intent, but make targeted fixes
4. Be specific and actionable in your steps
5. Do NOT add unnecessary complexity or bloat

Respond with ONLY the improved procedure text. No preamble, no explanation, just the new procedure."""

            try:
                new_text = self.llm_func(system_prompt, user_prompt)
                # Clean up: remove markdown code fences if present
                new_text = re.sub(r'^```\w*\n', '', new_text.strip())
                new_text = re.sub(r'\n```$', '', new_text.strip())

                if new_text and len(new_text) > 20:
                    new_texts[comp_id] = new_text
                    logger.debug(f"Proposed mutation for {comp_id}: {len(new_text)} chars")

            except Exception as e:
                logger.error(f"Reflection failed for {comp_id}: {e}")

        return new_texts


class MergeProposer:
    """
    Genealogy-aware crossover: finds two Pareto-optimal candidates
    that diverged from a common ancestor, then merges their
    independently-discovered improvements.
    """

    def __init__(self, config: GEPAConfig):
        self.config = config
        self._merges_done = 0

    def find_common_ancestor(
        self,
        cand_a: Candidate,
        cand_b: Candidate,
        all_candidates: Dict[int, Candidate],
    ) -> Optional[int]:
        """Find lowest common ancestor in the genealogy tree."""
        # Collect all ancestors of A
        ancestors_a = set()
        current = cand_a.id
        while current >= 0 and current in all_candidates:
            ancestors_a.add(current)
            current = all_candidates[current].parent_id

        # Walk up from B until we hit an ancestor of A
        current = cand_b.id
        while current >= 0 and current in all_candidates:
            if current in ancestors_a:
                return current
            current = all_candidates[current].parent_id

        return None

    def diff_candidates(
        self,
        ancestor: Candidate,
        descendant: Candidate,
    ) -> Dict[str, str]:
        """Find which components changed from ancestor to descendant."""
        mutations = {}
        for comp_id, text in descendant.components.items():
            ancestor_text = ancestor.components.get(comp_id, "")
            if text != ancestor_text:
                mutations[comp_id] = text
        return mutations

    def propose(
        self,
        pareto: ParetoFrontier,
        all_candidates: Dict[int, Candidate],
    ) -> Optional[Candidate]:
        """
        Try to merge two Pareto-optimal candidates.

        Returns:
            Merged candidate or None if merge not possible.
        """
        if self._merges_done >= self.config.max_merges:
            return None

        frontier_ids = list(pareto.get_frontier_ids())
        if len(frontier_ids) < 2:
            return None

        # Try pairs to find ones with a common ancestor
        random.shuffle(frontier_ids)

        for i in range(len(frontier_ids)):
            for j in range(i + 1, min(len(frontier_ids), i + 5)):
                cand_a = all_candidates.get(frontier_ids[i])
                cand_b = all_candidates.get(frontier_ids[j])

                if not cand_a or not cand_b:
                    continue

                ancestor_id = self.find_common_ancestor(cand_a, cand_b, all_candidates)
                if ancestor_id is None:
                    continue

                ancestor = all_candidates[ancestor_id]

                # Extract independent mutations
                mutations_a = self.diff_candidates(ancestor, cand_a)
                mutations_b = self.diff_candidates(ancestor, cand_b)

                if not mutations_a or not mutations_b:
                    continue

                # Merge: start from ancestor, apply both mutation sets
                merged_components = dict(ancestor.components)
                merged_components.update(mutations_a)
                merged_components.update(mutations_b)  # B wins on overlap

                # Create merged candidate
                new_id = max(all_candidates.keys()) + 1 if all_candidates else 0
                merged = Candidate(
                    id=new_id,
                    components=merged_components,
                    parent_id=ancestor_id,
                )

                self._merges_done += 1
                logger.info(
                    f"Merged candidates {cand_a.id} + {cand_b.id} "
                    f"(ancestor={ancestor_id}) -> {new_id}"
                )
                return merged

        return None
