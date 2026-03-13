"""
Pareto Frontier — Tracks non-dominated candidates across evaluation examples.
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .types import Candidate

logger = logging.getLogger(__name__)


class ParetoFrontier:
    """
    Instance-level Pareto frontier.

    A candidate is on the frontier if it achieves the best score
    on at least one evaluation example. This prevents premature
    convergence and preserves diverse strategies.
    """

    def __init__(self):
        # example_id -> best score seen
        self.best_scores: Dict[str, float] = {}
        # example_id -> set of candidate IDs achieving that score
        self.frontier_candidates: Dict[str, Set[int]] = defaultdict(set)
        # All candidates in the pool
        self.candidates: Dict[int, Candidate] = {}

    def update(self, candidate: Candidate) -> int:
        """
        Add a candidate and update the Pareto frontier.

        Returns:
            Number of examples where this candidate is now Pareto-optimal.
        """
        self.candidates[candidate.id] = candidate
        new_wins = 0

        for example_id, score in candidate.scores.items():
            current_best = self.best_scores.get(example_id, -1.0)

            if score > current_best:
                # New best — replace frontier for this example
                self.best_scores[example_id] = score
                self.frontier_candidates[example_id] = {candidate.id}
                new_wins += 1
            elif score == current_best:
                # Tie — add to frontier set
                self.frontier_candidates[example_id].add(candidate.id)
                new_wins += 1

        if new_wins > 0:
            logger.info(
                f"Candidate {candidate.id} is Pareto-optimal on "
                f"{new_wins}/{len(candidate.scores)} examples "
                f"(avg={candidate.avg_score:.3f})"
            )

        return new_wins

    def sample_candidate(self) -> Optional[Candidate]:
        """
        Sample a candidate from the Pareto frontier,
        weighted by how many examples it's best on.
        """
        if not self.frontier_candidates:
            return None

        # Count how many examples each candidate wins
        win_counts: Dict[int, int] = defaultdict(int)
        for cand_set in self.frontier_candidates.values():
            for cid in cand_set:
                win_counts[cid] += 1

        if not win_counts:
            return None

        # Weighted sample
        cids = list(win_counts.keys())
        weights = [win_counts[c] for c in cids]
        chosen_id = random.choices(cids, weights=weights, k=1)[0]

        return self.candidates.get(chosen_id)

    def get_frontier_ids(self) -> Set[int]:
        """Get all unique candidate IDs on the frontier."""
        all_ids = set()
        for cand_set in self.frontier_candidates.values():
            all_ids.update(cand_set)
        return all_ids

    def get_best_candidate(self) -> Optional[Candidate]:
        """Get the candidate with highest average score across all examples."""
        if not self.candidates:
            return None

        frontier_ids = self.get_frontier_ids()
        if not frontier_ids:
            return max(self.candidates.values(), key=lambda c: c.avg_score)

        frontier_cands = [self.candidates[cid] for cid in frontier_ids if cid in self.candidates]
        if not frontier_cands:
            return None

        return max(frontier_cands, key=lambda c: c.avg_score)

    @property
    def size(self) -> int:
        return len(self.get_frontier_ids())

    def summary(self) -> Dict:
        frontier_ids = self.get_frontier_ids()
        best = self.get_best_candidate()
        return {
            "frontier_size": len(frontier_ids),
            "total_candidates": len(self.candidates),
            "examples_tracked": len(self.best_scores),
            "best_avg_score": best.avg_score if best else 0.0,
            "best_candidate_id": best.id if best else None
        }
