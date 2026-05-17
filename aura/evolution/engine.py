"""
GEPA Engine — Genetic-Pareto Prompt Evolution for Aura.

Main optimization loop that evolves skill procedures by:
1. Evaluating candidates on tasks with full trace capture
2. Reflecting on failures to propose targeted mutations
3. Maintaining a Pareto frontier of diverse strategies
4. Merging independently-evolved improvements via genealogy
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .adapter import AuraSkillAdapter
from .cache import EvaluationCache
from .constraints import ConstraintValidator
from .pareto import ParetoFrontier
from .proposers import MergeProposer, ReflectiveMutationProposer
from .types import (
    Candidate,
    CandidateSelectionStrategy,
    EvalExample,
    GEPAConfig,
    GEPAResult,
)

logger = logging.getLogger(__name__)


class GEPAEngine:
    """
    GEPA optimization engine adapted for Aura's skill library.

    Instead of DSPy wrapping, operates directly on Aura Skill procedures.
    Uses Ollama (local models) for evaluation, stronger models for reflection.
    Cost: ~$0 with local models, $2-10 with cloud.
    """

    def __init__(
        self,
        config: GEPAConfig,
        adapter: AuraSkillAdapter,
        llm_func: Callable[[str, str], str],
    ):
        self.config = config
        self.adapter = adapter
        self.llm_func = llm_func

        # Core components
        self.pareto = ParetoFrontier()
        self.cache = EvaluationCache(config.run_dir)
        self.mutator = ReflectiveMutationProposer(config, llm_func)
        self.merger = MergeProposer(config)
        self.validator: Optional[ConstraintValidator] = None

        # State
        self.all_candidates: Dict[int, Candidate] = {}
        self.next_id = 0
        self._no_improvement_count = 0
        self._best_score = 0.0

        # Ensure run directory exists
        Path(config.run_dir).mkdir(parents=True, exist_ok=True)

    def optimize(
        self,
        seed_candidate: Candidate,
        eval_examples: List[EvalExample],
        val_examples: Optional[List[EvalExample]] = None,
    ) -> GEPAResult:
        """
        Run the GEPA optimization loop.

        Args:
            seed_candidate: Initial skill texts
            eval_examples: Training examples for reflection
            val_examples: Validation examples (uses eval_examples if None)

        Returns:
            GEPAResult with best candidate and stats
        """
        start_time = time.time()

        if val_examples is None:
            # Split: 60% train, 40% val
            random.shuffle(eval_examples)
            split = int(len(eval_examples) * 0.6)
            train_examples = eval_examples[:split]
            val_examples = eval_examples[split:]
        else:
            train_examples = eval_examples

        if not train_examples or not val_examples:
            raise ValueError("Need at least 1 train and 1 val example")

        # Initialize
        seed_candidate.id = self._get_next_id()
        self.all_candidates[seed_candidate.id] = seed_candidate
        self.validator = ConstraintValidator(self.config, seed_candidate)

        # Evaluate seed on full validation set
        logger.info("Evaluating seed candidate...")
        seed_scores = self._evaluate_cached(seed_candidate, val_examples)
        seed_candidate.scores = {
            ex.id: score for ex, score in zip(val_examples, seed_scores, strict=False)
        }
        self.pareto.update(seed_candidate)
        self._best_score = seed_candidate.avg_score
        logger.info(f"Seed score: {self._best_score:.3f}")

        # Save initial state
        self._save_state(0)

        # Main loop
        stop_reason = "max_iterations"
        iteration = 0
        for iteration in range(1, self.config.max_iterations + 1):
            elapsed = time.time() - start_time
            logger.info(f"\n=== Iteration {iteration}/{self.config.max_iterations} ===")

            # Check stopping conditions
            if elapsed > self.config.timeout_seconds:
                stop_reason = f"timeout ({self.config.timeout_seconds}s)"
                break

            if self.adapter.total_evals >= self.config.max_metric_calls:
                stop_reason = f"max_evals ({self.config.max_metric_calls})"
                break

            if self._best_score >= self.config.score_threshold:
                stop_reason = f"score_threshold ({self.config.score_threshold})"
                break

            if self._no_improvement_count >= self.config.no_improvement_patience:
                stop_reason = f"no_improvement ({self.config.no_improvement_patience} iters)"
                break

            # === OPTIONAL MERGE STEP ===
            if (
                self.config.use_merge
                and iteration % self.config.merge_interval == 0
                and self.pareto.size >= 2
            ):
                self._try_merge(val_examples)

            # === MUTATION STEP ===
            improved = self._mutation_step(
                iteration, train_examples, val_examples
            )

            if not improved:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            self._save_state(iteration)

        # Done
        duration = time.time() - start_time
        best = self.pareto.get_best_candidate() or seed_candidate

        result = GEPAResult(
            best_candidate=best,
            all_candidates=list(self.all_candidates.values()),
            iterations_run=iteration,
            total_evals=self.adapter.total_evals,
            pareto_front={
                ex_id: next(iter(cids)) if cids else -1
                for ex_id, cids in self.pareto.frontier_candidates.items()
            },
            improvement=best.avg_score - seed_candidate.avg_score,
            duration_seconds=duration,
            stop_reason=stop_reason,
        )

        # Save final state
        self.cache.save()
        self._save_result(result)

        logger.info(
            f"\nGEPA complete: {stop_reason}\n"
            f"  Seed: {seed_candidate.avg_score:.3f} -> Best: {best.avg_score:.3f} "
            f"(+{result.improvement:.3f})\n"
            f"  Candidates: {len(self.all_candidates)}, "
            f"Evals: {self.adapter.total_evals}, "
            f"Time: {duration:.1f}s"
        )

        return result

    def _mutation_step(
        self,
        iteration: int,
        train_examples: List[EvalExample],
        val_examples: List[EvalExample],
    ) -> bool:
        """
        One mutation iteration:
        1. Select candidate from Pareto frontier
        2. Select components to update
        3. Evaluate on minibatch with trace capture
        4. Reflect on failures and propose mutation
        5. Subsample acceptance check
        6. Full validation if accepted

        Returns True if an improvement was found.
        """
        # 1. Select parent candidate
        parent = self._select_candidate()
        if not parent:
            logger.warning("No candidate to mutate")
            return False

        # 2. Select components
        components = self.mutator.select_components(parent, iteration)
        if not components:
            return False

        logger.info(f"Mutating candidate {parent.id}, components: {components}")

        # 3. Sample minibatch and evaluate with traces
        minibatch = random.sample(
            train_examples,
            min(self.config.minibatch_size, len(train_examples)),
        )

        old_scores, trajectories = self.adapter.evaluate(
            parent, minibatch, capture_traces=True,
        )

        # 4. Build reflection dataset and propose mutation
        reflection_data = self.mutator.build_reflection_dataset(
            parent, trajectories, components,
        )

        new_texts = self.mutator.propose(parent, reflection_data, components)
        if not new_texts:
            logger.info("No mutations proposed")
            return False

        # 5. Build new candidate
        new_components = dict(parent.components)
        new_components.update(new_texts)

        new_candidate = Candidate(
            id=self._get_next_id(),
            components=new_components,
            parent_id=parent.id,
        )

        # Constraint check
        passed, violations = self.validator.validate(new_candidate)
        if not passed:
            logger.info(f"Candidate failed constraints: {violations}")
            return False

        # 6. Subsample acceptance: must beat parent on same minibatch
        new_scores, _ = self.adapter.evaluate(new_candidate, minibatch)

        if sum(new_scores) <= sum(old_scores):
            logger.info(
                f"Rejected: subsample {sum(new_scores):.2f} <= {sum(old_scores):.2f}"
            )
            return False

        logger.info(
            f"Subsample passed: {sum(new_scores):.2f} > {sum(old_scores):.2f}"
        )

        # 7. Full validation
        full_scores = self._evaluate_cached(new_candidate, val_examples)
        new_candidate.scores = {
            ex.id: score for ex, score in zip(val_examples, full_scores, strict=False)
        }

        # Register candidate
        self.all_candidates[new_candidate.id] = new_candidate
        wins = self.pareto.update(new_candidate)

        if new_candidate.avg_score > self._best_score:
            self._best_score = new_candidate.avg_score
            logger.info(
                f"NEW BEST: {new_candidate.avg_score:.3f} "
                f"(candidate {new_candidate.id}, {wins} Pareto wins)"
            )
            return True

        if wins > 0:
            logger.info(
                f"Pareto improvement: {wins} new wins "
                f"(avg={new_candidate.avg_score:.3f})"
            )
            return True

        return False

    def _try_merge(self, val_examples: List[EvalExample]):
        """Attempt a genealogy-aware merge of two Pareto candidates."""
        merged = self.merger.propose(self.pareto, self.all_candidates)
        if not merged:
            return

        # Constraint check
        passed, violations = self.validator.validate(merged)
        if not passed:
            logger.info(f"Merged candidate failed constraints: {violations}")
            return

        # Full validation
        scores = self._evaluate_cached(merged, val_examples)
        merged.scores = {
            ex.id: score for ex, score in zip(val_examples, scores, strict=False)
        }

        # Dual acceptance: must beat both parents
        parent_a = self.all_candidates.get(merged.parent_id)
        if parent_a and merged.avg_score <= parent_a.avg_score:
            logger.info("Merge rejected: doesn't beat parent")
            return

        self.all_candidates[merged.id] = merged
        wins = self.pareto.update(merged)

        if merged.avg_score > self._best_score:
            self._best_score = merged.avg_score

        logger.info(
            f"Merge accepted: avg={merged.avg_score:.3f}, "
            f"{wins} Pareto wins"
        )

    def _select_candidate(self) -> Optional[Candidate]:
        """Select a candidate based on configured strategy."""
        strategy = self.config.candidate_strategy

        if strategy == CandidateSelectionStrategy.PARETO:
            return self.pareto.sample_candidate()

        elif strategy == CandidateSelectionStrategy.CURRENT_BEST:
            return self.pareto.get_best_candidate()

        elif strategy == CandidateSelectionStrategy.EPSILON_GREEDY:
            if random.random() < self.config.epsilon:
                # Explore: random candidate from pool
                if self.all_candidates:
                    return random.choice(list(self.all_candidates.values()))
            return self.pareto.get_best_candidate()

        return self.pareto.get_best_candidate()

    def _evaluate_cached(
        self,
        candidate: Candidate,
        examples: List[EvalExample],
    ) -> List[float]:
        """Evaluate with cache lookup to avoid redundant LLM calls."""
        scores = []
        cand_hash = candidate.cache_key()

        uncached_indices = []
        uncached_examples = []

        for i, example in enumerate(examples):
            cached_score = self.cache.get(cand_hash, example.cache_key())
            if cached_score is not None:
                scores.append(cached_score)
            else:
                scores.append(None)
                uncached_indices.append(i)
                uncached_examples.append(example)

        if uncached_examples:
            new_scores, _ = self.adapter.evaluate(candidate, uncached_examples)
            for idx, score in zip(uncached_indices, new_scores, strict=False):
                scores[idx] = score
                self.cache.put(cand_hash, examples[idx].cache_key(), score)

        # Guard against None entries from partial evaluation failures
        scores = [s if s is not None else 0.0 for s in scores]
        return scores

    def _get_next_id(self) -> int:
        cid = self.next_id
        self.next_id += 1
        return cid

    def _save_state(self, iteration: int):
        """Save checkpoint for fault tolerance."""
        state_path = Path(self.config.run_dir) / f"checkpoint_{iteration}.json"
        state = {
            "iteration": iteration,
            "best_score": self._best_score,
            "total_candidates": len(self.all_candidates),
            "total_evals": self.adapter.total_evals,
            "pareto": self.pareto.summary(),
            "cache_stats": self.cache.stats,
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _save_result(self, result: GEPAResult):
        """Save final result with best candidate's skill texts."""
        result_path = Path(self.config.run_dir) / "result.json"
        data = {
            "best_candidate_id": result.best_candidate.id,
            "best_avg_score": result.best_candidate.avg_score,
            "best_components": result.best_candidate.components,
            "improvement": result.improvement,
            "iterations": result.iterations_run,
            "total_evals": result.total_evals,
            "duration_seconds": result.duration_seconds,
            "stop_reason": result.stop_reason,
            "pareto_front": result.pareto_front,
        }
        with open(result_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {result_path}")
