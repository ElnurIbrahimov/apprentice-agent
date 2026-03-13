"""
Aura Skill Adapter — Bridges Aura's skill library to GEPA.

Handles:
- Loading skills as GEPA candidates
- Running skills through Aura's agent and capturing traces
- Scoring outputs via LLM-as-judge
- Generating synthetic evaluation datasets
"""

import json
import logging
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from .types import Candidate, EvalExample, GEPAConfig, Trajectory

logger = logging.getLogger(__name__)


class AuraSkillAdapter:
    """
    Connects GEPA's optimize loop to Aura's skill library.

    evaluate() runs skills on tasks and captures traces.
    make_reflective_dataset() extracts failure context as ASI.
    """

    def __init__(
        self,
        config: GEPAConfig,
        llm_func: Callable[[str, str], str],  # (system, user) -> response
        eval_llm_func: Optional[Callable[[str, str], str]] = None,  # Cheaper model
    ):
        self.config = config
        self.llm_func = llm_func  # Strong model (reflection)
        self.eval_llm = eval_llm_func or llm_func  # Cheap model (evaluation)
        self._eval_count = 0

    def load_skills_as_candidate(self, skill_store) -> Candidate:
        """Load all skills from Aura's SkillStore as a seed candidate."""
        components = {}

        for skill_id, info in skill_store.index.items():
            skill = skill_store.load(skill_id)
            if skill:
                components[skill_id] = skill.procedure

        return Candidate(id=0, components=components, parent_id=-1)

    def generate_eval_dataset(
        self,
        candidate: Candidate,
        num_examples: int = 20,
    ) -> List[EvalExample]:
        """
        Generate synthetic evaluation tasks by having an LLM read each skill
        and produce (task_input, expected_behavior) pairs.
        """
        examples = []
        example_id = 0

        for skill_id, procedure in candidate.components.items():
            per_skill = max(2, num_examples // max(len(candidate.components), 1))

            prompt = f"""Given this skill procedure, generate {per_skill} diverse test tasks.
Each task should test a different aspect of the skill.

## Skill Procedure
{procedure[:2000]}

Generate tasks as a JSON array:
[
  {{"task": "description of what the user asks", "expected": "rubric for what a good response looks like"}},
  ...
]

Respond with ONLY the JSON array."""

            try:
                response = self.llm_func(
                    "You generate evaluation datasets for AI skills.",
                    prompt,
                )
                # Extract JSON
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    tasks = json.loads(json_match.group())
                    for task in tasks[:per_skill]:
                        examples.append(EvalExample(
                            id=f"eval_{example_id}",
                            task_input=task.get("task", ""),
                            expected_behavior=task.get("expected", ""),
                            source="synthetic",
                        ))
                        example_id += 1
            except Exception as e:
                logger.warning(f"Failed to generate eval for {skill_id}: {e}")

        logger.info(f"Generated {len(examples)} evaluation examples")
        return examples

    def evaluate(
        self,
        candidate: Candidate,
        examples: List[EvalExample],
        capture_traces: bool = False,
    ) -> Tuple[List[float], List[Trajectory]]:
        """
        Run the candidate's skills on evaluation examples and score results.

        Returns:
            (scores, trajectories)
        """
        scores = []
        trajectories = []

        for example in examples:
            self._eval_count += 1
            start = time.time()

            # Find which skill component is most relevant
            skill_id = self._match_skill(candidate, example.task_input)
            procedure = candidate.components.get(skill_id, "")

            # Run the skill
            try:
                output = self.eval_llm(
                    f"Follow this procedure to complete the task:\n\n{procedure}",
                    example.task_input,
                )
                error = None
                success = True
            except Exception as e:
                output = str(e)
                error = str(e)
                success = False

            exec_time = (time.time() - start) * 1000

            # Score the output
            score, feedback = self._score_output(
                example, output, success
            )
            scores.append(score)

            if capture_traces:
                trajectories.append(Trajectory(
                    skill_id=skill_id,
                    task_input=example.task_input,
                    task_output=output,
                    score=score,
                    success=success,
                    error=error,
                    execution_time_ms=exec_time,
                    evaluator_feedback=feedback,
                ))

        return scores, trajectories

    def _match_skill(self, candidate: Candidate, task_input: str) -> str:
        """Simple keyword matching to find relevant skill for a task."""
        task_lower = task_input.lower()
        best_id = ""
        best_score = -1

        for skill_id, procedure in candidate.components.items():
            # Count keyword overlap
            proc_words = set(procedure.lower().split())
            task_words = set(task_lower.split())
            overlap = len(proc_words & task_words)
            if overlap > best_score:
                best_score = overlap
                best_id = skill_id

        return best_id or list(candidate.components.keys())[0]

    def _score_output(
        self,
        example: EvalExample,
        output: str,
        success: bool,
    ) -> Tuple[float, str]:
        """
        Score an output against the expected behavior rubric.

        Uses a hybrid approach:
        - Fast keyword heuristic (0.3 weight)
        - LLM-as-judge for nuance (0.7 weight)
        """
        if not success:
            return 0.1, "Execution failed"

        # Fast heuristic: keyword overlap with expected behavior
        expected_words = set(example.expected_behavior.lower().split())
        output_words = set(output.lower().split())
        if expected_words:
            keyword_score = len(expected_words & output_words) / len(expected_words)
        else:
            keyword_score = 0.5

        # LLM-as-judge (using eval model for cost efficiency)
        judge_score = keyword_score  # Default fallback
        feedback = ""

        try:
            judge_prompt = f"""Rate this AI output on a scale of 0.0 to 1.0.

## Task
{example.task_input[:500]}

## Expected Behavior
{example.expected_behavior[:500]}

## Actual Output
{output[:800]}

## Scoring Criteria
- Correctness: Does it address the task correctly? (50%)
- Procedure following: Does it follow a clear method? (30%)
- Conciseness: Is it focused, not bloated? (20%)

Respond in this exact format:
SCORE: 0.XX
FEEDBACK: one sentence explaining the score"""

            response = self.eval_llm(
                "You are a strict but fair evaluator of AI outputs.",
                judge_prompt,
            )

            # Parse score
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response)
            if score_match:
                judge_score = float(score_match.group(1))
                judge_score = max(0.0, min(1.0, judge_score))

            # Parse feedback
            fb_match = re.search(r'FEEDBACK:\s*(.+)', response)
            if fb_match:
                feedback = fb_match.group(1).strip()

        except Exception as e:
            logger.debug(f"Judge scoring failed, using heuristic: {e}")
            feedback = "Judge unavailable, heuristic score used"

        # Composite: 30% heuristic + 70% judge
        final_score = 0.3 * keyword_score + 0.7 * judge_score

        # Length penalty: ramps from 0 at 90% max size to 0.3 at 100%+
        if len(output) > 10000:
            penalty = min(0.3, (len(output) - 9000) / 3000 * 0.3)
            final_score = max(0.0, final_score - penalty)

        return round(final_score, 3), feedback

    @property
    def total_evals(self) -> int:
        return self._eval_count
