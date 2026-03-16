"""Adaptive Planning — skip planning for simple tasks, re-plan for complex ones.

Roadmap 5.4: Classifies tasks as simple vs complex using heuristics.
Simple tasks go straight to ReAct execution. Complex tasks get a lightweight
plan that's refreshed every N steps (default N=3).
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that signal multi-step / complex tasks
_MULTI_STEP_PATTERNS = [
    r"\bfirst\b.*\bthen\b",
    r"\bstep\s*\d",
    r"\band\s+also\b",
    r"\bafter\s+that\b",
    r"\bnext\b.*\b(do|create|add|build|implement)\b",
    r"\b(1\.|2\.|3\.)",
    r"\bmultiple\b",
    r"\bseveral\b",
    r"\bboth\b.*\band\b",
    r"\beach\b",
    r"\ball\s+(the|of)\b",
]

# Patterns that signal code/technical tasks
_TECHNICAL_PATTERNS = [
    r"[A-Za-z_][A-Za-z0-9_]*\.(py|js|ts|jsx|tsx|rs|go|java|cpp|c|h|css|html|json|yaml|yml|toml)",
    r"```",
    r"[A-Z][a-z]+[A-Z]",  # CamelCase identifiers
    r"(def |class |function |const |let |var |import |from )",
    r"(/|\\)[A-Za-z_]",   # File paths
    r"https?://",
    r"\b(API|REST|GraphQL|SQL|database|schema|migration|deploy)\b",
]

# Imperative verbs that suggest actionable (but not necessarily complex) tasks
_IMPERATIVE_VERBS = {
    "create", "build", "implement", "write", "add", "fix", "update",
    "refactor", "migrate", "deploy", "configure", "setup", "install",
    "delete", "remove", "rename", "move", "copy", "merge", "split",
    "test", "debug", "optimize", "analyze", "review", "explain",
}


@dataclass
class PlanStep:
    """A single step in a task plan."""
    description: str
    expected_outcome: str = ""
    completed: bool = False
    result: str = ""


@dataclass
class TaskPlan:
    """A structured plan for a complex task."""
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    current_step_idx: int = 0
    created_at: float = field(default_factory=time.time)
    last_replanned_at: float = 0.0
    replan_count: int = 0

    @property
    def completed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.completed]

    @property
    def remaining_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if not s.completed]

    @property
    def progress_summary(self) -> str:
        done = len(self.completed_steps)
        total = len(self.steps)
        if total == 0:
            return "No plan steps."
        parts = [f"Progress: {done}/{total} steps completed."]
        if self.completed_steps:
            parts.append("Done:")
            for s in self.completed_steps:
                result_snippet = f" -> {s.result[:80]}" if s.result else ""
                parts.append(f"  [x] {s.description}{result_snippet}")
        if self.remaining_steps:
            parts.append("Remaining:")
            for s in self.remaining_steps:
                parts.append(f"  [ ] {s.description}")
        return "\n".join(parts)

    def mark_step_done(self, result: str = ""):
        """Mark the current step as completed and advance."""
        if self.current_step_idx < len(self.steps):
            self.steps[self.current_step_idx].completed = True
            self.steps[self.current_step_idx].result = result[:200]
            self.current_step_idx += 1

    def to_prompt_context(self) -> str:
        """Generate concise plan context for injection into the LLM prompt."""
        lines = ["[Plan]"]
        for i, step in enumerate(self.steps):
            marker = "x" if step.completed else " "
            current = " <-- current" if i == self.current_step_idx and not step.completed else ""
            lines.append(f"  [{marker}] {i+1}. {step.description}{current}")
        return "\n".join(lines)


def _count_complexity_signals(message: str) -> dict:
    """Count heuristic complexity signals in a message. Pure Python, no LLM."""
    msg_lower = message.lower()
    words = msg_lower.split()
    word_count = len(words)

    # 1. Length signal
    length_score = 0
    if word_count > 50:
        length_score = 2
    elif word_count > 25:
        length_score = 1

    # 2. Question marks / imperative verbs
    question_marks = message.count("?")
    imperative_count = sum(1 for w in words[:5] if w.strip(",.!") in _IMPERATIVE_VERBS)

    # 3. Multi-step indicators
    multi_step_hits = 0
    for pattern in _MULTI_STEP_PATTERNS:
        if re.search(pattern, msg_lower):
            multi_step_hits += 1

    # 4. Technical indicators
    technical_hits = 0
    for pattern in _TECHNICAL_PATTERNS:
        if re.search(pattern, message):
            technical_hits += 1

    # 5. Count ALL imperative verbs in the message (not just first 5 words)
    total_imperatives = sum(1 for w in words if w.strip(",.!:;") in _IMPERATIVE_VERBS)

    # 6. Comma-separated list of things to do (e.g., "fix X, add Y, and update Z")
    comma_count = message.count(",")
    # Multiple imperative verbs + commas = strong multi-action signal
    list_score = 0
    if comma_count >= 2 and total_imperatives >= 2:
        list_score = 2
    elif comma_count >= 1 and total_imperatives >= 2:
        list_score = 1

    # 7. Code block presence
    has_code_block = "```" in message

    return {
        "word_count": word_count,
        "length_score": length_score,
        "question_marks": question_marks,
        "imperative_count": imperative_count,
        "total_imperatives": total_imperatives,
        "multi_step_hits": multi_step_hits,
        "technical_hits": technical_hits,
        "list_score": list_score,
        "has_code_block": has_code_block,
    }


class AdaptivePlanner:
    """Adaptive planning: skip for simple, plan+replan for complex tasks.

    Usage:
        planner = AdaptivePlanner(brain=brain)
        is_complex = planner.classify(goal)
        if is_complex:
            plan = planner.generate_plan(goal)
            # ... inject plan.to_prompt_context() into LLM messages
            # After N ReAct steps:
            if planner.should_replan(iteration):
                plan = planner.replan(plan, iteration_results)
    """

    # Complexity threshold: sum of weighted signals must exceed this
    COMPLEXITY_THRESHOLD = 3

    def __init__(self, brain=None, planning_interval: int = 3):
        """
        Args:
            brain: OllamaBrain instance for plan generation (main model).
            planning_interval: Re-plan every N ReAct steps for complex tasks.
        """
        self.brain = brain
        self.planning_interval = planning_interval
        self._current_plan: Optional[TaskPlan] = None
        self._steps_since_last_plan: int = 0

    def classify(self, goal: str) -> bool:
        """Classify a task as complex (True) or simple (False).

        Uses fast heuristics only — no LLM call. ~0ms.
        """
        signals = _count_complexity_signals(goal)

        # Weighted score
        score = (
            signals["length_score"] * 1.0
            + signals["multi_step_hits"] * 2.0
            + signals["technical_hits"] * 0.5
            + signals["list_score"] * 1.5
            + (1.0 if signals["has_code_block"] else 0)
            + signals["imperative_count"] * 0.5
            + max(0, signals["total_imperatives"] - 1) * 1.0  # 2+ verbs = multi-action
        )

        # Short messages with no complexity signals -> simple
        if signals["word_count"] <= 10 and score < 1:
            logger.debug(f"[Planner] Simple (short, score={score:.1f}): {goal[:60]}")
            return False

        is_complex = score >= self.COMPLEXITY_THRESHOLD
        logger.debug(f"[Planner] {'Complex' if is_complex else 'Simple'} (score={score:.1f}): {goal[:60]}")
        return is_complex

    def generate_plan(self, goal: str) -> Optional[TaskPlan]:
        """Generate a structured plan for a complex task using the main LLM.

        Returns None if plan generation fails (agent should proceed without plan).
        """
        if not self.brain:
            return None

        prompt = (
            "Break down this task into 3-6 concrete steps. "
            "For each step, give a one-line description.\n"
            "Format: one step per line, numbered (1. 2. 3. etc.)\n"
            "Be specific and actionable. No preamble.\n\n"
            f"Task: {goal}"
        )

        try:
            response = self.brain.think(
                prompt,
                system_prompt="You are a task planner. Output only numbered steps, nothing else.",
                use_history=False,
                task_type=None,
            )
        except Exception as e:
            logger.warning(f"[Planner] Plan generation failed: {e}")
            return None

        if not response:
            return None

        # Parse numbered steps from response
        steps = self._parse_steps(response)
        if not steps:
            return None

        plan = TaskPlan(goal=goal, steps=steps)
        self._current_plan = plan
        self._steps_since_last_plan = 0

        logger.info(f"[Planner] Generated plan with {len(steps)} steps for: {goal[:60]}")
        return plan

    def tick(self) -> None:
        """Increment the step counter after a ReAct iteration.

        Call this once per ReAct step. Use should_replan() separately to check.
        """
        self._steps_since_last_plan += 1

    def should_replan(self, react_iteration: int = 0) -> bool:
        """Check if it's time to re-plan based on the planning interval.

        Pure check -- does NOT increment any counter. Call tick() first.
        """
        if not self._current_plan:
            return False
        return self._steps_since_last_plan >= self.planning_interval

    def replan(self, results_so_far: str = "") -> Optional[TaskPlan]:
        """Re-plan based on what's been accomplished so far.

        Lighter than initial planning — focuses on remaining work.
        """
        if not self.brain or not self._current_plan:
            return self._current_plan

        plan = self._current_plan
        progress = plan.progress_summary

        prompt = (
            f"Original goal: {plan.goal}\n\n"
            f"{progress}\n\n"
        )
        if results_so_far:
            prompt += f"Recent results:\n{results_so_far[:500]}\n\n"

        prompt += (
            "Given the progress so far, list the remaining steps needed to complete the goal.\n"
            "Format: one step per line, numbered. Be specific. No preamble."
        )

        try:
            response = self.brain.think(
                prompt,
                system_prompt="You are a task planner. Output only numbered steps for remaining work.",
                use_history=False,
                task_type=None,
            )
        except Exception as e:
            logger.warning(f"[Planner] Re-plan failed: {e}")
            return self._current_plan

        if not response:
            return self._current_plan

        new_steps = self._parse_steps(response)
        if not new_steps:
            return self._current_plan

        # Keep completed steps, replace remaining with new plan
        completed = [s for s in plan.steps if s.completed]
        plan.steps = completed + new_steps
        plan.current_step_idx = len(completed)
        plan.last_replanned_at = time.time()
        plan.replan_count += 1
        self._steps_since_last_plan = 0

        logger.info(f"[Planner] Re-planned: {len(completed)} done + {len(new_steps)} remaining")
        return plan

    def advance_step(self, result: str = ""):
        """Mark the current plan step as done after a successful ReAct iteration."""
        if self._current_plan and self._current_plan.current_step_idx < len(self._current_plan.steps):
            self._current_plan.mark_step_done(result)

    @property
    def current_plan(self) -> Optional[TaskPlan]:
        return self._current_plan

    def reset(self):
        """Clear the current plan."""
        self._current_plan = None
        self._steps_since_last_plan = 0

    @staticmethod
    def _parse_steps(response: str) -> list[PlanStep]:
        """Parse numbered steps from LLM response."""
        steps = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match patterns like "1. Do something" or "1) Do something" or "- Do something"
            match = re.match(r"^(?:\d+[\.\)]\s*|-\s*|\*\s*)(.*)", line)
            if match:
                desc = match.group(1).strip()
                if desc and len(desc) > 3:
                    steps.append(PlanStep(description=desc))
        return steps
