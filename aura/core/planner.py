"""Plan data structures, parsing, and prompts — shared between core and CLI.

This module contains the non-UI pieces of plan generation:
  - Data classes (StepStatus, PlanStep, ExecutionPlan)
  - LLM response parsing (parse_plan_from_llm)
  - Plan generation prompt (PLAN_GENERATION_PROMPT)

CLI rendering lives in aura.cli.plan_mode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    index: int
    description: str
    files: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    error: str = ""


@dataclass
class ExecutionPlan:
    """A structured execution plan with steps."""
    goal: str
    approach: str
    steps: List[PlanStep] = field(default_factory=list)

    @property
    def progress(self) -> str:
        done = sum(1 for s in self.steps if s.status in (StepStatus.DONE, StepStatus.SKIPPED))
        total = len(self.steps)
        return f"{done}/{total}"

    @property
    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED) for s in self.steps)

    @property
    def current_step(self) -> Optional[PlanStep]:
        for s in self.steps:
            if s.status == StepStatus.PENDING:
                return s
        return None

    def to_markdown(self) -> str:
        """Convert plan to Markdown checklist."""
        lines = [f"# Plan: {self.goal}", "", f"**Approach:** {self.approach}", ""]
        for step in self.steps:
            if step.status == StepStatus.DONE:
                checkbox = "[x]"
            elif step.status == StepStatus.RUNNING:
                checkbox = "[~]"
            elif step.status == StepStatus.FAILED:
                checkbox = "[!]"
            elif step.status == StepStatus.SKIPPED:
                checkbox = "[-]"
            else:
                checkbox = "[ ]"
            line = f"- {checkbox} **Step {step.index}:** {step.description}"
            if step.files:
                line += f" ({', '.join(step.files)})"
            lines.append(line)
            if step.error:
                lines.append(f"  - ⚠ {step.error}")
        return "\n".join(lines)


def parse_plan_from_llm(response: str) -> ExecutionPlan:
    """Parse an LLM-generated plan response into a structured ExecutionPlan."""
    lines = response.strip().splitlines()

    goal = ""
    approach = ""
    steps = []
    step_idx = 0

    for line in lines:
        line_stripped = line.strip()

        if not goal and (line_stripped.startswith("# ") or line_stripped.lower().startswith("goal:")):
            goal = re.sub(r'^#\s*|^goal:\s*', '', line_stripped, flags=re.IGNORECASE).strip()
            goal = re.sub(r'\*\*(.*?)\*\*', r'\1', goal)
            continue

        if not approach and line_stripped.lower().startswith("approach:"):
            approach = re.sub(r'^approach:\s*', '', line_stripped, flags=re.IGNORECASE).strip()
            continue

        step_match = re.match(r'^(?:\d+[\.\)]\s*|-\s*\[.\]\s*|-\s+)(.*)', line_stripped)
        if step_match:
            step_idx += 1
            desc = step_match.group(1).strip()
            desc = re.sub(r'\*\*(.*?)\*\*', r'\1', desc)
            files = re.findall(r'`([^`]+\.\w+)`', desc)
            steps.append(PlanStep(
                index=step_idx,
                description=desc,
                files=files,
            ))

    if not goal and steps:
        goal = "Execution Plan"
    if not approach:
        approach = "Step-by-step execution"

    return ExecutionPlan(goal=goal, approach=approach, steps=steps)


PLAN_GENERATION_PROMPT = """Analyze this task and create a step-by-step execution plan.

Task: {task}

IMPORTANT: Generate an ACTION plan, not a question list. Every step must be a concrete action (create, write, edit, run, install). Do NOT include steps that ask the user questions or present options. Make reasonable assumptions and proceed.

Respond with a structured plan in this format:
# [Task Title]

Approach: [1-2 sentence approach description]

1. [First step description] (`file.py` if applicable)
2. [Second step description] (`file.py` if applicable)
3. [Continue as needed...]

Keep steps concrete and actionable. Include file paths where relevant. 5-10 steps max. Every step should be something YOU execute, not something you ask the user about."""
