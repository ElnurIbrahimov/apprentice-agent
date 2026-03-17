"""Tests for editable plan mode."""
import pytest
from aura.cli.plan_mode import (
    PlanStep, ExecutionPlan, StepStatus,
    parse_plan_from_llm, render_plan, PLAN_GENERATION_PROMPT,
)
from rich.console import Console
from io import StringIO

def test_plan_step_defaults():
    step = PlanStep(index=1, description="Do something")
    assert step.status == StepStatus.PENDING
    assert step.files == []

def test_execution_plan_progress():
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[
            PlanStep(index=1, description="A", status=StepStatus.DONE),
            PlanStep(index=2, description="B", status=StepStatus.PENDING),
            PlanStep(index=3, description="C", status=StepStatus.PENDING),
        ]
    )
    assert plan.progress == "1/3"
    assert not plan.is_complete
    assert plan.current_step.index == 2

def test_execution_plan_complete():
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[
            PlanStep(index=1, description="A", status=StepStatus.DONE),
            PlanStep(index=2, description="B", status=StepStatus.DONE),
        ]
    )
    assert plan.is_complete

def test_plan_to_markdown():
    plan = ExecutionPlan(
        goal="Fix bug", approach="Debug and fix",
        steps=[
            PlanStep(index=1, description="Find the bug", status=StepStatus.DONE),
            PlanStep(index=2, description="Fix it", status=StepStatus.PENDING, files=["main.py"]),
        ]
    )
    md = plan.to_markdown()
    assert "Fix bug" in md
    assert "[x]" in md
    assert "[ ]" in md
    assert "main.py" in md

def test_parse_plan_from_llm():
    response = """# Fix the Login Bug

Approach: Debug the auth flow and fix the token refresh.

1. Read the auth middleware (`auth.py`)
2. Add token refresh logic (`auth.py`, `token.py`)
3. Write tests for the fix (`tests/test_auth.py`)
4. Run tests and verify"""

    plan = parse_plan_from_llm(response)
    assert plan.goal == "Fix the Login Bug"
    assert "token refresh" in plan.approach.lower() or "auth" in plan.approach.lower()
    assert len(plan.steps) == 4
    assert "auth.py" in plan.steps[0].files
    assert plan.steps[0].status == StepStatus.PENDING

def test_parse_plan_checkbox_format():
    response = """# Deploy
Approach: CI/CD

- [ ] Build the app
- [ ] Run tests
- [ ] Deploy to staging"""

    plan = parse_plan_from_llm(response)
    assert len(plan.steps) == 3

def test_render_plan():
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[PlanStep(index=1, description="Do it", status=StepStatus.RUNNING)]
    )
    console = Console(file=StringIO(), force_terminal=True, width=80)
    render_plan(console, plan)
    output = console.file.getvalue()
    assert "Test" in output
    assert "Do it" in output

def test_plan_generation_prompt():
    prompt = PLAN_GENERATION_PROMPT.format(task="fix the login bug")
    assert "fix the login bug" in prompt
