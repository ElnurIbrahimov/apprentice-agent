"""Tests for editable plan mode."""
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from aura.cli.plan_mode import (
    PLAN_GENERATION_PROMPT,
    ExecutionPlan,
    PlanStep,
    StepStatus,
    edit_plan_text,
    parse_plan_from_llm,
    render_plan,
    show_plan_approval,
)
from aura.core.agentic_loop_support import RecallResult


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

# ── Tests for plan-approve-execute helpers ──

def test_show_plan_approval_yes():
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[PlanStep(index=1, description="Step 1")]
    )
    console = Console(file=StringIO(), force_terminal=True, width=80)
    with patch("builtins.input", return_value="y"):
        result = show_plan_approval(console, plan)
    assert result == "y"

def test_show_plan_approval_no():
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[PlanStep(index=1, description="Step 1")]
    )
    console = Console(file=StringIO(), force_terminal=True, width=80)
    with patch("builtins.input", return_value="n"):
        result = show_plan_approval(console, plan)
    assert result == "n"

def test_show_plan_approval_edit():
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[PlanStep(index=1, description="Step 1")]
    )
    console = Console(file=StringIO(), force_terminal=True, width=80)
    with patch("builtins.input", return_value="e"):
        result = show_plan_approval(console, plan)
    assert result == "e"

def test_show_plan_approval_eof():
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[PlanStep(index=1, description="Step 1")]
    )
    console = Console(file=StringIO(), force_terminal=True, width=80)
    with patch("builtins.input", side_effect=EOFError):
        result = show_plan_approval(console, plan)
    assert result == "n"

def test_show_plan_approval_keyboard_interrupt():
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[PlanStep(index=1, description="Step 1")]
    )
    console = Console(file=StringIO(), force_terminal=True, width=80)
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        result = show_plan_approval(console, plan)
    assert result == "n"

def test_show_plan_approval_yes_verbose():
    """'yes' (full word) should also be accepted."""
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[PlanStep(index=1, description="Step 1")]
    )
    console = Console(file=StringIO(), force_terminal=True, width=80)
    with patch("builtins.input", return_value="yes"):
        result = show_plan_approval(console, plan)
    assert result == "y"

def test_show_plan_approval_random_input():
    """Random input should be treated as 'no'."""
    plan = ExecutionPlan(
        goal="Test", approach="Testing",
        steps=[PlanStep(index=1, description="Step 1")]
    )
    console = Console(file=StringIO(), force_terminal=True, width=80)
    with patch("builtins.input", return_value="maybe"):
        result = show_plan_approval(console, plan)
    assert result == "n"

def test_edit_plan_text_editor_fails():
    """If the editor fails, original text should be returned."""
    console = Console(file=StringIO(), force_terminal=True, width=80)
    original = "# Plan\n1. Step one"
    with patch("subprocess.call", side_effect=FileNotFoundError("no editor")):
        result = edit_plan_text(console, original)
    assert result == original


# ── plan_first tests (agentic loop) ─────────────────────────────────

@patch("aura.core.agentic_loop._recall_memories", return_value=RecallResult("", 0, ""))
@patch("aura.core.agentic_loop.os.path.getmtime", side_effect=OSError)
def test_plan_first_returns_plan_dict(mock_getmtime, mock_recall):
    """plan_first should return a dict with plan_text, plan, and prompt."""
    from unittest.mock import MagicMock

    from aura.core.agentic_loop import AgenticLoop

    brain = MagicMock()
    brain.think = MagicMock(return_value=(
        "# Fix the Bug\n\n"
        "Approach: Debug and fix.\n\n"
        "1. Find the bug (`main.py`)\n"
        "2. Fix it (`main.py`)\n"
        "3. Run tests\n"
    ))

    loop = AgenticLoop.__new__(AgenticLoop)
    loop.brain = brain
    loop.context = ""
    loop.tools = {}
    loop.memory = None
    loop.project_root = "."
    loop._current_action_mode = None
    loop._hot_files = {}
    loop._planner = None

    result = loop.plan_first("fix the login bug")
    assert "plan_text" in result
    assert "plan" in result
    assert "prompt" in result
    assert result["prompt"] == "fix the login bug"
    assert result["plan"] is not None
    assert len(result["plan"].steps) == 3
    assert "error" not in result


@patch("aura.core.agentic_loop._recall_memories", return_value=RecallResult("", 0, ""))
@patch("aura.core.agentic_loop.os.path.getmtime", side_effect=OSError)
def test_plan_first_empty_llm_response(mock_getmtime, mock_recall):
    """plan_first with empty LLM response should still return a plan (with defaults)."""
    from unittest.mock import MagicMock

    from aura.core.agentic_loop import AgenticLoop

    brain = MagicMock()
    brain.think = MagicMock(return_value="")

    loop = AgenticLoop.__new__(AgenticLoop)
    loop.brain = brain
    loop.context = ""
    loop.tools = {}
    loop.memory = None
    loop.project_root = "."
    loop._current_action_mode = None
    loop._hot_files = {}
    loop._planner = None

    result = loop.plan_first("do something")
    assert "plan" in result
    # Even with empty response, parse_plan_from_llm returns a plan with empty steps
    assert result["plan"] is not None
    assert result["plan"].steps == []


def test_plan_first_llm_error():
    """plan_first should catch LLM errors and return error key."""
    from unittest.mock import MagicMock

    from aura.core.agentic_loop import AgenticLoop

    brain = MagicMock()
    brain.think = MagicMock(side_effect=ConnectionError("offline"))

    loop = AgenticLoop.__new__(AgenticLoop)
    loop.brain = brain
    loop.context = ""
    loop.tools = {}
    loop.memory = None
    loop.project_root = "."
    loop._current_action_mode = None
    loop._hot_files = {}
    loop._planner = None

    result = loop.plan_first("test")
    assert "error" in result
    assert "offline" in result["error"]


# ── Permission mode cycling includes PLAN_APPROVE ────────────────────

def test_permission_mode_cycle_includes_plan_approve():
    from aura.cli.permissions_ui import _MODE_ORDER, PermissionMode, cycle_permission_mode

    assert PermissionMode.PLAN_APPROVE in _MODE_ORDER

    # Cycling from CAREFUL should reach AUTO_EDIT (order: careful -> auto_edit -> plan_approve -> full_auto)
    next_mode = cycle_permission_mode(PermissionMode.CAREFUL.value)
    assert next_mode == PermissionMode.AUTO_EDIT.value

    # Cycling from PLAN_APPROVE should reach FULL_AUTO
    next_mode2 = cycle_permission_mode(PermissionMode.PLAN_APPROVE.value)
    assert next_mode2 == PermissionMode.FULL_AUTO.value


def test_plan_approve_mode_description():
    from aura.cli.permissions_ui import PermissionMode, get_mode_description
    desc = get_mode_description(PermissionMode.PLAN_APPROVE.value)
    assert "plan" in desc.lower() or "approve" in desc.lower()
