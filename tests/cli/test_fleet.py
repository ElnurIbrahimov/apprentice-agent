"""Tests for parallel fleet execution."""
import pytest
import time
from aura.cli.fleet import (
    SubTask, SubAgentStatus, FleetRun, FleetExecutor,
    parse_decomposition, render_fleet_dashboard, DECOMPOSITION_PROMPT,
)
from rich.console import Console
from io import StringIO

def test_subtask_defaults():
    t = SubTask(id="t1", description="Do something")
    assert t.status == SubAgentStatus.PENDING
    assert t.files == []

def test_fleet_progress():
    fleet = FleetRun(goal="test", tasks=[
        SubTask(id="t1", description="A", status=SubAgentStatus.DONE),
        SubTask(id="t2", description="B", status=SubAgentStatus.PENDING),
        SubTask(id="t3", description="C", status=SubAgentStatus.FAILED),
    ])
    assert fleet.progress == "2/3"
    assert not fleet.is_complete

def test_fleet_complete():
    fleet = FleetRun(goal="test", tasks=[
        SubTask(id="t1", description="A", status=SubAgentStatus.DONE),
        SubTask(id="t2", description="B", status=SubAgentStatus.DONE),
    ])
    assert fleet.is_complete

def test_parse_decomposition():
    response = """1. Fix the auth middleware (`auth.py`)
2. Update the tests (`tests/test_auth.py`)
3. Update the docs (`README.md`)"""
    tasks = parse_decomposition(response)
    assert len(tasks) == 3
    assert "auth.py" in tasks[0].files
    assert tasks[0].id == "t1"

def test_parse_decomposition_checkbox():
    response = """- [ ] First task
- [ ] Second task"""
    tasks = parse_decomposition(response)
    assert len(tasks) == 2

def test_fleet_executor_parallel():
    fleet = FleetRun(goal="test", tasks=[
        SubTask(id="t1", description="Task A"),
        SubTask(id="t2", description="Task B"),
    ])
    def mock_execute(prompt):
        time.sleep(0.05)
        return {"success": True, "response": f"Done: {prompt}", "iterations": 2}

    executor = FleetExecutor(max_workers=2)
    result = executor.run(fleet, mock_execute)
    assert result.is_complete
    assert all(t.status == SubAgentStatus.DONE for t in result.tasks)

def test_fleet_executor_failure():
    fleet = FleetRun(goal="test", tasks=[
        SubTask(id="t1", description="Will fail"),
    ])
    def mock_fail(prompt):
        return {"success": False, "error": "Something broke"}

    executor = FleetExecutor(max_workers=1)
    result = executor.run(fleet, mock_fail)
    assert result.tasks[0].status == SubAgentStatus.FAILED
    assert "broke" in result.tasks[0].error

def test_fleet_executor_exception():
    fleet = FleetRun(goal="test", tasks=[
        SubTask(id="t1", description="Will crash"),
    ])
    def mock_crash(prompt):
        raise RuntimeError("Boom")

    executor = FleetExecutor(max_workers=1)
    result = executor.run(fleet, mock_crash)
    assert result.tasks[0].status == SubAgentStatus.FAILED
    assert "Boom" in result.tasks[0].error

def test_render_dashboard():
    fleet = FleetRun(goal="Fix bugs", start_time=time.time(), tasks=[
        SubTask(id="t1", description="Fix A", status=SubAgentStatus.DONE, elapsed=1.5),
        SubTask(id="t2", description="Fix B", status=SubAgentStatus.RUNNING),
    ])
    console = Console(file=StringIO(), force_terminal=True, width=100)
    render_fleet_dashboard(console, fleet)
    output = console.file.getvalue()
    assert "Fix bugs" in output
    assert "Fix A" in output

def test_decomposition_prompt():
    prompt = DECOMPOSITION_PROMPT.format(task="refactor the auth system")
    assert "refactor the auth system" in prompt
