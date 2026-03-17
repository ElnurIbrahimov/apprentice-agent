"""Tests for background agent mode."""
import pytest
import time
from aura.cli.background import (
    BackgroundTask, BackgroundManager, TaskState,
    render_tasks_table, create_background_indicator,
)
from rich.console import Console
from io import StringIO

def test_submit_and_complete():
    mgr = BackgroundManager()
    def mock_exec(prompt):
        time.sleep(0.05)
        return {"success": True, "response": "Done", "iterations": 3}
    task = mgr.submit("do something", mock_exec)
    assert task is not None
    assert task.state == TaskState.RUNNING
    time.sleep(0.2)
    assert task.state == TaskState.COMPLETED
    assert "Done" in task.result

def test_submit_failure():
    mgr = BackgroundManager()
    def mock_fail(prompt):
        return {"success": False, "error": "Broke"}
    task = mgr.submit("fail", mock_fail)
    time.sleep(0.2)
    assert task.state == TaskState.FAILED
    assert "Broke" in task.error

def test_submit_exception():
    mgr = BackgroundManager()
    def mock_crash(prompt):
        raise RuntimeError("Boom")
    task = mgr.submit("crash", mock_crash)
    time.sleep(0.2)
    assert task.state == TaskState.FAILED

def test_max_tasks():
    mgr = BackgroundManager(max_tasks=2)
    def slow(prompt):
        time.sleep(1)
        return {"success": True, "response": "ok"}
    t1 = mgr.submit("a", slow)
    t2 = mgr.submit("b", slow)
    t3 = mgr.submit("c", slow)  # should fail
    assert t1 is not None
    assert t2 is not None
    assert t3 is None

def test_list_tasks():
    mgr = BackgroundManager()
    def fast(prompt):
        return {"success": True, "response": "ok"}
    mgr.submit("first", fast)
    time.sleep(0.1)
    mgr.submit("second", fast)
    time.sleep(0.1)
    tasks = mgr.list_tasks()
    assert len(tasks) == 2

def test_cancel():
    mgr = BackgroundManager()
    def slow(prompt):
        time.sleep(5)
        return {"success": True, "response": "ok"}
    task = mgr.submit("slow", slow)
    assert mgr.cancel(task.id)
    assert task.state == TaskState.FAILED
    assert "Cancelled" in task.error

def test_completion_callback():
    completed = []
    mgr = BackgroundManager()
    mgr.set_completion_callback(lambda t: completed.append(t.id))
    def fast(prompt):
        return {"success": True, "response": "ok"}
    task = mgr.submit("test", fast)
    time.sleep(0.2)
    assert task.id in completed

def test_render_tasks():
    console = Console(file=StringIO(), force_terminal=True, width=100)
    tasks = [
        BackgroundTask(id="bg_abc", prompt="Do something", state=TaskState.COMPLETED, start_time=time.time()-10, end_time=time.time()),
    ]
    render_tasks_table(console, tasks)
    assert "Do something" in console.file.getvalue()

def test_background_indicator():
    mgr = BackgroundManager()
    assert create_background_indicator(mgr) == ""
    def slow(p):
        time.sleep(5)
        return {"success": True}
    mgr.submit("test", slow)
    indicator = create_background_indicator(mgr)
    assert "1" in indicator
    assert "bg" in indicator

def test_elapsed_str():
    task = BackgroundTask(id="t", prompt="p", start_time=time.time()-65, end_time=time.time())
    assert "1m" in task.elapsed_str
