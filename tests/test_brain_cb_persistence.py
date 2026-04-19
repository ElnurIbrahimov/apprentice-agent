"""Tests for circuit breaker state persistence across restarts."""

import json
import time

import pytest


@pytest.fixture
def brain_with_tmp_data(tmp_path, monkeypatch):
    """Construct OllamaBrain with an isolated data dir."""
    monkeypatch.setenv("AURA_DATA_DIR", str(tmp_path))
    from aura.brain import OllamaBrain
    b = OllamaBrain(warmup=False)
    yield b, tmp_path
    try:
        b.close()
    except Exception:
        pass


def test_cb_state_file_absent_means_closed_circuit(brain_with_tmp_data):
    brain, _ = brain_with_tmp_data
    assert brain._consecutive_think_failures == 0
    assert brain._think_circuit_open_at == 0.0


def test_save_then_load_restores_open_circuit(tmp_path, monkeypatch):
    """A process crash while the CB is open should preserve state on restart."""
    monkeypatch.setenv("AURA_DATA_DIR", str(tmp_path))
    from aura.brain import OllamaBrain

    # First process: open the circuit and persist
    b1 = OllamaBrain(warmup=False)
    with b1._cb_lock:
        b1._consecutive_think_failures = 3
        b1._think_circuit_open_at = time.time()
        b1._save_cb_state()
    b1.close()

    state_file = tmp_path / "brain_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["failures"] == 3

    # Second process: should restore state if still within cooldown
    b2 = OllamaBrain(warmup=False)
    assert b2._consecutive_think_failures == 3
    b2.close()


def test_stale_cb_state_is_discarded(tmp_path, monkeypatch):
    """State older than cooldown should be cleared on load, not honored."""
    monkeypatch.setenv("AURA_DATA_DIR", str(tmp_path))
    state_file = tmp_path / "brain_state.json"
    state_file.write_text(json.dumps({
        "failures": 5,
        "open_at": time.time() - 3600,  # 1 hour ago, way past 30s cooldown
    }))

    from aura.brain import OllamaBrain
    b = OllamaBrain(warmup=False)
    assert b._consecutive_think_failures == 0
    assert not state_file.exists(), "Stale state file should be cleaned up"
    b.close()


def test_save_with_zero_failures_removes_file(tmp_path, monkeypatch):
    """When the circuit resets, the state file should be deleted."""
    monkeypatch.setenv("AURA_DATA_DIR", str(tmp_path))
    from aura.brain import OllamaBrain

    b = OllamaBrain(warmup=False)
    with b._cb_lock:
        b._consecutive_think_failures = 2
        b._save_cb_state()
    state_file = tmp_path / "brain_state.json"
    assert state_file.exists()

    with b._cb_lock:
        b._consecutive_think_failures = 0
        b._save_cb_state()
    assert not state_file.exists()
    b.close()


def test_corrupt_state_file_is_tolerated(tmp_path, monkeypatch):
    """Garbage in the state file must not crash startup."""
    monkeypatch.setenv("AURA_DATA_DIR", str(tmp_path))
    state_file = tmp_path / "brain_state.json"
    state_file.write_text("not json {{{{")

    from aura.brain import OllamaBrain
    b = OllamaBrain(warmup=False)  # must not raise
    assert b._consecutive_think_failures == 0
    b.close()
