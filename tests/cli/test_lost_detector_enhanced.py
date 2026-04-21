"""Tests for the two new LostDetector heuristics added to SessionLoopGuard:
iterations-without-progress and wall-clock-without-progress."""
from __future__ import annotations

import time
from unittest.mock import patch

from aura.reliability.loop_guard import SessionLoopGuard


def _fresh_guard(iter_limit: int = 3, seconds_limit: float = 60.0) -> SessionLoopGuard:
    g = SessionLoopGuard(session_id="test")
    g.configure(
        no_progress_iter_limit=iter_limit,
        max_seconds_without_progress=seconds_limit,
    )
    return g


def test_note_progress_resets_iteration_counter():
    g = _fresh_guard(iter_limit=3)
    g.note_iteration()
    g.note_iteration()
    assert g._iterations_without_progress == 2
    g.note_progress()
    assert g._iterations_without_progress == 0


def test_no_progress_iteration_limit_trips():
    g = _fresh_guard(iter_limit=3)
    r1 = g.note_iteration()
    r2 = g.note_iteration()
    r3 = g.note_iteration()
    assert r1.triggered is False
    assert r2.triggered is False
    # Third iteration without progress trips the guard.
    assert r3.triggered is True
    assert r3.reason == "no_progress_iterations"


def test_seconds_without_progress_trips():
    g = _fresh_guard(iter_limit=999, seconds_limit=0.01)
    # Simulate time passing beyond the threshold.
    g._last_progress_time = time.monotonic() - 1.0
    r = g.note_iteration()
    assert r.triggered is True
    assert r.reason == "no_progress_timeout"


def test_note_progress_prevents_trip():
    g = _fresh_guard(iter_limit=3)
    g.note_iteration()
    g.note_iteration()
    g.note_progress()
    # After progress note, counter resets; should take 3 more to trip.
    r1 = g.note_iteration()
    r2 = g.note_iteration()
    assert r1.triggered is False
    assert r2.triggered is False


def test_disabled_guard_never_trips():
    g = SessionLoopGuard(session_id="test")
    g._enabled = False
    g.configure(no_progress_iter_limit=1, max_seconds_without_progress=0.001)
    for _ in range(5):
        r = g.note_iteration()
        assert r.triggered is False


def test_reset_clears_progress_state():
    g = _fresh_guard(iter_limit=3)
    g.note_iteration()
    g.note_iteration()
    assert g._iterations_without_progress == 2
    g.reset()
    assert g._iterations_without_progress == 0


def test_configure_rejects_nonpositive_values():
    g = _fresh_guard(iter_limit=4)
    g.configure(no_progress_iter_limit=0, max_seconds_without_progress=-1.0)
    # Unchanged because values were invalid.
    assert g._no_progress_iter_limit == 4
