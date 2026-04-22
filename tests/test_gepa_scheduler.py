"""Tests for the daemon's weekly GEPA scheduler.

Covers the hysteresis-based dedup that replaced the date-only comparison
(prevents same-day double-fire on NTP backward jumps / VM snapshot restores).
"""

import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


def _make_daemon():
    """Minimal AuraDaemon stub — bypass real __init__ to avoid IPC/event setup."""
    from aura_daemon import AuraDaemon
    d = AuraDaemon.__new__(AuraDaemon)
    d._gepa_running = False
    d._agent = MagicMock()  # presence matters, not identity
    d._last_gepa_run_at = 0.0
    d._bg_threads = []
    d._bg_threads_lock = threading.Lock()
    d._event_bus = MagicMock()
    d._save_daemon_state = MagicMock()
    return d


def _fixed_now(dt: datetime):
    """Build a MagicMock that stands in for datetime.now() and still exposes
    .hour and .timestamp() on its return value."""
    fake_now = MagicMock()
    fake_now.hour = dt.hour
    fake_now.timestamp.return_value = dt.timestamp()
    return fake_now


def test_gepa_skipped_when_agent_not_loaded():
    d = _make_daemon()
    d._agent = None
    with patch.object(type(d), "_spawn_bg") as mock_spawn:
        d._check_gepa_time()
    mock_spawn.assert_not_called()


def test_gepa_skipped_outside_gepa_hour():
    d = _make_daemon()
    with patch("aura_daemon.datetime") as mock_dt, \
         patch.object(type(d), "_spawn_bg") as mock_spawn:
        mock_dt.now.return_value = _fixed_now(datetime(2026, 4, 22, 15, 0, 0))
        d._check_gepa_time()
    mock_spawn.assert_not_called()


def test_gepa_skipped_if_already_ran_within_interval():
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    # Ran 3 days ago — inside the 7-day-minus-1h hysteresis window.
    d._last_gepa_run_at = time.time() - 3 * 86400
    now = datetime(2026, 4, 22, AuraDaemon.GEPA_HOUR, 5, 0)
    with patch("aura_daemon.datetime") as mock_dt, \
         patch.object(type(d), "_spawn_bg") as mock_spawn:
        mock_dt.now.return_value = _fixed_now(now)
        d._check_gepa_time()
    mock_spawn.assert_not_called()
    assert d._gepa_running is False


def test_gepa_fires_when_window_reached_and_stale():
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    # Ran 10 days ago — well past the interval.
    d._last_gepa_run_at = time.time() - 10 * 86400
    now = datetime(2026, 4, 22, AuraDaemon.GEPA_HOUR, 0, 0)
    with patch("aura_daemon.datetime") as mock_dt, \
         patch.object(type(d), "_spawn_bg") as mock_spawn:
        mock_dt.now.return_value = _fixed_now(now)
        d._check_gepa_time()
    assert d._gepa_running is True
    assert abs(d._last_gepa_run_at - now.timestamp()) < 1.0
    d._save_daemon_state.assert_called_once()
    mock_spawn.assert_called_once()


def test_gepa_fires_on_first_run_ever():
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    d._last_gepa_run_at = 0.0
    now = datetime(2026, 4, 22, AuraDaemon.GEPA_HOUR, 0, 0)
    with patch("aura_daemon.datetime") as mock_dt, \
         patch.object(type(d), "_spawn_bg"):
        mock_dt.now.return_value = _fixed_now(now)
        d._check_gepa_time()
    assert d._gepa_running is True


def test_gepa_reentry_prevented_while_running():
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    d._gepa_running = True
    now = datetime(2026, 4, 22, AuraDaemon.GEPA_HOUR, 0, 0)
    with patch("aura_daemon.datetime") as mock_dt, \
         patch.object(type(d), "_spawn_bg") as mock_spawn:
        mock_dt.now.return_value = _fixed_now(now)
        d._check_gepa_time()
    mock_spawn.assert_not_called()


def test_gepa_hysteresis_blocks_same_day_double_fire():
    """If GEPA just ran and the wall clock jumps backward (NTP / VM snapshot),
    the next invocation must still refuse to run until ~7 days have passed."""
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    now = datetime(2026, 4, 22, AuraDaemon.GEPA_HOUR, 0, 0)
    # Simulate GEPA ran 5 minutes ago.
    d._last_gepa_run_at = now.timestamp() - 300
    with patch("aura_daemon.datetime") as mock_dt, \
         patch.object(type(d), "_spawn_bg") as mock_spawn:
        mock_dt.now.return_value = _fixed_now(now)
        d._check_gepa_time()
    mock_spawn.assert_not_called()
