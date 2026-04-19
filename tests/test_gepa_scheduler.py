"""Tests for the daemon's weekly GEPA scheduler (#4 fix)."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch


def _make_daemon():
    """Minimal AuraDaemon stub — bypass real __init__ to avoid IPC/event setup."""
    from aura_daemon import AuraDaemon
    d = AuraDaemon.__new__(AuraDaemon)
    d._gepa_running = False
    d._agent = MagicMock()  # presence matters, not identity
    d._last_gepa_date = None
    d._event_bus = MagicMock()
    d._save_daemon_state = MagicMock()
    return d


def test_gepa_skipped_when_agent_not_loaded():
    d = _make_daemon()
    d._agent = None
    with patch("threading.Thread") as mock_thread:
        d._check_gepa_time()
    mock_thread.assert_not_called()


def test_gepa_skipped_outside_gepa_hour():
    d = _make_daemon()
    fake_now = MagicMock()
    fake_now.hour = 15  # not GEPA_HOUR
    fake_now.date.return_value = date.today()
    with patch("aura_daemon.datetime") as mock_dt, patch("threading.Thread") as mock_thread:
        mock_dt.now.return_value = fake_now
        d._check_gepa_time()
    mock_thread.assert_not_called()


def test_gepa_skipped_if_already_ran_within_interval():
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    d._last_gepa_date = date.today() - timedelta(days=3)  # < 7 day interval
    fake_now = MagicMock()
    fake_now.hour = AuraDaemon.GEPA_HOUR
    fake_now.date.return_value = date.today()
    with patch("aura_daemon.datetime") as mock_dt, patch("threading.Thread") as mock_thread:
        mock_dt.now.return_value = fake_now
        d._check_gepa_time()
    mock_thread.assert_not_called()


def test_gepa_fires_when_window_reached_and_stale():
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    d._last_gepa_date = date.today() - timedelta(days=10)  # older than interval
    fake_now = MagicMock()
    fake_now.hour = AuraDaemon.GEPA_HOUR
    fake_now.date.return_value = date.today()
    with patch("aura_daemon.datetime") as mock_dt, patch("threading.Thread") as mock_thread:
        mock_dt.now.return_value = fake_now
        d._check_gepa_time()
    assert d._gepa_running is True
    assert d._last_gepa_date == date.today()
    d._save_daemon_state.assert_called_once()
    mock_thread.assert_called_once()


def test_gepa_fires_on_first_run_ever():
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    d._last_gepa_date = None
    fake_now = MagicMock()
    fake_now.hour = AuraDaemon.GEPA_HOUR
    fake_now.date.return_value = date.today()
    with patch("aura_daemon.datetime") as mock_dt, patch("threading.Thread"):
        mock_dt.now.return_value = fake_now
        d._check_gepa_time()
    assert d._gepa_running is True


def test_gepa_reentry_prevented_while_running():
    from aura_daemon import AuraDaemon
    d = _make_daemon()
    d._gepa_running = True  # simulated: already running
    fake_now = MagicMock()
    fake_now.hour = AuraDaemon.GEPA_HOUR
    fake_now.date.return_value = date.today()
    with patch("aura_daemon.datetime") as mock_dt, patch("threading.Thread") as mock_thread:
        mock_dt.now.return_value = fake_now
        d._check_gepa_time()
    mock_thread.assert_not_called()
