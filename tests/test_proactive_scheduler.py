"""Tests for the daemon's ProactiveAwarenessEngine background tick."""

from unittest.mock import MagicMock, patch


def _make_daemon():
    """Construct AuraDaemon stub with just the attributes the tick methods read."""
    from aura_daemon import AuraDaemon
    d = AuraDaemon.__new__(AuraDaemon)
    d._proactive_running = False
    d._agent = MagicMock()
    d._event_bus = MagicMock()
    return d


def test_tick_skipped_when_agent_missing():
    d = _make_daemon()
    d._agent = None
    with patch("threading.Thread") as mock_thread:
        d._tick_proactive()
    mock_thread.assert_not_called()
    assert d._proactive_running is False


def test_tick_skipped_when_already_running():
    d = _make_daemon()
    d._proactive_running = True
    with patch("threading.Thread") as mock_thread:
        d._tick_proactive()
    mock_thread.assert_not_called()


def test_tick_starts_background_thread_when_ready():
    d = _make_daemon()
    with patch("threading.Thread") as mock_thread:
        d._tick_proactive()
    assert d._proactive_running is True
    mock_thread.assert_called_once()
    # Thread should be started as daemon
    kwargs = mock_thread.call_args.kwargs
    assert kwargs.get("daemon") is True


def test_run_skipped_with_too_few_projects():
    d = _make_daemon()
    d._proactive_running = True  # simulate tick just fired
    fake_wm = MagicMock()
    fake_wm.get_all_projects.return_value = [object(), object()]  # only 2

    with patch.dict("sys.modules", {
        "aura.consciousness.world_model": MagicMock(get_world_model=lambda: fake_wm),
    }):
        with patch("aura.consciousness.proactive_awareness.get_proactive_awareness_engine") as mock_engine:
            d._run_proactive_analysis()
    mock_engine.assert_not_called()
    assert d._proactive_running is False  # always cleared in finally


def test_run_calls_engine_when_projects_sufficient():
    d = _make_daemon()
    d._proactive_running = True
    fake_wm = MagicMock()
    fake_wm.get_all_projects.return_value = [object()] * 5

    fake_engine = MagicMock()
    fake_engine.run_full_analysis.return_value = []

    with patch.dict("sys.modules", {
        "aura.consciousness.world_model": MagicMock(get_world_model=lambda: fake_wm),
        "aura.consciousness.proactive_awareness": MagicMock(get_proactive_awareness_engine=lambda: fake_engine),
    }):
        d._run_proactive_analysis()
    fake_engine.run_full_analysis.assert_called_once()
    assert d._proactive_running is False


def test_tick_proactive_constant_is_30_minutes():
    from aura_daemon import AuraDaemon
    assert AuraDaemon.TICK_PROACTIVE == 1800
