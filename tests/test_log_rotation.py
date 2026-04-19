"""Smoke test that log files use RotatingFileHandler, not plain FileHandler."""

from pathlib import Path


def test_aura_daemon_uses_rotating_handler():
    src = (Path(__file__).resolve().parents[1] / "aura_daemon.py").read_text(encoding="utf-8")
    assert "RotatingFileHandler(" in src, "aura_daemon.py should use RotatingFileHandler"
    assert "maxBytes=" in src, "rotation must have a size cap"
    assert "backupCount=" in src, "rotation must keep backups"


def test_run_telegram_uses_rotating_handler():
    src = (Path(__file__).resolve().parents[1] / "run_telegram.py").read_text(encoding="utf-8")
    assert "RotatingFileHandler(" in src, "run_telegram.py should use RotatingFileHandler"
    # Telegram runner only rotates on server (when _log_dir exists) — still must be present.
    assert "maxBytes=" in src
    assert "backupCount=" in src


def test_rotating_handler_is_importable():
    from logging.handlers import RotatingFileHandler  # noqa: F401
