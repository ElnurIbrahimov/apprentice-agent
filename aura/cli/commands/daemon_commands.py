"""CLI subcommands for the Aura background daemon.

`aura start`  — detach and launch `aura_daemon.py`.
`aura stop`   — find PID, signal TERM, fall back to KILL.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from aura_daemon import PID_FILE, AuraDaemon

try:
    from aura.cli.display import console
except ImportError:
    from rich.console import Console
    console = Console()

logger = logging.getLogger(__name__)


def _daemon_entrypoint() -> str:
    """Absolute path to the aura_daemon.py entrypoint."""
    return str(Path(__file__).resolve().parents[3] / "aura_daemon.py")


def cmd_start(args: argparse.Namespace) -> int:
    """Detach a background AuraDaemon process. Reports PID + logfile."""
    if AuraDaemon.is_running():
        try:
            pid = PID_FILE.read_text().strip()
            console.print(f"  [yellow]Already running[/] (PID: {pid})")
        except Exception:
            console.print("  [yellow]Already running[/] (PID file unreadable)")
        return 0

    entry = _daemon_entrypoint()
    if not Path(entry).exists():
        console.print(f"  [red]Daemon entrypoint missing:[/] {entry}")
        return 1

    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL, "close_fds": True}
    if sys.platform.startswith("win"):
        # Fully detach from the current console on Windows
        flags = 0
        for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
            flags |= getattr(subprocess, name, 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    try:
        subprocess.Popen([sys.executable, entry], **kwargs)
    except Exception as e:
        console.print(f"  [red]Launch failed:[/] {e}")
        return 1

    # Poll for the PID file, up to 5s
    for _ in range(25):
        time.sleep(0.2)
        if AuraDaemon.is_running():
            try:
                pid = PID_FILE.read_text().strip()
            except Exception:
                pid = "?"
            console.print(f"  [green]Daemon started[/]  PID [cyan]{pid}[/]  (log: ~/.aura_daemon.log)")
            return 0

    console.print("  [yellow]Started, but PID file not yet visible. Check `aura status`.[/]")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Signal the daemon to exit, then remove the PID file if stuck."""
    if not AuraDaemon.is_running():
        console.print("  [dim]Not running[/]")
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return 0

    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception as e:
        console.print(f"  [red]PID file unreadable:[/] {e}")
        return 1

    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
            console.print(f"  [green]Stopped[/] (PID {pid})")
        except psutil.TimeoutExpired:
            proc.kill()
            console.print(f"  [yellow]Force-killed[/] (PID {pid})")
    except Exception:
        logger.debug("daemon_stop_psutil_fallback", exc_info=True)
        try:
            if sys.platform.startswith("win"):
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, 15)
                time.sleep(2)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, 9)  # still alive
                except ProcessLookupError:
                    pass
            console.print(f"  [green]Stopped[/] (PID {pid})")
        except Exception as e:
            console.print(f"  [red]Stop failed:[/] {e}")
            return 1

    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return 0
