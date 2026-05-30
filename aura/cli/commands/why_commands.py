"""/why <file> — Intent-to-Code Ledger v1.

Given a file path, show the timeline of edits Aura made to it, with the
triggering user prompt, model, and session. Reads from
`data/events/edits.jsonl` (append-only log written by
`aura.core.event_log.log_edit`) and enriches with session metadata from
`aura.core.session.AgenticSession.load`.

v1 scope — file-level. Line-level queries (`/why foo.py:42`) are v2 and
need diff-aware line tracking.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from ..display import console
from .common import command, TIER_BETA, TIER_EXPERIMENTAL, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/why",      "Show edit history + triggering prompts for a file", tier=TIER_EXPERIMENTAL)


def handle_why(agent, arg, context) -> Optional[str]:
    """Show the edit history of a file with triggering prompts."""
    file = (arg or "").strip()
    if not file:
        console.print("  [dim]Usage: /why <file>[/dim]")
        return None

    # Strip any ":N" line suffix — v1 ignores it but we accept the syntax
    # so users coming from v2 don't get confused.
    if ":" in file and not file.startswith(("/", "\\")) and not (len(file) > 1 and file[1] == ":"):
        # Preserve Windows drive letters (e.g. "C:\...") but split trailing ":42"
        pass
    file_path_part, line_part = _split_line_suffix(file)

    abs_path = _resolve_path(file_path_part)
    if line_part is not None:
        console.print(
            f"  [dim yellow]Note:[/dim yellow] line-level /why is v2 — showing file-level history for {os.path.basename(abs_path)}."
        )

    records = _read_edit_records(abs_path)
    if not records:
        console.print(
            f"  [dim]No Aura edit history recorded for "
            f"{os.path.basename(abs_path)}.[/dim]"
        )
        # If the file exists and is in a git repo, git blame may still help.
        git = _git_blame_summary(abs_path)
        if git:
            _render_git_section(git)
        return None

    sessions = _group_and_enrich(records)
    _render_why_panel(abs_path, sessions)

    git = _git_blame_summary(abs_path)
    if git:
        _render_git_section(git)
    return None


# ── helpers ───────────────────────────────────────────────────────────────


def _split_line_suffix(s: str) -> tuple[str, Optional[int]]:
    """Split 'foo.py:42' into ('foo.py', 42). Returns ('foo.py', None) for 'foo.py'.

    Handles Windows drive prefixes (C:\\foo.py:42 → ('C:\\foo.py', 42)).
    """
    if ":" not in s:
        return s, None
    # Windows drive: the first colon at index 1 belongs to the drive letter.
    drive_prefix = ""
    rest = s
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        drive_prefix = s[:2]
        rest = s[2:]
    if ":" not in rest:
        return s, None
    path_part, _, maybe_line = rest.rpartition(":")
    try:
        return drive_prefix + path_part, int(maybe_line)
    except ValueError:
        return s, None


def _resolve_path(file: str) -> str:
    """Normalize to absolute; tolerate missing file (log may still have it)."""
    try:
        p = Path(file).expanduser()
        return str(p.resolve() if p.exists() else p.absolute())
    except (OSError, ValueError):
        return file


def _edits_log_path() -> Path:
    data_dir = os.environ.get("AURA_DATA_DIR", "data")
    return Path(data_dir) / "events" / "edits.jsonl"


def _normalize_for_match(p: str) -> str:
    """Return a comparable form: absolute, case-folded on Windows, forward slashes."""
    try:
        abs_p = str(Path(p).resolve() if Path(p).exists() else Path(p).absolute())
    except (OSError, ValueError):
        abs_p = p
    abs_p = abs_p.replace("\\", "/")
    if os.name == "nt":
        abs_p = abs_p.lower()
    return abs_p


def _read_edit_records(abs_path: str) -> list[dict]:
    """Stream edits.jsonl, returning records whose path matches *abs_path*."""
    log_path = _edits_log_path()
    if not log_path.is_file():
        return []
    target = _normalize_for_match(abs_path)
    matches: list[dict] = []
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_path = rec.get("path") or ""
                if rec_path and _normalize_for_match(rec_path) == target:
                    matches.append(rec)
    except OSError:
        logger.debug("read_edit_records_failed", exc_info=True)
    return matches


def _group_and_enrich(records: list[dict]) -> list[dict]:
    """Group edits by session_id, resolve triggering prompt from AgenticSession.

    Returns groups newest-first. Each group:
      {session_id, first_ts, last_ts, model, edit_count, prompt, iterations}
    """
    groups: dict[str, dict] = {}
    for rec in records:
        sid = rec.get("session_id") or ""
        if sid not in groups:
            groups[sid] = {
                "session_id": sid,
                "first_ts": rec.get("ts", 0.0),
                "last_ts": rec.get("ts", 0.0),
                "model": rec.get("model") or "auto",
                "edit_count": 0,
                "prompt": rec.get("prompt") or "",
                "iterations": set(),
            }
        g = groups[sid]
        g["first_ts"] = min(g["first_ts"], rec.get("ts", g["first_ts"]))
        g["last_ts"] = max(g["last_ts"], rec.get("ts", g["last_ts"]))
        g["edit_count"] += 1
        it = rec.get("iteration")
        if isinstance(it, int):
            g["iterations"].add(it)
        # Prefer the longest recorded prompt from this session — edit events
        # sometimes get truncated differently depending on when in the run
        # they fire.
        if len(rec.get("prompt") or "") > len(g["prompt"]):
            g["prompt"] = rec.get("prompt") or g["prompt"]

    # Fallback prompt lookup via AgenticSession for groups that have an
    # empty or obviously truncated prompt.
    for sid, g in list(groups.items()):
        if not g["prompt"] and sid:
            prompt = _lookup_session_prompt(sid)
            if prompt:
                g["prompt"] = prompt

    return sorted(
        groups.values(), key=lambda g: g["last_ts"] or 0.0, reverse=True
    )


_SESSION_PROMPT_CACHE: dict[str, str] = {}


def _lookup_session_prompt(session_id: str) -> str:
    """Resolve the first user prompt for a session, cached across one /why call."""
    if session_id in _SESSION_PROMPT_CACHE:
        return _SESSION_PROMPT_CACHE[session_id]
    try:
        from aura.core.session import AgenticSession
        s = AgenticSession()
        s.load(session_id)
        for msg in getattr(s, "messages", []) or []:
            if msg.get("role") == "user" and msg.get("content"):
                prompt = str(msg["content"])[:500]
                _SESSION_PROMPT_CACHE[session_id] = prompt
                return prompt
    except Exception:
        logger.debug("session_prompt_lookup_failed for %s", session_id, exc_info=True)
    _SESSION_PROMPT_CACHE[session_id] = ""
    return ""


def _in_git_repo(path: str) -> bool:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=os.path.dirname(path) or ".",
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except (FileNotFoundError, OSError):
        return False


def _git_blame_summary(abs_path: str) -> Optional[dict]:
    """Return {last_commit_hash, last_commit_date, last_commit_message, top_author, top_author_pct}
    for *abs_path*, or None if the file isn't tracked or git isn't available."""
    if not _in_git_repo(abs_path):
        return None
    try:
        import subprocess
        # Last commit touching the file.
        r = subprocess.run(
            ["git", "log", "-1", "--format=%h|%ad|%s", "--date=short", "--", abs_path],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(abs_path) or ".",
        )
        last = {}
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split("|", 2)
            if len(parts) == 3:
                last = {
                    "last_commit_hash": parts[0],
                    "last_commit_date": parts[1],
                    "last_commit_message": parts[2][:80],
                }
        # Author distribution via blame.
        r2 = subprocess.run(
            ["git", "blame", "--line-porcelain", abs_path],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(abs_path) or ".",
        )
        authors: dict[str, int] = {}
        total_lines = 0
        if r2.returncode == 0:
            for line in r2.stdout.splitlines():
                if line.startswith("author "):
                    name = line[len("author "):].strip()
                    authors[name] = authors.get(name, 0) + 1
                    total_lines += 1
        top_author, top_pct = "", 0
        if authors and total_lines > 0:
            top_name, top_count = max(authors.items(), key=lambda kv: kv[1])
            top_author = top_name
            top_pct = int(round(100 * top_count / total_lines))
        result = {**last, "top_author": top_author, "top_author_pct": top_pct}
        if not any(result.values()):
            return None
        return result
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


# ── rendering ─────────────────────────────────────────────────────────────


def _render_why_panel(abs_path: str, sessions: list[dict]) -> None:
    """Render the edit-history timeline as a Rich panel."""
    import time as _t
    from rich.text import Text

    total_edits = sum(g["edit_count"] for g in sessions)
    header = Text()
    header.append("Edit history · ", style="bold")
    header.append(os.path.basename(abs_path), style="cyan bold")
    header.append(f" · {len(sessions)} session(s) · {total_edits} edit(s)\n", style="dim")
    console.print()
    console.print("  ", end="")
    console.print(header)

    for g in sessions:
        ts = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(g.get("last_ts") or 0.0))
        model = str(g.get("model") or "auto")[:28]
        edit_count = int(g.get("edit_count") or 0)
        iterations = sorted(g.get("iterations") or [])

        line = Text("  ")
        line.append(ts, style="bold")
        line.append("   ")
        line.append(f"{model:<24}", style="cyan")
        line.append(f"[{edit_count} edit{'s' if edit_count != 1 else ''}]", style="dim")
        if iterations:
            line.append(
                f"  iter {iterations[0]}-{iterations[-1]}" if len(iterations) > 1
                else f"  iter {iterations[0]}",
                style="dim",
            )
        console.print(line)

        prompt = (g.get("prompt") or "").strip().replace("\n", " ")
        if prompt:
            if len(prompt) > 110:
                prompt = prompt[:110] + "…"
            console.print(f"    [dim]↳[/dim] [italic]\"{prompt}\"[/italic]")
        sid = g.get("session_id") or ""
        if sid:
            console.print(f"    [dim]↳ session {sid[:16]}…[/dim]")
    console.print()


def _render_git_section(git: dict) -> None:
    console.print("  [bold]Git summary:[/bold]")
    last_hash = git.get("last_commit_hash")
    if last_hash:
        console.print(
            f"    [dim]Last commit:[/dim] "
            f"[cyan]{last_hash}[/cyan] "
            f"[dim]· {git.get('last_commit_date', '')} ·[/dim] "
            f"{git.get('last_commit_message', '')}"
        )
    top_author = git.get("top_author")
    pct = git.get("top_author_pct", 0)
    if top_author:
        console.print(
            f"    [dim]Top contributor:[/dim] "
            f"{top_author} [dim]({pct}%)[/dim]"
        )
    console.print()
