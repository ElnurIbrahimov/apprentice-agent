"""API endpoints for new AURA tools: Calendar, Spaced Repetition, Email, Screen Reader, Shell."""

import asyncio
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from api.auth import require_api_key
from api.utils import call_tool, get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tools"], dependencies=[Depends(require_api_key)])


def _safe_error(e: Exception, context: str = "") -> str:
    """Log full error internally, return generic message to client."""
    logger.error(f"[API] {context}: {e}", exc_info=True)
    return "Internal error — check server logs"


# ============================================================================
# CALENDAR
# ============================================================================

class AddEventRequest(BaseModel):
    title: str = Field(..., max_length=500)
    start: str = Field(..., max_length=64)
    end: Optional[str] = Field(None, max_length=64)
    description: str = Field("", max_length=5000)
    location: str = Field("", max_length=1000)
    recurrence: Optional[str] = Field(None, max_length=200)
    reminders: Optional[List[int]] = None


@router.get("/calendar/today")
async def calendar_today():
    """Get today's events."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _calendar_today_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "calendar_today")}


def _calendar_today_sync() -> dict:
    return call_tool("calendar", "today")


@router.get("/calendar/upcoming")
async def calendar_upcoming(days: int = 7):
    """Get upcoming events."""
    days = max(1, min(days, 365))  # Clamp to prevent abuse
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _calendar_upcoming_sync(days))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "calendar_upcoming")}


def _calendar_upcoming_sync(days: int) -> dict:
    return call_tool("calendar", "upcoming", days=days)


@router.post("/calendar/add")
async def calendar_add(request: AddEventRequest):
    """Add a calendar event."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _calendar_add_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "calendar_add")}


def _calendar_add_sync(request: AddEventRequest) -> dict:
    agent = get_agent()
    if "calendar" in agent.tools:
        return agent.tools["calendar"].add_event(
            title=request.title,
            start=request.start,
            end=request.end,
            description=request.description,
            location=request.location,
            recurrence=request.recurrence,
            reminders=request.reminders,
        )
    return {"success": False, "error": "Calendar tool not loaded"}


@router.delete("/calendar/{event_id}")
async def calendar_remove(event_id: str):
    """Remove a calendar event."""
    import re as _re
    if not event_id or not _re.match(r'^[a-zA-Z0-9_\-\.]{1,128}$', event_id):
        return {"success": False, "error": "Invalid event_id format"}
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _calendar_remove_sync(event_id))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "calendar_remove")}


def _calendar_remove_sync(event_id: str) -> dict:
    agent = get_agent()
    if "calendar" in agent.tools:
        return agent.tools["calendar"].remove_event(event_id)
    return {"success": False, "error": "Calendar tool not loaded"}


# ============================================================================
# SPACED REPETITION / FLASHCARDS
# ============================================================================

class AddCardRequest(BaseModel):
    front: str = Field(..., max_length=10000)
    back: str = Field(..., max_length=10000)
    tags: List[str] = Field(default_factory=list, max_length=50)
    deck: str = Field("default", max_length=200)


class AnswerRequest(BaseModel):
    card_id: str = Field(..., max_length=128)
    quality: int = Field(..., ge=0, le=5)


@router.get("/flashcards/due")
async def flashcards_due():
    """Get due cards count and next card."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _flashcards_due_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "flashcards_due")}


def _flashcards_due_sync() -> dict:
    agent = get_agent()
    if "spaced_repetition" in agent.tools:
        return agent.tools["spaced_repetition"].review()
    return {"success": False, "error": "Spaced repetition tool not loaded"}


@router.post("/flashcards/answer")
async def flashcards_answer(request: AnswerRequest):
    """Submit answer quality for a flashcard."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _flashcards_answer_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "flashcards_answer")}


def _flashcards_answer_sync(request: AnswerRequest) -> dict:
    agent = get_agent()
    if "spaced_repetition" in agent.tools:
        return agent.tools["spaced_repetition"].answer(request.card_id, request.quality)
    return {"success": False, "error": "Spaced repetition tool not loaded"}


@router.get("/flashcards/stats")
async def flashcards_stats():
    """Get deck statistics."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _flashcards_stats_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "flashcards_stats")}


def _flashcards_stats_sync() -> dict:
    agent = get_agent()
    if "spaced_repetition" in agent.tools:
        return agent.tools["spaced_repetition"].list_decks()
    return {"success": False, "error": "Spaced repetition tool not loaded"}


@router.post("/flashcards/add")
async def flashcards_add(request: AddCardRequest):
    """Add a flashcard."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _flashcards_add_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "flashcards_add")}


def _flashcards_add_sync(request: AddCardRequest) -> dict:
    agent = get_agent()
    if "spaced_repetition" in agent.tools:
        return agent.tools["spaced_repetition"].add_card(
            front=request.front,
            back=request.back,
            tags=request.tags,
            deck=request.deck,
        )
    return {"success": False, "error": "Spaced repetition tool not loaded"}


# ============================================================================
# EMAIL
# ============================================================================

class SendEmailRequest(BaseModel):
    to: str = Field(..., max_length=500)
    subject: str = Field(..., max_length=500)
    body: str = Field(..., max_length=100000)
    cc: Optional[str] = Field(None, max_length=500)
    bcc: Optional[str] = Field(None, max_length=500)


@router.get("/email/status")
async def email_status():
    """Check email configuration status."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _email_status_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "email_status")}


def _email_status_sync() -> dict:
    agent = get_agent()
    if "email" in agent.tools:
        return agent.tools["email"].get_config_status()
    return {"success": False, "error": "Email tool not loaded"}


@router.get("/email/inbox")
async def email_inbox(limit: int = 10):
    """Get recent emails."""
    limit = max(1, min(limit, 100))  # Clamp to prevent abuse
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _email_inbox_sync(limit))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "email_inbox")}


def _email_inbox_sync(limit: int) -> dict:
    agent = get_agent()
    if "email" in agent.tools:
        return agent.tools["email"].fetch_emails(limit=limit)
    return {"success": False, "error": "Email tool not loaded"}


@router.post("/email/send")
async def email_send(request: SendEmailRequest):
    """Send an email."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _email_send_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "email_send")}


def _email_send_sync(request: SendEmailRequest) -> dict:
    agent = get_agent()
    if "email" in agent.tools:
        return agent.tools["email"].send_email(
            to=request.to,
            subject=request.subject,
            body=request.body,
            cc=request.cc,
            bcc=request.bcc,
        )
    return {"success": False, "error": "Email tool not loaded"}


# ============================================================================
# SCREEN READER
# ============================================================================

@router.get("/screen/read")
async def screen_read():
    """Read current screen via OCR."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _screen_read_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "screen_read")}


def _screen_read_sync() -> dict:
    agent = get_agent()
    if "screen_reader" in agent.tools:
        return agent.tools["screen_reader"].read_screen()
    return {"success": False, "error": "Screen reader tool not loaded"}


@router.get("/screen/active-window")
async def screen_active_window():
    """Get active window info."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _screen_active_window_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "screen_active_window")}


def _screen_active_window_sync() -> dict:
    agent = get_agent()
    if "screen_reader" in agent.tools:
        return agent.tools["screen_reader"].get_active_window()
    return {"success": False, "error": "Screen reader tool not loaded"}


# ============================================================================
# SHELL EXECUTOR
# ============================================================================

class ShellRunRequest(BaseModel):
    command: str = Field(..., max_length=8192)
    session_id: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]{1,64}$')
    timeout: int = Field(60, ge=1, le=600)
    cwd: Optional[str] = Field(None, max_length=512)


@router.post("/shell/run")
async def shell_run(request: ShellRunRequest):
    """Execute a shell command."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _shell_run_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "shell_run")}


# ALLOWLIST: Only these commands may be executed. Everything else is blocked.
_SHELL_ALLOWED_COMMANDS = {
    # File inspection
    "ls", "dir", "cat", "head", "tail", "less", "more", "wc", "file", "stat",
    "grep", "rg", "sort", "uniq", "diff", "tr", "cut",
    # Directory navigation
    "pwd", "cd", "tree", "basename", "dirname", "realpath",
    # System info (read-only)
    "whoami", "date", "uptime", "uname", "hostname", "df", "du", "free", "top",
    "env", "printenv", "echo", "printf",
    # Dev tools (read-only / version control)
    "git",
    # File manipulation (safe subset)
    "mkdir", "touch",
    # Process info (read-only)
    "ps", "pgrep",
    # Network (safe read-only)
    "ping", "dig", "nslookup", "host", "curl", "wget",
}

# Extra patterns that are ALWAYS blocked even if the base command is allowed
_SHELL_DANGER_PATTERNS = [
    "rm ", "rm\t", "rmdir", "rm -rf /", "rm -rf /*", "mkfs.", "dd if=", ":(){", "fork bomb",
    "chmod -R 777 /", "shutdown", "reboot", "init 0", "init 6",
    "taskkill //F //IM node", "pkill -f node", "killall node",
    "> /dev/sd", "| base64 -d | sh", "| bash", "| sh",
    "| /bin/bash", "| /usr/bin/sh", "| /bin/sh",
    "$(", "`",  # Command substitution — block to prevent injection
    "sed -i", "sed -i=",  # In-place file modification
    "git push --force", "git push -f",  # Destructive remote operation
    "ln -s /", "ln -sf /",  # Symlink to sensitive dirs
]

# Explicitly blocked commands — shells and interpreters that enable RCE
_SHELL_BLOCKED_COMMANDS = {
    "powershell", "pwsh", "cmd", "cmd.exe",
    "python", "python3", "node", "ruby", "java", "javac", "go",
}

# Interpreters that accept code-execution flags — block these flag combos
# to prevent `python -c "os.system(...)"` style bypasses of the allowlist.
_INTERPRETER_EXEC_FLAGS = {
    "python": {"-c", "--command"},
    "python3": {"-c", "--command"},
    "node": {"-e", "--eval", "--print", "-p"},
    "ruby": {"-e"},
    "perl": {"-e"},
    "java": {},  # java doesn't have direct eval, safe
    "go": {"run", "generate", "build", "install", "test"},  # all can execute arbitrary code
}


def _validate_shell_cwd(cwd: str | None) -> str | None:
    """Validate and sanitize the cwd parameter for shell commands."""
    if cwd is None:
        return None
    from pathlib import Path as _Path
    try:
        resolved = _Path(cwd).resolve(strict=True)
    except (OSError, ValueError):
        return None  # Fall back to default cwd
    _cwd_str = str(resolved).lower().replace("\\", "/")
    _blocked_dirs = (
        "/windows/system32", "/windows/syswow64", "/etc", "/proc", "/sys",
        "/dev", "/root", "/boot", "/.ssh", "/.gnupg", "/program files",
    )
    if any(_cwd_str.startswith(d) or ("/" + d.lstrip("/")) in _cwd_str for d in _blocked_dirs):
        return None
    if not resolved.is_dir():
        return None
    return str(resolved)


def _extract_command_names(cmd: str) -> list[str]:
    """Extract base command names from a shell command string, handling pipes, &&, ||, ;."""
    import re as _re
    import shlex
    # Split on shell operators
    segments = _re.split(r'\s*(?:\|\||&&|[|;])\s*', cmd)
    commands = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # Handle env var prefixes like KEY=val command
        while '=' in seg.split()[0] if seg.split() else False:
            seg = ' '.join(seg.split()[1:])
            if not seg:
                break
        if not seg:
            continue
        # Get the first token (the command name)
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if tokens:
            # Strip path: /usr/bin/python -> python
            import os
            cmd_name = os.path.basename(tokens[0]).lower()
            # Strip extensions: python3.12 -> python3
            cmd_name = _re.sub(r'\.\d+$', '', cmd_name)
            # Strip .exe on Windows
            if cmd_name.endswith('.exe'):
                cmd_name = cmd_name[:-4]
            commands.append(cmd_name)
    return commands


def _shell_run_sync(request: ShellRunRequest) -> dict:
    import re as _re
    cmd_lower = request.command.lower().strip()

    # Step 1: Block dangerous patterns (always, regardless of allowlist)
    for pattern in _SHELL_DANGER_PATTERNS:
        if pattern in cmd_lower:
            return {"success": False, "error": "Blocked: command matches dangerous pattern"}

    # Step 2: ALLOWLIST — extract every command name and verify ALL are allowed
    cmd_names = _extract_command_names(request.command)
    if not cmd_names:
        return {"success": False, "error": "Blocked: could not parse command"}

    for cmd_name in cmd_names:
        # Explicit blocklist takes priority (shells, interpreters)
        if cmd_name in _SHELL_BLOCKED_COMMANDS:
            return {"success": False, "error": f"Blocked: '{cmd_name}' is not allowed (interpreter/shell)"}
        if cmd_name not in _SHELL_ALLOWED_COMMANDS:
            return {"success": False, "error": f"Blocked: '{cmd_name}' is not in the allowed commands list"}

    # Step 3: Block interpreter code-execution flags (e.g. `python -c "..."`, `node -e "..."`)
    # These bypass the allowlist by running arbitrary code through an allowed interpreter.
    import shlex as _shlex
    try:
        _tokens = _shlex.split(request.command)
    except ValueError:
        _tokens = request.command.split()
    for i, tok in enumerate(_tokens):
        _tok_base = os.path.basename(tok).lower().replace(".exe", "")
        _tok_base = _re.sub(r'\.\d+$', '', _tok_base)
        if _tok_base in _INTERPRETER_EXEC_FLAGS:
            blocked_flags = _INTERPRETER_EXEC_FLAGS[_tok_base]
            # Check remaining tokens for exec flags
            for subsequent in _tokens[i + 1:]:
                if subsequent in blocked_flags:
                    return {
                        "success": False,
                        "error": f"Blocked: '{_tok_base} {subsequent}' can execute arbitrary code",
                    }

    # Validate cwd to prevent path traversal into system directories
    validated_cwd = _validate_shell_cwd(request.cwd)

    agent = get_agent()
    if "shell_executor" in agent.tools:
        return agent.tools["shell_executor"].run(
            command=request.command,
            session_id=request.session_id,
            timeout=request.timeout,
            cwd=validated_cwd,
        )
    return {"success": False, "error": "Shell executor tool not loaded"}


@router.get("/shell/sessions")
async def shell_sessions():
    """List active shell sessions."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _shell_sessions_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "shell_sessions")}


def _shell_sessions_sync() -> dict:
    agent = get_agent()
    if "shell_executor" in agent.tools:
        return agent.tools["shell_executor"].list_sessions()
    return {"success": False, "error": "Shell executor tool not loaded"}


# ============================================================================
# TASK MANAGER
# ============================================================================

class AddTaskRequest(BaseModel):
    title: str = Field(..., max_length=500)
    description: str = Field("", max_length=5000)
    priority: str = Field("medium", max_length=20)
    project: Optional[str] = Field(None, max_length=200)
    due_date: Optional[str] = Field(None, max_length=64)
    tags: List[str] = Field(default_factory=list, max_length=50)


class UpdateTaskRequest(BaseModel):
    task_id: str = Field(..., max_length=128)
    status: Optional[str] = Field(None, max_length=20)
    priority: Optional[str] = Field(None, max_length=20)
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)


@router.get("/tasks/list")
async def tasks_list(status: Optional[str] = None, project: Optional[str] = None):
    """List tasks, optionally filtered by status or project."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _tasks_list_sync(status, project))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "tasks_list")}


def _tasks_list_sync(status: Optional[str], project: Optional[str]) -> dict:
    agent = get_agent()
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].list_tasks(status=status, project=project)
    return {"success": False, "error": "Task manager tool not loaded"}


@router.get("/tasks/board")
async def tasks_board():
    """Get kanban board view."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _tasks_board_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "tasks_board")}


def _tasks_board_sync() -> dict:
    agent = get_agent()
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].board()
    return {"success": False, "error": "Task manager tool not loaded"}


@router.post("/tasks/add")
async def tasks_add(request: AddTaskRequest):
    """Add a new task."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _tasks_add_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "tasks_add")}


def _tasks_add_sync(request: AddTaskRequest) -> dict:
    agent = get_agent()
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].add_task(
            title=request.title,
            description=request.description,
            priority=request.priority,
            project=request.project,
            due_date=request.due_date,
            tags=request.tags,
        )
    return {"success": False, "error": "Task manager tool not loaded"}


@router.put("/tasks/update")
async def tasks_update(request: UpdateTaskRequest):
    """Update a task."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _tasks_update_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "tasks_update")}


def _tasks_update_sync(request: UpdateTaskRequest) -> dict:
    agent = get_agent()
    if "task_manager" in agent.tools:
        kwargs = {}
        if request.status:
            kwargs["status"] = request.status
        if request.priority:
            kwargs["priority"] = request.priority
        if request.title:
            kwargs["title"] = request.title
        if request.description:
            kwargs["description"] = request.description
        return agent.tools["task_manager"].update_task(request.task_id, **kwargs)
    return {"success": False, "error": "Task manager tool not loaded"}


@router.delete("/tasks/{task_id}")
async def tasks_remove(task_id: str):
    """Remove a task."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _tasks_remove_sync(task_id))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "tasks_remove")}


def _tasks_remove_sync(task_id: str) -> dict:
    agent = get_agent()
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].remove_task(task_id)
    return {"success": False, "error": "Task manager tool not loaded"}


@router.get("/tasks/overdue")
async def tasks_overdue():
    """Get overdue tasks."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _tasks_overdue_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "tasks_overdue")}


def _tasks_overdue_sync() -> dict:
    agent = get_agent()
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].overdue()
    return {"success": False, "error": "Task manager tool not loaded"}


# ============================================================================
# API TESTER
# ============================================================================

class APITestRequest(BaseModel):
    method: str = "GET"
    url: str = Field(..., max_length=2048)
    headers: Optional[dict] = None
    body: Optional[str] = Field(None, max_length=1_000_000)
    timeout: int = Field(30, ge=1, le=120)

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str) -> str:
        """Reject non-HTTP(S) schemes and private/internal IPs to prevent SSRF."""
        import ipaddress
        from urllib.parse import urlparse
        lower = v.strip().lower()
        if not (lower.startswith("http://") or lower.startswith("https://")):
            raise ValueError("Only http:// and https:// URLs are allowed")
        # Block requests to private/internal networks
        parsed = urlparse(v.strip())
        hostname = parsed.hostname or ""
        _blocked_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}
        if hostname in _blocked_hosts or hostname.endswith(".local"):
            raise ValueError("Requests to localhost/internal hosts are blocked")
        # Check for private IP ranges (RFC 1918, link-local, loopback)
        _is_ip = True
        try:
            addr = ipaddress.ip_address(hostname)
        except ValueError:
            _is_ip = False
        if _is_ip:
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                raise ValueError("Requests to private/internal IP addresses are blocked")
        else:
            # hostname is a domain name — resolve and check all IPs
            import socket
            try:
                addrinfos = socket.getaddrinfo(hostname, None)
                for _family, _type, _proto, _canonname, sockaddr in addrinfos:
                    resolved_ip = ipaddress.ip_address(sockaddr[0])
                    if resolved_ip.is_private or resolved_ip.is_loopback or resolved_ip.is_link_local or resolved_ip.is_reserved:
                        raise ValueError(f"Domain {hostname} resolves to blocked IP {resolved_ip}")
            except socket.gaierror:
                raise ValueError(f"Cannot resolve hostname: {hostname}") from None
        # Block cloud metadata endpoints
        if hostname in ("169.254.169.254", "metadata.google.internal"):
            raise ValueError("Requests to cloud metadata endpoints are blocked")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
        if v.upper() not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        return v.upper()


@router.post("/api-tester/run")
async def api_tester_run(request: APITestRequest):
    """Execute an HTTP request."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _api_tester_run_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "api_tester_run")}


def _api_tester_run_sync(request: APITestRequest) -> dict:
    agent = get_agent()
    if "api_tester" in agent.tools:
        return agent.tools["api_tester"].request(
            method=request.method,
            url=request.url,
            headers=request.headers,
            body=request.body,
            timeout=request.timeout,
        )
    return {"success": False, "error": "API tester tool not loaded"}


@router.get("/api-tester/history")
async def api_tester_history(limit: int = 20):
    """Get API request history."""
    limit = max(1, min(limit, 100))  # Clamp to prevent abuse
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _api_tester_history_sync(limit))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "api_tester_history")}


def _api_tester_history_sync(limit: int) -> dict:
    agent = get_agent()
    if "api_tester" in agent.tools:
        return agent.tools["api_tester"].history(limit=limit)
    return {"success": False, "error": "API tester tool not loaded"}


# ============================================================================
# DATABASE
# ============================================================================

class SQLQueryRequest(BaseModel):
    sql: str = Field(..., max_length=10000)
    db: str = Field("default", max_length=200)


class CSVImportRequest(BaseModel):
    csv_path: str = Field(..., max_length=512)
    table: str = Field(..., max_length=200)
    db: str = Field("default", max_length=200)


# SQL statements allowed via the API — blocks DROP, DELETE, INSERT, UPDATE, ALTER, etc.
_SQL_ALLOWED_PREFIXES = ("select", "pragma", "explain", "with")


@router.post("/database/query")
async def database_query(request: SQLQueryRequest):
    """Execute a read-only SQL query (SELECT/PRAGMA/EXPLAIN only)."""
    # Restrict to read-only statements at the API layer
    sql_stripped = request.sql.strip().lower()
    if not sql_stripped.startswith(_SQL_ALLOWED_PREFIXES):
        return {"success": False, "error": "Only SELECT, PRAGMA, EXPLAIN, and WITH queries are allowed via the API"}
    # Block multi-statement injection (e.g., "SELECT 1; DROP TABLE foo")
    # Strip string literals first to avoid false positives on semicolons inside strings
    import re as _re
    _sql_no_strings = _re.sub(r"'[^']*'", "", request.sql)
    _sql_no_strings = _re.sub(r'"[^"]*"', "", _sql_no_strings)
    if ";" in _sql_no_strings:
        return {"success": False, "error": "Multi-statement queries are not allowed"}
    # Block DML keywords that can appear inside WITH CTEs without semicolons
    _DML_KEYWORDS = {"insert", "update", "delete", "drop", "alter", "create", "replace", "truncate"}
    _sql_words = set(_re.findall(r'\b\w+\b', _sql_no_strings.lower()))
    if _sql_words & _DML_KEYWORDS:
        return {"success": False, "error": "DML statements (INSERT/UPDATE/DELETE/DROP/etc.) are not allowed"}
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _database_query_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "database_query")}


def _database_query_sync(request: SQLQueryRequest) -> dict:
    agent = get_agent()
    if "database" in agent.tools:
        return agent.tools["database"].query(sql=request.sql, db=request.db)
    return {"success": False, "error": "Database tool not loaded"}


@router.get("/database/schema")
async def database_schema(db: str = "default", table: Optional[str] = None):
    """Get database schema."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _database_schema_sync(db, table))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "database_schema")}


def _database_schema_sync(db: str, table: Optional[str]) -> dict:
    agent = get_agent()
    if "database" in agent.tools:
        return agent.tools["database"].schema(db=db, table=table)
    return {"success": False, "error": "Database tool not loaded"}


@router.post("/database/import-csv")
async def database_import_csv(request: CSVImportRequest):
    """Import CSV into a database table."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _database_import_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "database_import_csv")}


def _database_import_sync(request: CSVImportRequest) -> dict:
    # Validate CSV path — restrict to project data directory
    import os as _os
    from pathlib import Path as _Path
    _safe_base = _Path(_os.getenv("AURA_DATA_DIR", str(_Path(__file__).parent.parent.parent / "data"))).resolve()
    try:
        csv_resolved = _Path(request.csv_path).resolve(strict=True)
    except OSError:
        return {"success": False, "error": "CSV file not found"}
    try:
        csv_resolved.relative_to(_safe_base)
    except ValueError:
        return {"success": False, "error": "CSV path must be within the data directory"}

    agent = get_agent()
    if "database" in agent.tools:
        return agent.tools["database"].import_csv(
            csv_path=str(csv_resolved), table=request.table, db=request.db
        )
    return {"success": False, "error": "Database tool not loaded"}


# ============================================================================
# AUDIO TRANSCRIBER
# ============================================================================

class TranscribeRequest(BaseModel):
    file_path: str = Field(..., max_length=512)
    language: Optional[str] = Field(None, max_length=10)
    model_size: Optional[str] = Field(None, max_length=20)


@router.post("/audio/transcribe")
async def audio_transcribe(request: TranscribeRequest):
    """Transcribe an audio/video file."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _audio_transcribe_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "audio_transcribe")}


def _audio_transcribe_sync(request: TranscribeRequest) -> dict:
    # Validate file path — restrict to project data directory
    import os as _os
    from pathlib import Path as _Path
    _safe_base = _Path(_os.getenv("AURA_DATA_DIR", str(_Path(__file__).parent.parent.parent / "data"))).resolve()
    fp = _Path(request.file_path).resolve()
    try:
        fp.relative_to(_safe_base)
    except ValueError:
        return {"success": False, "error": "File path must be within the data directory"}
    if not fp.exists():
        return {"success": False, "error": "File not found"}

    agent = get_agent()
    if "audio_transcriber" in agent.tools:
        return agent.tools["audio_transcriber"].transcribe(
            file_path=str(fp),
            language=request.language,
            model_size=request.model_size,
        )
    return {"success": False, "error": "Audio transcriber tool not loaded"}


@router.get("/audio/transcripts")
async def audio_transcripts():
    """List saved transcripts."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _audio_transcripts_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "audio_transcripts")}


def _audio_transcripts_sync() -> dict:
    agent = get_agent()
    if "audio_transcriber" in agent.tools:
        return agent.tools["audio_transcriber"].list_transcripts()
    return {"success": False, "error": "Audio transcriber tool not loaded"}


@router.get("/audio/status")
async def audio_status():
    """Check Whisper availability."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _audio_status_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "audio_status")}


def _audio_status_sync() -> dict:
    agent = get_agent()
    if "audio_transcriber" in agent.tools:
        return agent.tools["audio_transcriber"].status()
    return {"success": False, "error": "Audio transcriber tool not loaded"}


# ============================================================================
# CLIPBOARD HISTORY
# ============================================================================

@router.get("/clipboard/history")
async def clipboard_history(limit: int = 20, category: Optional[str] = None):
    """Get clipboard history."""
    limit = max(1, min(limit, 100))  # Clamp to prevent memory exhaustion
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _clipboard_history_sync(limit, category))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "clipboard_history")}


def _clipboard_history_sync(limit: int, category: Optional[str]) -> dict:
    agent = get_agent()
    if "clipboard" in agent.tools:
        return agent.tools["clipboard"].list_history(limit=limit, category=category)
    return {"success": False, "error": "Clipboard tool not loaded"}


@router.post("/clipboard/capture")
async def clipboard_capture():
    """Capture current clipboard content."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _clipboard_capture_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "clipboard_capture")}


def _clipboard_capture_sync() -> dict:
    agent = get_agent()
    if "clipboard" in agent.tools:
        return agent.tools["clipboard"].capture()
    return {"success": False, "error": "Clipboard tool not loaded"}


@router.get("/clipboard/search")
async def clipboard_search(query: str):
    """Search clipboard history."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _clipboard_search_sync(query))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "clipboard_search")}


def _clipboard_search_sync(query: str) -> dict:
    agent = get_agent()
    if "clipboard" in agent.tools:
        return agent.tools["clipboard"].search(query)
    return {"success": False, "error": "Clipboard tool not loaded"}


@router.get("/clipboard/stats")
async def clipboard_stats():
    """Clipboard usage statistics."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _clipboard_stats_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "clipboard_stats")}


def _clipboard_stats_sync() -> dict:
    agent = get_agent()
    if "clipboard" in agent.tools:
        return agent.tools["clipboard"].stats()
    return {"success": False, "error": "Clipboard tool not loaded"}


# ============================================================================
# RESEARCH
# ============================================================================

class SaveResearchRequest(BaseModel):
    title: str = Field(..., max_length=500)
    content: str = Field(..., max_length=100000)
    category: str = Field("tools", max_length=100)
    tags: Optional[List[str]] = None
    sources: Optional[List[str]] = None


@router.get("/research/list")
async def research_list(category: Optional[str] = None):
    """List research files."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _research_list_sync(category))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "research_list")}


def _research_list_sync(category: Optional[str]) -> dict:
    agent = get_agent()
    if "research" in agent.tools:
        return agent.tools["research"].list_research(category=category)
    return {"success": False, "error": "Research tool not loaded"}


@router.post("/research/save")
async def research_save(request: SaveResearchRequest):
    """Save a research note."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _research_save_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "research_save")}


def _research_save_sync(request: SaveResearchRequest) -> dict:
    agent = get_agent()
    if "research" in agent.tools:
        return agent.tools["research"].save(
            title=request.title,
            content=request.content,
            category=request.category,
            tags=request.tags,
            sources=request.sources,
        )
    return {"success": False, "error": "Research tool not loaded"}


@router.get("/research/search")
async def research_search(query: str, category: Optional[str] = None):
    """Search research files."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _research_search_sync(query, category))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "research_search")}


def _research_search_sync(query: str, category: Optional[str]) -> dict:
    agent = get_agent()
    if "research" in agent.tools:
        return agent.tools["research"].search(query, category=category)
    return {"success": False, "error": "Research tool not loaded"}


@router.get("/research/stats")
async def research_stats():
    """Research statistics."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _research_stats_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "research_stats")}


def _research_stats_sync() -> dict:
    agent = get_agent()
    if "research" in agent.tools:
        return agent.tools["research"].stats()
    return {"success": False, "error": "Research tool not loaded"}


@router.get("/research/skills")
async def research_skills():
    """List saved skills."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _research_skills_sync)
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "research_skills")}


def _research_skills_sync() -> dict:
    agent = get_agent()
    if "research" in agent.tools:
        return agent.tools["research"].list_skills()
    return {"success": False, "error": "Research tool not loaded"}
