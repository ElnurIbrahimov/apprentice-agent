"""Shell Executor tool for running shell commands with persistent sessions and security sandboxing."""

import logging
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

SHELL_METACHAR_RE = re.compile(r'[&|;`$(){}[\]<>!\\^%\r\n]')


def _contains_shell_injection(cmd: str) -> bool:
    """Detect shell metacharacters and dangerous flags."""
    if SHELL_METACHAR_RE.search(cmd):
        return True
    tokens = cmd.split()
    if len(tokens) >= 2 and tokens[1] in ('-c', '/c', '-e', '-enc'):
        return True
    return False

# Maximum output length before truncation
MAX_OUTPUT_CHARS = 10_000
# Default command timeout in seconds
DEFAULT_TIMEOUT = 60
# Maximum allowed timeout
MAX_TIMEOUT = 300
# Maximum concurrent sessions
MAX_SESSIONS = 5
# Session idle timeout in minutes
SESSION_IDLE_TIMEOUT = 30

# Security: blocked commands (exact)
BLOCKED_COMMANDS = {
    "rm -rf /", "rm -rf ~", "mkfs", ":(){ :|:& };:",
    "chmod -R 777 /", "> /dev/sda", "shutdown", "reboot",
    "format c:", "del /f /s /q c:\\",
}

# Security: blocked patterns (regex)
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/(?!\w)",          # rm -rf / but not rm -rf /tmp/foo
    r"rm\s+-rf\s+~",                # rm -rf ~
    r">\s*/dev/sd",                  # overwrite disk devices
    r"\|\s*(?:/(?:usr/)?(?:local/)?bin/)?(?:ba|da|z|k|fi|c)?sh\b",  # pipe to any shell variant
    r"mkfs\.",                       # format filesystem
    r"dd\s+if=.*/dev/",             # dd from devices
    r":\(\)\s*\{",                   # fork bomb
    r"format\s+[a-z]:",             # Windows format drive
    r"del\s+/[fq]\s+.*[cC]:\\",     # Windows delete system files
]

# Security: allowed command prefixes
# NOTE: python/pip/node/npm removed — use CodeExecutorTool for code execution
# NOTE: curl/wget removed — use WebSearchTool for HTTP requests
ALLOWED_COMMANDS_PREFIX = [
    "ls", "dir", "cd", "pwd", "echo", "cat", "head", "tail",
    "grep", "find", "wc", "sort", "uniq", "diff", "mkdir", "cp",
    "mv", "touch", "git",
    "docker", "tar", "zip", "unzip",
    "type", "where", "whoami", "hostname", "ping", "nslookup",
    "tree", "more", "less", "awk", "sed", "cut", "tr", "env",
    "export", "which", "man", "help", "cls", "clear",
    "cargo", "rustc", "go", "java", "javac", "dotnet", "cmake",
    "make", "gcc", "g++", "clang",
    "ipconfig", "ifconfig", "netstat", "ss",
    # Dev tools — needed for coding agent workflows
    "python", "python3", "pip", "pip3", "uv",
    "node", "npm", "npx", "yarn", "pnpm", "bun", "deno",
    "tsc", "eslint", "prettier", "vitest", "jest", "pytest",
    "rg", "fd", "ruff", "mypy", "black", "isort",
    "curl", "wget", "http",
]


@dataclass
class ShellSession:
    """A persistent shell session."""
    id: str
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    last_used: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_used:
            self.last_used = now

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ShellExecutorTool:
    """Execute shell commands with persistent sessions and security sandboxing."""

    name = "shell_executor"
    description = "Execute shell commands with persistent sessions"

    def __init__(self):
        self._sessions: Dict[str, ShellSession] = {}
        self._sessions_lock = threading.Lock()
        self._is_windows = sys.platform == "win32"

    def _validate_command(self, command: str) -> tuple:
        """Validate command against security rules. Returns (is_valid, reason)."""
        cmd_stripped = command.strip()
        cmd_lower = cmd_stripped.lower()

        # Check exact blocked commands
        for blocked in BLOCKED_COMMANDS:
            if blocked.lower() in cmd_lower:
                return False, f"Blocked command: {blocked}"

        # Check blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, cmd_stripped, re.IGNORECASE):
                return False, f"Command matches blocked pattern"

        # Check if command starts with an allowed prefix
        # Get the base command (first word, ignoring env vars and sudo)
        parts = cmd_stripped.split()
        base_cmd = None
        for part in parts:
            if "=" in part:
                continue  # Skip env variable assignments
            if part in ("sudo", "doas"):
                return False, f"Privilege escalation command '{part}' is disallowed"
            base_cmd = part.lower()
            break

        if base_cmd:
            # Strip path from command
            if "/" in base_cmd:
                base_cmd = base_cmd.rsplit("/", 1)[-1]
            if "\\" in base_cmd:
                base_cmd = base_cmd.rsplit("\\", 1)[-1]

            # Remove .exe or .cmd extension on Windows
            if self._is_windows:
                for ext in (".exe", ".cmd", ".bat", ".ps1"):
                    if base_cmd.endswith(ext):
                        base_cmd = base_cmd[:-len(ext)]

            is_allowed = any(base_cmd == prefix.lower() or base_cmd.startswith(prefix.lower() + ".")
                             for prefix in ALLOWED_COMMANDS_PREFIX)
            if not is_allowed:
                return False, f"Command '{base_cmd}' not in allowed list. Allowed: {', '.join(ALLOWED_COMMANDS_PREFIX[:15])}..."

        return True, "OK"

    def _get_session(self, session_id: str = None) -> ShellSession:
        """Get or create a session."""
        # Clean up expired sessions
        self._cleanup_sessions()

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_used = datetime.now().isoformat()
            return session

        if len(self._sessions) >= MAX_SESSIONS:
            # Remove oldest session
            oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_used)
            del self._sessions[oldest_id]

        new_id = session_id or uuid.uuid4().hex[:8]
        session = ShellSession(
            id=new_id,
            cwd=str(Path.cwd()),
        )
        self._sessions[new_id] = session
        return session

    def _cleanup_sessions(self):
        """Remove sessions idle for longer than SESSION_IDLE_TIMEOUT."""
        cutoff = datetime.now() - timedelta(minutes=SESSION_IDLE_TIMEOUT)
        expired = [
            sid for sid, s in self._sessions.items()
            if datetime.fromisoformat(s.last_used) < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]

    def _truncate_output(self, output: str) -> str:
        """Truncate output if it exceeds MAX_OUTPUT_CHARS."""
        if len(output) <= MAX_OUTPUT_CHARS:
            return output
        half = MAX_OUTPUT_CHARS // 2
        return (output[:half]
                + f"\n\n... [TRUNCATED {len(output) - MAX_OUTPUT_CHARS} chars] ...\n\n"
                + output[-half:])

    def run(self, command: str, session_id: str = None,
            timeout: int = DEFAULT_TIMEOUT, cwd: str = None) -> dict:
        """Execute a command in a session."""
        if not command or not command.strip():
            return {"success": False, "error": "No command provided"}

        # Security validation
        is_valid, reason = self._validate_command(command)
        if not is_valid:
            return {"success": False, "error": f"Security: {reason}"}

        if _contains_shell_injection(command):
            return {"success": False, "output": "", "error": "Command contains disallowed characters or flags", "exit_code": 1}

        # Clamp timeout
        timeout = min(max(1, timeout), MAX_TIMEOUT)

        with self._sessions_lock:
            session = self._get_session(session_id)

            # Use provided cwd or session cwd
            working_dir = cwd or session.cwd
            if not Path(working_dir).exists():
                working_dir = str(Path.cwd())

            # Handle cd commands to update session cwd
            cd_match = re.match(r'cd\s+(.+)', command.strip(), re.IGNORECASE)
            if cd_match:
                target = cd_match.group(1).strip().strip('"').strip("'")
                target_path = (Path(working_dir) / target).resolve()
                # SECURITY: Block navigation to system-critical directories
                target_str = str(target_path)
                blocked_roots = ["C:\\Windows", "C:\\System32", "/etc", "/sys", "/proc"]
                if any(target_str.startswith(root) for root in blocked_roots):
                    return {"success": False, "error": f"Access denied: {target_path}", "exit_code": 1, "session_id": session.id}
                if target_path.exists() and target_path.is_dir():
                    session.cwd = str(target_path)
                    return {
                        "success": True,
                        "stdout": f"Changed directory to {session.cwd}",
                        "stderr": "",
                        "exit_code": 0,
                        "session_id": session.id,
                        "cwd": session.cwd,
                        "response": f"cd {session.cwd}"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Directory not found: {target_path}",
                        "exit_code": 1,
                        "session_id": session.id,
                    }

        # Build command for platform
        if self._is_windows:
            shell_cmd = ["cmd.exe", "/c", command]
        else:
            shell_cmd = ["/bin/bash", "-c", command]

        start_time = time.time()

        try:
            proc = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                text=True,
                env=None,  # inherit current environment
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            elapsed = time.time() - start_time
            exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            elapsed = time.time() - start_time
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "elapsed": round(elapsed, 2),
                "session_id": session.id,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command failed: {str(e)}",
                "session_id": session.id,
            }

        # Truncate output
        stdout = self._truncate_output(stdout)
        stderr = self._truncate_output(stderr)

        # Record in session history
        with self._sessions_lock:
            history_entry = {
                "command": command,
                "stdout": stdout[:500],  # Keep history compact
                "stderr": stderr[:500],
                "exit_code": exit_code,
                "timestamp": datetime.now().isoformat(),
                "elapsed": round(elapsed, 2),
            }
            session.history.append(history_entry)
            # Keep history bounded
            if len(session.history) > 100:
                session.history = session.history[-100:]

        result = {
            "success": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "elapsed": round(elapsed, 2),
            "session_id": session.id,
            "cwd": session.cwd,
        }

        # Build response string
        output_parts = []
        if stdout.strip():
            output_parts.append(stdout.strip())
        if stderr.strip():
            output_parts.append(f"[stderr] {stderr.strip()}")
        result["response"] = "\n".join(output_parts) if output_parts else f"Command completed (exit code {exit_code})"

        return result

    def list_sessions(self) -> dict:
        """List active shell sessions."""
        with self._sessions_lock:
            self._cleanup_sessions()

            sessions = []
            for sid, s in self._sessions.items():
                sessions.append({
                    "id": s.id,
                    "cwd": s.cwd,
                    "commands_run": len(s.history),
                    "created_at": s.created_at,
                    "last_used": s.last_used,
                })

        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
            "response": f"{len(sessions)} active session(s)"
        }

    def close_session(self, session_id: str) -> dict:
        """Close a session."""
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return {"success": True, "response": f"Session {session_id} closed"}
        return {"success": False, "error": f"Session not found: {session_id}"}

    def get_history(self, session_id: str = None, limit: int = 10) -> dict:
        """Get command history for a session."""
        with self._sessions_lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
            elif self._sessions:
                session = list(self._sessions.values())[-1]
            else:
                return {"success": True, "history": [], "response": "No active sessions"}

            history = session.history[-limit:]
        formatted = []
        for h in history:
            status = "OK" if h["exit_code"] == 0 else f"ERR({h['exit_code']})"
            formatted.append(f"[{status}] {h['command']}")

        return {
            "success": True,
            "session_id": session.id,
            "history": history,
            "formatted": "\n".join(formatted) if formatted else "No commands in history",
            "response": f"Last {len(history)} command(s):\n" + "\n".join(formatted)
        }

    def run_streaming(self, command: str, session_id: str = None,
                      timeout: int = DEFAULT_TIMEOUT, cwd: str = None,
                      on_output=None) -> dict:
        """Execute a command with real-time streaming output.

        Args:
            command: Command to execute
            session_id: Optional session ID
            timeout: Timeout in seconds
            cwd: Working directory override
            on_output: Callback function(line: str) called for each line of output

        Returns:
            Same as run() but output was also streamed via on_output
        """
        if not command or not command.strip():
            return {"success": False, "error": "No command provided"}

        is_valid, reason = self._validate_command(command)
        if not is_valid:
            return {"success": False, "error": f"Security: {reason}"}

        if _contains_shell_injection(command):
            return {"success": False, "output": "", "error": "Command contains disallowed characters or flags", "exit_code": 1}

        timeout = min(max(1, timeout), MAX_TIMEOUT)

        with self._sessions_lock:
            session = self._get_session(session_id)
            working_dir = cwd or session.cwd
            if not Path(working_dir).exists():
                working_dir = str(Path.cwd())

        if self._is_windows:
            shell_cmd = ["cmd.exe", "/c", command]
        else:
            shell_cmd = ["/bin/bash", "-c", command]

        start_time = time.time()
        stdout_lines = []
        stderr_lines = []

        try:
            proc = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                text=True,
                bufsize=1,  # Line-buffered
                env=None,
            )

            # Read stdout in real-time
            def read_stream(stream, collector, is_stderr=False):
                for line in stream:
                    collector.append(line)
                    if on_output:
                        prefix = "[stderr] " if is_stderr else ""
                        on_output(prefix + line.rstrip())

            stderr_thread = threading.Thread(
                target=read_stream, args=(proc.stderr, stderr_lines, True), daemon=True
            )
            stderr_thread.start()

            # Read stdout on main thread
            read_stream(proc.stdout, stdout_lines)
            stderr_thread.join(timeout=5)

            proc.wait(timeout=timeout)
            elapsed = time.time() - start_time
            exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            elapsed = time.time() - start_time
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "elapsed": round(elapsed, 2),
                "session_id": session.id,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command failed: {str(e)}",
                "session_id": session.id,
            }

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        stdout = self._truncate_output(stdout)
        stderr = self._truncate_output(stderr)

        with self._sessions_lock:
            session.history.append({
                "command": command,
                "stdout": stdout[:500],
                "stderr": stderr[:500],
                "exit_code": exit_code,
                "timestamp": datetime.now().isoformat(),
                "elapsed": round(elapsed, 2),
            })
            if len(session.history) > 100:
                session.history = session.history[-100:]

        result = {
            "success": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "elapsed": round(elapsed, 2),
            "session_id": session.id,
            "cwd": session.cwd,
        }

        output_parts = []
        if stdout.strip():
            output_parts.append(stdout.strip())
        if stderr.strip():
            output_parts.append(f"[stderr] {stderr.strip()}")
        result["response"] = "\n".join(output_parts) if output_parts else f"Command completed (exit code {exit_code})"

        return result

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a shell action."""
        action_lower = action.lower().strip()

        # List sessions
        if action_lower in ("sessions", "list_sessions", "list sessions"):
            return self.list_sessions()

        # Close session
        if action_lower.startswith("close") or action_lower.startswith("kill"):
            session_id = kwargs.get("session_id")
            if not session_id:
                parts = action.split()
                session_id = parts[-1] if len(parts) > 1 else None
            if session_id:
                return self.close_session(session_id)
            return {"success": False, "error": "No session ID specified"}

        # History
        if action_lower.startswith("history"):
            session_id = kwargs.get("session_id")
            limit = kwargs.get("limit", 10)
            return self.get_history(session_id=session_id, limit=limit)

        # Default: run command (use streaming if callback provided)
        session_id = kwargs.get("session_id")
        timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
        cwd = kwargs.get("cwd")
        on_output = kwargs.get("on_output")

        if on_output:
            return self.run_streaming(
                command=action,
                session_id=session_id,
                timeout=timeout,
                cwd=cwd,
                on_output=on_output,
            )
        return self.run(
            command=action,
            session_id=session_id,
            timeout=timeout,
            cwd=cwd
        )


# Singleton
shell_executor_tool = ShellExecutorTool()
