"""Shell Executor tool for running shell commands with persistent sessions and security sandboxing."""

import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output parsers — regex-based, best-effort.
# Each parser returns a dict or None.
# ---------------------------------------------------------------------------

def _parse_pytest(stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
    """Parse pytest summary output."""
    combined = stdout + "\n" + stderr

    result: Dict[str, Any] = {
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_error": 0,
        "duration_seconds": 0.0,
        "failures": [],
    }

    # Summary line: "X passed, Y failed, Z errors in N.NNs"
    # Also handles: "X passed in N.NNs", "Y failed, Z errors in N.NNs", etc.
    summary = re.search(
        r'=+\s*(.*?)\s+in\s+([\d.]+)s?\s*=+',
        combined,
    )
    if summary:
        summary_text = summary.group(1)
        result["duration_seconds"] = float(summary.group(2))

        m = re.search(r'(\d+)\s+passed', summary_text)
        if m:
            result["tests_passed"] = int(m.group(1))
        m = re.search(r'(\d+)\s+failed', summary_text)
        if m:
            result["tests_failed"] = int(m.group(1))
        m = re.search(r'(\d+)\s+error', summary_text)
        if m:
            result["tests_error"] = int(m.group(1))

    # Extract failure names from "FAILED test_file.py::test_name" lines
    for m in re.finditer(r'FAILED\s+(\S+)', combined):
        result["failures"].append(m.group(1))

    # Only return structured data if we actually found the summary line
    if summary:
        return result
    return None


def _parse_git_status(stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
    """Parse git status output (both porcelain and human-readable)."""
    combined = stdout + "\n" + stderr

    result: Dict[str, Any] = {
        "staged": [],
        "modified": [],
        "untracked": [],
        "branch": "",
        "clean": False,
    }

    # Branch name from "On branch <name>" (human-readable)
    m = re.search(r'On branch\s+(\S+)', combined)
    if m:
        result["branch"] = m.group(1)

    # Porcelain-style two-char status codes (git status --short / --porcelain)
    # XY filename — X = staging, Y = worktree
    porcelain_lines = re.findall(r'^([MADRCU?! ]{2})\s+(.+)$', stdout, re.MULTILINE)
    if porcelain_lines:
        for status, filepath in porcelain_lines:
            x, y = status[0], status[1]
            filepath = filepath.strip().strip('"')
            if x == '?' and y == '?':
                result["untracked"].append(filepath)
            else:
                if x in ('M', 'A', 'D', 'R', 'C'):
                    result["staged"].append(filepath)
                if y in ('M', 'D'):
                    result["modified"].append(filepath)
    else:
        # Human-readable parsing fallback
        # "Changes to be committed:" section
        staged_section = re.search(
            r'Changes to be committed:.*?\n(.*?)(?:\n\n|\nChanges|\nUntracked|\Z)',
            combined, re.DOTALL,
        )
        if staged_section:
            for line in staged_section.group(1).splitlines():
                m2 = re.search(r'(?:new file|modified|deleted|renamed):\s+(.+)', line)
                if m2:
                    result["staged"].append(m2.group(1).strip())

        # "Changes not staged for commit:" section
        unstaged_section = re.search(
            r'Changes not staged for commit:.*?\n(.*?)(?:\n\n|\nUntracked|\Z)',
            combined, re.DOTALL,
        )
        if unstaged_section:
            for line in unstaged_section.group(1).splitlines():
                m2 = re.search(r'(?:modified|deleted):\s+(.+)', line)
                if m2:
                    result["modified"].append(m2.group(1).strip())

        # "Untracked files:" section
        untracked_section = re.search(
            r'Untracked files:.*?\n(.*?)(?:\n\n|\Z)',
            combined, re.DOTALL,
        )
        if untracked_section:
            for line in untracked_section.group(1).splitlines():
                line = line.strip()
                if line and not line.startswith('('):
                    result["untracked"].append(line)

    # Clean check
    if 'nothing to commit' in combined or (
        not result["staged"] and not result["modified"] and not result["untracked"]
        and ('working tree clean' in combined or 'nothing to commit' in combined)
    ):
        result["clean"] = True

    # Only return if we found branch info or any file status
    if result["branch"] or result["staged"] or result["modified"] or result["untracked"] or result["clean"]:
        return result
    return None


def _parse_git_diff_stat(stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
    """Parse git diff --stat summary output."""
    result: Dict[str, Any] = {
        "files_changed": 0,
        "insertions": 0,
        "deletions": 0,
        "files": [],
    }

    # Individual file lines: " src/foo.py | 10 ++++---"
    for m in re.finditer(r'^\s*(.+?)\s+\|\s+(\d+)', stdout, re.MULTILINE):
        result["files"].append(m.group(1).strip())

    # Summary line: "3 files changed, 10 insertions(+), 5 deletions(-)"
    summary = re.search(
        r'(\d+)\s+files?\s+changed',
        stdout,
    )
    if summary:
        result["files_changed"] = int(summary.group(1))

    m = re.search(r'(\d+)\s+insertions?\(\+\)', stdout)
    if m:
        result["insertions"] = int(m.group(1))

    m = re.search(r'(\d+)\s+deletions?\(-\)', stdout)
    if m:
        result["deletions"] = int(m.group(1))

    if result["files_changed"] > 0 or result["files"]:
        return result
    return None


def _parse_npm_pip_install(stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
    """Parse npm install or pip install output."""
    combined = stdout + "\n" + stderr

    result: Dict[str, Any] = {
        "packages_installed": 0,
        "warnings": [],
        "errors": [],
    }

    # npm: "added 5 packages" or "added 5 packages, removed 2 packages"
    m = re.search(r'added\s+(\d+)\s+package', combined)
    if m:
        result["packages_installed"] = int(m.group(1))

    # npm: "up to date" means 0 new
    if 'up to date' in combined and result["packages_installed"] == 0:
        result["packages_installed"] = 0

    # pip: "Successfully installed pkg1-1.0 pkg2-2.0"
    m = re.search(r'Successfully installed\s+(.+)', combined)
    if m:
        result["packages_installed"] = len(m.group(1).split())

    # pip: "Requirement already satisfied" — count unique packages
    already = re.findall(r'Requirement already satisfied:\s+(\S+)', combined)
    if already and result["packages_installed"] == 0:
        result["packages_installed"] = len(set(already))

    # Warnings: npm "WARN" or pip "WARNING"
    for m in re.finditer(r'(?:npm\s+)?(?:WARN|WARNING)[:\s]+(.+)', combined, re.IGNORECASE):
        result["warnings"].append(m.group(1).strip()[:200])

    # Errors: npm "ERR!" or pip "ERROR"
    for m in re.finditer(r'(?:npm\s+)?(?:ERR!|ERROR)[:\s]+(.+)', combined, re.IGNORECASE):
        result["errors"].append(m.group(1).strip()[:200])

    # Only return if we found install-related output
    if (result["packages_installed"] > 0 or result["warnings"]
            or result["errors"] or 'up to date' in combined
            or 'Successfully installed' in combined
            or 'Requirement already satisfied' in combined):
        return result
    return None


def _parse_ruff(stdout: str, stderr: str) -> Optional[Dict[str, Any]]:
    """Parse ruff check output."""
    combined = stdout + "\n" + stderr

    result: Dict[str, Any] = {
        "errors": 0,
        "warnings": 0,
        "fixable": 0,
        "rules": {},
    }

    # Individual violation lines: "file.py:10:1: E501 ..."
    for m in re.finditer(r':\d+:\d+:\s+([A-Z]+\d+)', combined):
        rule = m.group(1)
        result["rules"][rule] = result["rules"].get(rule, 0) + 1

    total_issues = sum(result["rules"].values())
    result["errors"] = total_issues

    # "Found N errors" line
    m = re.search(r'Found\s+(\d+)\s+error', combined)
    if m:
        result["errors"] = int(m.group(1))

    # "N fixable with" line
    m = re.search(r'\[(\d+)\s+fixable', combined)
    if not m:
        m = re.search(r'(\d+)\s+(?:potentially\s+)?fixable', combined)
    if m:
        result["fixable"] = int(m.group(1))

    # "All checks passed" means clean
    if 'All checks passed' in combined:
        result["errors"] = 0
        result["warnings"] = 0
        return result

    if result["errors"] > 0 or result["rules"] or result["fixable"] > 0:
        return result
    return None


def _parse_structured_output(
    command: str, stdout: str, stderr: str, exit_code: int
) -> Optional[Dict[str, Any]]:
    """Detect command type and return structured parsed data, or None."""
    cmd = command.strip().lower()

    # pytest
    if cmd.startswith('pytest') or cmd.startswith('python -m pytest'):
        return _parse_pytest(stdout, stderr)

    # git status
    if re.match(r'^git\s+status\b', cmd):
        return _parse_git_status(stdout, stderr)

    # git diff --stat
    if re.match(r'^git\s+diff\b', cmd) and '--stat' in cmd:
        return _parse_git_diff_stat(stdout, stderr)

    # npm install / pip install
    if re.match(r'^(?:npm|pnpm|yarn)\s+install\b', cmd) or re.match(r'^(?:pip|pip3)\s+install\b', cmd):
        return _parse_npm_pip_install(stdout, stderr)

    # ruff
    if re.match(r'^ruff\b', cmd):
        return _parse_ruff(stdout, stderr)

    return None


# Environment keys safe to pass to child processes.
# Everything else (API keys, tokens, secrets) is stripped.
_SAFE_ENV_KEYS = {
    "PATH", "TEMP", "TMP", "HOME", "SYSTEMROOT",
    "COMSPEC", "PATHEXT", "LANG", "USERPROFILE",
    # Required for many Windows tools and npm/node to function
    "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "WINDIR", "USERNAME", "HOMEDRIVE", "HOMEPATH",
    # Locale/encoding
    "LC_ALL", "LC_CTYPE", "PYTHONIOENCODING", "TERM",
}


def _get_sanitized_env() -> dict:
    """Return a copy of os.environ filtered to only safe keys."""
    return {
        k: v for k, v in os.environ.items()
        if k.upper() in _SAFE_ENV_KEYS
    }


# ---------------------------------------------------------------------------
# Injection detection — blocks actual shell exploits while allowing legitimate
# shell features: pipes (|), output redirection (>, >>), conditional chaining
# (&&, ||).
# ---------------------------------------------------------------------------

# Patterns that indicate real injection attempts (backtick subshell, $()
# expansion, eval/exec, process substitution, clobber, etc.)
_INJECTION_PATTERNS = [
    re.compile(r'`'),                       # backtick subshell
    re.compile(r'\$\('),                    # $() command substitution
    re.compile(r'\$\(\('),                  # $(()) arithmetic expansion
    re.compile(r'\$\{'),                    # ${} parameter expansion
    re.compile(r'>\|'),                     # >| clobber operator
    re.compile(r'&>'),                      # &> background redirect
    re.compile(r'<\('),                     # <() process substitution
    re.compile(r'>\('),                     # >() process substitution
    re.compile(r';\s*'),                    # raw semicolons (use && instead)
    re.compile(r'[\r\n]'),                  # newline injection
    re.compile(r'\beval\b'),               # eval command
    re.compile(r'\bexec\b'),               # exec command
    re.compile(r'\bsource\b'),             # source command
    re.compile(r'^\s*\.(?:\s|/)'),          # dot-sourcing (. script.sh)
]


def _contains_shell_injection(cmd: str) -> bool:
    """Detect actual injection patterns while allowing pipes, redirects, &&/||."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cmd):
            return True
    # Block interpreter code-execution flags at top level
    tokens = cmd.split()
    if len(tokens) >= 2 and tokens[1] in ('-c', '/c', '-e', '-enc'):
        return True
    return False


def _is_pipeline_or_chain(cmd: str) -> bool:
    """Check if the command uses pipes, redirects, or conditional chaining."""
    # Match |, >, >>, &&, || but NOT the injection patterns already caught
    return bool(re.search(r'\|(?!\|)|\|{2}|&&|>{1,2}', cmd))


def _extract_base_command(segment: str, is_windows: bool = False) -> str | None:
    """Extract the base command name from a pipeline/chain segment."""
    segment = segment.strip()
    # Strip output redirection from the end (e.g., "> file.txt", ">> log")
    segment = re.sub(r'>{1,2}\s*\S+\s*$', '', segment).strip()
    if not segment:
        return None
    parts = segment.split()
    for part in parts:
        if '=' in part:
            continue  # skip env vars like FOO=bar
        base = part.lower()
        # Strip path
        if '/' in base:
            base = base.rsplit('/', 1)[-1]
        if '\\' in base:
            base = base.rsplit('\\', 1)[-1]
        # Strip extension on Windows
        if is_windows:
            for ext in ('.exe', '.cmd', '.bat', '.ps1'):
                if base.endswith(ext):
                    base = base[:-len(ext)]
        return base
    return None

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
# NOTE: 'kill' is intentionally allowed — targeted PID kills are safe and needed.
# 'killall', 'pkill', and 'taskkill' remain blocked (blanket process killing).
BLOCKED_COMMANDS = {
    "rm -rf /", "rm -rf ~", "mkfs", ":(){ :|:& };:",
    "chmod -R 777 /", "> /dev/sda", "shutdown", "reboot",
    "format c:", "del /f /s /q c:\\",
    "powershell", "pwsh", "cmd", "cmd.exe",
    "taskkill", "pkill", "killall",
}

# Security: blocked patterns (regex)
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/(?!\w)",          # rm -rf / but not rm -rf /tmp/foo
    r"rm\s+-rf\s+~",                # rm -rf ~
    r">\s*/dev/sd",                  # overwrite disk devices
    r"\|\s*(?:/(?:usr/)?(?:local/)?bin/)?(?:ba|da|z|k|fi|c)?sh\b",  # pipe to any shell variant
    r"\|\s*/bin/bash\b",             # pipe to /bin/bash
    r"\|\s*/usr/bin/sh\b",           # pipe to /usr/bin/sh
    r"\|\s*/bin/sh\b",               # pipe to /bin/sh
    r"mkfs\.",                       # format filesystem
    r"dd\s+if=.*/dev/",             # dd from devices
    r":\(\)\s*\{",                   # fork bomb
    r"format\s+[a-z]:",             # Windows format drive
    r"del\s+/[fq]\s+.*[cC]:\\",     # Windows delete system files
]

# Security: allowed command prefixes (safe for direct execution).
# Compilers and build systems are DELIBERATELY excluded here — they execute
# arbitrary code via build scripts (cargo build runs build.rs; make runs Makefile
# recipes; gcc -x assembler can run raw machine code). They're in
# SANDBOX_REQUIRED_COMMANDS below, so they get auto-routed through the sandbox.
ALLOWED_COMMANDS_PREFIX = [
    "ls", "dir", "cd", "pwd", "echo", "cat", "head", "tail",
    "grep", "wc", "sort", "uniq", "diff", "mkdir", "cp",
    "mv", "touch", "git",
    "docker", "tar", "zip", "unzip",
    "type", "where", "whoami", "hostname", "ping", "nslookup",
    "tree", "more", "less", "cut", "tr",
    "export", "which", "man", "help", "cls", "clear",
    "ipconfig", "ifconfig", "netstat", "ss",
    "kill", "xargs",
    # Dev tools — read-only / linting / testing
    "tsc", "eslint", "prettier", "vitest", "jest", "pytest",
    "rg", "fd", "ruff", "mypy", "black", "isort",
    # Common read-only / informational commands
    "date", "cal", "uptime", "uname", "arch", "df", "du", "free",
    "id", "groups", "printenv", "file", "stat", "realpath", "basename",
    "dirname", "md5sum", "sha256sum", "sha1sum", "wc", "yes", "true",
    "false", "time", "timeout", "nproc", "lscpu",
    "ver", "systeminfo",
]

# Interpreter code-execution flags to block (e.g. python -c, node -e)
_INTERP_FLAGS = {
    "python": {"-c", "--command"}, "python3": {"-c", "--command"},
    "node": {"-e", "--eval", "--print", "-p"},
    "ruby": {"-e"}, "perl": {"-e"},
    "pwsh": {"-c", "-command", "-encodedcommand", "-enc"},
    "powershell": {"-c", "-command", "-encodedcommand", "-enc"},
    "git": {"-c"},
}

# Commands that allow arbitrary code execution or data exfiltration.
# These are auto-routed through run_sandboxed() instead of direct execution.
#
# Compilers and build systems belong here because build recipes / build.rs /
# Makefile recipes execute arbitrary code. A malicious Cargo.toml cloned into
# the cwd can exfiltrate secrets via a single `cargo build` if these ran
# unsandboxed. Set AURA_ALLOW_UNSAFE_COMPILERS=1 to opt out (not recommended).
SANDBOX_REQUIRED_COMMANDS = {
    "python", "python3", "pip", "pip3", "uv",
    "node", "npm", "npx", "yarn", "pnpm", "bun", "deno",
    "curl", "wget", "http",
    # Compilers / build systems — arbitrary-code-exec via build scripts.
    "cargo", "rustc", "go", "java", "javac", "dotnet",
    "cmake", "make", "gcc", "g++", "clang",
}
if os.environ.get("AURA_ALLOW_UNSAFE_COMPILERS") == "1":
    # Legacy behavior: treat compilers as direct-exec. Only for build boxes
    # where Aura is the only agent and the user accepts the risk.
    SANDBOX_REQUIRED_COMMANDS -= {
        "cargo", "rustc", "go", "java", "javac", "dotnet",
        "cmake", "make", "gcc", "g++", "clang",
    }
    ALLOWED_COMMANDS_PREFIX = [*ALLOWED_COMMANDS_PREFIX, "cargo", "rustc", "go", "java", "javac", "dotnet", "cmake", "make", "gcc", "g++", "clang"]


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

    def __init__(self, sandbox_executor=None):
        self._sessions: Dict[str, ShellSession] = {}
        self._sessions_lock = threading.Lock()
        self._sandbox_lock = threading.Lock()
        self._is_windows = sys.platform == "win32"
        self._sandbox = sandbox_executor  # Optional SandboxExecutor for sandboxed runs

    def _validate_command(self, command: str) -> tuple:
        """Validate command against security rules.

        Returns:
            (is_valid, reason) where reason is "OK", "SANDBOX_REQUIRED", or a block reason.

        For pipeline/chained commands (containing |, &&, ||), each segment
        is individually validated against the allowed list.
        """
        cmd_stripped = command.strip()

        # --- Global checks (apply to entire command) ---

        # Check blocked patterns (rm -rf /, fork bomb, dd, pipe-to-shell, etc.)
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, cmd_stripped, re.IGNORECASE):
                return False, "Command matches blocked pattern"

        # --- Split into segments for per-segment validation ---
        # Split on pipe (|), && and || while preserving the operators
        # for later shell execution.  We just need the segments for validation.
        segments = re.split(r'\s*(?:\|{2}|&&|\|)\s*', cmd_stripped)
        # Also strip trailing redirections from each segment for validation
        # (e.g. "echo hello > file.txt" -> validate "echo")

        has_sandbox_segment = False

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            seg_lower = segment.lower()

            # Check blocked commands against this segment
            for blocked in BLOCKED_COMMANDS:
                if blocked.lower() in seg_lower:
                    return False, f"Blocked command: {blocked}"

            # Extract base command for this segment
            base_cmd = _extract_base_command(segment, self._is_windows)
            if not base_cmd:
                continue

            # Check privilege escalation
            parts = segment.split()
            for part in parts:
                if '=' in part:
                    continue
                if part.lower() in ('sudo', 'doas'):
                    return False, f"Privilege escalation command '{part}' is disallowed"
                break

            # Check if command needs sandbox routing
            if base_cmd in SANDBOX_REQUIRED_COMMANDS:
                has_sandbox_segment = True
                continue  # Don't check allowed list for sandbox commands

            # Block interpreter code-execution flags (e.g. python -c, node -e)
            if base_cmd in _INTERP_FLAGS:
                blocked_flags = _INTERP_FLAGS[base_cmd]
                for token in parts[1:]:
                    if token.lower() in blocked_flags:
                        return False, f"Code-execution flag '{token}' blocked for '{base_cmd}'"

            is_allowed = any(base_cmd == prefix.lower()
                             for prefix in ALLOWED_COMMANDS_PREFIX)
            if not is_allowed:
                return False, (
                    f"Command '{base_cmd}' not in allowed list. "
                    f"Allowed: {', '.join(ALLOWED_COMMANDS_PREFIX[:15])}..."
                )

        # If any segment requires sandbox, route the whole command there
        if has_sandbox_segment:
            return True, "SANDBOX_REQUIRED"

        return True, "OK"

    def _get_session(self, session_id: str | None = None) -> ShellSession:
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

    def run(self, command: str, session_id: str | None = None,
            timeout: int = DEFAULT_TIMEOUT, cwd: str | None = None) -> dict:
        """Execute a command in a session."""
        if not command or not command.strip():
            return {"success": False, "error": "No command provided"}

        # Security validation
        is_valid, reason = self._validate_command(command)
        if not is_valid:
            return {"success": False, "error": f"Security: {reason}"}

        # Shell injection check — run BEFORE sandbox routing so all commands are checked
        if _contains_shell_injection(command):
            return {"success": False, "output": "", "error": "Command contains disallowed characters or flags", "exit_code": 1}

        # Route sandbox-required commands (python, curl, wget, etc.)
        if reason == "SANDBOX_REQUIRED":
            return self.run_sandboxed(command=command, cwd=cwd, timeout=timeout)

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
                blocked_roots = [
                    "C:\\Windows", "C:\\System32", "/etc", "/sys", "/proc",
                    "/root", "/home", "/.ssh", "/.gnupg", "/var/lib",
                    "C:\\Program Files", "C:\\ProgramData",
                ]
                target_lower = target_str.lower()
                if (any(target_lower.startswith(root.lower()) for root in blocked_roots)
                        or ".ssh" in target_str or ".gnupg" in target_str):
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

        # Build command for execution.
        # For pipeline/chained commands, we must use shell=True because
        # subprocess cannot handle |, &&, ||, > without a shell.
        # Security: _contains_shell_injection() has already rejected dangerous
        # patterns (backtick, $(), eval, exec, semicolons, etc.) and
        # _validate_command() verified every segment against the allowed list.
        use_shell = _is_pipeline_or_chain(command)
        if use_shell:
            shell_cmd = command
        else:
            shell_cmd = shlex.split(command)

        start_time = time.time()

        try:
            proc = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                text=True,
                shell=use_shell,
                env=_get_sanitized_env(),
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
                "error": f"Command failed: {e!s}",
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

        # Structured output parsing
        result["structured"] = _parse_structured_output(command, stdout, stderr, exit_code)

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
            for _sid, s in self._sessions.items():
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

    def get_history(self, session_id: str | None = None, limit: int = 10) -> dict:
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

    def run_streaming(self, command: str, session_id: str | None = None,
                      timeout: int = DEFAULT_TIMEOUT, cwd: str | None = None,
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

        # Route sandbox-required commands (python, curl, wget, etc.)
        if reason == "SANDBOX_REQUIRED":
            return self.run_sandboxed(command=command, cwd=cwd, timeout=timeout)

        timeout = min(max(1, timeout), MAX_TIMEOUT)

        with self._sessions_lock:
            session = self._get_session(session_id)
            working_dir = cwd or session.cwd
            if not Path(working_dir).exists():
                working_dir = str(Path.cwd())

        # Build command — same shell logic as run()
        use_shell = _is_pipeline_or_chain(command)
        if use_shell:
            shell_cmd = command
        else:
            shell_cmd = shlex.split(command)

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
                shell=use_shell,
                env=_get_sanitized_env(),
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
                "error": f"Command failed: {e!s}",
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

        # Structured output parsing
        result["structured"] = _parse_structured_output(command, stdout, stderr, exit_code)

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

    def run_sandboxed(self, command: str, cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
        """Execute command through the SandboxExecutor (E2B first, local fallback).

        Falls back to normal self.run() if no SandboxExecutor is wired in.
        """
        # Always validate before any execution path (not just ImportError fallback)
        is_valid, reason = self._validate_command(command)
        if not is_valid:
            return {"success": False, "error": f"Security: {reason}"}
        if _contains_shell_injection(command):
            return {"success": False, "error": "Command contains disallowed characters or flags", "exit_code": 1}

        with self._sandbox_lock:
            if self._sandbox is None:
                try:
                    from aura.sandbox import SandboxExecutor
                    self._sandbox = SandboxExecutor(
                        timeout=timeout,
                        workspace=cwd or str(Path.cwd()),
                    )
                except ImportError:
                    # Sandbox unavailable — safe fallback with shell=False
                    # (command already validated at top of run_sandboxed)
                    try:
                        cmd_args = shlex.split(command)
                        result = subprocess.run(
                            cmd_args, shell=False, capture_output=True, text=True,
                            timeout=min(timeout, MAX_TIMEOUT), cwd=cwd,
                            env=_get_sanitized_env(),
                        )
                        ret = {
                            "success": result.returncode == 0,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "exit_code": result.returncode,
                            "response": result.stdout.strip() or result.stderr.strip() or f"Exit code {result.returncode}",
                        }
                        ret["structured"] = _parse_structured_output(
                            command, result.stdout, result.stderr, result.returncode
                        )
                        return ret
                    except subprocess.TimeoutExpired:
                        return {"success": False, "error": f"Command timed out after {timeout}s"}
                    except Exception as e:
                        return {"success": False, "error": str(e)}

        result = self._sandbox.run_shell(command, cwd=cwd)
        ret = {
            "success": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "elapsed": result.execution_time,
            "sandbox": result.sandbox,
            "cwd": cwd or (self._sandbox.workspace if hasattr(self._sandbox, 'workspace') else ""),
            "response": result.stdout.strip() or result.stderr.strip() or f"Command completed (exit code {result.exit_code})",
            **({"error": result.error} if result.error else {}),
        }
        ret["structured"] = _parse_structured_output(
            command, result.stdout, result.stderr, result.exit_code
        )
        return ret


# Singleton
shell_executor_tool = ShellExecutorTool()
