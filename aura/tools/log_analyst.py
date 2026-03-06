"""Log Analyst Tool — parse and structure terminal output, logs, and error traces.

Extracts errors, warnings, tracebacks, and patterns from raw log text.
Returns structured data that AURA can reason about to suggest fixes.

Works with:
- Python tracebacks
- Shell command output / stderr
- Log files (any format)
- Build output (npm, cargo, make, etc.)
- Docker logs
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Patterns
# ------------------------------------------------------------------ #

PYTHON_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\):(.*?)(?=\n\S|\Z)",
    re.DOTALL,
)
PYTHON_ERROR_LINE_RE = re.compile(
    r'File "([^"]+)", line (\d+)(?:, in (\w+))?'
)
PYTHON_EXCEPTION_RE = re.compile(
    r"^(\w+(?:Error|Exception|Warning|Fault|Interrupt|Exit).*?)$",
    re.MULTILINE,
)

LOG_LEVEL_RE = re.compile(
    r"(?:^|\s)(ERROR|CRITICAL|FATAL|WARNING|WARN|INFO|DEBUG|EXCEPTION)\b",
    re.IGNORECASE | re.MULTILINE,
)

TIMESTAMP_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
)

NPM_ERROR_RE = re.compile(r"npm ERR!(.+)", re.MULTILINE)
CARGO_ERROR_RE = re.compile(r"error\[E(\d+)\]:\s*(.+)", re.MULTILINE)
MAKE_ERROR_RE = re.compile(r"make: \*\*\*\s*(.+)", re.MULTILINE)
DOCKER_RE = re.compile(r"Error response from daemon:\s*(.+)", re.MULTILINE)

# Common fix suggestions keyed by error type / pattern
FIX_HINTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"ModuleNotFoundError: No module named '(\w+)'"),
     "Install missing module: `pip install {1}`"),
    (re.compile(r"ImportError: cannot import name '(\w+)' from '(\w+)'"),
     "'{1}' may not exist in this version of '{2}' — check the package changelog"),
    (re.compile(r"PermissionError"),
     "Run as administrator or check file/directory permissions"),
    (re.compile(r"FileNotFoundError: \[Errno 2\] No such file or directory: '(.+)'"),
     "File '{1}' not found — verify the path exists"),
    (re.compile(r"ConnectionRefusedError"),
     "Service is not running or port is blocked — check the service status"),
    (re.compile(r"JSONDecodeError"),
     "Invalid JSON — validate the input with a JSON linter"),
    (re.compile(r"RecursionError"),
     "Stack overflow due to infinite recursion — add base case or increase sys.setrecursionlimit"),
    (re.compile(r"MemoryError"),
     "Out of memory — reduce batch size or use generators/streaming"),
    (re.compile(r"TimeoutError|asyncio.TimeoutError"),
     "Operation exceeded time limit — increase timeout or fix slow dependency"),
    (re.compile(r"UnicodeDecodeError"),
     "Encoding mismatch — add `encoding='utf-8'` or `errors='ignore'` when opening file"),
    (re.compile(r"KeyError: '(\w+)'"),
     "Key '{1}' missing from dict — use `.get('{1}')` or validate input"),
    (re.compile(r"TypeError: '(\w+)' object is not"),
     "Type mismatch — check what type '{1}' actually is at runtime"),
    (re.compile(r"ENOENT: no such file"),
     "File not found — check path exists and cwd is correct"),
    (re.compile(r"EADDRINUSE"),
     "Port already in use — find and kill the process using that port"),
    (re.compile(r"cannot find module '(.+)'", re.IGNORECASE),
     "Node module not found: run `npm install` or `npm install {1}`"),
]


class LogAnalystTool:
    """Analyze terminal output, error logs, and tracebacks — extract errors and suggest fixes."""

    name = "log_analyst"
    description = "Analyze terminal output, error logs, tracebacks — extract errors and suggest fixes"

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #

    def _extract_python_tracebacks(self, text: str) -> List[Dict]:
        tracebacks = []
        for match in PYTHON_TRACEBACK_RE.finditer(text):
            body = match.group(0)
            frames = []
            for frame in PYTHON_ERROR_LINE_RE.finditer(body):
                frames.append({
                    "file": frame.group(1),
                    "line": int(frame.group(2)),
                    "function": frame.group(3),
                })
            exception_match = PYTHON_EXCEPTION_RE.search(body)
            exception_text = exception_match.group(1) if exception_match else "Unknown exception"
            fix = self._suggest_fix(body)
            tracebacks.append({
                "type": "python_traceback",
                "exception": exception_text,
                "frames": frames,
                "innermost_frame": frames[-1] if frames else None,
                "fix_hint": fix,
                "raw": body[:600],
            })
        return tracebacks

    def _extract_log_levels(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for match in LOG_LEVEL_RE.finditer(text):
            level = match.group(1).upper()
            level = "WARNING" if level == "WARN" else level
            counts[level] = counts.get(level, 0) + 1
        return counts

    def _extract_build_errors(self, text: str) -> List[Dict]:
        errors = []
        for m in NPM_ERROR_RE.finditer(text):
            errors.append({"type": "npm", "message": m.group(1).strip()})
        for m in CARGO_ERROR_RE.finditer(text):
            errors.append({"type": "rust_cargo", "code": f"E{m.group(1)}", "message": m.group(2).strip()})
        for m in MAKE_ERROR_RE.finditer(text):
            errors.append({"type": "make", "message": m.group(1).strip()})
        for m in DOCKER_RE.finditer(text):
            errors.append({"type": "docker", "message": m.group(1).strip()})
        return errors

    def _suggest_fix(self, text: str) -> Optional[str]:
        for pattern, hint in FIX_HINTS:
            m = pattern.search(text)
            if m:
                try:
                    result = hint
                    for i, g in enumerate(m.groups(), 1):
                        result = result.replace(f"{{{i}}}", g or "")
                    return result
                except Exception:
                    return hint
        return None

    def _detect_source(self, text: str) -> str:
        """Detect the origin/type of the log."""
        if "Traceback (most recent call last)" in text:
            return "python"
        if "npm ERR!" in text:
            return "npm"
        if "error[E" in text and "-->" in text:
            return "rust"
        if "Error response from daemon" in text:
            return "docker"
        if "make: ***" in text:
            return "make"
        if any(kw in text for kw in ["WARN", "ERROR", "INFO", "DEBUG"]):
            return "structured_log"
        return "generic"

    def _extract_error_lines(self, text: str) -> List[str]:
        """Extract lines that look like errors."""
        lines = text.splitlines()
        error_lines = []
        for line in lines:
            if re.search(r"\b(error|fail|fatal|exception|critical)\b", line, re.IGNORECASE):
                stripped = line.strip()
                if stripped and len(stripped) > 5:
                    error_lines.append(stripped)
        return error_lines[:20]  # cap at 20

    def analyze(self, text: str, context: Optional[str] = None) -> Dict:
        """Analyze log/terminal output and extract structured information.

        Args:
            text: Raw log text, terminal output, or error trace
            context: Optional hint about what command was run (e.g., 'npm build')
        """
        if not text or not text.strip():
            return {"success": False, "error": "No text provided to analyze"}

        source = self._detect_source(text)
        tracebacks = self._extract_python_tracebacks(text)
        log_level_counts = self._extract_log_levels(text)
        build_errors = self._extract_build_errors(text)
        error_lines = self._extract_error_lines(text)

        # Overall severity
        error_count = log_level_counts.get("ERROR", 0) + log_level_counts.get("CRITICAL", 0) + log_level_counts.get("FATAL", 0)
        warning_count = log_level_counts.get("WARNING", 0)
        has_traceback = len(tracebacks) > 0

        severity = "ok"
        if error_count > 0 or has_traceback or build_errors:
            severity = "error"
        elif warning_count > 0:
            severity = "warning"

        # Top-level fix suggestion
        fix_hint = None
        if tracebacks:
            fix_hint = tracebacks[0].get("fix_hint")
        elif build_errors:
            fix_hint = self._suggest_fix(" ".join(e["message"] for e in build_errors))
        elif error_lines:
            fix_hint = self._suggest_fix(" ".join(error_lines[:3]))

        # Extract timestamps if present
        timestamps = list(dict.fromkeys(TIMESTAMP_RE.findall(text)))[:5]

        result = {
            "success": True,
            "source": source,
            "severity": severity,
            "summary": {
                "tracebacks": len(tracebacks),
                "build_errors": len(build_errors),
                "error_lines": len(error_lines),
                "log_levels": log_level_counts,
                "line_count": text.count("\n") + 1,
                "char_count": len(text),
            },
        }

        if tracebacks:
            result["tracebacks"] = tracebacks
        if build_errors:
            result["build_errors"] = build_errors
        if error_lines:
            result["error_lines"] = error_lines
        if timestamps:
            result["timestamps"] = timestamps
        if fix_hint:
            result["fix_hint"] = fix_hint
        if context:
            result["context"] = context

        return result

    def analyze_file(self, path: str, tail_lines: int = 500) -> Dict:
        """Analyze a log file.

        Args:
            path: Path to log file
            tail_lines: Number of lines to read from end of file
        """
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            if len(lines) > tail_lines:
                text = "\n".join(lines[-tail_lines:])
                truncated = True
            else:
                text = content
                truncated = False
            result = self.analyze(text, context=f"file: {path}")
            result["file"] = path
            result["total_lines"] = len(lines)
            result["lines_analyzed"] = min(tail_lines, len(lines))
            result["truncated"] = truncated
            return result
        except Exception as e:
            return {"success": False, "error": f"Failed to read file: {e}"}

    def compare(self, before: str, after: str) -> Dict:
        """Compare two log outputs to identify what changed.

        Args:
            before: Log text before a change
            after: Log text after a change
        """
        before_analysis = self.analyze(before)
        after_analysis = self.analyze(after)

        before_errors = set(before_analysis.get("error_lines", []))
        after_errors = set(after_analysis.get("error_lines", []))

        return {
            "success": True,
            "new_errors": list(after_errors - before_errors),
            "fixed_errors": list(before_errors - after_errors),
            "persisting_errors": list(before_errors & after_errors),
            "before_severity": before_analysis.get("severity"),
            "after_severity": after_analysis.get("severity"),
            "improved": (
                before_analysis.get("severity") in ("error", "warning")
                and after_analysis.get("severity") == "ok"
            ),
        }

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a log analysis action."""
        a = action.lower().strip()
        if "file" in a or kwargs.get("path"):
            return self.analyze_file(kwargs.get("path") or action, kwargs.get("tail_lines", 500))
        if "compare" in a:
            return self.compare(kwargs.get("before", ""), kwargs.get("after", ""))
        # Default: analyze provided text
        text = kwargs.get("text") or kwargs.get("log") or action
        return self.analyze(text, kwargs.get("context"))
