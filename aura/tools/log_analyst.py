"""Log Analyst Tool — parse and structure terminal output, logs, and error traces.

Extracts errors, warnings, tracebacks, and patterns from raw log text.
Returns structured data that AURA can reason about to suggest fixes.

Works with:
- Python tracebacks
- Shell command output / stderr
- Log files (any format)
- Build output (npm, cargo, make, etc.)
- Docker logs

Advanced capabilities:
- Error pattern clustering (group similar errors by signature)
- Temporal pattern detection (spikes, recurring errors)
- Stack trace deduplication (group by root cause)
- Log level summary with error rate metrics
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Patterns
# ------------------------------------------------------------------ #

PYTHON_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\):(.*?(?:Error|Exception|Warning|Fault|Interrupt|Exit)\b[^\n]*)",
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

# Patterns to strip variable parts from error messages for clustering
_VARIABLE_PATTERNS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"), "<TIMESTAMP>"),
    (re.compile(r"\b\d{10,13}\b"), "<EPOCH>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE), "<UUID>"),
    (re.compile(r"\b[0-9a-f]{24,64}\b"), "<HASH>"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"(?:/[\w.-]+)+(?:\.\w+)?"), "<PATH>"),
    (re.compile(r"[A-Za-z]:\\(?:[\w.-]+\\)*[\w.-]+"), "<PATH>"),
    (re.compile(r"line \d+"), "line <N>"),
    (re.compile(r"port \d+"), "port <N>"),
    (re.compile(r"pid[= ]\d+", re.IGNORECASE), "pid=<N>"),
    (re.compile(r"#\d+"), "#<N>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"'[^']{40,}'"), "'<LONG_STR>'"),
    (re.compile(r'"[^"]{40,}"'), '"<LONG_STR>"'),
]

# Extended timestamp patterns for temporal analysis
_DATETIME_FORMATS = [
    (re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?:\.\d+)?"), "%Y-%m-%dT%H:%M:%S"),
    (re.compile(r"(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})"), "%d/%b/%Y:%H:%M:%S"),
    (re.compile(r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"), None),  # syslog — needs special handling
]


def _normalize_error(line: str) -> str:
    """Strip variable parts from an error line to produce a clustering signature."""
    sig = line.strip()
    for pat, repl in _VARIABLE_PATTERNS:
        sig = pat.sub(repl, sig)
    # Collapse repeated whitespace
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig


def _parse_timestamp(line: str) -> Optional[datetime]:
    """Try to extract a datetime from a log line."""
    for pat, fmt in _DATETIME_FORMATS:
        m = pat.search(line)
        if m:
            raw = m.group(1).replace("T", " ")
            if fmt is None:
                # syslog format — assume current year
                try:
                    dt = datetime.strptime(raw, "%b %d %H:%M:%S")
                    return dt.replace(year=datetime.now().year)
                except ValueError:
                    continue
            try:
                return datetime.strptime(raw.replace("T", " "), fmt.replace("T", " "))
            except ValueError:
                continue
    return None


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

    # ------------------------------------------------------------------ #
    # A. Error Pattern Clustering
    # ------------------------------------------------------------------ #

    def _cluster_errors(self, text: str) -> List[Dict]:
        """Group similar error lines by their normalized signature.

        Strips timestamps, IDs, paths, hex addresses etc. so that
        'FileNotFoundError: /tmp/abc123.txt' and 'FileNotFoundError: /var/xyz.log'
        both map to the same cluster.

        Returns a list sorted by count (descending), each entry:
          { "signature": str, "count": int, "samples": [str, ...], "first_seen": str|None, "last_seen": str|None }
        """
        lines = text.splitlines()
        clusters: Dict[str, Dict] = {}  # signature -> { count, samples, first_ts, last_ts }

        for line in lines:
            if not re.search(r"\b(error|fail|fatal|exception|critical)\b", line, re.IGNORECASE):
                continue
            stripped = line.strip()
            if not stripped or len(stripped) <= 5:
                continue

            sig = _normalize_error(stripped)
            ts = _parse_timestamp(stripped)

            if sig not in clusters:
                clusters[sig] = {
                    "count": 0,
                    "samples": [],
                    "first_ts": ts,
                    "last_ts": ts,
                }
            c = clusters[sig]
            c["count"] += 1
            if len(c["samples"]) < 3:  # keep up to 3 sample lines
                c["samples"].append(stripped[:200])
            if ts:
                if c["first_ts"] is None or ts < c["first_ts"]:
                    c["first_ts"] = ts
                if c["last_ts"] is None or ts > c["last_ts"]:
                    c["last_ts"] = ts

        result = []
        for sig, data in sorted(clusters.items(), key=lambda x: x[1]["count"], reverse=True):
            entry: Dict[str, Any] = {
                "signature": sig,
                "count": data["count"],
                "samples": data["samples"],
            }
            if data["first_ts"]:
                entry["first_seen"] = data["first_ts"].isoformat()
            if data["last_ts"]:
                entry["last_seen"] = data["last_ts"].isoformat()
            result.append(entry)

        return result[:30]  # cap at 30 clusters

    # ------------------------------------------------------------------ #
    # B. Temporal Pattern Detection
    # ------------------------------------------------------------------ #

    def _temporal_patterns(self, text: str) -> Dict:
        """Analyze error frequency over time windows and detect spikes/recurring patterns.

        Returns:
          {
            "windows": { "5m": int, "15m": int, "1h": int, "24h": int },
            "error_rate_per_minute": float,
            "error_rate_per_hour": float,
            "spikes": [{ "signature": str, "window_5m": int, "avg_1h": float, "ratio": float }],
            "recurring": [{ "signature": str, "times_of_day": [str], "days_seen": int }],
            "time_range": { "earliest": str, "latest": str, "span_seconds": float } | None
          }
        """
        lines = text.splitlines()
        error_lines_with_ts: List[Tuple[datetime, str]] = []
        all_timestamps: List[datetime] = []

        for line in lines:
            ts = _parse_timestamp(line)
            if ts:
                all_timestamps.append(ts)
            if re.search(r"\b(error|fail|fatal|exception|critical)\b", line, re.IGNORECASE):
                if ts:
                    sig = _normalize_error(line.strip())
                    error_lines_with_ts.append((ts, sig))

        result: Dict[str, Any] = {
            "windows": {"5m": 0, "15m": 0, "1h": 0, "24h": 0},
            "error_rate_per_minute": 0.0,
            "error_rate_per_hour": 0.0,
            "spikes": [],
            "recurring": [],
            "time_range": None,
        }

        if not all_timestamps:
            return result

        earliest = min(all_timestamps)
        latest = max(all_timestamps)
        span = (latest - earliest).total_seconds()
        result["time_range"] = {
            "earliest": earliest.isoformat(),
            "latest": latest.isoformat(),
            "span_seconds": span,
        }

        if not error_lines_with_ts:
            return result

        # Window counts
        for ts, _sig in error_lines_with_ts:
            delta = (latest - ts).total_seconds()
            if delta <= 300:
                result["windows"]["5m"] += 1
            if delta <= 900:
                result["windows"]["15m"] += 1
            if delta <= 3600:
                result["windows"]["1h"] += 1
            if delta <= 86400:
                result["windows"]["24h"] += 1

        # Error rates
        if span > 0:
            total_errors = len(error_lines_with_ts)
            result["error_rate_per_minute"] = round(total_errors / (span / 60), 2) if span >= 60 else round(total_errors * 60 / max(span, 1), 2)
            result["error_rate_per_hour"] = round(total_errors / (span / 3600), 2) if span >= 3600 else round(total_errors * 3600 / max(span, 1), 2)

        # Spike detection: per-signature, compare 5min vs 1hr average rate
        sig_windows: Dict[str, Dict[str, int]] = defaultdict(lambda: {"5m": 0, "1h": 0})
        for ts, sig in error_lines_with_ts:
            delta = (latest - ts).total_seconds()
            if delta <= 300:
                sig_windows[sig]["5m"] += 1
            if delta <= 3600:
                sig_windows[sig]["1h"] += 1

        for sig, w in sig_windows.items():
            # Average 5-min rate over the last hour: (1h_count / 12)
            avg_5m_in_1h = w["1h"] / 12.0 if w["1h"] > 0 else 0
            if avg_5m_in_1h > 0 and w["5m"] > 2:
                ratio = w["5m"] / avg_5m_in_1h
                if ratio >= 3.0:
                    result["spikes"].append({
                        "signature": sig,
                        "window_5m": w["5m"],
                        "avg_5m_rate_in_1h": round(avg_5m_in_1h, 2),
                        "spike_ratio": round(ratio, 1),
                    })

        result["spikes"].sort(key=lambda x: x["spike_ratio"], reverse=True)
        result["spikes"] = result["spikes"][:10]

        # Recurring detection: same error at similar times across different days
        sig_by_day: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for ts, sig in error_lines_with_ts:
            day_key = ts.strftime("%Y-%m-%d")
            sig_by_day[sig][day_key].append(ts.hour)

        for sig, days in sig_by_day.items():
            if len(days) < 2:
                continue
            # Collect all hours across days
            all_hours: List[int] = []
            for hours in days.values():
                all_hours.extend(hours)
            hour_counter = Counter(all_hours)
            # If the same hour appears on multiple days, flag as recurring
            recurring_hours = [h for h, cnt in hour_counter.items() if cnt >= 2]
            if recurring_hours:
                result["recurring"].append({
                    "signature": sig,
                    "recurring_hours": sorted(recurring_hours),
                    "days_seen": len(days),
                    "total_occurrences": sum(len(h) for h in days.values()),
                })

        result["recurring"].sort(key=lambda x: x["days_seen"], reverse=True)
        result["recurring"] = result["recurring"][:10]

        return result

    # ------------------------------------------------------------------ #
    # C. Stack Trace Deduplication
    # ------------------------------------------------------------------ #

    def _dedup_tracebacks(self, tracebacks: List[Dict]) -> List[Dict]:
        """Group tracebacks by their root cause (bottom/innermost frame + exception type).

        Returns list of groups:
          { "root_cause": str, "exception": str, "count": int,
            "frames": [...], "fix_hint": str|None, "sample_raw": str }
        """
        if not tracebacks:
            return []

        groups: Dict[str, Dict] = {}  # root_cause_key -> group

        for tb in tracebacks:
            innermost = tb.get("innermost_frame")
            exc = tb.get("exception", "Unknown")

            if innermost:
                # Key: file + function + exception type (strip variable parts from exception)
                exc_type = exc.split(":")[0].strip() if ":" in exc else exc
                root_key = f"{innermost.get('file', '?')}:{innermost.get('function', '?')}|{exc_type}"
            else:
                root_key = _normalize_error(exc)

            if root_key not in groups:
                groups[root_key] = {
                    "root_cause": root_key,
                    "exception": exc,
                    "count": 0,
                    "frames": tb.get("frames", []),
                    "fix_hint": tb.get("fix_hint"),
                    "sample_raw": tb.get("raw", ""),
                }
            groups[root_key]["count"] += 1

        result = sorted(groups.values(), key=lambda x: x["count"], reverse=True)
        return result[:20]

    # ------------------------------------------------------------------ #
    # D. Log Level Summary
    # ------------------------------------------------------------------ #

    def _log_level_summary(self, text: str, log_level_counts: Dict[str, int]) -> Dict:
        """Produce a comprehensive log level summary with rates.

        Returns:
          {
            "total_lines": int,
            "levels": { "ERROR": int, "WARNING": int, "INFO": int, ... },
            "error_count": int,
            "warning_count": int,
            "info_count": int,
            "debug_count": int,
            "error_pct": float,
            "error_rate_per_minute": float | None,
            "error_rate_per_hour": float | None,
            "time_span_seconds": float | None,
          }
        """
        total_lines = text.count("\n") + 1

        error_count = sum(log_level_counts.get(k, 0) for k in ("ERROR", "CRITICAL", "FATAL", "EXCEPTION"))
        warning_count = log_level_counts.get("WARNING", 0)
        info_count = log_level_counts.get("INFO", 0)
        debug_count = log_level_counts.get("DEBUG", 0)

        error_pct = round((error_count / total_lines) * 100, 2) if total_lines > 0 else 0.0

        # Try to determine time span for rate calculation
        timestamps = []
        for line in text.splitlines():
            ts = _parse_timestamp(line)
            if ts:
                timestamps.append(ts)

        rate_per_min = None
        rate_per_hour = None
        span_seconds = None

        if len(timestamps) >= 2:
            earliest = min(timestamps)
            latest = max(timestamps)
            span_seconds = (latest - earliest).total_seconds()
            if span_seconds > 0:
                rate_per_min = round(error_count / (span_seconds / 60), 2) if span_seconds >= 60 else round(error_count * 60 / span_seconds, 2)
                rate_per_hour = round(error_count / (span_seconds / 3600), 2) if span_seconds >= 3600 else round(error_count * 3600 / span_seconds, 2)

        return {
            "total_lines": total_lines,
            "levels": dict(log_level_counts),
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "debug_count": debug_count,
            "error_pct": error_pct,
            "error_rate_per_minute": rate_per_min,
            "error_rate_per_hour": rate_per_hour,
            "time_span_seconds": span_seconds,
        }

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

        # --- New: advanced analysis ---
        error_clusters = self._cluster_errors(text)
        temporal = self._temporal_patterns(text)
        deduped_tracebacks = self._dedup_tracebacks(tracebacks)
        level_summary = self._log_level_summary(text, log_level_counts)

        # Overall severity
        error_count = level_summary["error_count"]
        warning_count = level_summary["warning_count"]
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

        result: Dict[str, Any] = {
            "success": True,
            "source": source,
            "severity": severity,
            "log_level_summary": level_summary,
            "summary": {
                "tracebacks": len(tracebacks),
                "unique_tracebacks": len(deduped_tracebacks),
                "build_errors": len(build_errors),
                "error_lines": len(error_lines),
                "error_clusters": len(error_clusters),
                "log_levels": log_level_counts,
                "line_count": level_summary["total_lines"],
                "char_count": len(text),
            },
        }

        # Error pattern clusters (A) — always include if there are errors
        if error_clusters:
            result["error_clusters"] = error_clusters
            # Build a human-readable top-errors summary
            top_msgs = []
            for c in error_clusters[:5]:
                top_msgs.append(f"{c['signature']} ({c['count']}x)")
            result["top_errors"] = top_msgs

        # Temporal patterns (B) — include if we found timestamps
        if temporal["time_range"]:
            result["temporal"] = temporal
            if temporal["spikes"]:
                result["spike_alert"] = [
                    f"SPIKE: '{s['signature'][:80]}' — {s['window_5m']}x in last 5min vs ~{s['avg_5m_rate_in_1h']}/5min avg ({s['spike_ratio']}x normal)"
                    for s in temporal["spikes"][:3]
                ]
            if temporal["recurring"]:
                result["recurring_alert"] = [
                    f"RECURRING: '{r['signature'][:80]}' — seen on {r['days_seen']} days at hours {r['recurring_hours']}"
                    for r in temporal["recurring"][:3]
                ]

        # Stack trace dedup (C) — show deduped instead of raw when there are duplicates
        if deduped_tracebacks:
            result["tracebacks_deduped"] = deduped_tracebacks
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
