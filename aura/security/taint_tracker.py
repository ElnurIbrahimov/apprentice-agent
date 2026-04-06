"""Information flow taint tracking for secrets and PII.

Stolen from OpenFang: tag data as tainted, track flow through the system,
block tainted data from reaching untrusted sinks (logs, external APIs,
unencrypted storage).

Prevents: API keys leaked to memory, PII in logs, secrets in tool outputs.
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TaintLabel(Enum):
    """Classification levels for sensitive data."""
    PUBLIC = "public"        # Safe to store/log/send anywhere
    INTERNAL = "internal"    # Safe to store, don't send externally
    PII = "pii"             # Redact in logs, encrypt in storage
    SECRET = "secret"       # Never store, never log, never send


# Severity ordering — used for dedup, highest_taint, and session elevation.
# Defined once here to avoid duplicate dicts scattered across functions.
TAINT_PRIORITY = {TaintLabel.PUBLIC: 0, TaintLabel.INTERNAL: 1, TaintLabel.PII: 2, TaintLabel.SECRET: 3}


# Regex patterns for secret detection (ordered by specificity)
_SECRET_PATTERNS = [
    # API keys (common prefixes)
    (r"sk-[a-zA-Z0-9]{20,}", TaintLabel.SECRET, "OpenAI API key"),
    (r"sk-ant-[a-zA-Z0-9\-]{20,}", TaintLabel.SECRET, "Anthropic API key"),
    (r"AIza[a-zA-Z0-9_\-]{20,}", TaintLabel.SECRET, "Google API key"),
    (r"ghp_[a-zA-Z0-9]{36}", TaintLabel.SECRET, "GitHub PAT"),
    (r"gho_[a-zA-Z0-9]{36}", TaintLabel.SECRET, "GitHub OAuth token"),
    (r"glpat-[a-zA-Z0-9\-_]{20,}", TaintLabel.SECRET, "GitLab PAT"),
    (r"xoxb-[a-zA-Z0-9\-]+", TaintLabel.SECRET, "Slack bot token"),
    (r"xoxp-[a-zA-Z0-9\-]+", TaintLabel.SECRET, "Slack user token"),
    (r"AKIA[A-Z0-9]{16}", TaintLabel.SECRET, "AWS access key"),
    (r"eyJ[a-zA-Z0-9_\-]{50,}\.[a-zA-Z0-9_\-]{50,}\.[a-zA-Z0-9_\-]{50,}", TaintLabel.SECRET, "JWT token"),
    # Generic patterns
    (r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?", TaintLabel.SECRET, "generic API key"),
    (r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?/~`]{8,64})['\"]?", TaintLabel.SECRET, "password"),
    (r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}", TaintLabel.SECRET, "Bearer token"),
    # Private keys
    (r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", TaintLabel.SECRET, "private key"),
    # Connection strings
    (r"(?i)(postgres|mysql|mongodb|redis)://\S+:\S+@\S+", TaintLabel.SECRET, "database connection string"),
]

_PII_PATTERNS = [
    # Email
    (r"[a-zA-Z0-9._%+\-]{3,}@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", TaintLabel.PII, "email address"),
    # Phone (various formats)
    (r"\+?1?\s*\(?[0-9]{3}\)?[\s\-\.][0-9]{3}[\s\-\.][0-9]{4}", TaintLabel.PII, "US phone number"),
    # SSN
    (r"\b\d{3}-\d{2}-\d{4}\b", TaintLabel.PII, "SSN"),
    # Credit card (basic Luhn-eligible patterns)
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", TaintLabel.PII, "credit card number"),
]

# Compile all patterns
_COMPILED_SECRETS = [(re.compile(p), label, desc) for p, label, desc in _SECRET_PATTERNS]
_COMPILED_PII = [(re.compile(p), label, desc) for p, label, desc in _PII_PATTERNS]


@dataclass
class TaintMatch:
    """A detected taint in content."""
    label: TaintLabel
    description: str
    matched_text: str  # The actual matched substring (truncated for safety)
    start: int
    end: int


def scan_for_secrets(text: str, check_pii: bool = True) -> list[TaintMatch]:
    """Scan text for secrets and optionally PII. Returns all matches."""
    if not text:
        return []

    matches = []

    for pattern, label, desc in _COMPILED_SECRETS:
        for m in pattern.finditer(text):
            matched = m.group(0)
            # Truncate matched text for safe logging
            safe_preview = matched[:8] + "..." + matched[-4:] if len(matched) > 16 else "***"
            matches.append(TaintMatch(
                label=label,
                description=desc,
                matched_text=safe_preview,
                start=m.start(),
                end=m.end(),
            ))

    if check_pii:
        for pattern, label, desc in _COMPILED_PII:
            for m in pattern.finditer(text):
                matched = m.group(0)
                safe_preview = matched[:4] + "..." + matched[-3:] if len(matched) > 10 else "***"
                matches.append(TaintMatch(
                    label=label,
                    description=desc,
                    matched_text=safe_preview,
                    start=m.start(),
                    end=m.end(),
                ))

    # Deduplicate overlapping matches (keep highest-severity per region)
    if len(matches) > 1:
        _priority = TAINT_PRIORITY
        matches.sort(key=lambda m: (m.start, -_priority[m.label]))
        deduped = []
        last_end = -1
        for m in matches:
            if m.start >= last_end:
                deduped.append(m)
                last_end = m.end
            elif deduped and _priority[m.label] > _priority[deduped[-1].label]:
                deduped[-1] = m
                last_end = max(last_end, m.end)
        matches = deduped

    return matches


def redact(text: str, matches: Optional[list[TaintMatch]] = None) -> str:
    """Redact all tainted regions from text.

    If matches not provided, scans text first.
    """
    if not text:
        return text

    if matches is None:
        matches = scan_for_secrets(text)

    if not matches:
        return text

    # Sort by start position descending so replacements don't shift indices
    sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
    result = text
    for m in sorted_matches:
        replacement = f"[REDACTED:{m.label.value}:{m.description}]"
        result = result[:m.start] + replacement + result[m.end:]

    return result


def highest_taint(matches: list[TaintMatch]) -> TaintLabel:
    """Return the highest taint level from a list of matches."""
    if not matches:
        return TaintLabel.PUBLIC

    priority = TAINT_PRIORITY
    return max(matches, key=lambda m: priority[m.label]).label


class TaintTracker:
    """Global taint tracking registry.

    Tracks which conversations/sessions have seen tainted data,
    and provides guards for sink operations.
    """

    _MAX_SESSIONS = 1000  # Cap to prevent unbounded memory growth

    def __init__(self):
        self._lock = threading.Lock()
        self._session_taints: dict[str, TaintLabel] = {}  # session_id → highest taint seen
        self._redaction_count = 0
        self._detection_count = 0

    def check_and_track(self, text: str, session_id: str = "default") -> tuple[list[TaintMatch], str]:
        """Scan text, track session taint level, return (matches, redacted_text)."""
        matches = scan_for_secrets(text)

        if matches:
            with self._lock:
                self._detection_count += len(matches)
                # Evict oldest PUBLIC sessions if at capacity
                if session_id not in self._session_taints and len(self._session_taints) >= self._MAX_SESSIONS:
                    public_keys = [k for k, v in self._session_taints.items() if v == TaintLabel.PUBLIC]
                    if public_keys:
                        del self._session_taints[public_keys[0]]
                    else:
                        # All elevated — evict first entry
                        del self._session_taints[next(iter(self._session_taints))]
                current = self._session_taints.get(session_id, TaintLabel.PUBLIC)
                new_level = highest_taint(matches)
                priority = TAINT_PRIORITY
                if priority[new_level] > priority[current]:
                    self._session_taints[session_id] = new_level
                    logger.warning(
                        f"[Taint] Session {session_id[:8]} taint level elevated: "
                        f"{current.value} → {new_level.value} ({len(matches)} matches)"
                    )

        redacted = redact(text, matches)
        if redacted != text:
            with self._lock:
                self._redaction_count += 1

        return matches, redacted

    def is_safe_for_sink(self, session_id: str, sink_type: str) -> bool:
        """Check if session data is safe for a given sink type.

        Sink types: "log", "memory", "external_api", "display"
        """
        with self._lock:
            level = self._session_taints.get(session_id, TaintLabel.PUBLIC)

        rules = {
            "display": {TaintLabel.PUBLIC, TaintLabel.INTERNAL, TaintLabel.PII},
            "memory": {TaintLabel.PUBLIC, TaintLabel.INTERNAL, TaintLabel.PII},
            "log": {TaintLabel.PUBLIC, TaintLabel.INTERNAL},
            "external_api": {TaintLabel.PUBLIC},
        }

        allowed = rules.get(sink_type, {TaintLabel.PUBLIC})
        return level in allowed

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_detections": self._detection_count,
                "total_redactions": self._redaction_count,
                "active_sessions": len(self._session_taints),
                "sessions_by_level": {
                    level.value: sum(1 for v in self._session_taints.values() if v == level)
                    for level in TaintLabel
                },
            }

    def clear_session(self, session_id: str):
        with self._lock:
            self._session_taints.pop(session_id, None)


# Global singleton
_tracker: Optional[TaintTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> TaintTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = TaintTracker()
    return _tracker
