"""Content sanitization for outgoing messages to prevent prompt injection exfiltration."""
import re
import logging

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 1000

_SUSPICIOUS_PATTERNS = re.compile(
    r'(?:'
    r'https?://\S+'           # URLs
    r'|click here'
    r'|verify your account'
    r'|your account (?:will|has)'
    r'|urgent[: ]'
    r'|immediately\b'
    r'|password\b'
    r'|credit card'
    r'|bank account'
    r')',
    re.IGNORECASE
)


def sanitize_outgoing(text: str, source: str = "unknown") -> tuple[str, bool]:
    """Sanitize LLM-generated text before sending to users.

    Returns:
        (sanitized_text, was_flagged)
    """
    if not text:
        return "", False

    text = text[:MAX_MESSAGE_LENGTH]
    flagged = bool(_SUSPICIOUS_PATTERNS.search(text))

    if flagged:
        logger.warning(
            f"[Sanitizer] Flagged outgoing message from source='{source}': {text[:100]}..."
        )

    return text, flagged
