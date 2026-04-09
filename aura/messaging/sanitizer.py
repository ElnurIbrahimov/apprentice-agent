"""Content sanitization for outgoing messages to prevent prompt injection exfiltration."""
import logging
import re

logger = logging.getLogger(__name__)

# Telegram message limit is 4096; use that as the ceiling
MAX_MESSAGE_LENGTH = 4000

# Only flag actual social engineering / phishing patterns — NOT URLs,
# which are legitimate in research results, citations, and tool output.
_SUSPICIOUS_PATTERNS = re.compile(
    r'(?:'
    r'click here'
    r'|verify your account'
    r'|your account (?:will|has)'
    r'|urgent[: ]action'
    r'|password\b.*\brequired'
    r'|credit card\b.*\bnumber'
    r'|bank account\b.*\bdetails'
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

    # Check patterns on FULL text before truncating for output
    flagged = bool(_SUSPICIOUS_PATTERNS.search(text))
    text = text[:MAX_MESSAGE_LENGTH]

    if flagged:
        logger.warning(
            f"[Sanitizer] Flagged outgoing message from source='{source}': {text[:100]}..."
        )

    return text, flagged
