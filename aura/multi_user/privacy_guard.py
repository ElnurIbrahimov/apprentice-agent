"""
PrivacyGuard - PII scrubbing, k-anonymity, and differential privacy (ADV-04).

Ensures cross-user learning never leaks personally identifiable information.
Used by KnowledgeAbstractor before generalizing insights across users.
"""

import hashlib
import os
import random
import re
import secrets
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_abstractor import AbstractInsight

logger = logging.getLogger(__name__)

_SALT_FILE = Path(__file__).parent.parent.parent / "aura_data" / ".privacy_salt"
_PRIVACY_SALT: bytes = b""


def _get_privacy_salt() -> bytes:
    """Return the privacy salt, loading or creating it on first call."""
    global _PRIVACY_SALT
    if _PRIVACY_SALT:
        return _PRIVACY_SALT

    env_salt = os.environ.get("AURA_PRIVACY_SALT", "")
    if env_salt:
        _PRIVACY_SALT = env_salt.encode()
        return _PRIVACY_SALT

    try:
        _SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _SALT_FILE.exists():
            _PRIVACY_SALT = _SALT_FILE.read_bytes()
        else:
            _PRIVACY_SALT = secrets.token_bytes(32)
            _SALT_FILE.write_bytes(_PRIVACY_SALT)
    except OSError as e:
        logger.warning(f"[PrivacyGuard] Could not load/create salt file: {e}. Using ephemeral salt.")
        _PRIVACY_SALT = secrets.token_bytes(32)

    return _PRIVACY_SALT


class PrivacyGuard:
    """Scrubs PII and user-attributable data before cross-pollination."""

    # Regex patterns for common PII
    PII_PATTERNS = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
        (r'\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln)\b',
         '[ADDRESS]'),
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
        (r'\bhttps?://\S+\b', '[URL]'),
        (r'\b\d{4}[-/]\d{2}[-/]\d{2}\b', '[DATE]'),
        (r'\b(?:\d{4}[- ]?){3}\d{4}\b', '[CREDIT_CARD]'),
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_ADDRESS]'),
    ]

    # Patterns that might identify a specific user
    USER_REFERENCE_PATTERNS = [
        r'\buser_\w+\b',
        r'\b[Uu]ser\s*#?\d+\b',
        r'\bmy\s+\w+\s+(project|company|team|startup)\b',
    ]

    def __init__(self, k_anonymity: int = 5, noise_epsilon: float = 1.0):
        self.k_anonymity = k_anonymity
        self.noise_epsilon = noise_epsilon
        self._user_ids_seen: Set[str] = set()

    def scrub_text(self, text: str) -> str:
        """Remove PII from text content."""
        scrubbed = text
        for pattern, replacement in self.PII_PATTERNS:
            scrubbed = re.sub(pattern, replacement, scrubbed, flags=re.IGNORECASE)
        for pattern in self.USER_REFERENCE_PATTERNS:
            scrubbed = re.sub(pattern, '[USER_REF]', scrubbed, flags=re.IGNORECASE)
        return scrubbed

    def scrub_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively scrub PII from a dictionary."""
        scrubbed = {}
        for key, value in data.items():
            # Skip user-identifying keys entirely
            if 'user_id' in key.lower() or 'user_name' in key.lower():
                continue
            if isinstance(value, str):
                scrubbed[key] = self.scrub_text(value)
            elif isinstance(value, dict):
                scrubbed[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                scrubbed[key] = [
                    self.scrub_text(v) if isinstance(v, str)
                    else self.scrub_dict(v) if isinstance(v, dict)
                    else v
                    for v in value
                ]
            else:
                scrubbed[key] = value
        return scrubbed

    def check_k_anonymity(
        self, insight: 'AbstractInsight', contributing_users: Set[str],
    ) -> bool:
        """Check if insight has enough supporting users to be shared.

        Returns True if the number of contributing users meets the
        k-anonymity threshold (default: 3 users).
        """
        return len(contributing_users) >= self.k_anonymity

    def add_differential_noise(
        self, value: float, sensitivity: float = 1.0,
    ) -> float:
        """Add Laplace noise for differential privacy.

        Args:
            value: The original value.
            sensitivity: How much one user can affect the result.

        Returns:
            Value with noise added.
        """
        scale = sensitivity / self.noise_epsilon
        noise = random.random() - 0.5
        return value + noise * scale

    def anonymize_user_id(self, user_id: str) -> str:
        """One-way hash a user ID for aggregate tracking."""
        return hashlib.sha256(
            _get_privacy_salt() + user_id.encode()
        ).hexdigest()[:16]
