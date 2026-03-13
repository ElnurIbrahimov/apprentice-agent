"""Shared utilities for the AURA API layer."""

import os


def safe_error_detail(e: Exception, default: str = "Internal server error") -> str:
    """Return detailed error in dev, generic in production."""
    if os.environ.get("AURA_ENV") == "production":
        return default
    return str(e)
