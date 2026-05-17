"""Aura reliability layer — loop guard, retry, error classification, rate-limit tracking."""

from .error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_api_error,
)
from .ollama_wrapper import ResilientOllamaClient
from .provider_shim import (
    ProviderGiveUp,
    all_rate_limit_snapshots,
    get_last_rate_limit,
    record_rate_limit,
    request_with_retry,
)
from .rate_limit_tracker import (
    RateLimitBucket,
    RateLimitState,
    format_rate_limit_compact,
    format_rate_limit_display,
    parse_rate_limit_headers,
)
from .retry_utils import jittered_backoff

__all__ = [
    "ClassifiedError",
    "FailoverReason",
    "ProviderGiveUp",
    "RateLimitBucket",
    "RateLimitState",
    "ResilientOllamaClient",
    "all_rate_limit_snapshots",
    "classify_api_error",
    "format_rate_limit_compact",
    "format_rate_limit_display",
    "get_last_rate_limit",
    "jittered_backoff",
    "parse_rate_limit_headers",
    "record_rate_limit",
    "request_with_retry",
]
