"""Aura reliability layer — loop guard, retry, error classification, rate-limit tracking."""

from .retry_utils import jittered_backoff
from .error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_api_error,
)
from .rate_limit_tracker import (
    RateLimitBucket,
    RateLimitState,
    format_rate_limit_compact,
    format_rate_limit_display,
    parse_rate_limit_headers,
)
from .provider_shim import (
    ProviderGiveUp,
    all_rate_limit_snapshots,
    get_last_rate_limit,
    record_rate_limit,
    request_with_retry,
)
from .ollama_wrapper import ResilientOllamaClient

__all__ = [
    "jittered_backoff",
    "ClassifiedError",
    "FailoverReason",
    "classify_api_error",
    "RateLimitBucket",
    "RateLimitState",
    "format_rate_limit_compact",
    "format_rate_limit_display",
    "parse_rate_limit_headers",
    "ProviderGiveUp",
    "all_rate_limit_snapshots",
    "get_last_rate_limit",
    "record_rate_limit",
    "request_with_retry",
    "ResilientOllamaClient",
]
