"""Custom exceptions and exception handling utilities.

SECURITY: Provides safe exception handling patterns that:
1. Log errors appropriately without exposing sensitive data
2. Provide meaningful error messages to users
3. Allow recovery from transient errors
"""

import logging
import functools
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
#                    CUSTOM EXCEPTIONS
# ============================================================================

class AURAException(Exception):
    """Base exception for all AURA errors."""
    pass


class ModelNotAvailableError(AURAException):
    """Raised when a required LLM model is not available."""
    pass


class ToolExecutionError(AURAException):
    """Raised when a tool fails to execute."""
    pass


class ValidationError(AURAException):
    """Raised when input validation fails."""
    pass


class RateLimitError(AURAException):
    """Raised when rate limit is exceeded."""
    pass


class AURATimeoutError(AURAException):
    """Raised when an operation times out."""
    pass


class SecurityError(AURAException):
    """Raised when a security check fails."""
    pass


class AURAMemoryError(AURAException):
    """Raised when memory operations fail."""
    pass


# Backward-compat aliases -- use AURA-prefixed names instead.
# WARNING: These previously shadowed Python builtins TimeoutError and MemoryError.
# They have been removed to prevent accidental masking of built-in exceptions.
# Use AURATimeoutError and AURAMemoryError directly.
# TimeoutError = AURATimeoutError   # REMOVED - shadows builtin
# MemoryError = AURAMemoryError     # REMOVED - shadows builtin


# ============================================================================
#                    EXCEPTION HANDLING DECORATORS
# ============================================================================

def safe_execute(
    default: Any = None,
    exceptions: tuple = (Exception,),
    log_level: int = logging.WARNING,
    reraise: bool = False
):
    """
    Decorator for safe exception handling.

    Args:
        default: Value to return on exception
        exceptions: Tuple of exception types to catch
        log_level: Logging level for caught exceptions
        reraise: If True, re-raise after logging

    Example:
        @safe_execute(default=[], exceptions=(IOError, ValueError))
        def load_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.log(log_level, f"{func.__name__} failed: {type(e).__name__}: {e}")
                if reraise:
                    raise
                return default
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    backoff_factor: float = 1.0,
    max_backoff: float = 30.0
):
    """
    Decorator to retry function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exception types to retry on
        backoff_factor: Initial backoff in seconds
        max_backoff: Maximum backoff time

    Example:
        @retry_on_failure(max_retries=3, exceptions=(ConnectionError,))
        def fetch_data():
            ...
    """
    import time

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        backoff = min(backoff_factor * (2 ** attempt), max_backoff)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {backoff:.1f}s: {e}"
                        )
                        time.sleep(backoff)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            raise last_exception
        return wrapper
    return decorator


def log_exceptions(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to log all exceptions with full traceback.

    Does not catch exceptions, only logs them.

    Example:
        @log_exceptions
        def important_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    return wrapper


# ============================================================================
#                    EXCEPTION CONTEXT MANAGERS
# ============================================================================

class suppress_and_log:
    """
    Context manager that suppresses specified exceptions and logs them.

    Usage:
        with suppress_and_log(IOError, ValueError, default="fallback"):
            result = risky_operation()
        # If exception occurs, result will be "fallback"
    """

    def __init__(self, *exceptions, default: Any = None, log_level: int = logging.WARNING):
        self.exceptions = exceptions or (Exception,)
        self.default = default
        self.log_level = log_level
        self.exception = None
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, self.exceptions):
            self.exception = exc_val
            self.value = self.default
            logger.log(self.log_level, f"Suppressed {exc_type.__name__}: {exc_val}")
            return True  # Suppress the exception
        return False


# ============================================================================
#                    ERROR MESSAGE FORMATTING
# ============================================================================

def format_user_error(exception: Exception) -> str:
    """
    Format an exception for display to the user.

    Hides technical details while providing useful information.
    """
    error_type = type(exception).__name__

    # Map technical exceptions to user-friendly messages
    user_messages = {
        'ConnectionError': "Unable to connect to the service. Please check your internet connection.",
        'TimeoutError': "The operation took too long. Please try again.",
        'FileNotFoundError': "The requested file was not found.",
        'PermissionError': "Permission denied. You don't have access to this resource.",
        'ValueError': "Invalid input provided. Please check your request.",
        'JSONDecodeError': "Received invalid data format. Please try again.",
        'ModelNotAvailableError': "The AI model is currently unavailable. Please try again later.",
        'RateLimitError': "Too many requests. Please wait a moment and try again.",
        'SecurityError': "This action was blocked for security reasons.",
        'ValidationError': str(exception),  # Validation errors should be user-friendly
    }

    return user_messages.get(error_type, f"An error occurred: {error_type}")


def safe_str(obj: Any, max_length: int = 200) -> str:
    """
    Safely convert any object to string with length limit.

    Useful for logging arbitrary objects without risk of huge output.
    """
    try:
        s = str(obj)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s
    except Exception:
        return f"<unprintable {type(obj).__name__}>"
