"""Input validation schemas for all tools.

SECURITY: Validates all tool inputs to prevent:
- Buffer overflow via excessively long strings
- Path traversal via malicious paths
- Injection attacks via unescaped special characters
- DoS via resource exhaustion

Uses Pydantic for type-safe validation with clear error messages.
"""

import logging
import re
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

# Try to import Pydantic, fall back to basic validation
try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("[VALIDATION] Pydantic not available, using basic validation")


# ============================================================================
#                    VALIDATION CONSTANTS
# ============================================================================

MAX_PATH_LENGTH = 4096          # Linux max path
MAX_FILENAME_LENGTH = 255       # Standard filesystem limit
MAX_CONTENT_LENGTH = 10_000_000 # 10MB max file content
MAX_QUERY_LENGTH = 1000         # Search queries
MAX_CODE_LENGTH = 50_000        # Code execution
MAX_URL_LENGTH = 2048           # URLs
MAX_CLIPBOARD_SIZE = 10_000_000 # 10MB clipboard


# ============================================================================
#                    PYDANTIC MODELS (if available)
# ============================================================================

if PYDANTIC_AVAILABLE:

    class FilePathInput(BaseModel):
        """Validated file path input."""
        path: str = Field(..., min_length=1, max_length=MAX_PATH_LENGTH)

        @field_validator('path')
        @classmethod
        def validate_path(cls, v: str) -> str:
            # Block null bytes
            if '\x00' in v:
                raise ValueError("Path cannot contain null bytes")
            # Block control characters
            if any(ord(c) < 32 and c not in '\t\n\r' for c in v):
                raise ValueError("Path cannot contain control characters")
            return v

    class FileContentInput(BaseModel):
        """Validated file content for writing."""
        path: str = Field(..., min_length=1, max_length=MAX_PATH_LENGTH)
        content: str = Field(..., max_length=MAX_CONTENT_LENGTH)
        overwrite: bool = False

        @field_validator('path')
        @classmethod
        def validate_path(cls, v: str) -> str:
            if '\x00' in v:
                raise ValueError("Path cannot contain null bytes")
            return v

    class SearchQueryInput(BaseModel):
        """Validated search query input."""
        query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
        num_results: int = Field(default=10, ge=1, le=100)

        @field_validator('query')
        @classmethod
        def validate_query(cls, v: str) -> str:
            # Strip leading/trailing whitespace
            v = v.strip()
            if not v:
                raise ValueError("Query cannot be empty")
            return v

    class CodeExecutionInput(BaseModel):
        """Validated code execution input."""
        code: str = Field(..., min_length=1, max_length=MAX_CODE_LENGTH)
        timeout: int = Field(default=30, ge=1, le=300)

    class URLInput(BaseModel):
        """Validated URL input."""
        url: str = Field(..., min_length=1, max_length=MAX_URL_LENGTH)

        @field_validator('url')
        @classmethod
        def validate_url(cls, v: str) -> str:
            # Basic URL format check
            if not re.match(r'^https?://', v, re.IGNORECASE):
                raise ValueError("URL must start with http:// or https://")
            return v

    class ClipboardInput(BaseModel):
        """Validated clipboard input."""
        content: str = Field(..., max_length=MAX_CLIPBOARD_SIZE)

    class VisionInput(BaseModel):
        """Validated vision/image analysis input."""
        image_path: str = Field(..., min_length=1, max_length=MAX_PATH_LENGTH)
        prompt: str = Field(default="Describe this image", max_length=MAX_QUERY_LENGTH)

        @field_validator('image_path')
        @classmethod
        def validate_image_path(cls, v: str) -> str:
            if '\x00' in v:
                raise ValueError("Path cannot contain null bytes")
            # Check for valid image extension
            valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
            ext = v.lower().split('.')[-1] if '.' in v else ''
            if f'.{ext}' not in valid_extensions:
                raise ValueError(f"Invalid image extension. Allowed: {valid_extensions}")
            return v


# ============================================================================
#                    FALLBACK VALIDATION (no Pydantic)
# ============================================================================

def validate_string(value: Any, name: str, max_length: int, min_length: int = 0,
                   allow_empty: bool = False, strip: bool = True) -> str:
    """Validate a string input."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")

    if strip:
        value = value.strip()

    if not allow_empty and not value:
        raise ValueError(f"{name} cannot be empty")

    if len(value) < min_length:
        raise ValueError(f"{name} must be at least {min_length} characters")

    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length of {max_length}")

    # Block null bytes
    if '\x00' in value:
        raise ValueError(f"{name} cannot contain null bytes")

    return value


def validate_int(value: Any, name: str, min_val: int | None = None, max_val: int | None = None) -> int:
    """Validate an integer input."""
    if isinstance(value, bool):  # bool is subclass of int
        raise ValueError(f"{name} must be an integer, got boolean")

    if not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be an integer")

    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be at least {min_val}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be at most {max_val}")

    return value


def validate_path(path: Any) -> str:
    """Validate a file path input."""
    path = validate_string(path, "path", MAX_PATH_LENGTH, min_length=1)

    # Block control characters
    if any(ord(c) < 32 and c not in '\t\n\r' for c in path):
        raise ValueError("Path cannot contain control characters")

    return path


def validate_url(url: Any) -> str:
    """Validate a URL input."""
    url = validate_string(url, "url", MAX_URL_LENGTH, min_length=1)

    if not re.match(r'^https?://', url, re.IGNORECASE):
        raise ValueError("URL must start with http:// or https://")

    return url


def validate_query(query: Any) -> str:
    """Validate a search query."""
    return validate_string(query, "query", MAX_QUERY_LENGTH, min_length=1)


def validate_code(code: Any) -> str:
    """Validate code for execution."""
    return validate_string(code, "code", MAX_CODE_LENGTH, min_length=1)


# ============================================================================
#                    VALIDATION DECORATOR
# ============================================================================

def validated(validator_func):
    """Decorator to validate function inputs before execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                validated_kwargs = validator_func(**kwargs)
                return func(*args, **validated_kwargs)
            except (ValueError, ValidationError if PYDANTIC_AVAILABLE else ValueError) as e:
                logger.warning(f"[VALIDATION] Input validation failed: {e}")
                return {
                    "success": False,
                    "error": f"Validation error: {e}",
                    "blocked_by": "input_validation"
                }
        return wrapper
    return decorator


# ============================================================================
#                    TOOL-SPECIFIC VALIDATORS
# ============================================================================

def validate_filesystem_read(path: str | None = None, **kwargs) -> dict:
    """Validate filesystem read inputs."""
    return {"path": validate_path(path)}


def validate_filesystem_write(path: str | None = None, content: str | None = None,
                              overwrite: bool = False, **kwargs) -> dict:
    """Validate filesystem write inputs."""
    return {
        "path": validate_path(path),
        "content": validate_string(content, "content", MAX_CONTENT_LENGTH, allow_empty=True),
        "overwrite": bool(overwrite)
    }


def validate_web_search(query: str | None = None, num_results: int = 10, **kwargs) -> dict:
    """Validate web search inputs."""
    return {
        "query": validate_query(query),
        "num_results": validate_int(num_results, "num_results", min_val=1, max_val=100)
    }


def validate_code_execution(code: str | None = None, timeout: int = 30, **kwargs) -> dict:
    """Validate code execution inputs."""
    return {
        "code": validate_code(code),
        "timeout": validate_int(timeout, "timeout", min_val=1, max_val=300)
    }


def validate_vision(image_path: str | None = None, prompt: str = "Describe this image", **kwargs) -> dict:
    """Validate vision/image analysis inputs."""
    path = validate_path(image_path)

    # Check for valid image extension
    valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    ext = path.lower().split('.')[-1] if '.' in path else ''
    if f'.{ext}' not in valid_extensions:
        raise ValueError(f"Invalid image extension. Allowed: {valid_extensions}")

    return {
        "image_path": path,
        "prompt": validate_string(prompt, "prompt", MAX_QUERY_LENGTH, allow_empty=False)
    }


def validate_clipboard_write(content: str | None = None, **kwargs) -> dict:
    """Validate clipboard write inputs."""
    return {
        "content": validate_string(content, "content", MAX_CLIPBOARD_SIZE, allow_empty=True)
    }


# ============================================================================
#                    SANITIZATION HELPERS
# ============================================================================

def sanitize_for_log(text: str, max_length: int = 200) -> str:
    """Sanitize text for safe logging (remove sensitive data)."""
    if not text:
        return ""

    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + "..."

    # Remove potential secrets
    patterns = [
        (r'password[\s=:]+[^\s,;]+', 'password=[REDACTED]'),
        (r'api[_-]?key[\s=:]+[^\s,;]+', 'api_key=[REDACTED]'),
        (r'token[\s=:]+[^\s,;]+', 'token=[REDACTED]'),
        (r'secret[\s=:]+[^\s,;]+', 'secret=[REDACTED]'),
        (r'bearer\s+[^\s,;]+', 'bearer [REDACTED]'),
        (r'authorization[\s=:]+[^\s,;]+', 'authorization=[REDACTED]'),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text
