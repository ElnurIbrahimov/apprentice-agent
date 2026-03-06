"""Secure logging with automatic sensitive data sanitization.

SECURITY: Prevents accidental logging of passwords, API keys, tokens, and other secrets.
"""

import logging
import re
from typing import Any, Optional

# Patterns to sanitize (compiled for performance)
SENSITIVE_PATTERNS = [
    # Passwords
    (re.compile(r'password[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'password=[REDACTED]'),
    (re.compile(r'passwd[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'passwd=[REDACTED]'),
    (re.compile(r'pwd[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'pwd=[REDACTED]'),

    # API Keys
    (re.compile(r'api[_-]?key[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'api_key=[REDACTED]'),
    (re.compile(r'apikey[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'apikey=[REDACTED]'),

    # Tokens
    (re.compile(r'token[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'token=[REDACTED]'),
    (re.compile(r'bearer\s+[^\s,;"\'\]}{]+', re.IGNORECASE), 'bearer [REDACTED]'),
    (re.compile(r'access[_-]?token[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'access_token=[REDACTED]'),
    (re.compile(r'refresh[_-]?token[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'refresh_token=[REDACTED]'),

    # Secrets
    (re.compile(r'secret[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'secret=[REDACTED]'),
    (re.compile(r'client[_-]?secret[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'client_secret=[REDACTED]'),

    # Authorization headers
    (re.compile(r'authorization[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'authorization=[REDACTED]'),
    (re.compile(r'auth[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'auth=[REDACTED]'),

    # SSH/Private keys (multi-line)
    (re.compile(r'-----BEGIN[^-]+PRIVATE KEY-----.*?-----END[^-]+PRIVATE KEY-----', re.DOTALL), '[PRIVATE_KEY_REDACTED]'),

    # Credit cards (basic pattern)
    (re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'), '[CARD_REDACTED]'),

    # SSN (US format)
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN_REDACTED]'),

    # Email in sensitive contexts
    (re.compile(r'email[\s=:]+[^\s,;"\'\]}{]+@[^\s,;"\'\]}{]+', re.IGNORECASE), 'email=[REDACTED]'),

    # Database connection strings
    (re.compile(r'(mysql|postgres|mongodb|redis)://[^\s,;"\'\]}{]+', re.IGNORECASE), '[DB_URL_REDACTED]'),

    # AWS credentials
    (re.compile(r'AKIA[0-9A-Z]{16}', re.IGNORECASE), '[AWS_KEY_REDACTED]'),
    (re.compile(r'aws[_-]?secret[\s=:]+[^\s,;"\'\]}{]+', re.IGNORECASE), 'aws_secret=[REDACTED]'),

    # HuggingFace tokens
    (re.compile(r'hf_[A-Za-z0-9]{30,}'), '[HF_TOKEN_REDACTED]'),

    # OpenAI API keys
    (re.compile(r'sk-[A-Za-z0-9]{20,}'), '[OPENAI_KEY_REDACTED]'),

    # Database connection strings with credentials
    (re.compile(r'(?:postgresql|mysql|mongodb|redis)://[^:@\s]+:[^@\s]+@', re.IGNORECASE), '[DB_CREDS_REDACTED]@'),

    # JWT tokens (start with eyJ)
    (re.compile(r'eyJ[A-Za-z0-9._-]{10,}'), '[JWT_REDACTED]'),

    # Authorization Bearer header
    (re.compile(r'(?i)authorization:\s*bearer\s+[A-Za-z0-9._\-]{20,}'), 'authorization: bearer [REDACTED]'),

    # Webhook secrets
    (re.compile(r'(?i)(?:webhook[_-]?secret|x-webhook-secret)[=:]\s*\S+'), 'webhook_secret=[REDACTED]'),

    # Ollama API keys in URLs
    (re.compile(r'(?i)ollama[_-]?api[_-]?key[=:]\s*\S+'), 'ollama_api_key=[REDACTED]'),
]


def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing/masking sensitive data.

    Args:
        text: The text to sanitize

    Returns:
        Sanitized text with sensitive data replaced
    """
    if not isinstance(text, str):
        return str(text)

    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)

    return result


def sanitize_dict(data: dict) -> dict:
    """Recursively sanitize all string values in a dictionary."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = sanitize_text(value)
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict(v) if isinstance(v, dict)
                else sanitize_text(v) if isinstance(v, str)
                else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def _sanitize_arg_recursive(arg, depth=0):
    """Recursively sanitize a logging argument (standalone version for formatters, max depth 50)."""
    if depth > 50:
        return arg
    if isinstance(arg, str):
        return sanitize_text(arg)
    elif isinstance(arg, dict):
        return {k: _sanitize_arg_recursive(v, depth + 1) for k, v in arg.items()}
    elif isinstance(arg, (list, tuple)):
        sanitized = [_sanitize_arg_recursive(item, depth + 1) for item in arg]
        return type(arg)(sanitized)
    return arg


class SanitizingFormatter(logging.Formatter):
    """
    Custom formatter that automatically sanitizes log messages.

    SECURITY: Prevents sensitive data from being logged.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Sanitize message and args (existing logic)
        if record.msg and isinstance(record.msg, str):
            record.msg = sanitize_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = _sanitize_arg_recursive(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(_sanitize_arg_recursive(a) for a in record.args)

        # Format (includes exc_info traceback appended at end)
        formatted = super().format(record)

        # Sanitize the entire formatted string to catch traceback secrets
        return sanitize_text(formatted)


class SecureLogger:
    """
    Wrapper around logging.Logger with automatic sanitization.

    Usage:
        logger = SecureLogger(__name__)
        logger.info(f"User password={password}")  # password will be redacted
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

    def _sanitize_arg(self, arg, depth=0):
        """Recursively sanitize a logging argument (max depth 50)."""
        if depth > 50:
            return arg
        if isinstance(arg, str):
            return sanitize_text(arg)
        elif isinstance(arg, dict):
            return {k: self._sanitize_arg(v, depth + 1) for k, v in arg.items()}
        elif isinstance(arg, (list, tuple)):
            sanitized = [self._sanitize_arg(item, depth + 1) for item in arg]
            return type(arg)(sanitized)
        return arg

    def _sanitize_message(self, msg: Any, *args) -> tuple:
        """Sanitize message and arguments."""
        if isinstance(msg, str):
            msg = sanitize_text(msg)
        else:
            msg = self._sanitize_arg(msg)
        args = tuple(self._sanitize_arg(a) for a in args)
        return msg, args

    def debug(self, msg: Any, *args, **kwargs):
        msg, args = self._sanitize_message(msg, *args)
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: Any, *args, **kwargs):
        msg, args = self._sanitize_message(msg, *args)
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: Any, *args, **kwargs):
        msg, args = self._sanitize_message(msg, *args)
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args, **kwargs):
        msg, args = self._sanitize_message(msg, *args)
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: Any, *args, **kwargs):
        msg, args = self._sanitize_message(msg, *args)
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: Any, *args, **kwargs):
        msg, args = self._sanitize_message(msg, *args)
        self._logger.exception(msg, *args, **kwargs)


def setup_secure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger to use sanitizing formatter.

    Call this at application startup to enable sanitization globally.
    """
    formatter = SanitizingFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Update all existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    # If no handlers, add a console handler
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def get_secure_logger(name: str) -> SecureLogger:
    """Get a secure logger instance."""
    return SecureLogger(name)
