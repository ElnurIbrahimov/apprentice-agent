"""LSP (Language Server Protocol) integration stub.

Mirrors Hermes Agent's lsp config:
  lsp:
    enabled: true
    wait_mode: document   # document | streaming
    wait_timeout: 5       # seconds
    servers: {}            # language-specific server configs

This is a thin wrapper that calls out to language servers via JSON-RPC
over stdio. For full LSP support, install the appropriate language
servers (pylsp, typescript-language-server, etc.) and configure
them in lsp.servers.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_lsp_config() -> dict:
    """Get the LSP config."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("lsp", {}) or {}
    except ImportError:
        return {}


def is_lsp_enabled() -> bool:
    """Check if LSP is enabled."""
    return get_lsp_config().get("enabled", False)


def get_lsp_servers() -> dict:
    """Get configured LSP server commands."""
    return get_lsp_config().get("servers", {})


def get_lsp_wait_mode() -> str:
    """Get LSP wait mode: 'document' or 'streaming'."""
    return get_lsp_config().get("wait_mode", "document")


def get_lsp_wait_timeout() -> int:
    """Get LSP wait timeout in seconds."""
    return int(get_lsp_config().get("wait_timeout", 5))


def detect_language(filename: str) -> Optional[str]:
    """Map a filename to a language identifier for LSP lookup."""
    ext_to_lang = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".cpp": "cpp",
        ".c": "c",
    }
    for ext, lang in ext_to_lang.items():
        if filename.endswith(ext):
            return lang
    return None


def get_server_command(language: str) -> Optional[list[str]]:
    """Get the command to start an LSP server for a given language."""
    servers = get_lsp_servers()
    cmd = servers.get(language)
    if isinstance(cmd, str):
        return cmd.split()
    if isinstance(cmd, list):
        return cmd
    return None


def has_lsp_for_file(filename: str) -> bool:
    """Check if an LSP server is configured for the given file's language."""
    lang = detect_language(filename)
    if not lang:
        return False
    return get_server_command(lang) is not None
