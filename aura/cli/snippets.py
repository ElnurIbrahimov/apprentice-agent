"""Prompt snippet manager — save/load/list reusable prompt templates."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SNIPPETS_PATH = Path.home() / ".aura" / "snippets.json"


class SnippetManager:
    """Manages reusable prompt snippets stored in ~/.aura/snippets.json."""

    def __init__(self, path: Path = _SNIPPETS_PATH):
        self._path = path
        self._snippets: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                self._snippets = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._snippets = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._snippets, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            logger.debug("snippet_save_failed", exc_info=True)

    def save_snippet(self, name: str, text: str) -> None:
        self._snippets[name] = text
        self._save()

    def get(self, name: str) -> Optional[str]:
        return self._snippets.get(name)

    def delete(self, name: str) -> bool:
        if name in self._snippets:
            del self._snippets[name]
            self._save()
            return True
        return False

    def list_all(self) -> dict[str, str]:
        return dict(self._snippets)
