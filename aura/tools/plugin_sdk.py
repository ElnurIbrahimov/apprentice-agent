"""AURA Plugin SDK

Lightweight tool registration system. Mirrors ClawdBot's skills marketplace
concept but for Python-native AURA tools.

Usage
-----
    from aura.tools.plugin_sdk import aura_tool

    @aura_tool(
        name="weather",
        description="Fetch current weather for a location",
        version="1.0.0",
        author="Elnur",
        category="data",
        tags=["weather", "real-time"],
    )
    class WeatherTool:
        def run(self, location: str) -> dict:
            ...

Discovery
---------
    from aura.tools.plugin_sdk import get_plugin_registry
    registry = get_plugin_registry()
    print(registry.list_tools())
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type


# ── Manifest ──────────────────────────────────────────────────────────────────

@dataclass
class ToolManifest:
    """Metadata for a registered AURA tool."""
    name: str
    description: str
    version: str = "0.1.0"
    author: str = "unknown"
    category: str = "general"          # data / system / knowledge / comms / creative
    tags: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)  # pip packages required
    entry_class: Optional[str] = None  # fully-qualified class name
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ToolManifest":
        d.pop("entry_class", None)  # non-serialisable ref — reconstructed on load
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Registry ──────────────────────────────────────────────────────────────────

class PluginRegistry:
    """Singleton registry of all registered AURA tools."""

    _MANIFEST_FILE = Path(__file__).parent / "plugin_manifests.json"

    def __init__(self):
        self._lock = threading.Lock()
        self._manifests: Dict[str, ToolManifest] = {}
        self._classes: Dict[str, Type] = {}
        self._load_persisted()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, cls: Type, manifest: ToolManifest) -> None:
        """Register a tool class with its manifest."""
        manifest.entry_class = f"{cls.__module__}.{cls.__qualname__}"
        with self._lock:
            self._manifests[manifest.name] = manifest
            self._classes[manifest.name] = cls
        self._persist()

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_class(self, name: str) -> Optional[Type]:
        with self._lock:
            return self._classes.get(name)

    def get_manifest(self, name: str) -> Optional[ToolManifest]:
        with self._lock:
            return self._manifests.get(name)

    def list_tools(self) -> List[ToolManifest]:
        with self._lock:
            return [m for m in self._manifests.values() if m.enabled]

    def search(self, query: str) -> List[ToolManifest]:
        """Search tools by name, description, or tag (case-insensitive)."""
        q = query.lower()
        with self._lock:
            return [
                m for m in self._manifests.values()
                if m.enabled and (
                    q in m.name.lower()
                    or q in m.description.lower()
                    or any(q in t.lower() for t in m.tags)
                    or q in m.category.lower()
                )
            ]

    def disable(self, name: str) -> bool:
        with self._lock:
            if name in self._manifests:
                self._manifests[name].enabled = False
                self._persist()
                return True
        return False

    def enable(self, name: str) -> bool:
        with self._lock:
            if name in self._manifests:
                self._manifests[name].enabled = True
                self._persist()
                return True
        return False

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        try:
            with self._lock:
                data = {n: m.to_dict() for n, m in self._manifests.items()}
            self._MANIFEST_FILE.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_persisted(self) -> None:
        if not self._MANIFEST_FILE.exists():
            return
        try:
            data = json.loads(self._MANIFEST_FILE.read_text(encoding="utf-8"))
            for name, d in data.items():
                try:
                    m = ToolManifest.from_dict(d)
                    self._manifests[name] = m
                    # _classes not restored from disk — classes re-register on import
                except Exception:
                    pass
        except Exception:
            pass

    def summary(self) -> str:
        tools = self.list_tools()
        lines = [f"AURA Plugin Registry — {len(tools)} tool(s) registered\n"]
        for m in sorted(tools, key=lambda x: x.category):
            req = f" [requires: {', '.join(m.requires)}]" if m.requires else ""
            lines.append(f"  [{m.category}] {m.name} v{m.version}  — {m.description}{req}")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────

_registry: Optional[PluginRegistry] = None
_registry_lock = threading.Lock()


def get_plugin_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = PluginRegistry()
    return _registry


# ── Decorator ─────────────────────────────────────────────────────────────────

def aura_tool(
    name: str,
    description: str,
    version: str = "0.1.0",
    author: str = "unknown",
    category: str = "general",
    tags: Optional[List[str]] = None,
    requires: Optional[List[str]] = None,
) -> Any:
    """Class decorator that registers a tool with the Plugin Registry.

    Example::

        @aura_tool(
            name="calculator",
            description="Evaluates math expressions safely",
            category="data",
        )
        class CalculatorTool:
            def run(self, expression: str) -> dict:
                ...
    """
    def decorator(cls: Type) -> Type:
        manifest = ToolManifest(
            name=name,
            description=description,
            version=version,
            author=author,
            category=category,
            tags=tags or [],
            requires=requires or [],
        )
        get_plugin_registry().register(cls, manifest)
        # Attach manifest to the class for introspection
        cls.__aura_manifest__ = manifest
        return cls
    return decorator
