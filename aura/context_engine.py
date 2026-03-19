# D:/Aura/aura/context_engine.py
"""
Always-On Context Engine (ACE) — auto-gathers context before every agent response.

Single entry point: context_engine.gather(message) -> ContextBundle
"""

import re
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError

try:
    import httpx as _httpx
except ImportError:
    _httpx = None

logger = logging.getLogger(__name__)

CONTEXT_BUDGET = 3000   # max tokens to inject
GATHER_TIMEOUT = 0.8    # 800ms hard deadline


@dataclass
class ContextBlock:
    label: str
    content: str
    priority: int       # 0-100, higher = include first when budget tight
    token_estimate: int = 0

    def __post_init__(self):
        self.token_estimate = len(self.content) // 4


@dataclass
class ContextBundle:
    query: str
    blocks: List[ContextBlock] = field(default_factory=list)
    gather_time_ms: float = 0.0

    def to_system_prompt(self) -> str:
        """Build <CONTEXT> block respecting token budget."""
        if not self.blocks:
            return ""
        sorted_blocks = sorted(self.blocks, key=lambda b: b.priority, reverse=True)
        included = []
        tokens_used = 0
        for block in sorted_blocks:
            if tokens_used + block.token_estimate > CONTEXT_BUDGET:
                # Truncate oversized block to fit remaining budget
                remaining = CONTEXT_BUDGET - tokens_used
                if remaining > 50:  # only include if we can fit something meaningful
                    char_limit = remaining * 4
                    truncated = ContextBlock(
                        label=block.label,
                        content=block.content[:char_limit] + "\n...(truncated to fit budget)",
                        priority=block.priority,
                    )
                    included.append(truncated)
                    tokens_used += truncated.token_estimate
                continue
            included.append(block)
            tokens_used += block.token_estimate
        if not included:
            return ""
        lines = ["<CONTEXT>"]
        for block in included:
            lines.append(f"[{block.label}] {block.content}")
        lines.append("</CONTEXT>")
        lines.append("Use this context where relevant.")
        return "\n".join(lines)

    def summary_line(self) -> str:
        parts = [b.label for b in self.blocks if b.content]
        return " | ".join(parts) if parts else ""


class AlwaysOnContextEngine:
    """
    Gathers context from all available sources before every agent response.

    Usage:
        engine = AlwaysOnContextEngine(agent)
        context = engine.gather(message)
        system_addon = context.to_system_prompt()
    """

    SAFE_EXTENSIONS = {
        '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css',
        '.json', '.yaml', '.yml', '.toml', '.md', '.txt',
        '.sh', '.bat', '.sql', '.csv',
    }
    MAX_FILE_CHARS = 50_000

    def __init__(self, agent):
        self.agent = agent
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ace")
        self._screen_cache = {"content": "", "ts": 0.0}
        self._screen_cache_ttl = 5.0
        self._screen_cache_lock = threading.Lock()

    def gather(self, message: str) -> ContextBundle:
        """Main entry point. Gather all context in parallel, return bundle."""
        t0 = time.time()
        bundle = ContextBundle(query=message)

        file_paths = self._extract_file_paths(message)
        urls = self._extract_urls(message)

        futures = {}
        futures["screen"] = self._executor.submit(self._get_screen_context)
        futures["profile"] = self._executor.submit(self._get_user_profile)

        if getattr(self.agent, 'memory', None):
            futures["memory"] = self._executor.submit(self._get_memory_context, message)

        if getattr(self.agent, 'episodic_bridge', None):
            futures["episodic"] = self._executor.submit(self._get_episodic_context, message)

        if len(file_paths) > 3:
            logger.warning(
                f"[ACE] {len(file_paths)} files attached — reading first 3 only. "
                f"Skipped: {[str(p) for p in file_paths[3:]]}"
            )
        for i, fp in enumerate(file_paths[:3]):
            futures[f"file_{i}"] = self._executor.submit(self._read_file, fp)

        for i, url in enumerate(urls[:2]):
            futures[f"url_{i}"] = self._executor.submit(self._fetch_url, url)

        deadline = time.time() + GATHER_TIMEOUT
        for key, future in futures.items():
            remaining = max(0.0, deadline - time.time())
            try:
                block = future.result(timeout=remaining)
                if block and block.content:
                    bundle.blocks.append(block)
            except (TimeoutError, Exception) as e:
                logger.debug(f"[ACE] {key} failed: {e}")

        bundle.gather_time_ms = (time.time() - t0) * 1000
        logger.debug(f"[ACE] {len(bundle.blocks)} blocks in {bundle.gather_time_ms:.0f}ms")
        return bundle

    def _extract_file_paths(self, text: str) -> List[str]:
        patterns = [
            r'[A-Za-z]:[/\\][\w/\\.\-]+',
            r'(?:^|[\s"])(/[\w/.\-]+\.[\w]+)',
            r'(?:^|[\s"])(\.{0,2}/[\w/.\-]+)',
        ]
        paths = []
        for pattern in patterns:
            paths.extend(re.findall(pattern, text))
        return list(dict.fromkeys(p.strip('"\'') for p in paths if p))

    def _extract_urls(self, text: str) -> List[str]:
        return re.findall(r'https?://[^\s<>"\']+[^\s<>"\'\.,;:!?)]', text)

    def _get_screen_context(self) -> Optional[ContextBlock]:
        now = time.time()
        with self._screen_cache_lock:
            if now - self._screen_cache["ts"] < self._screen_cache_ttl:
                cached = self._screen_cache["content"]
                return ContextBlock("Screen", cached, priority=85) if cached else None
        try:
            tools = getattr(self.agent, 'tools', {})
            if "screenpipe" in tools:
                result = tools["screenpipe"].get_current_context()
                if result and result.get("success"):
                    content = (result.get("text") or result.get("content", ""))[:800]
                    with self._screen_cache_lock:
                        self._screen_cache.update({"content": content, "ts": now})
                    return ContextBlock("Screen", content, priority=85)
        except Exception as e:
            logger.debug(f"[ACE] Screen failed: {e}")
        with self._screen_cache_lock:
            self._screen_cache.update({"content": "", "ts": now})
        return None

    def _get_user_profile(self) -> Optional[ContextBlock]:
        try:
            mr = getattr(self.agent, 'memory_retriever', None)
            if mr and mr.user_profile:
                facts = ", ".join(f"{k}: {v}" for k, v in list(mr.user_profile.items())[:8])
                if facts:
                    return ContextBlock("User", facts, priority=100)
            # Fallback: read file
            profile_path = Path(__file__).parent.parent / "data" / "memory" / "user_profile.md"
            if profile_path.exists():
                lines = [
                    l.strip() for l in profile_path.read_text(encoding="utf-8").splitlines()
                    if ":" in l and not l.startswith("#") and l.strip()
                ]
                if lines:
                    return ContextBlock("User", " | ".join(lines[:6]), priority=100)
        except Exception as e:
            logger.debug(f"[ACE] Profile failed: {e}")
        return None

    def _get_memory_context(self, query: str) -> Optional[ContextBlock]:
        try:
            results = self.agent.memory.query(query, k=4)
            if results:
                content = "\n".join(f"- {r.content[:120]}" for r in results[:4])
                return ContextBlock("Memory", content, priority=70)
        except Exception as e:
            logger.debug(f"[ACE] Memory failed: {e}")
        return None

    def _get_episodic_context(self, query: str) -> Optional[ContextBlock]:
        try:
            context = self.agent.episodic_bridge.get_context_for_query(query)
            if context:
                return ContextBlock("Episodes", context[:500], priority=65)
        except Exception as e:
            logger.debug(f"[ACE] Episodic failed: {e}")
        return None

    def _read_file(self, path_str: str) -> Optional[ContextBlock]:
        try:
            p = Path(path_str)
            if not p.exists() or not p.is_file():
                return None
            if p.suffix.lower() not in self.SAFE_EXTENSIONS:
                return None
            if p.stat().st_size > self.MAX_FILE_CHARS * 4:
                return None
            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > self.MAX_FILE_CHARS:
                content = content[:self.MAX_FILE_CHARS] + f"\n...(truncated)"
            return ContextBlock(f"File:{p.name}", content, priority=80)
        except Exception as e:
            logger.debug(f"[ACE] File read failed {path_str}: {e}")
        return None

    def _fetch_url(self, url: str) -> Optional[ContextBlock]:
        if _httpx is None:
            return None
        try:
            resp = _httpx.get(url, timeout=3.0, follow_redirects=True)
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()[:600]
            return ContextBlock(f"URL:{url[:40]}", text, priority=50)
        except Exception as e:
            logger.debug(f"[ACE] URL fetch failed {url}: {e}")
        return None
