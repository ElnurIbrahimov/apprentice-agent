"""
Shared thread pool registry for Aura.

Three pools, each with a specific role:
  - llm_pool()   (4 workers): All Ollama/LLM inference calls
                                (kept low to respect Ollama Pro's 3-concurrent-model limit)
  - bg_pool()     (8 workers): Non-urgent background work (writes, state updates,
                                memory indexing, parallel context gather)
  - tool_pool()   (4 workers): Synchronous tool execution (CLI tools, sub-agents,
                                search/fetch in deep_research)

Usage:
    from aura.pools import llm_pool, bg_pool, tool_pool
    future = llm_pool().submit(some_function, arg1, arg2)

Pools are created lazily on first access to avoid spawning threads at import time.
Cleanup is registered via atexit.
"""

import asyncio
import atexit
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_llm_pool: Optional[ThreadPoolExecutor] = None
_bg_pool: Optional[ThreadPoolExecutor] = None
_tool_pool: Optional[ThreadPoolExecutor] = None


def llm_pool() -> ThreadPoolExecutor:
    """Pool for all LLM/Ollama calls. 4 workers.

    Kept low to avoid overwhelming Ollama Pro's 3-concurrent-model limit.
    More workers just queue behind each other and waste threads.
    """
    global _llm_pool
    if _llm_pool is None:
        with _lock:
            if _llm_pool is None:
                _llm_pool = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="llm_worker"
                )
                logger.debug("[pools] LLM pool created (4 workers)")
    return _llm_pool


def bg_pool() -> ThreadPoolExecutor:
    """Pool for background non-urgent work. 8 workers."""
    global _bg_pool
    if _bg_pool is None:
        with _lock:
            if _bg_pool is None:
                _bg_pool = ThreadPoolExecutor(
                    max_workers=8, thread_name_prefix="aura-bg"
                )
                logger.debug("[pools] BG pool created (8 workers)")
    return _bg_pool


def tool_pool() -> ThreadPoolExecutor:
    """Pool for synchronous tool execution. 4 workers."""
    global _tool_pool
    if _tool_pool is None:
        with _lock:
            if _tool_pool is None:
                _tool_pool = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="aura-tool"
                )
                logger.debug("[pools] Tool pool created (4 workers)")
    return _tool_pool


def _shutdown_all():
    kwargs = {"wait": False}
    if sys.version_info >= (3, 9):
        kwargs["cancel_futures"] = True
    # Snapshot pool refs under lock to avoid race with lazy initialization
    with _lock:
        pools = [("llm", _llm_pool), ("bg", _bg_pool), ("tool", _tool_pool)]
    for name, pool in pools:
        if pool is not None:
            try:
                pool.shutdown(**kwargs)
                logger.debug(f"[pools] {name} pool shut down")
            except Exception:
                logger.warning(f"[pools] {name} pool shutdown failed", exc_info=True)


atexit.register(_shutdown_all)


# ---- Fire-and-forget async tasks (prevents GC of unfinished tasks) --------

_async_background_tasks: set[asyncio.Task] = set()


def fire_and_forget(coro) -> asyncio.Task | None:
    """Schedule a coroutine as a background task with GC protection.

    Returns the task so callers can optionally await it, or None if no
    running event loop is available (e.g., called from a sync context).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("[pools] fire_and_forget: no running event loop, dropping coroutine")
        coro.close()  # Prevent "coroutine was never awaited" warning
        return None
    task = loop.create_task(coro)
    _async_background_tasks.add(task)
    task.add_done_callback(_async_background_tasks.discard)
    return task


# ---- Background submit with fallback (moved from brain.py 2026-04-06) ----

_BG_FALLBACK_SEM = threading.Semaphore(8)


def bg_submit(fn, *args, **kwargs):
    """Submit work to the background pool, with fallback for shutdown."""
    try:
        bg_pool().submit(fn, *args, **kwargs)
    except RuntimeError:
        # Pool shut down — run in daemon thread as last resort, capped at 8
        if not _BG_FALLBACK_SEM.acquire(blocking=False):
            logger.warning(
                "bg_submit: fallback thread cap reached (8), dropping task %s", fn
            )
            return
        def _run_and_release():
            try:
                fn(*args, **kwargs)
            finally:
                _BG_FALLBACK_SEM.release()
        threading.Thread(target=_run_and_release, daemon=True).start()
