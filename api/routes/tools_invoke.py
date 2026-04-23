"""Generic tool introspection + invocation endpoint.

Complements the typed endpoints in ``tools_new.py``. Those remain canonical
for tools with request-layer safety checks (database SQL allowlist, shell
allowlist, SSRF-blocked API tester, path-containment audio/CSV, etc.). This
module exposes a *narrow* generic path for read-only / low-risk tools that
don't have typed routes yet — primarily for the Tool Playground UI.

Design:
  * **Explicit allowlist** of ``(tool_name, method_name)`` pairs. Everything
    else is rejected with ``"not invokable"``. Tools with typed routes in
    ``tools_new.py`` are NOT included here — that would regress their
    safety layer.
  * **Deferred-tool resolution** via ``loader.ensure_tool()`` so the list
    covers tools that aren't eagerly loaded into ``agent.tools``.
  * **Signature filtering** — ``inspect.signature`` drops unknown kwargs
    rather than passing them to the tool (which would raise TypeError).
  * **Rate limited** — generic dispatch is a foot-gun without a cap.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import EndpointRateLimiter, get_agent, safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["tools"], dependencies=[Depends(require_api_key)])

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------
# Only (tool, method) pairs listed here are invokable via this generic path.
# Gate rule: read-only OR low-risk idempotent actions only. Any method that
# writes, sends, executes, or leaves the sandbox MUST go through a typed
# endpoint in tools_new.py (where request-layer checks live), not here.
#
# Excluded on purpose (have typed routes — DO NOT ADD HERE):
#   database, shell_executor, email, api_tester, code_executor,
#   audio_transcriber, calendar (add/remove), task_manager (add/update/remove),
#   spaced_repetition (add/answer), clipboard (capture), screen_reader,
#   research (save), filesystem (write), code_edit, git (commit/push/pull),
#   tool_builder, marketplace (install), deploy, scaffold, windows_control,
#   system_control, image_gen, voice, voice_synth, notifications (create),
#   meeting_intel, local_rag (index).

_INVOKABLE: Dict[str, frozenset] = {
    # --- Web search variants ---
    "web_search": frozenset({"search", "news", "images", "instant_answer", "run"}),
    "brave_search": frozenset({"search", "news", "images", "instant_answer", "run"}),
    "tavily_search": frozenset({"search", "run"}),
    "firecrawl": frozenset({"scrape", "search"}),

    # --- Market / reference data ---
    "crypto_price": frozenset({"get_price", "get_multiple_prices", "run"}),

    # --- Academic / research ---
    "arxiv_search": frozenset({
        "search", "get_paper", "get_abstract", "search_by_author",
        "get_recent", "summarize_search", "related_papers", "bibtex",
        "get_cache_stats", "run",
    }),
    "research": frozenset({
        "search", "list_research", "stats", "search_by_tag",
        "read", "list_skills",
    }),

    # --- Read-only code / git ---
    "code_search": frozenset({
        "grep", "glob", "find_definition", "find_references",
        "project_structure", "detect_project_type", "semantic_search", "search",
    }),
    "git": frozenset({
        "status", "log", "diff", "branch", "list_branches",
        "show", "current_branch", "recent_commits",
    }),

    # --- Knowledge / context read ---
    "knowledge_graph": frozenset({
        "query", "get_node", "get_node_by_label", "get_clusters",
    }),
    "obsidian": frozenset({"search", "list_notes", "read", "stats"}),
    "life_logger": frozenset({
        "search", "summary", "daily_summary", "weekly_summary",
    }),

    # --- GitHub / log / predictive (read-only) ---
    "github": frozenset({
        "weekly_summary", "list_prs", "list_issues", "recent_commits",
        "repo_stats", "search_code", "my_notifications",
    }),
    "log_analyst": frozenset({"analyze"}),
    "predictive_tasks": frozenset({"predict", "stats"}),

    # --- Skills / tool discovery ---
    "tool_search": frozenset({"search"}),
    "load_skill": frozenset({"list_skills", "load", "search"}),
}


_rate_limiter = EndpointRateLimiter(max_per_minute=30)


# ---------------------------------------------------------------------------
# Introspection — /api/tools/registry
# ---------------------------------------------------------------------------


class MethodParamSpec(BaseModel):
    name: str
    type: str
    required: bool
    default: Optional[Any] = None


class MethodSpec(BaseModel):
    name: str
    doc: Optional[str] = None
    params: List[MethodParamSpec]


class ToolSpec(BaseModel):
    name: str
    description: str
    loaded: bool
    methods: List[MethodSpec]


def _describe_param(name: str, p: inspect.Parameter) -> MethodParamSpec:
    """Render a parameter into a JSON-safe description."""
    has_default = p.default is not inspect.Parameter.empty
    default: Any = None
    if has_default and p.default is not None:
        try:
            # Round-trip through JSON to drop non-serializable defaults.
            import json as _json
            _json.dumps(p.default)
            default = p.default
        except (TypeError, ValueError):
            default = str(p.default)

    # Annotation -> short string
    ann = p.annotation
    if ann is inspect.Parameter.empty:
        type_str = "any"
    else:
        type_str = getattr(ann, "__name__", None) or str(ann)
    return MethodParamSpec(name=name, type=type_str, required=not has_default, default=default)


def _describe_method(method_name: str, fn: Any) -> Optional[MethodSpec]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params: List[MethodParamSpec] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        params.append(_describe_param(pname, param))
    doc = (inspect.getdoc(fn) or "").strip().split("\n\n", 1)[0] or None
    return MethodSpec(name=method_name, doc=doc, params=params)


def _get_tool_instance(tool_name: str) -> Optional[Any]:
    """Return a loaded tool instance, resolving from the deferred registry if needed."""
    agent = get_agent()
    if agent is None:
        return None
    if tool_name in agent.tools:
        return agent.tools[tool_name]
    # Try the deferred registry via loader.ensure_tool
    try:
        from aura.tools.loader import ensure_tool
    except ImportError:
        return None
    try:
        brain = getattr(agent, "brain", None)
        return ensure_tool(agent.tools, tool_name, brain=brain)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TOOLS-INVOKE] ensure_tool(%s) failed: %s", tool_name, exc)
        return None


def _deferred_description(tool_name: str) -> str:
    """Fetch the deferred-registry description (or empty string)."""
    try:
        from aura.tools.deferred_registry import deferred_registry
        for entry in deferred_registry.list_all():
            if entry.get("name") == tool_name:
                return entry.get("description", "") or ""
    except Exception:  # noqa: BLE001
        pass
    return ""


@router.get("/registry", response_model=List[ToolSpec])
async def tools_registry():
    """List invokable tools and their allowlisted method signatures.

    Only methods in the hardcoded allowlist are returned — this is not a
    full reflection API, it's the playground's menu.
    """
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _build_registry_sync)
    except Exception as exc:  # noqa: BLE001
        logger.error("[TOOLS-INVOKE] registry failed: %s", exc, exc_info=True)
        raise HTTPException(500, safe_error_detail(exc, "Failed to build tool registry"))


def _build_registry_sync() -> List[ToolSpec]:
    agent = get_agent()
    already_loaded = set(getattr(agent, "tools", {}).keys()) if agent is not None else set()

    specs: List[ToolSpec] = []
    for tool_name in sorted(_INVOKABLE.keys()):
        # Don't resolve deferred tools just to list them — describe from
        # the class if we can, and flag loaded=False. The invoke endpoint
        # resolves on demand.
        tool = agent.tools.get(tool_name) if agent is not None else None
        loaded = tool is not None

        methods: List[MethodSpec] = []
        if tool is None:
            # Try to peek at the class without fully instantiating heavy deps.
            # Best-effort: if loader registered it, resolving is cheap enough
            # for the common tools in the allowlist.
            tool = _get_tool_instance(tool_name)
            loaded = tool is not None and tool_name in already_loaded

        if tool is not None:
            for method_name in sorted(_INVOKABLE[tool_name]):
                fn = getattr(tool, method_name, None)
                if fn is None or not callable(fn):
                    continue
                spec = _describe_method(method_name, fn)
                if spec is not None:
                    methods.append(spec)

        description = (
            (inspect.getdoc(tool) or "").strip().split("\n\n", 1)[0]
            if tool is not None
            else _deferred_description(tool_name)
        )

        specs.append(ToolSpec(
            name=tool_name,
            description=description or tool_name,
            loaded=loaded,
            methods=methods,
        ))
    return specs


# ---------------------------------------------------------------------------
# Invocation — /api/tools/invoke
# ---------------------------------------------------------------------------


class InvokeRequest(BaseModel):
    tool: str = Field(..., max_length=64)
    method: str = Field(..., max_length=64)
    args: Optional[Dict[str, Any]] = None
    timeout: int = Field(30, ge=1, le=120)


class InvokeResponse(BaseModel):
    success: bool
    tool: str
    method: str
    result: Optional[Any] = None
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None


@router.post("/invoke", response_model=InvokeResponse)
async def tools_invoke(request: InvokeRequest):
    """Invoke an allowlisted tool method.

    Rejects anything outside ``_INVOKABLE``. Filters unknown kwargs via
    ``inspect.signature`` before calling. Runs the blocking tool call in
    the default executor with a bounded timeout.
    """
    _rate_limiter.check()

    allowed_methods = _INVOKABLE.get(request.tool)
    if allowed_methods is None:
        raise HTTPException(403, f"Tool '{request.tool}' is not exposed via generic invoke")
    if request.method not in allowed_methods:
        raise HTTPException(403, f"Method '{request.method}' on '{request.tool}' is not invokable")

    import time as _time
    t0 = _time.monotonic()

    try:
        tool = _get_tool_instance(request.tool)
    except Exception as exc:  # noqa: BLE001
        logger.error("[TOOLS-INVOKE] resolve %s failed: %s", request.tool, exc, exc_info=True)
        raise HTTPException(500, safe_error_detail(exc, "Tool resolution failed"))

    if tool is None:
        raise HTTPException(503, f"Tool '{request.tool}' is not available")

    fn = getattr(tool, request.method, None)
    if fn is None or not callable(fn):
        raise HTTPException(404, f"Tool '{request.tool}' has no method '{request.method}'")

    # Filter kwargs to the subset the method actually accepts.
    try:
        sig = inspect.signature(fn)
        accepts_var_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        raw_args = request.args or {}
        if accepts_var_kwargs:
            filtered_args = dict(raw_args)
        else:
            accepted = {
                name for name, p in sig.parameters.items()
                if name != "self" and p.kind is not inspect.Parameter.VAR_POSITIONAL
            }
            filtered_args = {k: v for k, v in raw_args.items() if k in accepted}
    except (TypeError, ValueError):
        filtered_args = request.args or {}

    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: fn(**filtered_args)),
            timeout=request.timeout,
        )
    except asyncio.TimeoutError:
        return InvokeResponse(
            success=False,
            tool=request.tool,
            method=request.method,
            error=f"Tool call exceeded {request.timeout}s timeout",
            elapsed_ms=int((_time.monotonic() - t0) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[TOOLS-INVOKE] %s.%s failed: %s",
            request.tool, request.method, exc, exc_info=True,
        )
        return InvokeResponse(
            success=False,
            tool=request.tool,
            method=request.method,
            error=safe_error_detail(exc, "Tool call failed"),
            elapsed_ms=int((_time.monotonic() - t0) * 1000),
        )

    elapsed_ms = int((_time.monotonic() - t0) * 1000)

    # Tool methods generally return dicts. Preserve structure; if they return
    # a non-serializable object, coerce to string so the API contract holds.
    import json as _json
    try:
        _json.dumps(result)
        safe_result: Any = result
    except (TypeError, ValueError):
        safe_result = str(result)

    # Infer success: most Aura tools return {"success": bool, ...}. If absent,
    # treat a returned value as success unless it was None.
    inferred_success = True
    if isinstance(result, dict) and "success" in result:
        inferred_success = bool(result.get("success"))
    elif result is None:
        inferred_success = False

    return InvokeResponse(
        success=inferred_success,
        tool=request.tool,
        method=request.method,
        result=safe_result,
        elapsed_ms=elapsed_ms,
    )
