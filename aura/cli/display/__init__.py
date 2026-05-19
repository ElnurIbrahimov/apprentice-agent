"""Rich-based display for AURA CLI — clean, polished, Claude Code / OpenCode aesthetic.

This package replaces the former single-file ``aura/cli/display.py``. The
public API is unchanged — every name that used to be importable from
``aura.cli.display`` still is. Submodules group related rendering chunks:

  - ``display.help`` — the /? help screen
  - ``display.streaming`` — ``StreamingResponse`` + streaming splitters
  - ``display.checkpoint_picker`` — checkpoint UI + rewind result

Consumers should continue to ``from aura.cli.display import show_response`` etc.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from aura.cli.tool_icons import get_tool_icon
from aura.cli.tool_output import ToolOutputRenderer

logger = logging.getLogger(__name__)

_no_color = os.environ.get("NO_COLOR") is not None
console: Console = Console(highlight=True, soft_wrap=True, no_color=_no_color)


@functools.lru_cache(maxsize=1)
def _get_tool_renderer() -> ToolOutputRenderer:
    return ToolOutputRenderer(console=console)


def _get_code_theme() -> str:
    """Get the current code theme from the theme system."""
    try:
        from aura.cli.themes import get_theme
        return get_theme().code_theme
    except (ImportError, AttributeError):
        return "monokai"


def _get_theme_colors() -> dict:
    """Get semantic colors from current theme."""
    try:
        from aura.cli.themes import get_theme
        theme = get_theme()
        return {
            "accent": theme.accent,
            "accent_dim": theme.accent_dim,
            "success": theme.success,
            "error": theme.error,
            "warning": theme.warning,
            "info": theme.info,
            "tool": theme.tool_color,
            "tool_success": theme.tool_success,
            "tool_error": theme.tool_error,
            "tool_pending": theme.tool_pending,
            "permission_border": theme.permission_border,
            "permission_accent": theme.permission_accent,
            "text_secondary": theme.text_secondary,
            "text_muted": theme.text_muted,
        }
    except (ImportError, AttributeError):
        from aura.cli.themes import AuraTheme
        _f = AuraTheme(name="fallback")
        return {
            "accent": _f.accent,
            "accent_dim": _f.accent_dim,
            "success": _f.success,
            "error": _f.error,
            "warning": _f.warning,
            "info": _f.info,
            "tool": _f.tool_color,
            "tool_success": _f.tool_success,
            "tool_error": _f.tool_error,
            "tool_pending": _f.tool_pending,
            "permission_border": _f.permission_border,
            "permission_accent": _f.permission_accent,
            "text_secondary": _f.text_secondary,
            "text_muted": _f.text_muted,
        }


# ─────────────────────────────────────────────────────────
# Banner & Welcome
# ─────────────────────────────────────────────────────────

def show_banner() -> None:
    """Display startup screen — cohesive bordered panel with logo, info, shortcuts."""
    from aura import __version__

    from ..banner import _LOGO_LINES, _apply_gradient

    colors = _get_theme_colors()
    try:
        from aura.cli.themes import get_theme
        gradient = get_theme().gradient
    except (ImportError, AttributeError):
        from aura.cli.themes import AuraTheme
        _fb = AuraTheme(name="fallback")
        gradient = _fb.gradient

    content = Text()
    for line in _LOGO_LINES:
        content.append(" ")
        content.append_text(_apply_gradient(line, gradient))
        content.append("\n")
    content.append(f" \u2728 v{__version__}", style="dim")

    panel = Panel(
        content,
        border_style=colors["accent"],
        padding=(0, 1),
        width=min(console.width - 2, 50),
    )
    console.print()
    console.print(panel)


def show_welcome_info(agent: Any) -> None:
    """Show model, tools, and keyboard shortcuts below the banner."""
    colors = _get_theme_colors()
    model = "auto"
    try:
        model = agent.brain._model_override or "auto"
    except AttributeError:
        logger.debug("welcome_model_read_failed", exc_info=True)

    tool_count = 0
    try:
        tool_count = len(agent.tools)
    except (TypeError, AttributeError):
        logger.debug("welcome_tool_count_failed", exc_info=True)

    t = Text("  ")
    t.append(model, style=f"bold {colors['accent']}")
    if tool_count:
        t.append(f" \u00b7 {tool_count} tools", style="dim")
    console.print(t)

    info = Text("  ")
    # Collapse $HOME → ~ and cap at 60 chars to stop long paths wrapping
    # badly on 80-col terminals.
    cwd = os.getcwd().replace("\\", "/")
    _home = os.path.expanduser("~").replace("\\", "/")
    if cwd.lower().startswith(_home.lower()):
        cwd = "~" + cwd[len(_home):]
    if len(cwd) > 60:
        cwd = "…" + cwd[-59:]
    info.append(cwd, style="dim")

    has_cloud_key = bool(os.environ.get("OLLAMA_API_KEY"))
    if has_cloud_key:
        info.append(" \u00b7 ", style="dim")
        info.append("Ollama cloud", style="dim")
    else:
        info.append(" \u00b7 ", style="dim")
        info.append("Ollama local", style="dim")

    try:
        from aura.providers import list_all_provider_models
        model_count = len(list_all_provider_models())
        if model_count:
            info.append(f" ({model_count} models)", style="dim")
    except Exception:
        logger.debug("welcome_model_count_failed", exc_info=True)

    console.print(info)

    h = Text("  ")
    shortcuts = [
        ("/", "commands"),
        ("Alt+M", "model"),
        ("?", "help"),
        ("Shift+Tab", "perms"),
        ("Ctrl+K", "palette"),
    ]
    for i, (key, desc) in enumerate(shortcuts):
        if i > 0:
            h.append("  ", style="dim")
        h.append(key, style=f"bold {colors['accent']}")
        h.append(f" {desc}", style="dim")
    console.print(h)
    console.print()


# ─────────────────────────────────────────────────────────
# Status Bar
# ─────────────────────────────────────────────────────────

def show_status_bar(
    model: str = "auto",
    cost_usd: float = 0.0,
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    bg_indicator: str = "",
    research_indicator: str = "",
    watch_indicator: str = "",
    steering_queue: object = None,
    session_title: str = "",
    message_count: int = 0,
    project_type: str = "",
    **_unused: object,  # swallow legacy kwargs (mood_indicator) for back-compat
) -> None:
    """Update the persistent bottom toolbar."""
    from ..input import set_bottom_toolbar
    from ..status_bar import build_status_bar

    toolbar_parts = build_status_bar(
        model=model,
        cost_usd=cost_usd,
        token_used=token_used,
        token_limit=token_limit,
        permission_mode=permission_mode,
        bg_indicator=bg_indicator,
        research_indicator=research_indicator,
        watch_indicator=watch_indicator,
        steering_queue=steering_queue,
        session_title=session_title,
        message_count=message_count,
        project_type=project_type,
        as_ansi=True,
    )
    set_bottom_toolbar(toolbar_parts)


# ─────────────────────────────────────────────────────────
# Thinking / Spinner
# ─────────────────────────────────────────────────────────

def show_thinking(label: str | None = None, step: int | None = None) -> Live:
    """Context manager — shows animated shimmer spinner while agent runs."""
    from ..spinner import AuraSpinner
    spinner = AuraSpinner(label=label, step=step)
    live = Live(spinner, console=console, refresh_per_second=12, transient=True)
    live._aura_spinner = spinner
    return live


_TOOL_PHASE_VERBS: dict[str, str] = {
    "web_search": "Searching the web",
    "search_web": "Searching the web",
    "browse": "Browsing",
    "browse_url": "Browsing",
    "read_file": "Reading files",
    "edit_file": "Editing code",
    "write_file": "Writing code",
    "execute": "Running code",
    "run_command": "Running command",
    "shell": "Running shell",
    "analyze": "Analyzing",
    "summarize": "Summarizing",
    "translate": "Translating",
    "calculate": "Calculating",
    "research": "Researching",
    "deep_research": "Deep researching",
    "memory_recall": "Remembering",
    "memory_store": "Storing memory",
    "git": "Running git",
    "grep": "Searching code",
    "glob": "Finding files",
    "code_search": "Searching codebase",
}


def get_thinking_label(tool_name: str | None = None) -> str:
    """Return a phase-aware thinking label based on the current tool."""
    if tool_name and tool_name in _TOOL_PHASE_VERBS:
        return _TOOL_PHASE_VERBS[tool_name]
    return "Thinking..."


# ─────────────────────────────────────────────────────────
# Tool Call Display
# ─────────────────────────────────────────────────────────

def show_tool_call(
    tool_name: str,
    description: str = "",
    result: Any = None,
    elapsed: float = 0.0,
    status: str = "success",
    step: int = 0,
    max_steps: int = 0,
) -> None:
    """Print a tool call with step counter, status dot, label, and description."""
    colors = _get_theme_colors()
    time_str = f" ({elapsed:.1f}s)" if elapsed >= 0.5 else ""

    if tool_name in ("edit_file", "write_file") and result and isinstance(result, dict) and result.get("diff"):
        try:
            from aura.cli.diff_viewer import render_diff_compact
            filename = result.get("path", "file")
            filename = filename.split("/")[-1].split("\\")[-1]
            summary = render_diff_compact(
                result.get("old_content", ""),
                result.get("new_content", ""),
                filename=filename,
                elapsed=elapsed,
            )
            dot_char, dot_style = _get_status_dot(status, colors)
            step_prefix = _step_prefix(step, max_steps, colors)
            console.print(f"  {step_prefix}[{dot_style}]{dot_char}[/{dot_style}] {summary}")
            return
        except (ImportError, ValueError, KeyError, TypeError):
            pass

    dot_char, dot_style = _get_status_dot(status, colors)
    label = get_tool_icon(tool_name)

    line = Text()
    line.append("  \u2503 ", style=f"{colors['accent']}")
    if step > 0:
        line.append(f"[{step}] ", style=f"bold {colors['accent']}")
    line.append(dot_char, style=dot_style)
    line.append(f" {label}", style=f"bold {colors['tool']}")
    if description:
        line.append(f" {description}", style="dim")
    if time_str:
        line.append(time_str, style="dim")
    console.print(line)

    # Register as a numbered block so /blocks and /copy can reference it
    _register_block("tool_call", f"{label} {description}", description or label)


def show_tool_result_inline(tool_name: str, result: Any) -> None:
    """Show a compact inline tool result — just enough to see what happened."""
    if not result:
        return

    colors = _get_theme_colors()
    B = f"  [{colors['accent']}]\u2503[/{colors['accent']}]   "

    try:
        import json
        if isinstance(result, str):
            parsed = json.loads(result)
        else:
            parsed = result
        if not isinstance(parsed, dict):
            return

        if parsed.get("error"):
            err_msg = str(parsed["error"])
            # Tail-truncate: the last 120 chars of a stack trace or stderr
            # carry the actual failure line; the head is boilerplate.
            if len(err_msg) > 120:
                err_msg = "\u2026" + err_msg[-120:]
            console.print(f"{B}[{colors['error']}]\u2717 {err_msg}[/{colors['error']}]")
            return

        if tool_name in ("grep", "glob", "code_search", "search", "find"):
            results_list = parsed.get("results", parsed.get("matches", []))
            if isinstance(results_list, list):
                count = len(results_list)
                console.print(f"{B}[dim]{count} {'match' if count == 1 else 'matches'}[/dim]")
                return

        if tool_name in ("web_search", "browse", "web_fetch", "browse_url"):
            status_code = parsed.get("status_code", 200)
            url = parsed.get("url", "")
            if url:
                short_url = url[:60] + "\u2026" if len(url) > 60 else url
                console.print(f"{B}[dim]{status_code} {short_url}[/dim]")
                return

        output = parsed.get("output", parsed.get("content", parsed.get("result", "")))
        if not output or not isinstance(output, str):
            return

        lines = output.splitlines()
        line_count = len(lines)

        if tool_name in ("read_file", "cat"):
            lang = parsed.get("language", "")
            extras = []
            if line_count > 0:
                extras.append(f"{line_count} lines")
            if lang:
                extras.append(lang)
            if extras:
                sep = " │ "
                console.print(f"{B}[dim]{sep.join(extras)}[/dim]")

        elif tool_name in ("shell", "shell_executor", "bash", "run", "run_command", "execute"):
            exit_code = parsed.get("exit_code", parsed.get("returncode", 0)) or 0
            if exit_code == 0:
                ic = f"[{colors['success']}]\u2713[/{colors['success']}]"
            else:
                ic = f"[{colors['error']}]\u2717 exit {exit_code}[/{colors['error']}]"
            meaningful = [l.strip() for l in lines if l.strip()][-3:]
            if meaningful:
                console.print(f"{B}{ic} [dim]({line_count} lines)[/dim]")
                for ml in meaningful:
                    if len(ml) > 100:
                        ml = ml[:100] + "\u2026"
                    console.print(f"{B}[dim]{ml}[/dim]")
            else:
                console.print(f"{B}{ic} [dim](no output)[/dim]")

        elif tool_name in ("edit_file", "write_file"):
            added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
            if added or removed:
                console.print(f"{B}[{colors['success']}]+{added}[/{colors['success']}]/[{colors['error']}]-{removed}[/{colors['error']}] [dim]lines changed[/dim]")
            elif line_count > 0:
                console.print(f"{B}[dim]{line_count} lines written[/dim]")

        elif tool_name in ("list_dir", "ls"):
            items = [l.strip() for l in lines if l.strip()]
            count = len(items)
            preview = ", ".join(items[:6])
            if count > 6:
                preview += f" +{count - 6} more"
            if preview:
                console.print(f"{B}[dim]{count} items: {preview}[/dim]")

        else:
            if line_count > 3:
                console.print(f"{B}[dim]{line_count} lines[/dim]")
            elif line_count > 0:
                for l in lines[:3]:
                    if len(l) > 100:
                        l = l[:100] + "\u2026"
                    console.print(f"{B}[dim]{l}[/dim]")

    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        pass

    # Register tool result as a block for /blocks navigation
    try:
        result_str = json.dumps(result) if not isinstance(result, str) else result
        title = f"{tool_name} result"
        if isinstance(result, dict):
            summary = result.get("output", result.get("content", result.get("result", "")))
            if isinstance(summary, str) and summary:
                title = summary[:80].replace("\n", " ").strip()
        _register_block("tool_result", title, result_str[:5000])
    except Exception:
        pass


def _step_prefix(step: int, max_steps: int, colors: dict) -> str:
    """Build a step counter prefix string."""
    if step <= 0:
        return ""
    return f"[bold {colors['accent']}]Step {step}[/bold {colors['accent']}] [dim]\u00b7[/dim] "


def _get_status_dot(status: str, colors: dict) -> tuple[str, str]:
    """Return (character, style) for a tool status indicator."""
    if status == "success":
        return "\u25cf", colors["tool_success"]
    elif status == "error":
        return "\u25cf", colors["tool_error"]
    elif status == "running":
        return "\u25cb", colors["tool_pending"]
    elif status == "pending":
        return "\u25cb", colors["text_muted"]
    else:
        return "\u25cf", colors["tool_success"]


# ─────────────────────────────────────────────────────────
# Block Output Helpers
# ─────────────────────────────────────────────────────────

def _register_block(block_type: str, title: str, content: str, metadata: dict | None = None) -> int | None:
    """Register an output block and return its ID, or None if no BlockManager is active."""
    try:
        from aura.cli.context import get_ctx
        ctx = get_ctx()
        if ctx is None or ctx.blocks is None:
            return None
        return ctx.blocks.add(block_type, title, content, metadata=metadata)
    except Exception:
        return None


def _show_block_id(block_id: int | None) -> None:
    """Print a dim block ID marker after the output line."""
    if block_id is not None:
        console.print(f"  [dim]#{block_id}[/dim]")


# ─────────────────────────────────────────────────────────
# Permission Prompt
# ─────────────────────────────────────────────────────────

def show_permission_prompt(
    action: str,
    detail: str = "",
    options: list[tuple[str, str]] | None = None,
) -> str:
    """Show a styled permission prompt with numbered options."""
    colors = _get_theme_colors()

    if options is None:
        options = [
            ("allow", "Allow this action"),
            ("session", "Allow for this session"),
            ("deny", "No, skip this"),
        ]

    console.print()

    header = Text("  ")
    header.append("\u25b3 ", style=f"bold {colors['warning']}")
    header.append("Permission required", style=f"bold {colors['permission_accent']}")
    console.print(header)

    if action:
        console.print(f"    {action}", style=f"{colors['tool']}")
    if detail:
        if len(detail) > 200:
            detail = detail[:200] + "\u2026"
        console.print(f"    {detail}", style="dim")

    console.print()

    for i, (_key, label) in enumerate(options, 1):
        if i == 1:
            console.print(f"    [{colors['permission_accent']}]> {i}. {label}[/{colors['permission_accent']}]")
        else:
            console.print(f"      {i}. {label}", style="dim")

    console.print()

    try:
        choice = console.input(f"  [{colors['text_muted']}]Choose (1-{len(options)}), Enter for 1, Esc to deny: [/{colors['text_muted']}]")
        choice = choice.strip()

        if not choice or choice == "1":
            return options[0][0]
        elif choice == "y" or choice == "yes":
            return options[0][0]
        elif choice == "n" or choice == "no":
            return "deny"

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            pass

        return "deny"
    except (EOFError, KeyboardInterrupt):
        return "deny"


# ─────────────────────────────────────────────────────────
# Response Rendering
# ─────────────────────────────────────────────────────────

def show_response(text: str, model: str = "", stream: bool = True) -> None:
    """Render agent response as clean markdown.

    The ``stream`` parameter is accepted for API compatibility but no longer
    simulates a streaming effect — the real stream comes via
    ``StreamingResponse`` when the LLM emits chunks. Simulated streaming
    (time.sleep between word chunks of an already-complete string) only
    slowed down slow terminals for no real user benefit.
    """
    code_theme = _get_code_theme()

    console.print()
    max_width = min(console.width - 4, 100)

    try:
        md = Markdown(text, code_theme=code_theme)
    except (ValueError, TypeError):
        md = Text(text)
    console.print(Padding(md, (0, 2)), width=max_width)

    if model:
        console.print(f"  [dim]{model}[/dim]")

    console.print()

    # Register the response as a numbered block
    _register_block("response", text[:80].replace("\n", " ").strip(), text)


def show_response_attribution(model: str, elapsed: float, tokens: int = 0) -> None:
    """Show model + time + tokens after each response."""
    from ..spinner import _format_elapsed
    _get_theme_colors()

    t = Text("  ")
    model_short = model.replace(":cloud", "").replace(":latest", "")
    if len(model_short) > 30:
        model_short = model_short[:27] + "\u2026"
    t.append(model_short, style="dim")
    t.append(f" \u00b7 {_format_elapsed(elapsed)}", style="dim")
    if tokens > 0:
        if tokens >= 1000:
            t.append(f" \u00b7 {tokens / 1000:.1f}K tokens", style="dim")
        else:
            t.append(f" \u00b7 {tokens} tokens", style="dim")
    console.print(t)


def show_context_summary(
    memory_count: int = 0,
    kg_topic: str = "",
    mood: str = "",
    model: str = "",
    tool_count: int = 0,
    memory_snippets: Optional[list[str]] = None,
) -> None:
    """Show an ultra-minimal context line before responses."""
    if memory_snippets is None:
        memory_snippets = []
    colors = _get_theme_colors()

    parts = []
    if memory_count > 0:
        parts.append(f"{memory_count} memories")
    if kg_topic:
        parts.append(f'KG: "{kg_topic}"')
    if mood:
        parts.append(mood)
    if model:
        short = model.replace(":cloud", "").replace(":latest", "")
        if len(short) > 25:
            short = short[:22] + "\u2026"
        parts.append(short)

    if not parts:
        return

    t = Text("  ")
    t.append("context: ", style=f"{colors['text_muted']}")
    t.append("  \u2502  ".join(parts), style="dim")
    console.print(t)


# ─────────────────────────────────────────────────────────
# Errors / Info / Warnings
# ─────────────────────────────────────────────────────────

def classify_exception(exc: BaseException) -> tuple[str, str | None]:
    """Classify a raw exception into (user-facing message, hint).

    Raw ``str(exc)`` surfaces noise like "Only one live display may be
    active at once" to end users. This maps common failure modes to short
    friendly messages with an actionable next step. If no rule matches,
    returns the raw string and no hint.
    """
    name = type(exc).__name__
    msg = str(exc) or name

    if name == "LiveError" or "one live display" in msg.lower():
        return ("Display glitch while rendering the last turn.",
                "Try again. If it persists, run /clear to reset.")

    if name in ("ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError",
                "ConnectionError", "ConnectionRefusedError", "ConnectionResetError"):
        return (f"Network hiccup reaching the model provider ({name}).",
                "Check your connection and retry. `ollama serve` may need restarting.")
    if name == "HTTPStatusError" or " 429" in msg or "Too Many Requests" in msg:
        return ("Model provider rate-limited the request.",
                "Wait a few seconds and retry, or switch models via /model.")
    if any(code in msg for code in (" 500", " 502", " 503", " 504")):
        return ("Model provider returned a server error.",
                "Retry in a moment, or switch models via /model.")

    if name == "JSONDecodeError":
        return ("Model returned malformed JSON.",
                "Retry; smaller models sometimes struggle with strict JSON.")

    if name == "FileNotFoundError":
        return (f"File not found: {msg}", None)
    if name == "PermissionError":
        return (f"Permission denied: {msg}",
                "Check ACLs or whether another process holds the file.")

    if name in ("KeyboardInterrupt", "CancelledError"):
        return ("Aborted.", None)

    return (msg, None)


def show_error(message, *, hint: str | None = None) -> None:
    """Display error — themed line, optional dim hint below.

    If an exception is passed, it is run through classify_exception() so
    callers don't have to repeat the mapping. Pass a plain string to
    bypass classification.
    """
    colors = _get_theme_colors()
    if isinstance(message, BaseException):
        text, auto_hint = classify_exception(message)
        if hint is None:
            hint = auto_hint
        message = text
    console.print(f"  [{colors['error']}]\u2717[/{colors['error']}] {message}")
    if hint:
        console.print(f"    [dim]{hint}[/dim]")
    _register_block("error", str(message)[:120], str(message))


def show_info(message: str) -> None:
    """Display info message — minimal dim text."""
    console.print(f"  [dim]{message}[/dim]")
    _register_block("info", str(message)[:120], str(message))


def show_warning(message: str) -> None:
    """Display warning with triangle icon."""
    colors = _get_theme_colors()
    console.print(f"  [{colors['warning']}]\u25b3[/{colors['warning']}] {message}")
    _register_block("info", str(message)[:120], str(message))


# ─────────────────────────────────────────────────────────
# Re-exports from submodules (preserves the original public API)
# ─────────────────────────────────────────────────────────
# Imported last so the submodules can freely `from aura.cli.display import ...`
# the names defined above without hitting a partial-module state.
from .checkpoint_picker import show_checkpoint_picker, show_rewind_result  # noqa: E402
from .help import show_help  # noqa: E402
from .streaming import StreamingResponse, _split_for_streaming, _split_into_blocks  # noqa: E402

__all__ = [
    "StreamingResponse",
    "_split_for_streaming",
    "_split_into_blocks",
    "console",
    "get_thinking_label",
    "show_banner",
    "show_checkpoint_picker",
    "show_context_summary",
    "show_error",
    "show_help",
    "show_info",
    "show_permission_prompt",
    "show_response",
    "show_response_attribution",
    "show_rewind_result",
    "show_status_bar",
    "show_thinking",
    "show_tool_call",
    "show_tool_result_inline",
    "show_warning",
    "show_welcome_info",
]
