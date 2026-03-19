"""Rich-based display for AURA CLI — panels, syntax highlighting, streaming."""

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

from aura.cli.tool_output import ToolOutputRenderer, format_elapsed

console = Console(highlight=True, soft_wrap=True)

# Module-level tool output renderer (lazy init)
_tool_renderer = None

def _get_tool_renderer():
    global _tool_renderer
    if _tool_renderer is None:
        _tool_renderer = ToolOutputRenderer(console=console)
    return _tool_renderer


def show_banner():
    """Display ASCII art banner with gradient colors."""
    from .banner import get_banner, get_welcome_line
    from aura import __version__
    width = console.size.width
    console.print(get_banner(width))
    console.print(get_welcome_line(__version__))
    console.print()


def show_welcome_info(agent):
    """Show a brief info line after the banner: model, session, tool count."""
    model = "auto"
    try:
        model = agent.brain._model_override or "auto"
    except Exception:
        pass

    tool_count = 0
    try:
        tool_count = len(agent.tools)
    except Exception:
        pass

    session = "new"
    try:
        if hasattr(agent, 'memory') and hasattr(agent.memory, 'session_id'):
            sid = agent.memory.session_id
            if sid:
                session = str(sid)[:8]
    except Exception:
        pass

    info = Text()
    info.append("  Model: ", style="dim")
    info.append(model, style="dim bold")
    info.append("  |  ", style="dim")
    info.append("Session: ", style="dim")
    info.append(session, style="dim bold")
    info.append("  |  ", style="dim")
    info.append("Tools: ", style="dim")
    info.append(f"{tool_count} loaded", style="dim bold")
    console.print(info)
    console.print()


def show_status_bar(
    model: str = "auto",
    project_type: str = "",
    session_title: str = "",
    message_count: int = 0,
    thinking: bool = False,
    elapsed: float = 0.0,
    cost_usd: float = 0.0,
    tier: str = "",
    token_used: int = 0,
    token_limit: int = 128000,
    permission_mode: str = "careful",
    bg_indicator: str = "",
    research_indicator: str = "",
    mood_indicator: str = "",
    watch_indicator: str = "",
    steering_queue=None,
):
    """Print the status bar line."""
    from .status_bar import build_status_bar
    width = console.size.width
    bar = build_status_bar(
        model=model, project_type=project_type,
        session_title=session_title, message_count=message_count,
        width=width, thinking=thinking, elapsed=elapsed,
        cost_usd=cost_usd, tier=tier,
        token_used=token_used, token_limit=token_limit,
        permission_mode=permission_mode,
        bg_indicator=bg_indicator,
        research_indicator=research_indicator,
        mood_indicator=mood_indicator,
        watch_indicator=watch_indicator,
        steering_queue=steering_queue,
    )
    console.print(bar, style="on grey11", end="\n")


def show_thinking(label: str = "Working..."):
    """Context manager — shows spinner while agent runs. Disappears when done."""
    return Live(
        Spinner("dots", text=f"  [dim]{label}[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    )


# Phase-aware verb mapping for contextual spinner labels
_TOOL_PHASE_VERBS: dict[str, str] = {
    "web_search": "Searching the web...",
    "search_web": "Searching the web...",
    "browse": "Browsing...",
    "browse_url": "Browsing...",
    "read_file": "Reading files...",
    "edit_file": "Editing code...",
    "write_file": "Writing code...",
    "execute": "Running code...",
    "run_command": "Running command...",
    "shell": "Running shell...",
    "analyze": "Analyzing...",
    "summarize": "Summarizing...",
    "translate": "Translating...",
    "calculate": "Calculating...",
    "research": "Researching...",
    "deep_research": "Deep researching...",
    "memory_recall": "Remembering...",
    "memory_store": "Storing memory...",
}


def get_thinking_label(tool_name: str | None = None) -> str:
    """Return a phase-aware thinking label based on the current tool being used."""
    if tool_name and tool_name in _TOOL_PHASE_VERBS:
        return _TOOL_PHASE_VERBS[tool_name]
    return "Thinking..."


def show_tool_call(tool_name: str, description: str = "", result=None, elapsed: float = 0.0):
    """Print a tool call in a compact styled format with elapsed time.

    If result is provided and contains diff info for edit/write, show a compact diff summary.
    If result is provided with substantial output, render via ToolOutputRenderer.
    """
    # Format the elapsed time suffix
    time_str = f" {format_elapsed(elapsed)}" if elapsed > 0 else ""

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
            console.print(f"  {summary}")
            return
        except Exception:
            pass  # Fall through to default display

    # Get theme-aware tool color
    try:
        from aura.cli.themes import get_theme
        tool_color = get_theme().tool_color
    except Exception:
        tool_color = "yellow"

    line = Text()
    line.append("  ▸ ", style=f"dim {tool_color}")
    line.append(tool_name, style=f"bold {tool_color}")
    if description:
        line.append(f" {description}", style=f"dim {tool_color}")
    if time_str:
        line.append(time_str, style="dim")
    console.print(line)

    # Render substantial tool output via ToolOutputRenderer
    if result and isinstance(result, (dict, str)):
        try:
            import json
            if isinstance(result, str):
                parsed = json.loads(result)
            else:
                parsed = result
            if isinstance(parsed, dict) and not parsed.get("error"):
                output = parsed.get("output", parsed.get("content", parsed.get("result", "")))
                if isinstance(output, str) and len(output) > 50:
                    _get_tool_renderer().render_tool_result(tool_name, parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass


def show_response(text: str, model: str = "", stream: bool = True):
    """Render agent response as markdown with a left-border panel.

    Args:
        text: Response text (markdown)
        model: Model name to display
        stream: If True, use block-level streaming (only re-renders active block)
    """
    # Get theme colors
    try:
        from aura.cli.themes import get_theme
        theme = get_theme()
        border_style = f"dim {theme.response_border}"
        header_style = theme.response_header
        code_theme = theme.code_theme
    except Exception:
        border_style = "dim cyan"
        header_style = "bold cyan"
        code_theme = "monokai"

    console.print()

    # Header: label + model name
    label = Text()
    label.append(" ◆ ", style=header_style)
    label.append("AURA", style=header_style)
    if model:
        label.append(f"  ({model})", style="dim")

    if stream and len(text) > 20:
        # Block-level streaming: freeze finalized blocks, only re-render the active one
        import time

        # Print header once
        console.print(Padding(Text.from_markup(f"  [dim {border_style.replace('dim ', '')}]{'─' * 60}[/]"), (0, 2)))
        console.print(Padding(label, (0, 2)))
        console.print(Padding(Text.from_markup(f"  [dim {border_style.replace('dim ', '')}]{'─' * 60}[/]"), (0, 2)))

        chunks = _split_for_streaming(text)
        accumulated = ""
        finalized_count = 0  # how many blocks we've already printed

        with Live(console=console, refresh_per_second=15, transient=True) as live:
            for chunk in chunks:
                accumulated += chunk
                blocks = _split_into_blocks(accumulated)

                # Print any newly finalized blocks (all except the last)
                while finalized_count < len(blocks) - 1:
                    block_text = blocks[finalized_count]
                    try:
                        block_md = Markdown(block_text, code_theme=code_theme)
                    except Exception:
                        block_md = Text(block_text)
                    # Exit live temporarily to print finalized block permanently
                    live.update(Text(""))
                    console.print(Padding(block_md, (0, 4)))
                    finalized_count += 1

                # Live-update only the active (last) block
                if blocks:
                    active_block = blocks[-1]
                    try:
                        active_md = Markdown(active_block, code_theme=code_theme)
                    except Exception:
                        active_md = Text(active_block)
                    live.update(Padding(active_md, (0, 4)))

                time.sleep(0.008)

        # Print the final active block permanently
        blocks = _split_into_blocks(accumulated)
        if blocks and finalized_count < len(blocks):
            for i in range(finalized_count, len(blocks)):
                try:
                    block_md = Markdown(blocks[i], code_theme=code_theme)
                except Exception:
                    block_md = Text(blocks[i])
                console.print(Padding(block_md, (0, 4)))

        console.print(Padding(Text.from_markup(f"  [dim {border_style.replace('dim ', '')}]{'─' * 60}[/]"), (0, 2)))
        console.print()
    else:
        # Non-streaming: render the full panel as before
        try:
            md = Markdown(text, code_theme=code_theme)
        except Exception:
            md = Text(text)

        panel = Panel(
            md,
            title=label,
            title_align="left",
            border_style=border_style,
            padding=(0, 2),
            expand=True,
        )
        console.print(Padding(panel, (0, 2)))
        console.print()


def _split_for_streaming(text: str) -> list[str]:
    """Split text into word-based chunks for streaming display."""
    words = text.split(' ')
    chunks = []
    # Use larger chunks for longer texts to keep animation snappy
    chunk_size = 1 if len(text) < 200 else (3 if len(text) < 1000 else 5)
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk = ' '.join(chunk_words)
        if i + chunk_size < len(words):
            chunk += ' '
        chunks.append(chunk)
    return chunks


def _split_into_blocks(text: str) -> list[str]:
    """Split markdown text into top-level blocks by double-newlines.

    Respects code fences (``` blocks stay together as a single block).
    Returns a list of block strings.
    """
    lines = text.split('\n')
    blocks: list[str] = []
    current_lines: list[str] = []
    in_code_fence = False
    prev_was_empty = False

    for line in lines:
        stripped = line.strip()

        # Track code fence boundaries
        if stripped.startswith('```'):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            prev_was_empty = False
            continue

        # Inside a code fence — everything stays in the current block
        if in_code_fence:
            current_lines.append(line)
            prev_was_empty = False
            continue

        # Outside code fence: detect blank line (block boundary)
        if stripped == '':
            if current_lines and any(l.strip() for l in current_lines):
                # Blank line after content — finalize current block
                while current_lines and current_lines[-1].strip() == '':
                    current_lines.pop()
                if current_lines:
                    blocks.append('\n'.join(current_lines))
                current_lines = []
            else:
                current_lines.append(line)
            prev_was_empty = True
        else:
            current_lines.append(line)
            prev_was_empty = False

    # Whatever remains is the current (possibly incomplete) block
    if current_lines:
        remaining = '\n'.join(current_lines).strip()
        if remaining:
            blocks.append(remaining)

    # If nothing was parsed, return the whole text as one block
    if not blocks and text.strip():
        blocks.append(text.strip())

    return blocks


def show_context_summary(
    memory_count: int = 0,
    kg_topic: str = "",
    mood: str = "",
    model: str = "",
    tool_count: int = 0,
    memory_snippets: list[str] | None = None,
):
    """Show a one-line context summary before the response, with optional memory snippets.

    Example:
      context: 3 memories | mood: curious | model: devstral-2
      remembered: "prefers Python" | "working on BroadMind" | "uses RTX 4060"
    """
    if memory_snippets is None:
        memory_snippets = []

    parts = []
    if memory_count > 0:
        parts.append(f"[dim]{memory_count} memories[/dim]")
    if kg_topic:
        topic = kg_topic[:30]
        parts.append(f'[dim]KG: [/dim][dim italic]"{topic}"[/dim italic]')
    if mood:
        parts.append(f"[dim]mood: [/dim][dim]{mood}[/dim]")
    if model:
        short = model.replace(":cloud", "").replace(":latest", "")
        if len(short) > 25:
            short = short[:22] + "..."
        parts.append(f"[dim]model: [/dim][dim cyan]{short}[/dim cyan]")
    if tool_count > 0:
        parts.append(f"[dim]{tool_count} tools[/dim]")

    if not parts:
        return

    line = "  [dim]context:[/dim] " + " [dim]|[/dim] ".join(parts)
    console.print(Text.from_markup(line))

    # Show memory snippets on a second line (max 3, truncated to 40 chars each)
    if memory_snippets:
        snippets = memory_snippets[:3]
        formatted = []
        for s in snippets:
            truncated = s.strip()
            if len(truncated) > 40:
                truncated = truncated[:37] + "..."
            formatted.append(f'[dim italic]"{truncated}"[/dim italic]')
        snippet_line = "  [dim]remembered:[/dim] " + " [dim]|[/dim] ".join(formatted)
        console.print(Text.from_markup(snippet_line))


def show_error(message: str):
    """Display error in a styled format."""
    try:
        from aura.cli.themes import get_theme
        error_color = get_theme().error_color
    except Exception:
        error_color = "red"
    err = Text()
    err.append("  ✗ ", style=f"bold {error_color}")
    err.append(message, style=error_color)
    console.print(err)


def show_info(message: str):
    """Display info message."""
    info = Text()
    info.append("  ● ", style="dim cyan")
    info.append(message, style="dim")
    console.print(info)


def show_help():
    """Display quick help with available commands and shortcuts."""
    from rich.table import Table

    table = Table(
        show_header=True, header_style="bold cyan",
        border_style="dim", padding=(0, 2),
        title="[bold]Commands & Shortcuts[/bold]",
    )
    table.add_column("Key / Command", style="cyan", width=24)
    table.add_column("Action", style="white")

    # --- Keyboard shortcuts ---
    table.add_row("Alt+M", "Model picker (interactive)")
    table.add_row("Ctrl+L", "Clear screen")
    table.add_row("Ctrl+N", "New session")
    table.add_row("Ctrl+K", "Command palette (fuzzy search)")
    table.add_row("Ctrl+G", "Open editor for long prompt")
    table.add_row("Shift+Tab", "Cycle permission mode (Plan / Careful / Auto-Edit / Full Auto)")
    table.add_row("Esc Esc", "Rewind to checkpoint")
    table.add_row("Ctrl+C / Ctrl+D", "Exit")
    table.add_row("?", "Show this help")

    table.add_row("", "")

    # --- Model & session ---
    table.add_row("/model [name]", "Pick model interactively or set by name")
    table.add_row("/sessions", "Manage sessions (list, switch, delete)")
    table.add_row("/compact", "Compress conversation history")
    table.add_row("/clear", "Clear conversation history")

    table.add_row("", "")

    # --- Code & files ---
    table.add_row("/grep <pattern>", "Search code content")
    table.add_row("/search, /find <query>", "Search files and definitions")
    table.add_row("/edit <file>", "Read file with line numbers")
    table.add_row("/project [info|index|search]", "Project context, indexing, semantic search")

    table.add_row("", "")

    # --- Execution ---
    table.add_row("/shell, /bash, /run <cmd>", "Execute shell command")
    table.add_row("/plan <task>", "Create and execute a plan")
    table.add_row("/agent <name> <task>", "Run specialist agent")
    table.add_row("/goal <objective>", "Run a goal")

    table.add_row("", "")

    # --- Parallel & background ---
    table.add_row("/fleet <task>", "Run parallel sub-agents on decomposed task")
    table.add_row("& <prompt>", "Run prompt as background task")
    table.add_row("/tasks", "Show background tasks")

    table.add_row("", "")

    # --- Research ---
    table.add_row("/research <topic>", "Start research mode with citation tracking")
    table.add_row("/sources", "Show collected research sources")
    table.add_row("/export research", "Export research session to Markdown")

    table.add_row("", "")

    # --- Emotional & hooks ---
    table.add_row("/mood", "Show current emotional state")
    table.add_row("/hook [list|add|remove]", "Manage automation hooks")

    table.add_row("", "")

    # --- Git power tools ---
    table.add_row("/pr", "Create pull request with AI-generated description")
    table.add_row("/branch <name>", "Create and switch to a new git branch")
    table.add_row("/stash [desc]", "Smart stash with description")
    table.add_row("/blame file:N", "Explain why a line exists using git history")

    table.add_row("", "")

    # --- Testing & watch ---
    table.add_row("/test [cmd]", "Run tests with formatted output")
    table.add_row("/watch", "Monitor files for AURA:/AI: comments")

    table.add_row("", "")

    # --- Utilities ---
    table.add_row("/browse <url>", "Browse web pages")
    table.add_row("/speak <text>", "Text-to-speech")
    table.add_row("/recall <query>", "Search memories")
    table.add_row("/context", "Show context window usage")
    table.add_row("/rewind", "Rewind file changes to a checkpoint")
    table.add_row("/theme [name]", "Switch color theme (dark, light, monokai, dracula, solarized, nord)")

    table.add_row("", "")

    # --- Exit ---
    table.add_row("/quit, /exit", "Exit AURA")

    console.print()
    console.print(table)
    console.print()


def show_checkpoint_picker(checkpoints: list[dict]) -> str | None:
    """Display checkpoints in a numbered list and let the user pick one.

    Args:
        checkpoints: List of checkpoint dicts from CheckpointManager.list_checkpoints()

    Returns:
        Selected checkpoint ID, or None if cancelled.
    """
    from rich.table import Table
    import time as _time

    if not checkpoints:
        show_info("No checkpoints available.")
        return None

    # Show last 10 checkpoints
    display = checkpoints[:10]

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
        title="[bold]Checkpoints[/bold]",
    )
    table.add_column("#", style="bold white", width=4, justify="right")
    table.add_column("Time", style="dim", width=16)
    table.add_column("Label", style="white", min_width=20)
    table.add_column("Files", style="dim cyan", width=30)

    now = _time.time()
    for i, cp in enumerate(display, 1):
        # Relative time
        delta = now - cp.get("timestamp", now)
        if delta < 60:
            rel = f"{int(delta)}s ago"
        elif delta < 3600:
            rel = f"{int(delta / 60)}m ago"
        elif delta < 86400:
            rel = f"{int(delta / 3600)}h ago"
        else:
            rel = f"{int(delta / 86400)}d ago"

        label = cp.get("label", "") or "-"
        files = cp.get("files", [])
        file_names = ", ".join(
            f.get("backup_name", "?") for f in files[:3]
        )
        if len(files) > 3:
            file_names += f" +{len(files) - 3}"

        table.add_row(str(i), rel, label, file_names)

    console.print()
    console.print(table)
    console.print()

    # Prompt for selection
    try:
        raw = console.input("[dim]  Pick checkpoint # (or Enter to cancel): [/dim]")
        raw = raw.strip()
        if not raw:
            return None
        idx = int(raw) - 1
        if 0 <= idx < len(display):
            return display[idx]["id"]
        else:
            show_error(f"Invalid selection: {raw}")
            return None
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


def show_rewind_result(success: bool, checkpoint_id: str):
    """Display the result of a rewind operation.

    Args:
        success: Whether the rewind succeeded.
        checkpoint_id: The checkpoint ID that was targeted.
    """
    if success:
        msg = Text()
        msg.append("  ✓ ", style="bold green")
        msg.append(f"Rewound to checkpoint {checkpoint_id}", style="green")
        console.print(msg)
    else:
        msg = Text()
        msg.append("  ✗ ", style="bold red")
        msg.append("Failed to rewind", style="red")
        console.print(msg)
