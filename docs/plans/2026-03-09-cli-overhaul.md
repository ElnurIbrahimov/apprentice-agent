# AURA CLI Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform AURA's basic print-based CLI into a polished terminal UI with ASCII banner, status bar, model selector, keyboard hints, streaming responses, and styled tool calls — matching the quality of OpenCode, Codex CLI, and Gemini CLI.

**Architecture:** Rich-based upgrade to the existing `aura/cli/` package. New `banner.py` for ASCII art, rewritten `display.py` with Panel/Layout/Live components, enhanced `input.py` with placeholder + keybinding hints, new `status_bar.py` for persistent footer, new `model_picker.py` for interactive model selection. Main.py chat loop updated to use streaming + status bar.

**Tech Stack:** Python 3.12, Rich (Console, Panel, Layout, Live, Markdown, Syntax, Table, Text, Columns), prompt_toolkit (PromptSession, key_bindings)

---

### Task 1: Suppress Warnings & Bump Version

**Files:**
- Modify: `D:/Aura/main.py:1-8`
- Modify: `D:/Aura/aura/__init__.py:3`

**Step 1: Suppress warnings at the top of main.py**

In `D:/Aura/main.py`, add `import warnings` and `warnings.filterwarnings("ignore")` right after the existing `os.environ["TQDM_DISABLE"] = "1"` line (line 5):

```python
#!/usr/bin/env python3
"""Main entry point for the Apprentice Agent."""

import os
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
```

**Step 2: Bump version to 4.3.0**

In `D:/Aura/aura/__init__.py`, change line 3:

```python
__version__ = "4.3.0"
```

**Step 3: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('main.py', doraise=True); py_compile.compile('aura/__init__.py', doraise=True); print('OK')"`

**Step 4: Commit**

```bash
git add main.py aura/__init__.py
git commit -m "chore: suppress warnings on startup, bump version to 4.3.0"
```

---

### Task 2: ASCII Art Banner

**Files:**
- Create: `D:/Aura/aura/cli/banner.py`
- Modify: `D:/Aura/aura/cli/display.py:12-13` (replace show_banner)

**Step 1: Create banner.py with 3 responsive sizes**

Create `D:/Aura/aura/cli/banner.py`:

```python
"""ASCII art banners for AURA CLI — 3 responsive sizes."""

from rich.text import Text

# Full banner for terminals >= 60 columns
BANNER_FULL = r"""
    ╔═══╗ ╔╗ ╔╗ ╔═══╗  ╔═══╗
    ║╔═╗║ ║║ ║║ ║╔═╗║  ║╔═╗║
    ║║ ║║ ║║ ║║ ║╔═╝╝  ║║ ║║
    ║╔═╗║ ║║ ║║ ║║╔═╗  ║╔═╗║
    ║║ ║║ ║╚═╝║ ║║║╚╗  ║║ ║║
    ╚╝ ╚╝ ╚═══╝ ╚╝╚═╝  ╚╝ ╚╝"""

# Compact banner for terminals 40-59 columns
BANNER_COMPACT = r"""
   ╔══╗╔╗╔╗╔══╗╔══╗
   ║╔╗║║║║║║╔═╝║╔╗║
   ║╔╗║║╚╝║║║╔╗║╔╗║
   ╚╝╚╝╚══╝╚╝╚╝╚╝╚╝"""

# Tiny banner for terminals < 40 columns
BANNER_TINY = "[bold cyan]◆ AURA[/bold cyan]"

# Gradient colors for the banner (cyan → blue → magenta)
_GRADIENT = ["cyan", "deep_sky_blue1", "dodger_blue1", "blue1", "dark_violet", "magenta"]


def get_banner(width: int = 80) -> Text:
    """Return a Rich Text banner sized for the given terminal width."""
    if width >= 60:
        raw = BANNER_FULL
    elif width >= 40:
        raw = BANNER_COMPACT
    else:
        return Text.from_markup(BANNER_TINY)

    lines = raw.strip("\n").split("\n")
    text = Text()

    for li, line in enumerate(lines):
        # Pick gradient color based on line position
        color = _GRADIENT[li % len(_GRADIENT)]
        text.append(line, style=f"bold {color}")
        text.append("\n")

    return text


def get_welcome_line(version: str = "4.3.0") -> Text:
    """Return the one-line welcome below the banner."""
    t = Text()
    t.append(f"  v{version}", style="dim")
    t.append("  │  ", style="dim")
    t.append("/", style="bold cyan")
    t.append(" commands", style="dim")
    t.append("  │  ", style="dim")
    t.append("Ctrl+M", style="bold cyan")
    t.append(" model", style="dim")
    t.append("  │  ", style="dim")
    t.append("?", style="bold cyan")
    t.append(" help", style="dim")
    return t
```

**Step 2: Update show_banner in display.py**

Replace the `show_banner` function in `D:/Aura/aura/cli/display.py` (line 12-13):

```python
def show_banner():
    from .banner import get_banner, get_welcome_line
    from aura import __version__
    width = console.size.width
    console.print(get_banner(width))
    console.print(get_welcome_line(__version__))
    console.print()
```

**Step 3: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('aura/cli/banner.py', doraise=True); py_compile.compile('aura/cli/display.py', doraise=True); print('OK')"`

**Step 4: Commit**

```bash
git add aura/cli/banner.py aura/cli/display.py
git commit -m "feat: responsive ASCII art banner with gradient colors"
```

---

### Task 3: Status Bar

**Files:**
- Create: `D:/Aura/aura/cli/status_bar.py`

**Step 1: Create status_bar.py**

Create `D:/Aura/aura/cli/status_bar.py`:

```python
"""Persistent status bar for AURA CLI — shows model, project, session info."""

import os
from pathlib import Path
from rich.text import Text


def build_status_bar(
    model: str = "auto",
    project_type: str = "",
    session_title: str = "",
    message_count: int = 0,
    width: int = 80,
    thinking: bool = False,
    elapsed: float = 0.0,
) -> Text:
    """Build a single-line status bar for the AURA CLI footer.

    Returns a Rich Text object ready to print.
    """
    bar = Text()

    # Left: CWD + project type
    cwd = os.getcwd()
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    cwd = cwd.replace("\\", "/")
    # Truncate long paths
    if len(cwd) > 30:
        cwd = "..." + cwd[-27:]

    bar.append(f" {cwd}", style="dim white")
    if project_type and project_type != "unknown":
        bar.append(f" ({project_type})", style="dim cyan")

    bar.append("  │  ", style="dim")

    # Center: model or thinking indicator
    if thinking:
        bar.append(f"Thinking... ({elapsed:.1f}s)", style="bold yellow")
    else:
        # Show model name, trimmed
        model_short = model.replace(":cloud", "").replace(":latest", "")
        if len(model_short) > 25:
            model_short = model_short[:22] + "..."
        bar.append(model_short, style="bold cyan")

    bar.append("  │  ", style="dim")

    # Right: session info + hint
    if session_title:
        title = session_title[:20]
        bar.append(f'"{title}"', style="dim white")
        if message_count:
            bar.append(f" ({message_count} msgs)", style="dim")
    else:
        bar.append(f"{message_count} msgs", style="dim")

    # Keyboard hint (right-aligned padding)
    hint = "  Ctrl+M model"
    remaining = width - bar.cell_len - len(hint) - 1
    if remaining > 0:
        bar.append(" " * remaining)
        bar.append("Ctrl+M", style="dim bold")
        bar.append(" model", style="dim")

    return bar
```

**Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('aura/cli/status_bar.py', doraise=True); print('OK')"`

**Step 3: Commit**

```bash
git add aura/cli/status_bar.py
git commit -m "feat: status bar component — model, project type, session, keyboard hints"
```

---

### Task 4: Model Picker

**Files:**
- Create: `D:/Aura/aura/cli/model_picker.py`

**Step 1: Create model_picker.py**

Create `D:/Aura/aura/cli/model_picker.py`:

```python
"""Interactive model picker for AURA CLI — select model mid-session."""

from rich.console import Console
from rich.text import Text
from rich.panel import Panel

# Model roles with display info
MODEL_ROLES = [
    ("fast", "gemini-3-flash-preview:cloud", "1M ctx"),
    ("reason", "kimi-k2.5:cloud", "256K ctx"),
    ("code", "minimax-m2.5:cloud", "196K ctx"),
    ("think", "kimi-k2-thinking:cloud", "256K ctx"),
    ("vision", "qwen3-vl:235b-cloud", "256K ctx"),
    ("longctx", "gemini-3-flash-preview:cloud", "1M ctx"),
]


def pick_model(console: Console, current_model: str = "auto") -> "str | None":
    """Show interactive model picker. Returns selected model name or None if cancelled.

    Args:
        console: Rich Console instance
        current_model: Currently active model name

    Returns:
        Model name string, "auto" for auto-routing, or None if user cancelled
    """
    # Build display
    lines = Text()
    lines.append("\n")

    for i, (role, model, ctx) in enumerate(MODEL_ROLES, 1):
        model_short = model.replace(":cloud", "")
        marker = " ← current" if model == current_model else ""
        is_current = model == current_model

        num_style = "bold cyan" if not is_current else "bold green"
        model_style = "white" if not is_current else "bold green"
        role_style = "dim yellow"
        ctx_style = "dim"

        lines.append(f"    {i}", style=num_style)
        lines.append(f". {model_short:<35s}", style=model_style)
        lines.append(f" {role:<8s}", style=role_style)
        lines.append(f" {ctx}", style=ctx_style)
        if marker:
            lines.append(marker, style="green")
        lines.append("\n")

    lines.append("\n")
    lines.append("  [1-6]", style="bold cyan")
    lines.append(" select  ", style="dim")
    lines.append("[a]", style="bold cyan")
    lines.append(" auto  ", style="dim")
    lines.append("[Esc/q]", style="bold cyan")
    lines.append(" cancel", style="dim")

    header = Text()
    header.append("  Select model", style="bold white")
    cur = current_model.replace(":cloud", "").replace(":latest", "")
    header.append(f"  (current: {cur})", style="dim")

    panel = Panel(
        lines,
        title="[bold cyan]Model Picker[/bold cyan]",
        subtitle=header,
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)

    # Get choice
    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice or choice in ("q", "esc", "escape"):
        console.print("  [dim]Cancelled.[/dim]")
        return None
    elif choice == "a" or choice == "auto":
        return "auto"
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODEL_ROLES):
                _, model, _ = MODEL_ROLES[idx]
                return model
        except ValueError:
            pass

        # Try as model name
        for _, model, _ in MODEL_ROLES:
            if choice in model:
                return model

    console.print(f"  [dim]Invalid choice: {choice}[/dim]")
    return None


def update_model_roles_from_config():
    """Refresh MODEL_ROLES from Config at runtime."""
    global MODEL_ROLES
    try:
        from aura.config import Config
        MODEL_ROLES = [
            ("fast", Config.MODEL_FAST, "1M ctx"),
            ("reason", Config.MODEL_REASON, "256K ctx"),
            ("code", Config.MODEL_CODE, "196K ctx"),
            ("think", Config.MODEL_THINK, "256K ctx"),
            ("vision", Config.MODEL_VISION, "256K ctx"),
            ("longctx", Config.MODEL_LONGCTX, "1M ctx"),
        ]
    except Exception:
        pass  # Keep defaults
```

**Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('aura/cli/model_picker.py', doraise=True); print('OK')"`

**Step 3: Commit**

```bash
git add aura/cli/model_picker.py
git commit -m "feat: interactive model picker — select model mid-session with Ctrl+M"
```

---

### Task 5: Enhanced Display (Panels, Syntax Highlighting, Streaming)

**Files:**
- Modify: `D:/Aura/aura/cli/display.py` (full rewrite)

**Step 1: Rewrite display.py with enhanced components**

Replace entire contents of `D:/Aura/aura/cli/display.py`:

```python
"""Rich-based display for AURA CLI — panels, syntax highlighting, streaming."""

import shutil
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from rich.syntax import Syntax

console = Console(highlight=True, soft_wrap=True)


def show_banner():
    """Display ASCII art banner with gradient colors."""
    from .banner import get_banner, get_welcome_line
    from aura import __version__
    width = console.size.width
    console.print(get_banner(width))
    console.print(get_welcome_line(__version__))
    console.print()


def show_status_bar(
    model: str = "auto",
    project_type: str = "",
    session_title: str = "",
    message_count: int = 0,
    thinking: bool = False,
    elapsed: float = 0.0,
):
    """Print the status bar line."""
    from .status_bar import build_status_bar
    width = console.size.width
    bar = build_status_bar(
        model=model, project_type=project_type,
        session_title=session_title, message_count=message_count,
        width=width, thinking=thinking, elapsed=elapsed,
    )
    # Print with background
    bg_bar = Text(" " * width, style="on grey11")
    bg_bar = bar
    console.print(bg_bar, style="on grey11", end="\n")


def show_thinking(label: str = "Working..."):
    """Context manager — shows spinner while agent runs. Disappears when done."""
    return Live(
        Spinner("dots", text=f"  [dim]{label}[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    )


def show_tool_call(tool_name: str, description: str = ""):
    """Print a tool call in a compact panel."""
    desc = f"  {description}" if description else ""
    line = Text()
    line.append("  ▸ ", style="cyan")
    line.append(tool_name, style="bold cyan")
    line.append(desc, style="dim")
    console.print(line)


def show_response(text: str):
    """Render agent response as markdown with syntax highlighting."""
    console.print()
    label = Text()
    label.append("  ◆ ", style="bold cyan")
    label.append("AURA", style="bold cyan")
    console.print(label)
    try:
        md = Markdown(text, code_theme="monokai")
        console.print(md, padding=(0, 4))
    except Exception:
        console.print(text, padding=(0, 4))
    console.print()


def show_error(message: str):
    """Display error in a styled format."""
    err = Text()
    err.append("  ✗ ", style="bold red")
    err.append(message, style="red")
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
    table.add_column("Key / Command", style="cyan", width=22)
    table.add_column("Action", style="white")

    # Shortcuts
    table.add_row("Ctrl+M", "Switch model mid-session")
    table.add_row("Ctrl+C / Ctrl+D", "Exit")
    table.add_row("", "")
    # Chat commands
    table.add_row("/model [name]", "Show or set model")
    table.add_row("/sessions", "List / switch sessions")
    table.add_row("/compact", "Compress conversation history")
    table.add_row("/clear", "Clear history")
    table.add_row("", "")
    # Dev commands
    table.add_row("/grep <pattern>", "Search code content")
    table.add_row("/find def <name>", "Find definition")
    table.add_row("/edit <file>", "View file with line numbers")
    table.add_row("/project index", "Build semantic code index")
    table.add_row("/project search <q>", "Semantic code search")
    table.add_row("/shell <cmd>", "Run shell command")
    table.add_row("/agent <name> <task>", "Route to specialist agent")
    table.add_row("", "")
    table.add_row("/plan <task>", "Generate execution plan")
    table.add_row("/browse <url>", "Open URL in browser")

    console.print()
    console.print(table)
    console.print()
```

**Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('aura/cli/display.py', doraise=True); print('OK')"`

**Step 3: Commit**

```bash
git add aura/cli/display.py
git commit -m "feat: enhanced display — panels, syntax highlighting, help table, styled responses"
```

---

### Task 6: Enhanced Input with Placeholder & Keybindings

**Files:**
- Modify: `D:/Aura/aura/cli/input.py` (rewrite)

**Step 1: Rewrite input.py with modern prompt, placeholder, Ctrl+M**

Replace entire contents of `D:/Aura/aura/cli/input.py`:

```python
"""prompt_toolkit-based input for AURA CLI — styled prompt, keybindings, model picker.

Falls back to plain input() when prompt_toolkit can't attach to the console.
"""

from pathlib import Path

HISTORY_FILE = Path.home() / ".aura_history"

_session_ok = True
_model_pick_requested = False  # flag for Ctrl+M


def create_session():
    """Create a prompt_toolkit session with styled prompt, history, and keybindings.

    Returns a PromptSession if prompt_toolkit works, otherwise None.
    """
    global _session_ok
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.styles import Style
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings

        _style = Style.from_dict({
            "prompt": "bold cyan",
            "placeholder": "#666666 italic",
        })

        kb = KeyBindings()

        @kb.add("c-m")  # Ctrl+M
        def _ctrl_m(event):
            global _model_pick_requested
            _model_pick_requested = True
            event.app.exit(result="__MODEL_PICK__")

        session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            style=_style,
            key_bindings=kb,
            placeholder=HTML('<style fg="#666666"><i>Type a message, / for commands...</i></style>'),
        )
        _session_ok = True
        return session
    except Exception:
        _session_ok = False
        return None


def get_input(session) -> "str | None":
    """Get user input. Returns None on exit, '__MODEL_PICK__' for Ctrl+M."""
    try:
        if session is not None and _session_ok:
            result = session.prompt([("class:prompt", "\n  ❯ ")]).strip()
            return result
        else:
            return input("\n  ❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
```

**Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('aura/cli/input.py', doraise=True); print('OK')"`

**Step 3: Commit**

```bash
git add aura/cli/input.py
git commit -m "feat: styled input with placeholder, Ctrl+M model picker keybinding"
```

---

### Task 7: Wire Everything Into main.py Chat Loop

**Files:**
- Modify: `D:/Aura/main.py:158-210` (run_chat_mode function)
- Modify: `D:/Aura/main.py:294-296` (add ? help command)

**Step 1: Update run_chat_mode to use status bar, model picker, and streaming**

Replace the `run_chat_mode` function (from line 158) with:

```python
def run_chat_mode(agent, speak: bool = False):
    """Interactive CLI — full agent loop with status bar, model picker, streaming."""
    import io
    import sys
    import time
    import threading
    from aura.cli.display import (
        console, show_banner, show_thinking, show_response,
        show_error, show_info, show_status_bar, show_help,
    )
    from aura.cli.input import create_session, get_input
    from aura.cli.model_picker import pick_model, update_model_roles_from_config

    show_banner()

    # Detect project type for status bar
    _project_type = ""
    try:
        from aura.tools.project_context import detect_and_load_context
        ctx = detect_and_load_context(".")
        _project_type = ctx.get("project_type", "") if isinstance(ctx, dict) else ""
    except Exception:
        pass

    # Show initial status bar
    _current_model = agent.brain._model_override or "auto"
    _session_title = ""
    _msg_count = len(agent.brain.conversation_history) if hasattr(agent.brain, 'conversation_history') else 0
    show_status_bar(
        model=_current_model, project_type=_project_type,
        session_title=_session_title, message_count=_msg_count,
    )

    if speak:
        show_info("Voice output enabled")

    # Register CLI permission callback for destructive actions
    def _cli_confirm(tool_name: str, action: str) -> bool:
        if tool_name == "code_edit_preview":
            print(f"\n  Proposed edit:\n")
            for line in action.split("\n")[:40]:
                if line.startswith("+") and not line.startswith("+++"):
                    print(f"  \033[32m{line}\033[0m")
                elif line.startswith("-") and not line.startswith("---"):
                    print(f"  \033[31m{line}\033[0m")
                else:
                    print(f"  {line}")
            if action.count("\n") > 40:
                print(f"  ... ({action.count(chr(10)) - 40} more lines)")
        else:
            print(f"\n  ⚠ Permission required:")
            print(f"    Tool: {tool_name}")
            print(f"    Action: {action[:200]}")
        try:
            response = input("    Allow? (y/n/always): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response == "always":
            for word in action.lower().split()[:3]:
                if len(word) > 3:
                    agent._approved_patterns.add(word)
            return True
        return response in ("y", "yes")

    agent.set_cli_confirm_callback(_cli_confirm)

    # Initialize model picker roles from config
    update_model_roles_from_config()

    session = create_session()

    while True:
        user_input = get_input(session)

        if user_input is None:
            console.print("\n[dim]Goodbye.[/dim]\n")
            break

        # Handle Ctrl+M model picker
        if user_input == "__MODEL_PICK__":
            _current_model = agent.brain._model_override or "auto"
            choice = pick_model(console, _current_model)
            if choice:
                if choice == "auto":
                    agent.brain.set_model_override(None)
                    _current_model = "auto"
                    show_info("Model set to auto-routing")
                else:
                    agent.brain.set_model_override(choice)
                    _current_model = choice
                    show_info(f"Model set to {choice}")
            show_status_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
            )
            continue

        if not user_input:
            continue

        # Signal activity to daemon
        try:
            import socket, json
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                s.connect(("127.0.0.1", 19733))
                s.send((json.dumps({"type": "activity"}) + "\n").encode())
        except Exception:
            pass

        # Handle ? for help
        if user_input.strip() == "?":
            show_help()
            continue

        if user_input.startswith("/"):
            handle_command(agent, user_input, speak=speak)
            # Update message count after command
            _msg_count = len(agent.brain.conversation_history) if hasattr(agent.brain, 'conversation_history') else 0
            _current_model = agent.brain._model_override or "auto"
            show_status_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
            )
            continue

        # Run agent in thread with spinner
        result_holder = {}
        captured_output = io.StringIO()

        def _run():
            old_stdout = sys.stdout
            sys.stdout = captured_output
            try:
                result_holder["result"] = agent.run(user_input)
            except Exception as exc:
                result_holder["error"] = str(exc)
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        with show_thinking():
            thread.join()

        if "error" in result_holder:
            show_error(result_holder["error"])
            continue

        result = result_holder["result"]
        response_text = result.get("response", "")

        show_response(response_text)

        # Update status bar after response
        _msg_count = len(agent.brain.conversation_history) if hasattr(agent.brain, 'conversation_history') else 0
        _current_model = agent.brain._model_override or "auto"
        show_status_bar(
            model=_current_model, project_type=_project_type,
            session_title=_session_title, message_count=_msg_count,
        )

        if speak and response_text:
            try:
                agent._speak(response_text)
            except Exception:
                pass
```

**Step 2: Add ? help command to handle_command**

In `handle_command`, before the `else: Unknown command` block, add:

```python
    elif cmd == "?":
        from aura.cli.display import show_help
        show_help()
```

**Step 3: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('main.py', doraise=True); print('OK')"`

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire status bar, model picker (Ctrl+M), help (?), and enhanced display into chat loop"
```

---

## Summary

| Task | Feature | Files | Est. Lines |
|------|---------|-------|-----------|
| 1 | Suppress warnings + version bump | main.py, __init__.py | ~5 |
| 2 | ASCII art banner (3 responsive sizes) | NEW banner.py, display.py | ~70 |
| 3 | Status bar component | NEW status_bar.py | ~70 |
| 4 | Model picker (interactive, Ctrl+M) | NEW model_picker.py | ~120 |
| 5 | Enhanced display (panels, syntax, help) | display.py (rewrite) | ~120 |
| 6 | Enhanced input (placeholder, keybindings) | input.py (rewrite) | ~70 |
| 7 | Wire into main.py chat loop | main.py | ~130 |

**Total: ~585 lines, 3 new files, 4 modified files, 7 commits.**
