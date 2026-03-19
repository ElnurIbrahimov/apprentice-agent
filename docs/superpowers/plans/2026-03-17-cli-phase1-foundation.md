# AURA CLI Phase 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Aura CLI to 2026 table-stakes parity with Claude Code, Aider, and Codex CLI — adding context visibility, keyboard shortcuts, permission tiers, inline diffs, and checkpoint/rewind.

**Architecture:** Five independent features layered onto the existing CLI. Each adds a new module (or extends an existing one) with minimal coupling. The status bar, input handler, and display modules are the primary integration points. All features are opt-in and backwards-compatible.

**Tech Stack:** Python 3.12+, Rich (terminal rendering), prompt_toolkit (input handling), difflib (diff generation), threading (background token counting), JSON (checkpoint storage)

---

## File Structure

### New Files
| File | Purpose |
|------|---------|
| `aura/cli/context_bar.py` | Token counting, context gauge rendering, /context command |
| `aura/cli/keybindings.py` | Keyboard shortcut registry, customizable keybindings, Ctrl+K palette |
| `aura/cli/diff_viewer.py` | Syntax-highlighted unified diff renderer |
| `aura/cli/checkpoint.py` | File snapshot, rewind picker, checkpoint management |
| `aura/cli/permissions_ui.py` | Permission tier UI, Shift+Tab cycling, mode indicator |
| `aura/cli/command_palette.py` | Fuzzy-search command palette (Ctrl+K) |

### Modified Files
| File | Changes |
|------|---------|
| `aura/cli/status_bar.py` | Add token gauge, permission mode indicator |
| `aura/cli/input.py` | Register new keybindings (Ctrl+L, Ctrl+N, Ctrl+K, Ctrl+G, Shift+Tab, Esc Esc, Shift+Enter) |
| `aura/cli/display.py` | Use diff_viewer for edit display, add context warning |
| `aura/core/agentic_loop.py` | Emit token counts via callback, call checkpoint before edits |
| `aura/core/permissions.py` | Add permission tiers (PLAN, CAREFUL, AUTO_EDIT, FULL_AUTO) |
| `aura/tools/code_edit.py` | Call checkpoint.snapshot() before edits |
| `main.py` | Wire up new keybindings, permission cycling, context display, rewind command |

### Test Files
| File | Tests |
|------|-------|
| `tests/cli/test_context_bar.py` | Token estimation, gauge rendering, budget warnings |
| `tests/cli/test_diff_viewer.py` | Diff generation, syntax highlighting, edge cases |
| `tests/cli/test_checkpoint.py` | Snapshot/restore, rewind picker, pruning |
| `tests/cli/test_permissions_ui.py` | Tier cycling, mode persistence, AURA.md overrides |
| `tests/cli/test_keybindings.py` | Shortcut registration, customization, conflicts |

---

## Chunk 1: Context Window Visibility

### Task 1: Token Estimation Engine

**Files:**
- Create: `aura/cli/context_bar.py`
- Test: `tests/cli/test_context_bar.py`

- [ ] **Step 1: Write failing test for token estimation**

```python
# tests/cli/test_context_bar.py
import pytest
from aura.cli.context_bar import estimate_tokens, format_token_count

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0

def test_estimate_tokens_simple():
    # ~1 token per 4 chars for English text
    result = estimate_tokens("hello world this is a test")
    assert 5 <= result <= 10

def test_estimate_tokens_code():
    code = "def hello():\n    return 'world'\n"
    result = estimate_tokens(code)
    assert result > 0

def test_format_token_count_small():
    assert format_token_count(500) == "500"

def test_format_token_count_thousands():
    assert format_token_count(12400) == "12.4K"

def test_format_token_count_millions():
    assert format_token_count(1200000) == "1.2M"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Aura && python -m pytest tests/cli/test_context_bar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aura.cli.context_bar'`

- [ ] **Step 3: Implement token estimation**

```python
# aura/cli/context_bar.py
"""Context window visibility — token counting and budget display."""

from __future__ import annotations
import math
from typing import List, Dict, Optional

# Average tokens per character varies by content type.
# English text: ~4 chars/token. Code: ~3.5 chars/token. JSON: ~3 chars/token.
_CHARS_PER_TOKEN = 3.8


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using char-based heuristic."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: List[Dict]) -> int:
    """Estimate total tokens across a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        # Add overhead for role, separators (~4 tokens per message)
        total += 4
        # Tool calls add extra tokens
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            import json
            total += estimate_tokens(json.dumps(tool_calls))
    return total


def format_token_count(count: int) -> str:
    """Format token count for display: 500, 12.4K, 1.2M."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /d/Aura && python -m pytest tests/cli/test_context_bar.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add aura/cli/context_bar.py tests/cli/test_context_bar.py
git commit -m "feat(cli): add token estimation engine for context visibility"
```

### Task 2: Context Gauge and Budget Tracking

**Files:**
- Modify: `aura/cli/context_bar.py`
- Test: `tests/cli/test_context_bar.py`

- [ ] **Step 1: Write failing test for context gauge**

```python
# tests/cli/test_context_bar.py — append these tests

def test_context_gauge_low_usage():
    gauge = build_context_gauge(used=5000, limit=128000)
    assert "green" in gauge or "5.0K" in gauge

def test_context_gauge_medium_usage():
    gauge = build_context_gauge(used=70000, limit=128000)
    assert "yellow" in gauge

def test_context_gauge_high_usage():
    gauge = build_context_gauge(used=110000, limit=128000)
    assert "red" in gauge

def test_context_gauge_zero_limit():
    gauge = build_context_gauge(used=0, limit=0)
    assert "0" in gauge

def test_context_breakdown():
    breakdown = build_context_breakdown(
        system_tokens=2000,
        history_tokens=8000,
        tools_tokens=1500,
        limit=128000
    )
    assert "System" in breakdown
    assert "History" in breakdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Aura && python -m pytest tests/cli/test_context_bar.py::test_context_gauge_low_usage -v`
Expected: FAIL — `ImportError: cannot import name 'build_context_gauge'`

- [ ] **Step 3: Implement context gauge**

```python
# aura/cli/context_bar.py — append to existing file

from rich.text import Text


# Context window limits by model family (conservative estimates)
MODEL_CONTEXT_LIMITS = {
    "default": 128_000,
    "qwen": 128_000,
    "deepseek": 128_000,
    "gemma": 128_000,
    "llama": 128_000,
    "minimax": 1_000_000,
    "chatgpt": 128_000,
    "gpt-5": 1_000_000,
}


def get_context_limit(model_name: str) -> int:
    """Get context window limit for a model."""
    model_lower = (model_name or "").lower()
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if prefix in model_lower:
            return limit
    return MODEL_CONTEXT_LIMITS["default"]


def _usage_color(pct: float) -> str:
    """Return color name based on usage percentage."""
    if pct < 0.50:
        return "green"
    elif pct < 0.80:
        return "yellow"
    else:
        return "red"


def build_context_gauge(used: int, limit: int) -> str:
    """Build a compact context gauge string for the status bar.

    Returns something like: 'Ctx: 12.4K/128K [████░░░░] 10%'
    """
    if limit <= 0:
        return f"Ctx: {format_token_count(used)}/0"

    pct = min(used / limit, 1.0)
    pct_int = int(pct * 100)
    color = _usage_color(pct)

    used_str = format_token_count(used)
    limit_str = format_token_count(limit)

    # 8-char bar
    filled = int(pct * 8)
    bar = "█" * filled + "░" * (8 - filled)

    return f"[{color}]{used_str}[/{color}]/{limit_str} [{color}]{bar}[/{color}] {pct_int}%"


def build_context_breakdown(
    system_tokens: int,
    history_tokens: int,
    tools_tokens: int,
    limit: int,
) -> str:
    """Build a detailed context breakdown for /context command."""
    total = system_tokens + history_tokens + tools_tokens
    lines = [
        f"  System prompt:  {format_token_count(system_tokens):>8}",
        f"  Conversation:   {format_token_count(history_tokens):>8}",
        f"  Tool schemas:   {format_token_count(tools_tokens):>8}",
        f"  ─────────────────────",
        f"  Total:          {format_token_count(total):>8} / {format_token_count(limit)}",
    ]
    pct = (total / limit * 100) if limit > 0 else 0
    color = _usage_color(total / limit if limit > 0 else 0)
    lines.append(f"  Usage:          [{color}]{pct:.0f}%[/{color}]")

    if pct > 80:
        lines.append(f"\n  [yellow]⚠ Context is {pct:.0f}% full — consider /compact[/yellow]")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/Aura && python -m pytest tests/cli/test_context_bar.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add aura/cli/context_bar.py tests/cli/test_context_bar.py
git commit -m "feat(cli): add context gauge and breakdown for token visibility"
```

### Task 3: Wire Context Gauge into Status Bar

**Files:**
- Modify: `aura/cli/status_bar.py`
- Modify: `aura/core/agentic_loop.py`
- Modify: `main.py`

- [ ] **Step 1: Read current status_bar.py**

Read `D:\Aura\aura\cli\status_bar.py` to understand `build_status_bar()` signature and layout.

- [ ] **Step 2: Add token_used and token_limit params to build_status_bar()**

Add `token_used: int = 0` and `token_limit: int = 128000` parameters. Insert the context gauge into the center section of the status bar, after the cost display.

The gauge should appear as: `| Ctx: 12.4K/128K [████░░░░] 10% |`

- [ ] **Step 3: Update main.py to pass token counts to status bar**

In `run_chat_mode()`, after `agentic.run()` completes (around line 543), estimate token usage from the conversation history and pass to `show_status_bar()`.

```python
# After agentic.run() returns, estimate context usage:
from aura.cli.context_bar import estimate_messages_tokens, get_context_limit
token_used = estimate_messages_tokens(agentic._conversation_history)
token_limit = get_context_limit(agent.brain.current_model or "default")
```

- [ ] **Step 4: Add /context slash command**

In `handle_command()` in main.py, add a `/context` case that calls `build_context_breakdown()` and prints the result.

- [ ] **Step 5: Test manually**

Run: `cd /d/Aura && python main.py`
Expected: Status bar now shows token gauge. `/context` shows detailed breakdown.

- [ ] **Step 6: Commit**

```bash
cd /d/Aura && git add aura/cli/status_bar.py aura/cli/context_bar.py main.py
git commit -m "feat(cli): wire context gauge into status bar and add /context command"
```

---

## Chunk 2: Keyboard Shortcuts

### Task 4: Keybindings Registry

**Files:**
- Create: `aura/cli/keybindings.py`
- Test: `tests/cli/test_keybindings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/cli/test_keybindings.py
import pytest
from aura.cli.keybindings import KeybindingsRegistry, DEFAULT_KEYBINDINGS

def test_default_keybindings_exist():
    assert "ctrl+l" in DEFAULT_KEYBINDINGS
    assert "ctrl+n" in DEFAULT_KEYBINDINGS
    assert "ctrl+k" in DEFAULT_KEYBINDINGS
    assert "ctrl+g" in DEFAULT_KEYBINDINGS
    assert "escape escape" in DEFAULT_KEYBINDINGS
    assert "shift+tab" in DEFAULT_KEYBINDINGS

def test_registry_get_action():
    reg = KeybindingsRegistry()
    assert reg.get_action("ctrl+l") == "clear_screen"
    assert reg.get_action("ctrl+n") == "new_session"

def test_registry_custom_override():
    custom = {"ctrl+l": "new_session"}
    reg = KeybindingsRegistry(overrides=custom)
    assert reg.get_action("ctrl+l") == "new_session"

def test_registry_unknown_key():
    reg = KeybindingsRegistry()
    assert reg.get_action("ctrl+q") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Aura && python -m pytest tests/cli/test_keybindings.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement keybindings registry**

```python
# aura/cli/keybindings.py
"""Keyboard shortcut registry with customization support."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Action constants
ACTION_CLEAR_SCREEN = "clear_screen"
ACTION_NEW_SESSION = "new_session"
ACTION_COMMAND_PALETTE = "command_palette"
ACTION_OPEN_EDITOR = "open_editor"
ACTION_REWIND = "rewind"
ACTION_CYCLE_PERMISSIONS = "cycle_permissions"
ACTION_MODEL_PICKER = "model_picker"
ACTION_SEARCH_HISTORY = "search_history"

DEFAULT_KEYBINDINGS: Dict[str, str] = {
    "ctrl+l": ACTION_CLEAR_SCREEN,
    "ctrl+n": ACTION_NEW_SESSION,
    "ctrl+k": ACTION_COMMAND_PALETTE,
    "ctrl+g": ACTION_OPEN_EDITOR,
    "ctrl+r": ACTION_SEARCH_HISTORY,
    "escape escape": ACTION_REWIND,
    "shift+tab": ACTION_CYCLE_PERMISSIONS,
    "alt+m": ACTION_MODEL_PICKER,
}

_KEYBINDINGS_PATH = Path.home() / ".aura" / "keybindings.json"


class KeybindingsRegistry:
    """Manages keyboard shortcuts with user customization."""

    def __init__(self, overrides: Optional[Dict[str, str]] = None):
        self._bindings: Dict[str, str] = dict(DEFAULT_KEYBINDINGS)
        # Load user overrides from file
        if overrides is None:
            overrides = self._load_user_overrides()
        if overrides:
            self._bindings.update(overrides)

    def _load_user_overrides(self) -> Dict[str, str]:
        if _KEYBINDINGS_PATH.exists():
            try:
                return json.loads(_KEYBINDINGS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def get_action(self, key_combo: str) -> Optional[str]:
        """Get the action bound to a key combination."""
        return self._bindings.get(key_combo.lower())

    def get_key_for_action(self, action: str) -> Optional[str]:
        """Get the key combo bound to an action (first match)."""
        for key, act in self._bindings.items():
            if act == action:
                return key
        return None

    def all_bindings(self) -> Dict[str, str]:
        """Return all current bindings."""
        return dict(self._bindings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/Aura && python -m pytest tests/cli/test_keybindings.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add aura/cli/keybindings.py tests/cli/test_keybindings.py
git commit -m "feat(cli): add keybindings registry with customization support"
```

### Task 5: Wire Keybindings into prompt_toolkit

**Files:**
- Modify: `aura/cli/input.py`
- Modify: `main.py`

- [ ] **Step 1: Read current input.py keybinding setup**

Read `D:\Aura\aura\cli\input.py` lines 80-163 to understand how `create_session()` sets up prompt_toolkit keybindings.

- [ ] **Step 2: Add new key signals to input.py**

In `create_session()`, after the existing Alt+M binding, add bindings for each shortcut. Each binding sets the buffer text to a signal string and validates (submits):

```python
# Signal constants (add at module top)
SIGNAL_MODEL_PICK = "__MODEL_PICK__"
SIGNAL_CLEAR_SCREEN = "__CLEAR_SCREEN__"
SIGNAL_NEW_SESSION = "__NEW_SESSION__"
SIGNAL_COMMAND_PALETTE = "__CMD_PALETTE__"
SIGNAL_OPEN_EDITOR = "__OPEN_EDITOR__"
SIGNAL_REWIND = "__REWIND__"
SIGNAL_CYCLE_PERMS = "__CYCLE_PERMS__"
```

For each new binding in `create_session()`:
```python
@kb.add('c-l')
def _clear(event):
    event.app.current_buffer.text = SIGNAL_CLEAR_SCREEN
    event.app.current_buffer.validate_and_handle()

@kb.add('c-n')
def _new(event):
    event.app.current_buffer.text = SIGNAL_NEW_SESSION
    event.app.current_buffer.validate_and_handle()

@kb.add('c-k')
def _palette(event):
    event.app.current_buffer.text = SIGNAL_COMMAND_PALETTE
    event.app.current_buffer.validate_and_handle()

@kb.add('c-g')
def _editor(event):
    event.app.current_buffer.text = SIGNAL_OPEN_EDITOR
    event.app.current_buffer.validate_and_handle()

@kb.add('s-tab')
def _cycle_perms(event):
    event.app.current_buffer.text = SIGNAL_CYCLE_PERMS
    event.app.current_buffer.validate_and_handle()
```

For `Esc Esc` (double escape), use prompt_toolkit's escape sequence:
```python
@kb.add('escape', 'escape')
def _rewind(event):
    event.app.current_buffer.text = SIGNAL_REWIND
    event.app.current_buffer.validate_and_handle()
```

- [ ] **Step 3: Handle signals in main.py run_chat_mode()**

In the main loop (around line 462-512), add handlers for each new signal before the existing `__MODEL_PICK__` check:

```python
if user_input == SIGNAL_CLEAR_SCREEN:
    console.clear()
    continue
elif user_input == SIGNAL_NEW_SESSION:
    # Create new session, reset history
    agentic = AgenticLoop(agent.brain, ...)  # reinitialize
    console.print("[dim]● New session started[/dim]")
    continue
elif user_input == SIGNAL_COMMAND_PALETTE:
    # Will be implemented in Task later (command_palette.py)
    console.print("[dim]● Command palette coming soon[/dim]")
    continue
elif user_input == SIGNAL_OPEN_EDITOR:
    # Open $EDITOR for multi-line input
    import tempfile, subprocess
    editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        f.write("")
        tmp_path = f.name
    subprocess.call([editor, tmp_path])
    user_input = Path(tmp_path).read_text().strip()
    Path(tmp_path).unlink(missing_ok=True)
    if not user_input:
        continue
    # Fall through to normal processing
elif user_input == SIGNAL_REWIND:
    # Will be implemented in Task 9 (checkpoint.py)
    console.print("[dim]● Rewind coming soon[/dim]")
    continue
elif user_input == SIGNAL_CYCLE_PERMS:
    # Will be implemented in Task 7 (permissions_ui.py)
    console.print("[dim]● Permission cycling coming soon[/dim]")
    continue
```

- [ ] **Step 4: Update show_help() with new shortcuts**

In `display.py` `show_help()`, add the new keyboard shortcuts to the help table.

- [ ] **Step 5: Test manually**

Run: `cd /d/Aura && python main.py`
Test: Press `Ctrl+L` (clear screen), `Ctrl+N` (new session), `Ctrl+G` (editor). Verify each works.

- [ ] **Step 6: Commit**

```bash
cd /d/Aura && git add aura/cli/input.py aura/cli/display.py main.py
git commit -m "feat(cli): wire keyboard shortcuts (Ctrl+L/N/K/G, Shift+Tab, Esc Esc)"
```

---

## Chunk 3: Permission Tiers

### Task 6: Permission Tier System

**Files:**
- Create: `aura/cli/permissions_ui.py`
- Modify: `aura/core/permissions.py`
- Test: `tests/cli/test_permissions_ui.py`

- [ ] **Step 1: Write failing test**

```python
# tests/cli/test_permissions_ui.py
import pytest
from aura.cli.permissions_ui import PermissionMode, cycle_permission_mode

def test_permission_modes_exist():
    assert PermissionMode.PLAN == "plan"
    assert PermissionMode.CAREFUL == "careful"
    assert PermissionMode.AUTO_EDIT == "auto_edit"
    assert PermissionMode.FULL_AUTO == "full_auto"

def test_cycle_forward():
    assert cycle_permission_mode("careful") == "auto_edit"
    assert cycle_permission_mode("auto_edit") == "full_auto"
    assert cycle_permission_mode("full_auto") == "plan"
    assert cycle_permission_mode("plan") == "careful"

def test_mode_description():
    from aura.cli.permissions_ui import get_mode_description
    desc = get_mode_description("plan")
    assert "read-only" in desc.lower() or "plan" in desc.lower()

def test_mode_indicator():
    from aura.cli.permissions_ui import get_mode_indicator
    indicator = get_mode_indicator("careful")
    assert len(indicator) > 0  # Non-empty string for status bar
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Aura && python -m pytest tests/cli/test_permissions_ui.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement permission tier UI**

```python
# aura/cli/permissions_ui.py
"""Permission tier UI — mode cycling and status display."""

from __future__ import annotations
from enum import Enum
from typing import Optional


class PermissionMode(str, Enum):
    PLAN = "plan"           # Read-only — no edits, no commands
    CAREFUL = "careful"     # Approve every edit and command (default)
    AUTO_EDIT = "auto_edit" # File edits auto-apply, commands still ask
    FULL_AUTO = "full_auto" # Everything runs, budget is the only guard


_MODE_ORDER = [
    PermissionMode.PLAN,
    PermissionMode.CAREFUL,
    PermissionMode.AUTO_EDIT,
    PermissionMode.FULL_AUTO,
]

_MODE_DESCRIPTIONS = {
    PermissionMode.PLAN: "Plan Mode — read-only, no file edits or commands",
    PermissionMode.CAREFUL: "Careful — approve every edit and shell command",
    PermissionMode.AUTO_EDIT: "Auto-Edit — file edits auto-apply, commands ask",
    PermissionMode.FULL_AUTO: "Full Auto — everything runs autonomously",
}

_MODE_INDICATORS = {
    PermissionMode.PLAN: "[blue]◎ PLAN[/blue]",
    PermissionMode.CAREFUL: "[yellow]◉ CAREFUL[/yellow]",
    PermissionMode.AUTO_EDIT: "[green]◉ AUTO-EDIT[/green]",
    PermissionMode.FULL_AUTO: "[red]● FULL-AUTO[/red]",
}

_MODE_SHORT = {
    PermissionMode.PLAN: "[blue]PLAN[/blue]",
    PermissionMode.CAREFUL: "[yellow]CARE[/yellow]",
    PermissionMode.AUTO_EDIT: "[green]AUTO[/green]",
    PermissionMode.FULL_AUTO: "[red]FULL[/red]",
}


def cycle_permission_mode(current: str) -> str:
    """Cycle to the next permission mode. Returns mode name string."""
    try:
        current_mode = PermissionMode(current)
    except ValueError:
        return PermissionMode.CAREFUL.value

    idx = _MODE_ORDER.index(current_mode)
    next_idx = (idx + 1) % len(_MODE_ORDER)
    return _MODE_ORDER[next_idx].value


def get_mode_description(mode: str) -> str:
    """Get human-readable description of a permission mode."""
    try:
        return _MODE_DESCRIPTIONS[PermissionMode(mode)]
    except (ValueError, KeyError):
        return "Unknown mode"


def get_mode_indicator(mode: str) -> str:
    """Get short indicator string for status bar."""
    try:
        return _MODE_SHORT[PermissionMode(mode)]
    except (ValueError, KeyError):
        return "[dim]???[/dim]"


def should_auto_approve_edit(mode: str) -> bool:
    """Whether file edits should be auto-approved in this mode."""
    return mode in (PermissionMode.AUTO_EDIT.value, PermissionMode.FULL_AUTO.value)


def should_auto_approve_command(mode: str) -> bool:
    """Whether shell commands should be auto-approved in this mode."""
    return mode == PermissionMode.FULL_AUTO.value


def should_block_mutations(mode: str) -> bool:
    """Whether all mutations (edits, commands) should be blocked."""
    return mode == PermissionMode.PLAN.value
```

- [ ] **Step 4: Run tests**

Run: `cd /d/Aura && python -m pytest tests/cli/test_permissions_ui.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add aura/cli/permissions_ui.py tests/cli/test_permissions_ui.py
git commit -m "feat(cli): add permission tier system (plan/careful/auto-edit/full-auto)"
```

### Task 7: Wire Permission Tiers into CLI

**Files:**
- Modify: `aura/core/permissions.py`
- Modify: `aura/cli/status_bar.py`
- Modify: `main.py`

- [ ] **Step 1: Read current permissions.py**

Read `D:\Aura\aura\core\permissions.py` to understand `PermissionManager.check()`.

- [ ] **Step 2: Integrate permission modes into PermissionManager**

Add a `_mode` attribute to `PermissionManager`. Override `check()` behavior based on mode:
- PLAN: always return False for mutating tools
- CAREFUL: existing behavior (use callback)
- AUTO_EDIT: auto-approve `edit_file` and `write_file`, callback for `shell`
- FULL_AUTO: always return True

- [ ] **Step 3: Add mode indicator to status bar**

In `build_status_bar()`, add the permission mode indicator after the model info: `| CARE |`

- [ ] **Step 4: Handle Shift+Tab in main.py**

When `SIGNAL_CYCLE_PERMS` is received, call `cycle_permission_mode()`, update the PermissionManager, and show a brief notification:

```python
elif user_input == SIGNAL_CYCLE_PERMS:
    current_mode = permissions.get_mode()
    new_mode = cycle_permission_mode(current_mode)
    permissions.set_mode(new_mode)
    console.print(f"[dim]● {get_mode_description(new_mode)}[/dim]")
    continue
```

- [ ] **Step 5: Test manually**

Run Aura, press `Shift+Tab` repeatedly. Verify mode cycles through all 4 tiers and status bar updates.

- [ ] **Step 6: Commit**

```bash
cd /d/Aura && git add aura/core/permissions.py aura/cli/permissions_ui.py aura/cli/status_bar.py main.py
git commit -m "feat(cli): wire permission tiers with Shift+Tab cycling and status bar indicator"
```

---

## Chunk 4: Inline Diff Viewer

### Task 8: Diff Rendering Engine

**Files:**
- Create: `aura/cli/diff_viewer.py`
- Test: `tests/cli/test_diff_viewer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/cli/test_diff_viewer.py
import pytest
from aura.cli.diff_viewer import render_diff, generate_diff

def test_generate_diff_basic():
    old = "line1\nline2\nline3\n"
    new = "line1\nmodified\nline3\n"
    diff = generate_diff(old, new, filename="test.py")
    assert "line2" in diff
    assert "modified" in diff
    assert "-" in diff or "+" in diff

def test_generate_diff_empty_old():
    diff = generate_diff("", "new content\n", filename="new.py")
    assert "new content" in diff

def test_generate_diff_no_change():
    diff = generate_diff("same\n", "same\n", filename="test.py")
    assert diff == "" or diff.strip() == ""

def test_render_diff_returns_rich_text():
    old = "def hello():\n    pass\n"
    new = "def hello():\n    return 'world'\n"
    result = render_diff(old, new, filename="test.py")
    # Should return a Rich renderable (Panel, Text, or similar)
    assert result is not None

def test_diff_summary_line():
    from aura.cli.diff_viewer import diff_summary
    old = "a\nb\nc\n"
    new = "a\nB\nc\nd\n"
    summary = diff_summary(old, new, filename="test.py")
    assert "test.py" in summary
    assert "+" in summary or "add" in summary.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Aura && python -m pytest tests/cli/test_diff_viewer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement diff viewer**

```python
# aura/cli/diff_viewer.py
"""Syntax-highlighted inline diff viewer for the terminal."""

from __future__ import annotations
import difflib
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


def generate_diff(old: str, new: str, filename: str = "file") -> str:
    """Generate unified diff string between old and new content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    ))

    if not diff_lines:
        return ""
    return "\n".join(diff_lines)


def diff_summary(old: str, new: str, filename: str = "file") -> str:
    """One-line summary of changes: 'test.py (+3/-1)'."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    added = 0
    removed = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    if added == 0 and removed == 0:
        return f"{filename} (no changes)"
    return f"{filename} ([green]+{added}[/green]/[red]-{removed}[/red])"


def render_diff(
    old: str,
    new: str,
    filename: str = "file",
    context_lines: int = 3,
) -> Optional[Panel]:
    """Render a syntax-highlighted diff as a Rich Panel.

    Returns None if no changes detected.
    """
    diff_text = generate_diff(old, new, filename)
    if not diff_text:
        return None

    # Build a Rich Text with colored +/- lines
    text = Text()
    for line in diff_text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            text.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        else:
            text.append(line + "\n", style="dim")

    summary = diff_summary(old, new, filename)
    return Panel(
        text,
        title=f"[bold]{summary}[/bold]",
        border_style="dim",
        padding=(0, 1),
    )


def render_diff_compact(old: str, new: str, filename: str = "file") -> str:
    """Render a compact one-line diff summary for tool call display."""
    return f"▸ [yellow]edit[/yellow] {diff_summary(old, new, filename)}"
```

- [ ] **Step 4: Run tests**

Run: `cd /d/Aura && python -m pytest tests/cli/test_diff_viewer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add aura/cli/diff_viewer.py tests/cli/test_diff_viewer.py
git commit -m "feat(cli): add syntax-highlighted inline diff viewer"
```

### Task 9: Wire Diff Viewer into Edit Flow

**Files:**
- Modify: `aura/cli/display.py`
- Modify: `aura/core/agentic_loop.py`

- [ ] **Step 1: Read agentic_loop.py _edit_file method**

Read `D:\Aura\aura\core\agentic_loop.py` around the `_edit_file()` method (line 389-427) to understand how edits are displayed.

- [ ] **Step 2: Modify _edit_file to use diff_viewer**

After the dry-run edit returns a diff, render it using `render_diff()` instead of printing raw text. The `on_tool_call` callback should receive the rendered diff for display.

In `_edit_file()`:
```python
from aura.cli.diff_viewer import render_diff, render_diff_compact

# After dry_run returns result with diff:
if result.get("diff"):
    # Parse the old/new from the diff or read the file before/after
    # Show compact summary via on_tool_call callback
    # Show full diff panel via console
    ...
```

- [ ] **Step 3: Update show_tool_call in display.py**

When the tool is `edit_file` and a diff is available, show the diff panel instead of just `▸ edit_file description`.

- [ ] **Step 4: Test manually**

Run Aura and ask it to edit a file. Verify the diff is shown with green/red highlighting before and after the edit.

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add aura/cli/display.py aura/cli/diff_viewer.py aura/core/agentic_loop.py
git commit -m "feat(cli): show syntax-highlighted diffs on file edits"
```

---

## Chunk 5: Checkpoint & Rewind

### Task 10: Checkpoint System

**Files:**
- Create: `aura/cli/checkpoint.py`
- Test: `tests/cli/test_checkpoint.py`

- [ ] **Step 1: Write failing test**

```python
# tests/cli/test_checkpoint.py
import pytest
import tempfile
import os
from pathlib import Path

def test_snapshot_and_restore(tmp_path):
    from aura.cli.checkpoint import CheckpointManager

    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")

    # Create a test file
    test_file = tmp_path / "test.py"
    test_file.write_text("original content")

    # Snapshot it
    cp_id = mgr.snapshot(str(test_file), label="before edit")
    assert cp_id is not None

    # Modify the file
    test_file.write_text("modified content")
    assert test_file.read_text() == "modified content"

    # Restore from checkpoint
    mgr.restore(cp_id)
    assert test_file.read_text() == "original content"

def test_list_checkpoints(tmp_path):
    from aura.cli.checkpoint import CheckpointManager

    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")
    test_file = tmp_path / "test.py"
    test_file.write_text("v1")
    mgr.snapshot(str(test_file), label="version 1")
    test_file.write_text("v2")
    mgr.snapshot(str(test_file), label="version 2")

    cps = mgr.list_checkpoints()
    assert len(cps) == 2
    assert cps[0]["label"] == "version 2"  # Most recent first

def test_prune_old_checkpoints(tmp_path):
    from aura.cli.checkpoint import CheckpointManager

    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints", max_checkpoints=3)
    test_file = tmp_path / "test.py"

    for i in range(5):
        test_file.write_text(f"version {i}")
        mgr.snapshot(str(test_file), label=f"v{i}")

    cps = mgr.list_checkpoints()
    assert len(cps) <= 3

def test_multi_file_snapshot(tmp_path):
    from aura.cli.checkpoint import CheckpointManager

    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("file a")
    f2.write_text("file b")

    cp_id = mgr.snapshot_multi([str(f1), str(f2)], label="multi edit")
    f1.write_text("changed a")
    f2.write_text("changed b")

    mgr.restore(cp_id)
    assert f1.read_text() == "file a"
    assert f2.read_text() == "file b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Aura && python -m pytest tests/cli/test_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement checkpoint manager**

```python
# aura/cli/checkpoint.py
"""File checkpoint system for rewind/undo support."""

from __future__ import annotations
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional


class CheckpointManager:
    """Manages file snapshots for checkpoint/rewind."""

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = 50,
    ):
        self._dir = Path(checkpoint_dir or Path.cwd() / ".aura_checkpoints")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_checkpoints
        self._index_path = self._dir / "index.json"
        self._index: List[Dict] = self._load_index()

    def _load_index(self) -> List[Dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def snapshot(self, file_path: str, label: str = "") -> str:
        """Snapshot a single file. Returns checkpoint ID."""
        return self.snapshot_multi([file_path], label=label)

    def snapshot_multi(self, file_paths: List[str], label: str = "") -> str:
        """Snapshot multiple files atomically. Returns checkpoint ID."""
        cp_id = f"cp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        cp_dir = self._dir / cp_id
        cp_dir.mkdir(parents=True, exist_ok=True)

        files_info = []
        for fp in file_paths:
            src = Path(fp)
            if src.exists():
                dest = cp_dir / src.name
                # Handle name collisions by adding a counter
                counter = 0
                while dest.exists():
                    counter += 1
                    dest = cp_dir / f"{src.stem}_{counter}{src.suffix}"
                shutil.copy2(str(src), str(dest))
                files_info.append({
                    "original_path": str(src.resolve()),
                    "backup_name": dest.name,
                })

        entry = {
            "id": cp_id,
            "timestamp": time.time(),
            "label": label,
            "files": files_info,
        }
        self._index.insert(0, entry)  # Most recent first

        # Prune old checkpoints
        while len(self._index) > self._max:
            old = self._index.pop()
            old_dir = self._dir / old["id"]
            if old_dir.exists():
                shutil.rmtree(old_dir, ignore_errors=True)

        self._save_index()
        return cp_id

    def restore(self, checkpoint_id: str) -> bool:
        """Restore files from a checkpoint."""
        entry = next((e for e in self._index if e["id"] == checkpoint_id), None)
        if not entry:
            return False

        cp_dir = self._dir / checkpoint_id
        if not cp_dir.exists():
            return False

        for f_info in entry["files"]:
            src = cp_dir / f_info["backup_name"]
            dest = Path(f_info["original_path"])
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dest))

        return True

    def list_checkpoints(self) -> List[Dict]:
        """Return all checkpoints, most recent first."""
        return list(self._index)

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Get a specific checkpoint's metadata."""
        return next((e for e in self._index if e["id"] == checkpoint_id), None)

    def clear(self):
        """Remove all checkpoints."""
        for entry in self._index:
            cp_dir = self._dir / entry["id"]
            if cp_dir.exists():
                shutil.rmtree(cp_dir, ignore_errors=True)
        self._index.clear()
        self._save_index()
```

- [ ] **Step 4: Run tests**

Run: `cd /d/Aura && python -m pytest tests/cli/test_checkpoint.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add aura/cli/checkpoint.py tests/cli/test_checkpoint.py
git commit -m "feat(cli): add checkpoint manager for file snapshot/rewind"
```

### Task 11: Wire Checkpoint into Edit Flow + Rewind UI

**Files:**
- Modify: `aura/core/agentic_loop.py`
- Modify: `main.py`

- [ ] **Step 1: Add checkpoint calls before file edits**

In `agentic_loop.py`, in the `_edit_file()` method, before applying the edit:

```python
# Before the actual edit:
if hasattr(self, '_checkpoint_mgr') and self._checkpoint_mgr:
    self._checkpoint_mgr.snapshot(file_path, label=f"before edit: {file_path}")
```

Similarly in `_write_file()` if the file already exists.

Initialize `_checkpoint_mgr` in `AgenticLoop.__init__()`:
```python
from aura.cli.checkpoint import CheckpointManager
self._checkpoint_mgr = CheckpointManager()
```

- [ ] **Step 2: Implement rewind picker in main.py**

When `SIGNAL_REWIND` is received (Esc Esc), show an interactive picker of recent checkpoints:

```python
elif user_input == SIGNAL_REWIND:
    cps = agentic._checkpoint_mgr.list_checkpoints()
    if not cps:
        console.print("[dim]● No checkpoints available[/dim]")
        continue
    # Show last 10 checkpoints
    console.print("\n[bold]Rewind to checkpoint:[/bold]")
    for i, cp in enumerate(cps[:10]):
        ts = time.strftime("%H:%M:%S", time.localtime(cp["timestamp"]))
        files = ", ".join(f["original_path"].split("/")[-1] for f in cp["files"])
        console.print(f"  {i+1}. [{ts}] {cp['label']} ({files})")
    console.print(f"  0. Cancel")
    choice = input("\nSelect checkpoint: ").strip()
    if choice.isdigit() and 0 < int(choice) <= min(10, len(cps)):
        selected = cps[int(choice) - 1]
        agentic._checkpoint_mgr.restore(selected["id"])
        files = ", ".join(f["original_path"].split("/")[-1] for f in selected["files"])
        console.print(f"[green]✓ Restored: {files}[/green]")
    continue
```

- [ ] **Step 3: Add /rewind slash command as alias**

In `handle_command()`, add `/rewind` that sends the same signal.

- [ ] **Step 4: Test manually**

Run Aura, ask it to edit a file, then press Esc Esc. Verify you can see checkpoints and restore.

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add aura/core/agentic_loop.py main.py
git commit -m "feat(cli): wire checkpoint/rewind — Esc Esc to undo file changes"
```

---

## Chunk 6: Integration & Polish

### Task 12: Final Integration and Testing

**Files:**
- Modify: `aura/cli/display.py` — Update help table with all new features
- Modify: `aura/cli/status_bar.py` — Final layout with token gauge + permission mode
- Modify: `main.py` — Ensure all signals, commands, and displays work together

- [ ] **Step 1: Update help table**

Add all new shortcuts and commands to `show_help()`:
```
Ctrl+L          Clear screen
Ctrl+N          New session
Ctrl+K          Command palette
Ctrl+G          Open editor for long prompt
Shift+Tab       Cycle permission mode
Esc Esc         Rewind to checkpoint
/context        Show context window usage
/rewind         Rewind file changes
```

- [ ] **Step 2: Final status bar layout**

Verify the status bar now shows:
```
[~/project (python) main] | kimi-k2.5:cloud | 42 tools | balanced | $0.02 | 12.4K/128K [████░░░░] | CARE | session (5 msgs) Alt+M
```

- [ ] **Step 3: Run all tests**

Run: `cd /d/Aura && python -m pytest tests/cli/ -v`
Expected: All tests pass.

- [ ] **Step 4: Manual integration test**

Test the full flow:
1. Start Aura → banner + status bar with token gauge + permission mode
2. Press `Shift+Tab` → mode cycles, status bar updates
3. Ask Aura to edit a file → see inline diff with syntax highlighting
4. Press `Esc Esc` → see checkpoint picker, restore works
5. Type `/context` → see detailed token breakdown
6. Press `Ctrl+L` → screen clears
7. Press `Ctrl+N` → new session starts
8. Press `Ctrl+G` → editor opens
9. Type `?` → help table shows all new shortcuts

- [ ] **Step 5: Commit**

```bash
cd /d/Aura && git add -A
git commit -m "feat(cli): Phase 1 complete — context gauge, shortcuts, permissions, diffs, rewind"
```
