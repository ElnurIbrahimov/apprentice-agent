# AURA CLI Interface Design
## "The Living Terminal" -- A CLI UX for an OS-Level Agent

**Date:** 2026-02-28
**Status:** Design specification
**Goal:** Replace AURA's raw `print()/input()` CLI with a polished, professional terminal interface that makes AURA feel alive.

---

## 1. What's Wrong With the Current CLI

The current `main.py` uses bare `print()` and `input()`. Problems:

- **No visual distinction** between phases (OBSERVE, PLAN, ACT, EVALUATE are just `[OBSERVE]` text prefixes)
- **No streaming display** -- response text just dumps raw with `print(chunk, end="", flush=True)`
- **No tool-call visibility** -- when the agent loop runs, you see debug-style `[ACT] Deciding...` text that looks like logs, not UI
- **No context awareness display** -- user can't see what AURA loaded (memories, KG context, emotions, screen state)
- **No progress indication** -- long operations have no spinners or progress bars
- **`input()` is primitive** -- no history, no multiline, no autocomplete, no key bindings
- **Startup is bare** -- just one line: `AURA  Autonomous Universal Reasoning Agent`

---

## 2. Library Stack

```
prompt_toolkit    -- Input handling (history, multiline, keybinds, autocomplete)
rich              -- Output rendering (markdown, panels, spinners, live display, tables)
```

**Why this pair, not Textual:**
Textual is a full TUI framework (widgets, layouts, event loops). AURA is a REPL, not a dashboard. A REPL needs excellent *input* (prompt_toolkit) and excellent *output* (rich). Textual would over-engineer the I/O loop and fight with AURA's existing asyncio/threading for LLM streaming. The prompt_toolkit + rich combo is what tools like Aider, Claude Code, and most serious CLI agents use under the hood.

**Install:**
```
pip install rich prompt_toolkit
```

Both are pure Python, zero native deps, and already widely used.

---

## 3. REPL Loop Architecture

### 3.1 High-Level Flow

```
STARTUP
  |
  v
[Load Config] -> [Load Identity] -> [Load Memories] -> [Check Screen/Clipboard]
  |
  v
[Display Startup Banner]  (shows loaded context summary)
  |
  v
+------- REPL LOOP -------+
|                          |
|  [1] PROMPT (user input) |  <-- prompt_toolkit: history, multiline, autocomplete
|          |               |
|          v               |
|  [2] PRE-PROCESS         |  <-- fast path check, emotion analysis, context gather
|          |               |      (shown as dim status line)
|          v               |
|  [3] DISPATCH            |  <-- route to: chat_stream, agent run, or slash command
|          |               |
|          v               |
|  [4] DISPLAY RESPONSE    |  <-- rich: streaming markdown, tool call panels, thinking
|          |               |
|          v               |
|  [5] POST-PROCESS        |  <-- KG extraction, memory save, strategy eval
|          |               |      (shown as dim status line, then vanishes)
|          v               |
|  [LOOP BACK TO 1]        |
+---------------------------+
```

### 3.2 The Display Module: `aura/cli/display.py`

A single module that owns all terminal output. Nothing in `agent.py` should ever call `print()` directly. Instead, the agent emits events and the display module renders them.

```python
# aura/cli/display.py

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.rule import Rule
from rich.status import Status

console = Console()

class AuraDisplay:
    """Centralized display renderer for AURA's CLI."""

    def __init__(self):
        self.console = Console()
        self._live: Optional[Live] = None

    # --- Startup ---
    def show_banner(self, identity: dict, loaded_context: dict):
        """Render the startup banner."""
        ...

    # --- Phase indicators ---
    def show_thinking(self, thought_type: str, content: str):
        """Show a thinking indicator (dim, transient)."""
        ...

    def show_tool_call(self, tool_name: str, action: str, status: str = "running"):
        """Show a tool call with spinner."""
        ...

    def show_tool_result(self, tool_name: str, success: bool, summary: str):
        """Show tool result (replaces spinner with checkmark/x)."""
        ...

    # --- Streaming response ---
    def start_response_stream(self):
        """Begin streaming a response (sets up live display)."""
        ...

    def stream_chunk(self, chunk: str):
        """Append a chunk to the streaming response."""
        ...

    def end_response_stream(self):
        """Finalize the streamed response (render as full markdown)."""
        ...

    # --- Context display ---
    def show_context_loaded(self, context_items: list):
        """Show what context was auto-loaded (transient)."""
        ...

    # --- Agent loop ---
    def show_phase(self, phase: str, iteration: int, detail: str = ""):
        """Show agent phase transition."""
        ...
```

---

## 4. Specific Display Patterns

### 4.1 Startup Banner

What it shows on `aura` launch:

```
  ╭──────────────────────────────────────────────╮
  │             A U R A   v3.0                   │
  │    Autonomous Universal Reasoning Agent      │
  ╰──────────────────────────────────────────────╯

  Model: devstral-2:123b-cloud
  Memory: 847 episodes, 12 recent
  Screen: VS Code - agent.py (detected)
  Mood: curious (from last session)

  Type /help for commands. Ctrl+C to cancel. Ctrl+D to exit.
```

Implementation:
```python
def show_banner(self, identity: dict, loaded_context: dict):
    self.console.print()
    self.console.print(
        Panel(
            Text("A U R A   v3.0", style="bold cyan", justify="center"),
            subtitle="Autonomous Universal Reasoning Agent",
            border_style="cyan",
            padding=(0, 4),
        )
    )
    self.console.print()

    # Context summary as a compact table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()

    model = loaded_context.get("model", "auto")
    table.add_row("Model:", f"[green]{model}[/green]")

    mem_count = loaded_context.get("memory_count", 0)
    recent = loaded_context.get("recent_memories", 0)
    table.add_row("Memory:", f"{mem_count} episodes, {recent} recent")

    if loaded_context.get("screen_context"):
        table.add_row("Screen:", f"[dim]{loaded_context['screen_context']}[/dim]")

    if loaded_context.get("mood"):
        table.add_row("Mood:", f"{loaded_context['mood']}")

    self.console.print(table)
    self.console.print()
    self.console.print("[dim]Type /help for commands. Ctrl+C to cancel. Ctrl+D to exit.[/dim]")
    self.console.print()
```

### 4.2 User Input (prompt_toolkit)

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

bindings = KeyBindings()

@bindings.add('escape', 'enter')  # Alt+Enter for multiline submit
def _(event):
    event.current_buffer.validate_and_handle()

session = PromptSession(
    message=HTML('<ansibrightcyan><b>you</b></ansibrightcyan> <ansigray>></ansigray> '),
    history=FileHistory(str(Path.home() / ".aura" / "chat_history")),
    auto_suggest=AutoSuggestFromHistory(),
    multiline=False,  # single line by default
    key_bindings=bindings,
    enable_history_search=True,  # Ctrl+R to search history
    complete_while_typing=False,
)
```

**Key behaviors:**
- `Enter` submits single-line input
- `{` or triple-backtick auto-switches to multiline mode (Enter = newline, Alt+Enter = submit)
- `Ctrl+R` searches history
- `Tab` on `/` prefix autocompletes slash commands
- `Ctrl+C` interrupts current response (not exit)
- `Ctrl+D` exits AURA

**Slash command completion:**
```python
from prompt_toolkit.completion import WordCompleter

command_completer = WordCompleter([
    '/help', '/quit', '/exit', '/goal', '/recall', '/clear',
    '/model', '/compact', '/plan', '/browse', '/agent', '/hook',
    '/speak', '/voice', '/dream', '/status', '/tools', '/memory',
], sentence=True)
```

### 4.3 Thinking Display (Transient)

When AURA is pre-processing (emotion analysis, context gathering, fast-path check), show a transient status that disappears when the response starts:

```
  you > What's the best way to structure a FastAPI project?

  thinking...  Recalling 3 relevant memories  Loading KG context
```

This line uses `rich.status.Status` and gets replaced by the actual response. It should be DIM text, not prominent -- it's ambient awareness, not content.

```python
def show_preprocessing(self, steps: list[str]):
    """Show transient preprocessing steps."""
    # Using rich Status with custom spinner
    with self.console.status(
        "[dim]thinking...[/dim]",
        spinner="dots",
        spinner_style="cyan",
    ) as status:
        for step in steps:
            status.update(f"[dim]{step}[/dim]")
            yield  # caller advances when each step completes
```

### 4.4 Tool Call Display (Inline, Persistent)

When the agent loop runs tools, each tool call appears as a compact inline block:

```
  you > Research the latest attention mechanisms in transformers

  aura  thinking...

  > web_search  Searching "attention mechanisms transformers 2026"...
  > web_search  Found 8 results (0.3s)

  > arxiv  Searching arxiv for "multi-head attention alternatives"...
  > arxiv  Found 12 papers (1.2s)

  > memory  Recalling related notes...
  > memory  Found 2 relevant memories

  aura  Based on my research, here are the key developments...
        [streaming markdown response continues]
```

**Design rules for tool calls:**
- Prefix with `>` in dim style (visually subordinate to conversation)
- Tool name in bold-dim
- Action description as running text
- Result replaces the spinner line (in-place update via `rich.live.Live`)
- Timing shown in parentheses
- If tool errors, show in red: `> web_search  ERROR: Connection timeout (2.1s)`

```python
def show_tool_call(self, tool_name: str, description: str):
    """Show a tool call starting."""
    self.console.print(
        f"  [dim]>[/dim] [bold dim]{tool_name}[/bold dim]  "
        f"[dim]{description}[/dim]",
    )

def show_tool_result(self, tool_name: str, success: bool, summary: str, elapsed: float):
    """Show tool call result (printed on new line after the call)."""
    status_icon = "[green]done[/green]" if success else "[red]ERROR[/red]"
    self.console.print(
        f"  [dim]>[/dim] [bold dim]{tool_name}[/bold dim]  "
        f"{status_icon} {summary} [dim]({elapsed:.1f}s)[/dim]",
    )
```

### 4.5 Streaming Response Display

The main response streams as markdown, rendered incrementally:

```python
def stream_response(self, chunk_generator):
    """Stream a response with live markdown rendering."""
    buffer = ""

    self.console.print()  # blank line before response
    self.console.print("  [bold cyan]aura[/bold cyan]  ", end="")

    # For short responses, just print inline
    # For long responses, use Live display for markdown rendering
    with Live(console=self.console, refresh_per_second=15, transient=False) as live:
        for chunk in chunk_generator:
            buffer += chunk
            # Render accumulated markdown
            live.update(Markdown(buffer))

    self.console.print()  # blank line after response
```

**But there's a subtlety.** Rich's `Markdown` re-renders the entire buffer each time, which can cause visual jitter on long responses. The better approach:

```python
def stream_response(self, chunk_generator):
    """Stream response: raw text while streaming, markdown once complete."""
    buffer = ""
    self.console.print("  [bold cyan]aura[/bold cyan]  ", end="")

    for chunk in chunk_generator:
        buffer += chunk
        self.console.print(chunk, end="", highlight=False)

    # After streaming completes, optionally re-render as formatted markdown
    # Only if the response contains markdown syntax
    if any(marker in buffer for marker in ['```', '**', '# ', '| ', '- [']):
        self.console.print()
        self.console.print(Panel(Markdown(buffer), border_style="dim", padding=(0, 1)))
    else:
        self.console.print()
```

### 4.6 Agent Loop Display (Goal Mode)

When running `/goal` or `aura "task"`, the OBSERVE/PLAN/ACT/EVALUATE loop needs distinct visual treatment:

```
  ╭─ Goal: Deploy the RentEase auth module ──────────────────╮
  │                                                          │
  │  Iteration 1/10                                          │
  │                                                          │
  │  OBSERVE  Analyzing project structure...                 │
  │  PLAN     1. Check auth directory  2. Run tests          │
  │  ACT      > filesystem  Reading RentEase/auth/...        │
  │           > code_executor  Running pytest...             │
  │  EVALUATE Tests passing. Auth module ready.              │
  │                                                          │
  │  Iteration 2/10                                          │
  │  ...                                                     │
  ╰──────────────────────────────────────────────────────────╯
```

```python
PHASE_STYLES = {
    "OBSERVE":  ("bold blue",    "eye"),
    "PLAN":     ("bold yellow",  "clipboard"),
    "ACT":      ("bold green",   "wrench"),
    "EVALUATE": ("bold magenta", "check"),
    "REMEMBER": ("bold dim",     "brain"),
}

def show_phase(self, phase: str, detail: str = ""):
    style, icon_name = PHASE_STYLES.get(phase, ("dim", "circle"))
    icon = {"eye": "O", "clipboard": "P", "wrench": "A", "check": "E", "brain": "R"}[icon_name]
    self.console.print(
        f"  [{style}]{phase:10s}[/{style}] [dim]{detail}[/dim]"
    )
```

### 4.7 Context Panel (What AURA auto-loaded)

After pre-processing and before the response starts, briefly show what context AURA pulled in:

```
  context  3 memories | KG: "FastAPI", "Pydantic" | Emotion: curious | Screen: VS Code
```

This is a single dim line. It answers "what did AURA know when it answered this?" -- critical for trust.

```python
def show_context_summary(self, context: dict):
    parts = []
    if context.get("memories"):
        parts.append(f"{len(context['memories'])} memories")
    if context.get("kg_entities"):
        entities = ", ".join(f'"{e}"' for e in context["kg_entities"][:3])
        parts.append(f"KG: {entities}")
    if context.get("emotion"):
        parts.append(f"Emotion: {context['emotion']}")
    if context.get("screen"):
        parts.append(f"Screen: {context['screen']}")

    if parts:
        summary = " | ".join(parts)
        self.console.print(f"  [dim]context[/dim]  [dim]{summary}[/dim]")
```

### 4.8 Thinking vs Acting vs Responding

Three distinct visual states:

| State | Visual | When |
|-------|--------|------|
| **Thinking** | Dim spinner + italic text: `thinking... Analyzing query structure` | Pre-processing, fast-path check, emotion analysis, context gathering |
| **Acting** | `>` prefixed lines with tool name in bold: `> web_search Searching...` | Tool execution during agent loop or chat pre-handlers |
| **Responding** | `aura` label in cyan bold, then streaming text | LLM response streaming |

The key insight: **thinking is transient** (vanishes), **acting is persistent** (stays in scrollback), **responding is the main content**.

---

## 5. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Submit input |
| `Alt+Enter` | Newline (multiline mode) |
| `Ctrl+C` | Cancel current response / clear input |
| `Ctrl+D` | Exit AURA |
| `Ctrl+R` | Search input history |
| `Ctrl+L` | Clear screen (keeps session) |
| `Up/Down` | Navigate input history |
| `Tab` | Autocomplete slash commands |
| `Ctrl+K` | Toggle thinking visibility (show/hide internal reasoning) |
| `Ctrl+T` | Toggle tool call visibility (show/hide tool executions) |

```python
@bindings.add('c-k')  # Ctrl+K
def toggle_thinking(event):
    display.show_thinking_enabled = not display.show_thinking_enabled
    status = "on" if display.show_thinking_enabled else "off"
    console.print(f"  [dim]Thinking display: {status}[/dim]")

@bindings.add('c-t')  # Ctrl+T
def toggle_tools(event):
    display.show_tools_enabled = not display.show_tools_enabled
    status = "on" if display.show_tools_enabled else "off"
    console.print(f"  [dim]Tool display: {status}[/dim]")

@bindings.add('c-l')  # Ctrl+L
def clear_screen(event):
    console.clear()
```

---

## 6. Voice Mode Integration

Voice mode changes the input method but keeps the same display:

```
  ╭──────────────────────────────────────────────╮
  │             A U R A   v3.0                   │
  │         Voice mode active                    │
  ╰──────────────────────────────────────────────╯

  [MIC] Listening...               <- animated dots spinner
  [MIC] "What's the weather like?" <- transcription appears

  aura  [SPEAKING] Let me check...
        The weather in Baku is currently...
```

**Push-to-talk vs wake word:**
- Default: push-to-talk via `Space` key (hold to record, release to submit)
- Optional: wake word "Hey Aura" (always listening, no key needed)
- Visual indicator: `[MIC]` label turns green when recording, red when processing
- During TTS playback: `[SPEAKING]` label shown before response

```python
# Voice mode prompt (replaces text input)
def voice_input_display(self, state: str):
    states = {
        "listening":    "[bold green][MIC][/bold green] [dim]Listening...[/dim]",
        "processing":   "[bold yellow][MIC][/bold yellow] [dim]Processing...[/dim]",
        "transcribed":  "[bold blue][MIC][/bold blue]",
        "speaking":     "[bold cyan][SPK][/bold cyan]",
        "idle":         "[dim][MIC][/dim] [dim]Press Space to talk[/dim]",
    }
    self.console.print(f"  {states.get(state, states['idle'])}", end="\r")
```

---

## 7. Slash Commands Display

Slash commands get their own styled output:

```
  you > /help

  ╭─ AURA Commands ─────────────────────────────────────────╮
  │                                                         │
  │  /help          Show this help                          │
  │  /model [name]  Show or set model                       │
  │  /goal <task>   Run agent loop on a task                │
  │  /plan <task>   Preview execution plan before running   │
  │  /recall <q>    Search memories                         │
  │  /clear         Clear conversation history              │
  │  /compact [f]   Compress history (optional focus)       │
  │  /tools         List available tools                    │
  │  /status        Show system status                      │
  │  /browse <url>  Open URL in browser tool                │
  │  /agent <s> <t> Run specialist agent                    │
  │  /hook ...      Manage event hooks                      │
  │  /voice         Toggle voice mode                       │
  │  /dream         Run dream consolidation                 │
  │  /quit          Exit AURA                               │
  │                                                         │
  ╰─────────────────────────────────────────────────────────╯
```

```python
def show_help(self):
    table = Table(title="AURA Commands", box=rich.box.ROUNDED, border_style="cyan")
    table.add_column("Command", style="bold")
    table.add_column("Description")

    commands = [
        ("/help", "Show this help"),
        ("/model [name]", "Show or set model"),
        ("/goal <task>", "Run agent loop on a task"),
        # ... etc
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    self.console.print(table)
```

---

## 8. Error Display

Errors are categorized and styled:

```python
def show_error(self, error_type: str, message: str, recoverable: bool = True):
    style = "yellow" if recoverable else "red"
    label = "warning" if recoverable else "error"
    self.console.print(f"  [{style}]{label}[/{style}]  {message}")
    if recoverable:
        self.console.print(f"  [dim]AURA will retry or use fallback.[/dim]")
```

```
  warning  Model timeout, falling back to qwen2.5-coder:7b
  AURA will retry or use fallback.

  error  No API connection available. Check Ollama is running.
```

---

## 9. Status Bar (Optional, Advanced)

A persistent bottom-of-screen status bar showing live system state:

```
 AURA v3.0 | devstral-2:123b | Mem: 847 | KG: 2.1k entities | Uptime: 1h23m | CPU: 12% GPU: 45%
```

This uses prompt_toolkit's bottom toolbar:

```python
def bottom_toolbar():
    model = agent.brain.get_current_model()
    mem = agent.memory.count()
    return HTML(
        f'<b>AURA</b> v3.0 | '
        f'<ansicyan>{model}</ansicyan> | '
        f'Mem: {mem} | '
        f'<ansigreen>ready</ansigreen>'
    )

session = PromptSession(
    message=HTML('<ansibrightcyan><b>you</b></ansibrightcyan> > '),
    bottom_toolbar=bottom_toolbar,
    refresh_interval=5,  # update every 5 seconds
    ...
)
```

---

## 10. File Structure

```
aura/
  cli/
    __init__.py
    display.py        # AuraDisplay class - all rendering logic
    input.py          # AuraInput class - prompt_toolkit session setup
    banner.py         # Startup banner rendering
    themes.py         # Color schemes (dark, light, minimal)
    formatters.py     # Tool result formatters, markdown helpers
main.py               # Thin orchestrator: creates AuraInput + AuraDisplay + Agent, runs loop
```

### 10.1 The New `main.py` REPL

```python
def run_chat_mode(agent, speak=False):
    display = AuraDisplay()
    input_session = AuraInput()

    # Startup
    context = gather_startup_context(agent)
    display.show_banner(agent.identity, context)

    # REPL
    while True:
        try:
            user_input = input_session.prompt()
        except EOFError:
            display.show_exit()
            break
        except KeyboardInterrupt:
            continue  # Ctrl+C clears input, doesn't exit

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            handle_command(agent, user_input, display)
        else:
            # Pre-processing with status display
            with display.thinking_context() as thinking:
                thinking.update("Analyzing...")
                # emotion, context, fast-path checks happen here
                preprocessed = agent.preprocess(user_input)
                if preprocessed.context_items:
                    display.show_context_summary(preprocessed.context_items)

            # Stream response
            display.show_response_label()
            for chunk in agent.chat_stream(user_input, speak=speak):
                display.stream_chunk(chunk)
            display.end_response()
```

---

## 11. Event-Driven Architecture: Agent -> Display Bridge

The agent should NOT call `print()`. Instead, it emits events that the display layer consumes. This decouples the agent logic from the presentation.

```python
# aura/cli/events.py

from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class EventType(Enum):
    THINKING = "thinking"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    PHASE_CHANGE = "phase_change"
    STREAM_CHUNK = "stream_chunk"
    CONTEXT_LOADED = "context_loaded"
    ERROR = "error"
    STATUS = "status"

@dataclass
class DisplayEvent:
    type: EventType
    data: dict
    timestamp: float

class EventBus:
    """Simple pub/sub for agent -> display communication."""
    def __init__(self):
        self._handlers = {}

    def on(self, event_type: EventType, handler):
        self._handlers.setdefault(event_type, []).append(handler)

    def emit(self, event: DisplayEvent):
        for handler in self._handlers.get(event.type, []):
            handler(event)
```

In `agent.py`, replace every `print(f"[OBSERVE]...")` with:

```python
self.event_bus.emit(DisplayEvent(
    type=EventType.PHASE_CHANGE,
    data={"phase": "OBSERVE", "detail": "Analyzing current context..."},
    timestamp=time.time()
))
```

The display module subscribes:

```python
bus.on(EventType.PHASE_CHANGE, lambda e: display.show_phase(e.data["phase"], e.data["detail"]))
bus.on(EventType.TOOL_START, lambda e: display.show_tool_call(e.data["tool"], e.data["action"]))
bus.on(EventType.TOOL_END, lambda e: display.show_tool_result(e.data["tool"], e.data["success"], e.data["summary"], e.data["elapsed"]))
bus.on(EventType.THINKING, lambda e: display.show_thinking(e.data["type"], e.data["content"]))
```

This means the same agent can drive the CLI display, the web UI (via WebSocket), and the API -- all by subscribing different handlers to the same event bus.

---

## 12. Comparison: AURA vs Claude Code UX

| Feature | Claude Code | AURA (this design) |
|---------|-------------|---------------------|
| Input | prompt_toolkit with multiline | Same, plus slash autocomplete, bottom toolbar |
| Thinking | Shimmer spinner with "thinking verb" | Dim transient status with specific step names |
| Tool calls | Hidden by default (can peek via claude-esp) | Visible by default, dim inline `>` prefix |
| Response | Streamed markdown with rich formatting | Same, raw stream then optional markdown reformat |
| Context | Not shown | Explicit one-line summary before response |
| Agent loop | N/A (single turn) | Phase indicators (OBSERVE/PLAN/ACT/EVALUATE) |
| Memory | N/A | Shown in banner, searchable via /recall |
| Voice | N/A | Integrated mode with visual MIC/SPK indicators |
| Keyboard | Ctrl+C cancel, Ctrl+D exit | Same + Ctrl+K thinking toggle, Ctrl+T tools toggle |
| Status bar | None | Bottom toolbar with model, memory count, state |
| Startup | "Claude Code" title | Full context summary: model, memories, screen, mood |
| Event system | Internal | Event bus (drives CLI, web, API from same source) |

---

## 13. Implementation Priority

### Phase 1: Core REPL (replace current main.py)
1. `aura/cli/display.py` -- AuraDisplay with `show_banner`, `stream_chunk`, `show_error`
2. `aura/cli/input.py` -- AuraInput with prompt_toolkit session, history, slash completion
3. Rewrite `run_chat_mode()` in `main.py` to use AuraDisplay + AuraInput
4. Keep all existing agent logic untouched

### Phase 2: Tool Call Visibility
1. Add EventBus to agent
2. Replace `print()` calls in `agent.py` with event emissions
3. Wire AuraDisplay to EventBus
4. Show tool calls inline during both chat and goal modes

### Phase 3: Context & Thinking
1. Show context summary line before responses
2. Show preprocessing status (thinking spinner)
3. Add Ctrl+K / Ctrl+T toggles

### Phase 4: Polish
1. Bottom status toolbar
2. Voice mode visual integration
3. Theme support (dark/light/minimal)
4. Agent loop panel display for /goal mode

---

## 14. Dependencies

```
# Already likely installed:
rich>=13.0           # Terminal formatting
prompt_toolkit>=3.0  # Input handling

# No new deps needed. Both are pure Python.
```

---

## 15. Design Principles

1. **Ambient, not noisy.** Thinking and context are dim/transient. Tool calls are subordinate. Only the response is prominent.
2. **Trust through transparency.** Show what AURA loaded (context line) and what it did (tool calls). User can toggle these off if they want clean output.
3. **Interrupt-friendly.** Ctrl+C always cancels gracefully. No stuck states.
4. **Progressive disclosure.** Simple chat looks simple. Agent loop shows more. /status shows everything.
5. **Decouple agent from display.** Event bus means the same agent drives CLI, web UI, and API.
6. **Input history is memory.** prompt_toolkit history file means Ctrl+R searches past inputs across sessions.
