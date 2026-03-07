# AURA Living OS — Full Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform AURA from a lobotomized chatbot into a living OS-layer agent — always-on, fully tool-capable CLI, parallel intelligence modules, proactive daemon.

**Architecture:** Four phases building on each other. Phase 1 unlocks all tools in every conversation (one line change + rich CLI). Phase 2 adds always-on context awareness. Phase 3 wires 12 intelligence modules into a parliament. Phase 4 makes AURA a persistent background service.

**Tech Stack:** Python, rich 14.3.2, prompt_toolkit 3.0.52, threading/concurrent.futures, NSSM (Windows service), existing AURA tools/memory infrastructure.

**Critical facts from codebase analysis:**
- `agent.run(message)` returns `dict` with `result["response"]` = final text
- `agent.chat_stream()` = zero tools, zero memory write — being killed
- `rich` and `prompt_toolkit` already installed
- All 20+ tools already exist and work — just not called in chat mode
- `D:/Aura/main.py:122` is the exact line to fix: `chat_stream()` → `agent.run()`
- Fast path: simple queries bypass the agent loop (preserved)
- `brain.conversation_history` persists between `run()` calls (context continuity)

---

## PHASE 1: Kill Chat Mode — Wire Real Agent Loop + Rich CLI
**Every input now goes through the full agent loop. AURA can read files, search web, run code.**

---

### Task 1: Create CLI package skeleton

**Files:**
- Create: `D:/Aura/aura/cli/__init__.py`

**Step 1: Create the package init**
```python
# D:/Aura/aura/cli/__init__.py
"""AURA CLI interface — rich display + prompt_toolkit input."""
```

**Step 2: Verify it exists**
```bash
python -c "from aura.cli import __init__; print('ok')"
```
Expected: `ok`

**Step 3: Commit**
```bash
git -C D:/Aura add aura/cli/__init__.py
git -C D:/Aura commit -m "feat: add aura/cli package skeleton"
```

---

### Task 2: Create display module

**Files:**
- Create: `D:/Aura/aura/cli/display.py`

**Step 1: Write display.py**
```python
# D:/Aura/aura/cli/display.py
"""Rich-based display for AURA CLI."""

from rich.console import Console
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

console = Console(highlight=False)


def show_banner():
    console.print("\n[bold cyan]AURA[/bold cyan]  Autonomous Universal Reasoning Agent\n")


def show_thinking(label: str = "Working..."):
    """Context manager — shows spinner while agent runs."""
    return Live(
        Spinner("dots", text=f"  [dim]{label}[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,   # disappears when done
    )


def show_tool_call(tool_name: str, description: str = ""):
    """Print a tool call line inline."""
    console.print(f"  [dim cyan]▸[/dim cyan] [cyan]{tool_name}[/cyan]  [dim]{description}[/dim]")


def show_response(text: str):
    """Render agent response as markdown."""
    console.print()
    console.print("[bold cyan]AURA[/bold cyan]")
    try:
        console.print(Markdown(text))
    except Exception:
        console.print(text)
    console.print()


def show_error(message: str):
    console.print(f"\n[bold red]Error:[/bold red] {message}\n")


def show_info(message: str):
    console.print(f"[dim]{message}[/dim]")
```

**Step 2: Quick smoke test**
```bash
python -c "
from aura.cli.display import show_banner, show_response
show_banner()
show_response('Hello **world**. This is `markdown`.')
print('display ok')
"
```
Expected: Banner + formatted markdown + "display ok"

**Step 3: Commit**
```bash
git -C D:/Aura add aura/cli/display.py
git -C D:/Aura commit -m "feat: add rich display module for AURA CLI"
```

---

### Task 3: Create input module

**Files:**
- Create: `D:/Aura/aura/cli/input.py`

**Step 1: Write input.py**
```python
# D:/Aura/aura/cli/input.py
"""prompt_toolkit-based input for AURA CLI — history, autocomplete."""

from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

HISTORY_FILE = Path.home() / ".aura_history"

_STYLE = Style.from_dict({
    "prompt": "bold cyan",
})

SLASH_COMMANDS = [
    "/quit", "/exit", "/goal", "/recall", "/clear",
    "/speak", "/model", "/compact", "/plan", "/browse",
    "/agent", "/hook",
]


def create_session() -> PromptSession:
    """Create a prompt_toolkit session with persistent history."""
    return PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        style=_STYLE,
    )


def get_input(session: PromptSession) -> str | None:
    """
    Get one line of user input.
    Returns None on Ctrl+D / Ctrl+C (signal to exit).
    """
    try:
        return session.prompt("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
```

**Step 2: Smoke test**
```bash
python -c "from aura.cli.input import create_session; s = create_session(); print('input ok')"
```
Expected: `input ok`

**Step 3: Commit**
```bash
git -C D:/Aura add aura/cli/input.py
git -C D:/Aura commit -m "feat: add prompt_toolkit input module for AURA CLI"
```

---

### Task 4: Rewrite run_chat_mode() in main.py

This is the critical change. `chat_stream()` dies here. `agent.run()` takes over.

**Files:**
- Modify: `D:/Aura/main.py:96-124`

**Step 1: Replace `run_chat_mode()` function**

Find this exact block in main.py (lines 96-124):
```python
def run_chat_mode(agent, speak: bool = False):
    """Run the agent in interactive chat mode.

    Args:
        agent: The agent instance
        speak: If True, speak responses using TTS
    """
    print("\033[1;36mAURA\033[0m  Autonomous Universal Reasoning Agent")
    if speak:
        print("\033[33mVoice output enabled\033[0m")
    print()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(agent, user_input, speak=speak)
        else:
            print("\nAURA: ", end="", flush=True)
            for chunk in agent.chat_stream(user_input, speak=speak):
                print(chunk, end="", flush=True)
            print()
```

Replace with:
```python
def run_chat_mode(agent, speak: bool = False):
    """Interactive CLI — every input goes through the full agent loop."""
    import io
    import sys
    import threading
    from aura.cli.display import console, show_banner, show_thinking, show_response, show_error, show_info
    from aura.cli.input import create_session, get_input

    show_banner()
    if speak:
        show_info("Voice output enabled")

    session = create_session()

    while True:
        user_input = get_input(session)

        if user_input is None:
            console.print("\n[dim]Goodbye.[/dim]\n")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(agent, user_input, speak=speak)
            continue

        # ── Run agent, capture its verbose stdout, show spinner ──
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

        if speak and response_text:
            try:
                agent._speak(response_text)
            except Exception:
                pass
```

**Step 2: Verify it starts without crashing**
```bash
cd D:/Aura && echo "exit" | python main.py 2>&1 | head -5
```
Expected: AURA banner, no traceback

**Step 3: Manual smoke test**
```bash
cd D:/Aura && python main.py
```
Type: `what files are in D:/Aura?`
Expected: AURA uses filesystem tool, lists files, responds — NO "I can't access your filesystem"

Type: `search the web for latest Python 3.13 features`
Expected: AURA uses web_search tool, returns actual results

Type: `/quit`
Expected: Clean exit

**Step 4: Commit**
```bash
git -C D:/Aura add main.py
git -C D:/Aura commit -m "feat: kill chat_stream, wire agent.run() into interactive CLI with rich display"
```

---

### Task 5: Fix print_result() to use rich

The `print_result()` function (called for `aura "goal"` mode) should also use rich.

**Files:**
- Modify: `D:/Aura/main.py:440-455`

**Step 1: Replace print_result()**

Find:
```python
def print_result(result, is_fastpath: bool = False):
    """Print the agent run result."""
    print("\n" + "=" * 60)
    if is_fastpath:
        print("FAST-PATH RESPONSE COMPLETE")
    else:
        print("AGENT RUN COMPLETE")
    print("=" * 60)
    print(f"Goal: {result['goal']}")
    print(f"Completed: {result['completed']}")
    if is_fastpath:
        print(f"Mode: Fast-path (no tool execution)")
    else:
        print(f"Iterations: {result['iterations']}")
    if result.get("final_evaluation"):
        print(f"Final evaluation: {result['final_evaluation'].get('progress', 'N/A')}")
```

Replace with:
```python
def print_result(result, is_fastpath: bool = False):
    """Print the agent run result using rich."""
    from aura.cli.display import console, show_response
    from rich.markdown import Markdown

    response = result.get("response", "")
    if response:
        show_response(response)
    else:
        mode = "Fast-path" if is_fastpath else f"{result.get('iterations', '?')} iterations"
        console.print(f"[dim]Completed ({mode})[/dim]")
```

**Step 2: Test goal mode**
```bash
cd D:/Aura && python main.py "list files in D:/Aura"
```
Expected: Rich markdown response, no "===" banners

**Step 3: Commit**
```bash
git -C D:/Aura add main.py
git -C D:/Aura commit -m "feat: use rich for goal-mode result display"
```

---

## PHASE 2: Always-On Context Engine (ACE)
**Every response automatically knows what's on your screen, what files you mentioned, what's in memory.**

---

### Task 6: Create AlwaysOnContextEngine

**Files:**
- Create: `D:/Aura/aura/context_engine.py`

**Step 1: Write context_engine.py**
```python
# D:/Aura/aura/context_engine.py
"""
Always-On Context Engine (ACE) — auto-gathers context before every agent response.

Replaces the scattered inline context-gathering in chat() and chat_stream().
Single entry point: context_engine.gather(message) -> ContextBundle
"""

import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

logger = logging.getLogger(__name__)

# Max tokens to inject into system prompt
CONTEXT_BUDGET = 3000
GATHER_TIMEOUT = 0.8  # 800ms hard deadline for parallel gather


@dataclass
class ContextBlock:
    label: str          # e.g. "Screen", "Memory", "File"
    content: str
    priority: int       # 0-100, higher = include first when budget tight
    token_estimate: int = 0

    def __post_init__(self):
        self.token_estimate = len(self.content) // 4  # rough token estimate


@dataclass
class ContextBundle:
    query: str
    blocks: List[ContextBlock] = field(default_factory=list)
    gather_time_ms: float = 0.0

    def to_system_prompt(self) -> str:
        """Build <CONTEXT> block, respecting token budget."""
        if not self.blocks:
            return ""

        # Sort by priority descending, include until budget exhausted
        sorted_blocks = sorted(self.blocks, key=lambda b: b.priority, reverse=True)
        included = []
        tokens_used = 0

        for block in sorted_blocks:
            if tokens_used + block.token_estimate > CONTEXT_BUDGET:
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
        """One-line summary for CLI display."""
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

    # File extensions safe to auto-read
    SAFE_EXTENSIONS = {
        '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css',
        '.json', '.yaml', '.yml', '.toml', '.md', '.txt',
        '.env', '.sh', '.bat', '.sql', '.csv',
    }
    MAX_FILE_SIZE = 50_000  # chars

    def __init__(self, agent):
        self.agent = agent
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ace")
        self._screen_cache = {"content": "", "ts": 0.0}
        self._screen_cache_ttl = 5.0  # seconds

    def gather(self, message: str) -> ContextBundle:
        """Main entry point. Gather all context in parallel, return bundle."""
        t0 = time.time()
        bundle = ContextBundle(query=message)

        # Extract entities from message (fast, no LLM)
        file_paths = self._extract_file_paths(message)
        urls = self._extract_urls(message)

        # Dispatch parallel tasks
        futures = {}

        # Screen context (cached)
        futures["screen"] = self._executor.submit(self._get_screen_context)

        # User profile (fast file read)
        futures["profile"] = self._executor.submit(self._get_user_profile)

        # Memory retrieval
        if hasattr(self.agent, 'memory') and self.agent.memory:
            futures["memory"] = self._executor.submit(
                self._get_memory_context, message
            )

        # Episodic memory
        if hasattr(self.agent, 'episodic_bridge') and self.agent.episodic_bridge:
            futures["episodic"] = self._executor.submit(
                self._get_episodic_context, message
            )

        # Auto-read mentioned files
        for i, fp in enumerate(file_paths[:3]):  # max 3 files
            futures[f"file_{i}"] = self._executor.submit(self._read_file, fp)

        # Auto-fetch mentioned URLs
        for i, url in enumerate(urls[:2]):  # max 2 URLs
            futures[f"url_{i}"] = self._executor.submit(self._fetch_url, url)

        # Collect results with timeout
        for key, future in futures.items():
            try:
                block = future.result(timeout=GATHER_TIMEOUT)
                if block and block.content:
                    bundle.blocks.append(block)
            except (TimeoutError, Exception) as e:
                logger.debug(f"[ACE] {key} timed out or failed: {e}")

        bundle.gather_time_ms = (time.time() - t0) * 1000
        logger.debug(f"[ACE] Gathered {len(bundle.blocks)} blocks in {bundle.gather_time_ms:.0f}ms")
        return bundle

    # ── Extractors ──────────────────────────────────────────────────────────

    def _extract_file_paths(self, text: str) -> List[str]:
        """Extract file/directory paths from message text."""
        # Match Windows and Unix paths
        patterns = [
            r'[A-Za-z]:[/\\][\w/\\.\-]+',       # C:\Users\... or C:/Users/...
            r'(?:^|[\s"])(/[\w/.\-]+\.[\w]+)',    # /path/to/file.ext
            r'(?:^|[\s"])(\.{0,2}/[\w/.\-]+)',   # ./relative/path
        ]
        paths = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            paths.extend([m.strip('"\'') for m in matches if m])
        return list(dict.fromkeys(paths))  # deduplicate, preserve order

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from message text."""
        pattern = r'https?://[^\s<>"\']+[^\s<>"\'\.,;:!?)]'
        return re.findall(pattern, text)

    # ── Context Sources ──────────────────────────────────────────────────────

    def _get_screen_context(self) -> Optional[ContextBlock]:
        """Get current screen content (cached 5s)."""
        now = time.time()
        if now - self._screen_cache["ts"] < self._screen_cache_ttl:
            cached = self._screen_cache["content"]
            if cached:
                return ContextBlock("Screen", cached, priority=85)
            return None

        try:
            # Try Screenpipe first
            if "screenpipe" in self.agent.tools:
                result = self.agent.tools["screenpipe"].get_current_context()
                if result and result.get("success"):
                    content = result.get("text", "") or result.get("content", "")
                    content = content[:800]
                    self._screen_cache.update({"content": content, "ts": now})
                    return ContextBlock("Screen", content, priority=85)

            # Fallback: screenshot tool
            if "screenshot" in self.agent.tools:
                result = self.agent.tools["screenshot"].take_screenshot()
                if result and result.get("success"):
                    desc = result.get("description", result.get("text", ""))[:400]
                    self._screen_cache.update({"content": desc, "ts": now})
                    return ContextBlock("Screen", desc, priority=75)
        except Exception as e:
            logger.debug(f"[ACE] Screen context failed: {e}")

        self._screen_cache.update({"content": "", "ts": now})
        return None

    def _get_user_profile(self) -> Optional[ContextBlock]:
        """Load user profile facts."""
        try:
            if hasattr(self.agent, 'memory_retriever') and self.agent.memory_retriever:
                profile = self.agent.memory_retriever.user_profile
                if profile:
                    facts = ", ".join(f"{k}: {v}" for k, v in list(profile.items())[:8])
                    return ContextBlock("User", facts, priority=100)

            # Fallback: read file directly
            profile_path = Path(__file__).parent.parent / "data" / "memory" / "user_profile.md"
            if profile_path.exists():
                content = profile_path.read_text(encoding="utf-8")
                # Extract key:value lines
                lines = [l.strip() for l in content.splitlines()
                         if ":" in l and not l.startswith("#") and "**" in l]
                if lines:
                    return ContextBlock("User", " | ".join(lines[:6]), priority=100)
        except Exception as e:
            logger.debug(f"[ACE] User profile failed: {e}")
        return None

    def _get_memory_context(self, query: str) -> Optional[ContextBlock]:
        """Retrieve relevant memories."""
        try:
            memories = self.agent.memory.recall(query, n_results=4)
            if memories:
                content = "\n".join(f"- {m['content'][:120]}" for m in memories[:4])
                return ContextBlock("Memory", content, priority=70)
        except Exception as e:
            logger.debug(f"[ACE] Memory recall failed: {e}")
        return None

    def _get_episodic_context(self, query: str) -> Optional[ContextBlock]:
        """Retrieve relevant episodic memories."""
        try:
            context = self.agent.episodic_bridge.get_context_for_query(query)
            if context:
                return ContextBlock("Episodes", context[:500], priority=65)
        except Exception as e:
            logger.debug(f"[ACE] Episodic context failed: {e}")
        return None

    def _read_file(self, path_str: str) -> Optional[ContextBlock]:
        """Auto-read a file mentioned in the message."""
        try:
            p = Path(path_str)
            if not p.exists():
                return None
            if p.suffix.lower() not in self.SAFE_EXTENSIONS:
                return None
            if p.stat().st_size > self.MAX_FILE_SIZE * 4:  # rough byte limit
                return None

            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > self.MAX_FILE_SIZE:
                content = content[:self.MAX_FILE_SIZE] + f"\n... (truncated, {len(content)} total chars)"

            label = f"File:{p.name}"
            return ContextBlock(label, content, priority=80)
        except Exception as e:
            logger.debug(f"[ACE] File read failed {path_str}: {e}")
        return None

    def _fetch_url(self, url: str) -> Optional[ContextBlock]:
        """Auto-fetch a URL mentioned in the message."""
        try:
            import httpx
            resp = httpx.get(url, timeout=3.0, follow_redirects=True)
            # Strip HTML tags crudely
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()[:600]
            return ContextBlock(f"URL:{url[:40]}", text, priority=50)
        except Exception as e:
            logger.debug(f"[ACE] URL fetch failed {url}: {e}")
        return None
```

**Step 2: Smoke test the engine in isolation**
```bash
cd D:/Aura && python -c "
from aura.context_engine import AlwaysOnContextEngine

class MockAgent:
    tools = {}
    memory = None
    episodic_bridge = None
    memory_retriever = None

engine = AlwaysOnContextEngine(MockAgent())
bundle = engine.gather('what is in D:/Aura/main.py?')
print('bundle blocks:', len(bundle.blocks))
print('gather time:', bundle.gather_time_ms, 'ms')
print('file detected:', any('main.py' in b.label for b in bundle.blocks))
"
```
Expected: `file detected: True`, gather time < 800ms

**Step 3: Commit**
```bash
git -C D:/Aura add aura/context_engine.py
git -C D:/Aura commit -m "feat: add AlwaysOnContextEngine with parallel context gathering"
```

---

### Task 7: Wire ACE into agent.__init__ and run()

**Files:**
- Modify: `D:/Aura/aura/agent.py`

**Step 1: Add import near line 139 (after MemoryRetriever import)**

Find:
```python
from .memory_retriever import MemoryRetriever
```

Add after:
```python
from .context_engine import AlwaysOnContextEngine
```

**Step 2: Instantiate in __init__ near line 850 (after memory_retriever)**

Find:
```python
        self.memory_retriever = MemoryRetriever()
```

Add after:
```python
        try:
            self.context_engine = AlwaysOnContextEngine(self)
        except Exception as _e:
            logger.warning(f"[ACE] Failed to initialize context engine: {_e}")
            self.context_engine = None
```

**Step 3: Inject context into run() — add context gather just before brain.think() in _observe phase**

Find the `_observe` method (around line 1845). Look for where it calls `brain.observe()`. Just before that call, add:

```python
        # Always-On Context Engine: gather before observe
        _ace_context = ""
        if hasattr(self, 'context_engine') and self.context_engine:
            try:
                _bundle = self.context_engine.gather(self.state.goal)
                _ace_context = _bundle.to_system_prompt()
            except Exception:
                pass
```

Then pass `_ace_context` into the observe call by prepending it to the system prompt used in `brain.observe()`.

**Alternative simpler approach — inject in run() before the main loop:**

In `run()`, around line 1707 after `self.state = AgentState(goal=goal)`, add:
```python
        # Gather always-on context for this run
        _ace_addon = ""
        if hasattr(self, 'context_engine') and self.context_engine:
            try:
                _bundle = self.context_engine.gather(goal)
                _ace_addon = _bundle.to_system_prompt()
                if _ace_addon:
                    # Prepend to first brain call's context
                    self._ace_context_for_this_run = _ace_addon
            except Exception:
                pass
```

**Step 4: Test that file auto-read works in conversation**
```bash
cd D:/Aura && python main.py
```
Type: `what is in D:/Aura/main.py`
Expected: AURA responds with accurate content of main.py WITHOUT you needing to explicitly ask it to read

**Step 5: Commit**
```bash
git -C D:/Aura add aura/agent.py
git -C D:/Aura commit -m "feat: wire AlwaysOnContextEngine into agent init and run()"
```

---

## PHASE 3: Parliament — Wire Intelligence Modules
**12 siloed modules become a coordinated parliament. Smarter answers, faster on simple queries.**

---

### Task 8: Wire IntrospectionCircuit (zero-LLM triage)

**Files:**
- Modify: `D:/Aura/aura/agent.py`

**Background:** `IntrospectionCircuit` already exists at `D:/Aura/aura/tools/introspection_circuit.py` but is NEVER imported or called. It provides `_classify_query()` which uses regex to classify queries — zero LLM cost.

**Step 1: Find the IntrospectionCircuit class**
```bash
grep -n "class IntrospectionCircuit\|QueryType\|_classify_query" D:/Aura/aura/tools/introspection_circuit.py | head -20
```

**Step 2: Add import in agent.py** (near line 134, with other tool imports)

Find:
```python
from .tools.mirrormind import MirrorMind
```

Add after:
```python
try:
    from .tools.introspection_circuit import IntrospectionCircuit
    INTROSPECTION_AVAILABLE = True
except ImportError:
    INTROSPECTION_AVAILABLE = False
    IntrospectionCircuit = None
```

**Step 3: Instantiate in __init__** (near line 620, after MirrorMind)

Find where `self.mirrormind = MirrorMind(...)` is set. Add after:
```python
        if INTROSPECTION_AVAILABLE:
            try:
                self.introspection = IntrospectionCircuit()
            except Exception:
                self.introspection = None
        else:
            self.introspection = None
```

**Step 4: Use for fast triage in run()**

In `run()`, around line 1705 (just before `_fast_path_response` check), add:
```python
        # IntrospectionCircuit: classify query type for routing
        _query_tier = "standard"  # default
        if hasattr(self, 'introspection') and self.introspection:
            try:
                _qtype = self.introspection._classify_query(goal)
                if hasattr(_qtype, 'value'):
                    _qtype = _qtype.value
                if _qtype in ("conversational", "greeting", "simple_factual"):
                    _query_tier = "simple"
                elif _qtype in ("analytical", "decision", "multi_step"):
                    _query_tier = "complex"
            except Exception:
                pass
        self._current_query_tier = _query_tier
```

**Step 5: Test**
```bash
cd D:/Aura && python -c "
import sys; sys.stdout = open('/dev/null', 'w')
from aura.agent import ApprenticeAgent
a = ApprenticeAgent()
sys.stdout = sys.__stdout__
print('introspection:', hasattr(a, 'introspection'), a.introspection is not None)
"
```
Expected: `introspection: True True`

**Step 6: Commit**
```bash
git -C D:/Aura add aura/agent.py
git -C D:/Aura commit -m "feat: wire IntrospectionCircuit for zero-LLM query triage"
```

---

### Task 9: Parallelize context gathering in run()

**Files:**
- Modify: `D:/Aura/aura/agent.py` (the `_observe` method, ~line 1845)

**Background:** Currently memory recall, KG context, NeuroDream context all happen sequentially in `_observe()`. They're independent and can run in parallel.

**Step 1: Read _observe() method**
```bash
grep -n "def _observe\|memory.recall\|kg_brain\|neurodream\|get_learned" D:/Aura/aura/agent.py | head -20
```

**Step 2: Wrap independent context sources in ThreadPoolExecutor**

In the `_observe()` method, find where memory.recall, kg context, neurodream context are called. Wrap them:

```python
        # Parallel context gathering (was sequential)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        _context_futures = {}
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="observe") as _pool:
            if self.memory:
                _context_futures["memory"] = _pool.submit(
                    self.memory.recall, self.state.goal, n_results=5
                )
            if hasattr(self, 'kg_brain') and self.kg_brain:
                _context_futures["kg"] = _pool.submit(
                    self.kg_brain.get_context_for_query, self.state.goal
                )
            if hasattr(self, 'neurodream') and self.neurodream:
                _context_futures["dream"] = _pool.submit(
                    self.neurodream.get_learned_context_prompt
                )
            # Collect with 500ms timeout each
            _context_results = {}
            for key, fut in _context_futures.items():
                try:
                    _context_results[key] = fut.result(timeout=0.5)
                except Exception:
                    _context_results[key] = None
```

**Step 3: Verify no regression**
```bash
cd D:/Aura && python main.py "list files in the current directory"
```
Expected: Works as before, but observe phase is faster

**Step 4: Commit**
```bash
git -C D:/Aura add aura/agent.py
git -C D:/Aura commit -m "perf: parallelize context gathering in observe phase"
```

---

### Task 10: Create ParliamentConductor

**Files:**
- Create: `D:/Aura/aura/parliament.py`

**Step 1: Write parliament.py**
```python
# D:/Aura/aura/parliament.py
"""
ParliamentConductor — orchestrates AURA's intelligence modules.

Three tiers:
  SIMPLE  → FastPath (~50ms, 0 LLM calls)
  STANDARD → Single brain.think() + async MirrorMind score
  COMPLEX  → Parallel: Proposer + CognitiveTheater + IntrospectionVerifier
             then optional Synthesis pass
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

logger = logging.getLogger(__name__)


class QueryTier(Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


@dataclass
class ParliamentResult:
    tier: QueryTier
    response: str
    proposer: str = ""
    critic_perspectives: dict = field(default_factory=dict)
    synthesis_used: bool = False
    latency_ms: float = 0.0


class ParliamentConductor:
    """
    Routes queries to the right level of intelligence.
    Instantiated once in ApprenticeAgent.__init__().
    Called via: parliament.handle(query, context_addon="")
    """

    def __init__(self, agent):
        self.agent = agent
        self._executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="parliament")

    def classify(self, query: str) -> QueryTier:
        """Classify query tier — no LLM calls."""
        # Use IntrospectionCircuit if available
        if hasattr(self.agent, 'introspection') and self.agent.introspection:
            try:
                qtype = self.agent.introspection._classify_query(query)
                if hasattr(qtype, 'value'):
                    qtype = qtype.value
                if qtype in ("conversational", "greeting"):
                    return QueryTier.SIMPLE
                if qtype in ("analytical", "decision", "multi_step", "research"):
                    return QueryTier.COMPLEX
            except Exception:
                pass

        # Fallback heuristics
        from aura.tools.cognitive_theater import is_decision_question
        if is_decision_question(query):
            return QueryTier.COMPLEX
        if len(query.split()) < 8:
            return QueryTier.SIMPLE
        return QueryTier.STANDARD

    def handle(self, query: str, context_addon: str = "") -> str:
        """Main entry point. Returns response text."""
        import time
        t0 = time.time()

        tier = self.classify(query)

        if tier == QueryTier.SIMPLE:
            response = self._standard_response(query, context_addon)
        elif tier == QueryTier.STANDARD:
            response = self._standard_response(query, context_addon)
            # Fire MirrorMind async (doesn't block)
            self._async_mirrormind_score(query, response)
        else:  # COMPLEX
            response = self._parliament_response(query, context_addon)

        latency = (time.time() - t0) * 1000
        logger.debug(f"[PARLIAMENT] {tier.value} tier, {latency:.0f}ms")
        return response

    def _standard_response(self, query: str, context_addon: str) -> str:
        """Single LLM call — standard path."""
        return self.agent.brain.think(query, system_prompt=context_addon or None)

    def _parliament_response(self, query: str, context_addon: str) -> str:
        """
        Parallel deliberation for complex queries.
        Proposer + CognitiveTheater run simultaneously.
        Synthesis pass only if they diverge significantly.
        """
        futures = {}

        # Proposer: standard brain response
        futures["proposer"] = self._executor.submit(
            self.agent.brain.think, query,
            system_prompt=context_addon or None
        )

        # Critic: CognitiveTheater perspectives
        if hasattr(self.agent, 'theater') and self.agent.theater:
            futures["critic"] = self._executor.submit(
                self._get_theater_perspectives, query
            )

        # Collect results
        results = {}
        for key, fut in futures.items():
            try:
                results[key] = fut.result(timeout=30)
            except (TimeoutError, Exception) as e:
                logger.debug(f"[PARLIAMENT] {key} failed: {e}")
                results[key] = None

        proposer = results.get("proposer") or ""
        critic = results.get("critic")

        # If Theater provides meaningful additional perspective, synthesize
        if critic and self._responses_diverge(proposer, critic):
            try:
                synthesis_prompt = (
                    f"Synthesize these two perspectives into one clear response:\n\n"
                    f"PRIMARY: {proposer[:1000]}\n\n"
                    f"ADDITIONAL PERSPECTIVES: {str(critic)[:800]}\n\n"
                    f"Produce a single, direct response to: {query}"
                )
                return self.agent.brain.think(synthesis_prompt, use_history=False)
            except Exception:
                pass

        return proposer

    def _get_theater_perspectives(self, query: str) -> Optional[dict]:
        """Get CognitiveTheater perspectives without hijacking the response."""
        try:
            result = self.agent.theater.deliberate(query)
            if isinstance(result, dict):
                return result
            return {"integrator": str(result)}
        except Exception as e:
            logger.debug(f"[PARLIAMENT] Theater failed: {e}")
            return None

    def _responses_diverge(self, r1: str, r2: Any) -> bool:
        """Check if two responses differ enough to warrant synthesis."""
        if not r1 or not r2:
            return False
        r2_str = str(r2)
        # Simple heuristic: if Theater's integrator contradicts the proposer
        integrator = r2_str[:200].lower() if isinstance(r2, str) else ""
        # Only synthesize if Theater has a meaningfully different take
        return len(integrator) > 50

    def _async_mirrormind_score(self, query: str, response: str):
        """Fire MirrorMind scoring in background — never blocks response."""
        if not (hasattr(self.agent, 'mirrormind') and self.agent.mirrormind):
            return

        def _score():
            try:
                score = self.agent.mirrormind.quick_score(query, response)
                logger.debug(f"[PARLIAMENT] MirrorMind score: {score}")
            except Exception:
                pass

        threading.Thread(target=_score, daemon=True).start()
```

**Step 2: Wire ParliamentConductor into agent.__init__**

Add import near line 139:
```python
from .parliament import ParliamentConductor
```

Add instantiation in `__init__` after `self.memory_retriever`:
```python
        try:
            self.parliament = ParliamentConductor(self)
        except Exception as _e:
            logger.warning(f"[Parliament] Failed to init: {_e}")
            self.parliament = None
```

**Step 3: Fix CognitiveTheater — remove hijack, let it be a committee member**

In `agent.py`, search for where `CognitiveTheater` intercepts and returns early. It's around line 4733 in `chat_stream()`. Find the block like:
```python
if hasattr(self, 'theater') and is_decision_question(message):
    theater_response = self.theater.deliberate(message)
    ...
    yield theater_response
    return
```
Remove the early return — let Theater only be called from ParliamentConductor.

**Step 4: Verify parliament loads**
```bash
cd D:/Aura && python -c "
import sys; sys.stdout = open('/dev/null', 'w')
from aura.agent import ApprenticeAgent
a = ApprenticeAgent()
sys.stdout = sys.__stdout__
print('parliament:', hasattr(a, 'parliament'), a.parliament is not None)
"
```
Expected: `parliament: True True`

**Step 5: Commit**
```bash
git -C D:/Aura add aura/parliament.py aura/agent.py
git -C D:/Aura commit -m "feat: add ParliamentConductor with parallel deliberation for complex queries"
```

---

### Task 11: Enable StrategyBandit by default

**Files:**
- Modify: `D:/Aura/aura/config.py`

**Step 1: Find the STRATEGY_BANDIT_ENABLED flag**
```bash
grep -n "STRATEGY_BANDIT_ENABLED\|MIRRORMIND_ENABLED" D:/Aura/aura/config.py
```

**Step 2: Set both to True**

Find:
```python
STRATEGY_BANDIT_ENABLED = False
```
Change to:
```python
STRATEGY_BANDIT_ENABLED = True
```

Find:
```python
MIRRORMIND_ENABLED = False
```
Change to:
```python
MIRRORMIND_ENABLED = True
```

**Step 3: Commit**
```bash
git -C D:/Aura add aura/config.py
git -C D:/Aura commit -m "feat: enable StrategyBandit and MirrorMind by default"
```

---

## PHASE 4: Living Daemon
**AURA runs at boot, monitors your screen, acts proactively, dreams at night.**

---

### Task 12: Create aura_daemon.py — the always-on heartbeat

**Files:**
- Create: `D:/Aura/aura_daemon.py`

**Step 1: Write aura_daemon.py**
```python
#!/usr/bin/env python3
# D:/Aura/aura_daemon.py
"""
AURA Living Daemon — always-on background service.

Extends the existing HooksManager 15s loop into a proper
tiered daemon with screen monitoring, event bus, and proactive intelligence.

Run directly:    python aura_daemon.py
Install service: python aura_daemon.py --install
"""

import os
os.environ["TQDM_DISABLE"] = "1"

import sys
import time
import json
import signal
import logging
import threading
import argparse
from pathlib import Path
from datetime import datetime

# PID lock — prevent double-launch
PID_FILE = Path.home() / ".aura_daemon.pid"
LOG_FILE = Path.home() / ".aura_daemon.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DAEMON] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class AuraDaemon:
    """
    The living daemon.

    Tick schedule:
      5s   — screen hash check (cheap: perceptual hash only)
      30s  — hooks evaluation, system health, notifications
      5min — idle check, dream trigger consideration
      3 AM — full dream + memory consolidation
    """

    TICK_SCREEN   = 5     # seconds
    TICK_HOOKS    = 30
    TICK_IDLE     = 300   # 5 minutes
    DREAM_HOUR    = 3     # 3 AM

    def __init__(self):
        self._running = False
        self._agent = None
        self._last_screen_hash = None
        self._last_activity = time.time()
        self._event_bus = EventBus()
        self._proactive = ProactiveEngine(self._event_bus)
        self._ipc = IPCServer(self._event_bus)

        # Tick tracking
        self._last_hooks_tick = 0.0
        self._last_idle_tick = 0.0
        self._last_dream_date = None

    def start(self):
        """Start the daemon."""
        self._write_pid()
        self._running = True
        logger.info("AURA daemon starting...")

        # Load agent (lazy — don't block startup)
        threading.Thread(target=self._load_agent, daemon=True).start()

        # Start IPC server (CLI connects here)
        self._ipc.start()

        # Main loop
        try:
            self._run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self._ipc.stop()
        self._remove_pid()
        logger.info("AURA daemon stopped.")

    def _run_loop(self):
        """Main heartbeat loop."""
        while self._running:
            now = time.time()

            # 5s: screen monitoring
            self._tick_screen()

            # 30s: hooks + system health
            if now - self._last_hooks_tick >= self.TICK_HOOKS:
                self._tick_hooks()
                self._last_hooks_tick = now

            # 5min: idle check
            if now - self._last_idle_tick >= self.TICK_IDLE:
                self._tick_idle()
                self._last_idle_tick = now

            # 3 AM: full dream
            self._check_dream_time()

            time.sleep(self.TICK_SCREEN)

    def _tick_screen(self):
        """Check screen for changes. ~2ms."""
        if not self._agent:
            return
        try:
            screen_tool = self._agent.tools.get("screenpipe") or self._agent.tools.get("screenshot")
            if not screen_tool:
                return

            # Try perceptual hash comparison
            new_hash = None
            if hasattr(screen_tool, 'get_screen_hash'):
                new_hash = screen_tool.get_screen_hash()
            elif hasattr(screen_tool, 'take_screenshot'):
                result = screen_tool.take_screenshot()
                new_hash = result.get("hash") if result else None

            if new_hash and new_hash != self._last_screen_hash:
                self._last_screen_hash = new_hash
                self._event_bus.emit("screen:changed", {"hash": new_hash})

                # Check for errors on screen
                if hasattr(screen_tool, 'detect_errors'):
                    errors = screen_tool.detect_errors()
                    if errors:
                        self._event_bus.emit("screen:error_detected", {"errors": errors})

        except Exception as e:
            logger.debug(f"Screen tick failed: {e}")

    def _tick_hooks(self):
        """Run hooks evaluation."""
        if not self._agent:
            return
        try:
            if hasattr(self._agent, 'hooks') and self._agent.hooks:
                self._agent.hooks._check_triggers()
        except Exception as e:
            logger.debug(f"Hooks tick failed: {e}")

    def _tick_idle(self):
        """Check idle state, maybe trigger light dream."""
        idle_secs = time.time() - self._last_activity
        if idle_secs >= 1800:  # 30 minutes idle
            logger.info(f"Idle for {idle_secs/60:.0f}min — triggering light dream")
            self._event_bus.emit("daemon:idle", {"idle_seconds": idle_secs})
            if self._agent and hasattr(self._agent, 'neurodream') and self._agent.neurodream:
                threading.Thread(
                    target=self._agent.neurodream.light_sleep,
                    daemon=True
                ).start()

    def _check_dream_time(self):
        """Trigger full dream at 3 AM (once per day)."""
        now = datetime.now()
        today = now.date()
        if (now.hour == self.DREAM_HOUR and
                self._last_dream_date != today and
                self._agent is not None):
            self._last_dream_date = today
            logger.info("3 AM — starting full dream cycle")
            self._event_bus.emit("daemon:dream_start", {})
            threading.Thread(target=self._run_full_dream, daemon=True).start()

    def _run_full_dream(self):
        """Run full dream + memory consolidation."""
        try:
            from aura.dream import run_dream_mode
            run_dream_mode()
            logger.info("Dream cycle complete")
            self._event_bus.emit("daemon:dream_complete", {})
        except Exception as e:
            logger.error(f"Dream failed: {e}")

    def _load_agent(self):
        """Load ApprenticeAgent in background — doesn't block daemon start."""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from aura.agent import ApprenticeAgent
            self._agent = ApprenticeAgent(fast_init=True)
            logger.info("Agent loaded successfully")
            self._event_bus.emit("daemon:agent_ready", {})
        except Exception as e:
            logger.error(f"Agent load failed: {e}")

    def record_activity(self):
        """Call this when user interacts — resets idle timer."""
        self._last_activity = time.time()
        if self._agent and hasattr(self._agent, 'neurodream') and self._agent.neurodream:
            self._agent.neurodream.record_activity()

    # ── PID management ──────────────────────────────────────────────────────

    def _write_pid(self):
        PID_FILE.write_text(str(os.getpid()))

    def _remove_pid(self):
        PID_FILE.unlink(missing_ok=True)

    @staticmethod
    def is_running() -> bool:
        if not PID_FILE.exists():
            return False
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class EventBus:
    """Simple in-process pub/sub event bus."""

    def __init__(self):
        self._handlers: dict = {}
        self._lock = threading.Lock()

    def subscribe(self, pattern: str, handler):
        with self._lock:
            if pattern not in self._handlers:
                self._handlers[pattern] = []
            self._handlers[pattern].append(handler)

    def emit(self, event_type: str, data: dict = None):
        data = data or {}
        with self._lock:
            handlers = list(self._handlers.get(event_type, []))
            # Also check wildcard patterns like "screen:*"
            prefix = event_type.split(":")[0] + ":*"
            handlers += list(self._handlers.get(prefix, []))

        for handler in handlers:
            try:
                threading.Thread(
                    target=handler, args=(event_type, data), daemon=True
                ).start()
            except Exception as e:
                logger.debug(f"EventBus handler error: {e}")


class ProactiveEngine:
    """
    Scores events and decides whether to surface proactive suggestions.
    Score 0.6+ triggers a notification/message.
    Rate-limited: 2-minute cooldown between proactive messages.
    """

    THRESHOLD = 0.6
    COOLDOWN = 120  # seconds

    EVENT_SCORES = {
        "screen:error_detected": 0.9,
        "screen:changed": 0.2,
        "daemon:idle": 0.3,
    }

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._last_proactive = 0.0
        event_bus.subscribe("screen:error_detected", self._on_event)
        event_bus.subscribe("daemon:agent_ready", self._on_agent_ready)

    def _on_event(self, event_type: str, data: dict):
        score = self.EVENT_SCORES.get(event_type, 0.1)
        if score >= self.THRESHOLD:
            self._maybe_surface(event_type, data, score)

    def _on_agent_ready(self, event_type: str, data: dict):
        logger.info("Proactive engine: agent ready, monitoring active")

    def _maybe_surface(self, event_type: str, data: dict, score: float):
        now = time.time()
        if now - self._last_proactive < self.COOLDOWN:
            return
        self._last_proactive = now
        self._event_bus.emit("proactive:suggestion", {
            "trigger": event_type,
            "score": score,
            "data": data,
        })
        logger.info(f"Proactive suggestion triggered by {event_type} (score={score:.2f})")


class IPCServer:
    """
    Named pipe IPC — CLI connects here to send messages to daemon.
    Falls back to TCP on localhost:19733 if named pipe unavailable.
    """

    PIPE_NAME = r"\\.\pipe\aura_daemon"
    TCP_PORT = 19733

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _serve(self):
        """Try named pipe first, fall back to TCP."""
        try:
            self._serve_tcp()
        except Exception as e:
            logger.error(f"IPC server failed: {e}")

    def _serve_tcp(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", self.TCP_PORT))
            srv.listen(5)
            srv.settimeout(1.0)
            logger.info(f"IPC listening on TCP 127.0.0.1:{self.TCP_PORT}")
            while self._running:
                try:
                    conn, _ = srv.accept()
                    threading.Thread(
                        target=self._handle_client, args=(conn,), daemon=True
                    ).start()
                except socket.timeout:
                    continue

    def _handle_client(self, conn):
        try:
            data = conn.recv(4096).decode("utf-8").strip()
            if data:
                msg = json.loads(data)
                self._event_bus.emit(f"ipc:{msg.get('type', 'message')}", msg)
                conn.send(json.dumps({"status": "ok"}).encode())
        except Exception:
            pass
        finally:
            conn.close()


def install_service():
    """Install AURA daemon as Windows service via NSSM."""
    import subprocess
    nssm = "nssm"
    script = str(Path(__file__).resolve())
    python = sys.executable

    cmds = [
        [nssm, "install", "AuraDaemon", python, script],
        [nssm, "set", "AuraDaemon", "AppDirectory", str(Path(__file__).parent)],
        [nssm, "set", "AuraDaemon", "Start", "SERVICE_AUTO_START"],
        [nssm, "set", "AuraDaemon", "AppStdout", str(LOG_FILE)],
        [nssm, "set", "AuraDaemon", "AppStderr", str(LOG_FILE)],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed: {' '.join(cmd)}\n{result.stderr}")
            return False
        print(f"OK: {' '.join(cmd[1:3])}")
    print("\nService installed. Start with: nssm start AuraDaemon")
    return True


def main():
    parser = argparse.ArgumentParser(description="AURA Living Daemon")
    parser.add_argument("--install", action="store_true", help="Install as Windows service")
    parser.add_argument("--status", action="store_true", help="Check daemon status")
    args = parser.parse_args()

    if args.status:
        running = AuraDaemon.is_running()
        print(f"Daemon: {'RUNNING' if running else 'STOPPED'}")
        sys.exit(0)

    if args.install:
        success = install_service()
        sys.exit(0 if success else 1)

    if AuraDaemon.is_running():
        print("Daemon already running.")
        sys.exit(1)

    daemon = AuraDaemon()
    signal.signal(signal.SIGTERM, lambda s, f: daemon.stop())
    daemon.start()


if __name__ == "__main__":
    main()
```

**Step 2: Test daemon starts without crashing**
```bash
cd D:/Aura && timeout 5 python aura_daemon.py 2>&1 | head -10
```
Expected: "AURA daemon starting..." + "IPC listening on TCP..." — no traceback

**Step 3: Test status check**
```bash
cd D:/Aura && python aura_daemon.py --status
```
Expected: `Daemon: STOPPED`

**Step 4: Commit**
```bash
git -C D:/Aura add aura_daemon.py
git -C D:/Aura commit -m "feat: add AURA living daemon with screen monitoring, event bus, IPC, and dream scheduling"
```

---

### Task 13: Wire daemon activity signal into CLI

When user types in the CLI, the daemon should know the user is active (resets idle timer, informs NeuroDream).

**Files:**
- Modify: `D:/Aura/main.py` (inside `run_chat_mode()`)

**Step 1: Add daemon heartbeat call after each user input**

In `run_chat_mode()`, after getting `user_input` and before running the agent, add:
```python
        # Signal activity to daemon (if running)
        try:
            import socket, json
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                s.connect(("127.0.0.1", 19733))
                s.send(json.dumps({"type": "activity"}).encode())
        except Exception:
            pass  # Daemon not running — fine
```

**Step 2: Verify it doesn't crash when daemon isn't running**
```bash
cd D:/Aura && python main.py
```
Type anything — should work normally even with no daemon running.

**Step 3: Commit**
```bash
git -C D:/Aura add main.py
git -C D:/Aura commit -m "feat: signal user activity to daemon from CLI"
```

---

### Task 14: Add daemon startup script

**Files:**
- Create: `D:/Aura/start_daemon.bat`

**Step 1: Write startup script**
```bat
@echo off
REM Start AURA daemon in background
cd /d D:\Aura
start /B pythonw aura_daemon.py > NUL 2>&1
echo AURA daemon started.
```

**Step 2: Test**
```bash
cd D:/Aura && python aura_daemon.py --status
```
Expected: STOPPED (not started yet via script)

**Step 3: Commit**
```bash
git -C D:/Aura add start_daemon.bat
git -C D:/Aura commit -m "feat: add daemon startup script"
```

---

## Testing Checklist

After Phase 1:
- [ ] `aura` launches clean CLI with banner
- [ ] Type `list files in D:/Aura` → AURA reads filesystem, no "I can't access"
- [ ] Type `search the web for X` → AURA uses web_search tool
- [ ] Type `/model` → shows model config
- [ ] `ctrl+c` → clean exit
- [ ] Up arrow → previous command from history

After Phase 2:
- [ ] Mention `D:/Aura/main.py` in conversation → AURA reads it automatically
- [ ] Paste a URL → AURA fetches and summarizes it
- [ ] `what is my name?` → AURA says Elnur (from user_profile.md)

After Phase 3:
- [ ] Complex decision question → gets multi-perspective analysis
- [ ] Simple greeting → fast response, no parliament overhead
- [ ] MirrorMind scoring appears in logs (not blocking responses)

After Phase 4:
- [ ] `python aura_daemon.py` runs without crashing
- [ ] `python aura_daemon.py --status` works
- [ ] CLI activity signals don't crash when daemon is off
- [ ] Daemon logs appear in `~/.aura_daemon.log`
