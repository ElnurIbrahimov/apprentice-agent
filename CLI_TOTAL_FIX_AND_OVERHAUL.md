# AURA CLI v4.6.0 — Total Fix & Overhaul Plan

> Generated: 2026-03-25 by 10-agent parallel audit + 4-agent competitive CLI research
> Scope: Every file in `aura/cli/`, `aura/core/`, `aura/providers/`, `aura/tools/`, `aura/multi_agent/`

---

## Executive Summary

**Overall Score: 6/10** — Good architecture, impressive features (plan mode, chain mode, security, proactive system), but critical bugs in parallel execution, streaming hangs, UI polish, and dead code make it feel broken in practice. The gap to state-of-the-art (Claude Code, Gemini CLI, Codex, OpenCode) is primarily in **UI polish** and **reliability**, not in missing features.

### What's Actually Good
- Plan mode, chain mode, oneshot, pipe mode — all production-quality
- Security module (SSRF guard, taint tracker, audit chain, tool signing) — excellent
- Proactive system (curiosity scanner, theory of mind, motivation accumulator) — impressive
- Lazy loading throughout — fast startup
- Diff viewer — proper syntax-highlighted diffs
- Memory consolidation — successfully unified to single SQLite store
- Reliability (loop guard, telemetry, routing stats) — solid

### What's Broken
- All parallel execution is fake (fleet, debate, multi-agent)
- Multiple infinite-hang scenarios in streaming
- UI is amateur compared to competitors
- Hooks system 90% dead, autocomplete lies about capabilities
- Sessions are fragmented across two stores
- No cleanup on shutdown for daemon/background tasks

---

## Part 1: Critical Bugs

### 1.1 `_think_lock` Serializes All Parallel Execution
**Root cause of fleet/debate/multi-agent being fake.**

- **File:** `brain.py:2393`
- **Problem:** `brain.think()` acquires `self._think_lock` (non-reentrant `threading.Lock()`) and holds it for the ENTIRE LLM call duration. All fleet workers, debate threads, and orchestrator agents contend on this single lock.
- **Impact:** Fleet mode shows "3 running" but only 1 runs at a time. Debate serializes all debaters. Orchestrator parallel mode is sequential.
- **Fix:** Hold lock only for history read/write (microseconds), release before the actual LLM call. Use per-call history isolation so parallel calls don't share state.

```python
# CURRENT (broken):
with self._think_lock:
    # ... prepare messages ...
    response = client.chat(...)  # BLOCKS LOCK FOR MINUTES
    # ... update history ...

# FIXED:
with self._think_lock:
    messages = self._prepare_messages(...)  # Copy what we need
history_snapshot = list(self.conversation_history)  # Snapshot

# No lock held during LLM call
response = client.chat(model=model, messages=messages)

with self._think_lock:
    self._update_history(response)  # Brief lock for write
```

### 1.2 `think_with_tools_stream` Has No Stale-Stream Timeout
- **File:** `brain.py:1015-1047`
- **Problem:** `think_stream` has `_STREAM_STALE_TIMEOUT = 90s`, but `think_with_tools_stream` has NO timeout. If server stalls mid-stream, the `for chunk in stream:` loop blocks forever.
- **Fix:** Add the same 90s stale-stream timeout logic from `think_stream` (line 2578-2596) to `think_with_tools_stream`.

### 1.3 Provider Models Crash in Agentic Loop
- **File:** `brain.py:999-1008`, `brain.py:588-594`
- **Problem:** `client.chat(tools=tools)` called on providers, but `BaseProvider.chat()` has no `tools` parameter → `TypeError`.
- **Fix:** Either add `tools` param to `BaseProvider.chat()` with native tool calling support, or strip tools before provider calls and use prompt-based tool calling (like ChatGPT path).

### 1.4 Provider Streaming Has No Read Timeout
- **Files:** `providers/openai_compat.py:148`, `anthropic_provider.py:177`, `gemini_provider.py:173`
- **Problem:** `resp.iter_lines()` with no socket-level read timeout. Connection stalls hang forever.
- **Fix:** Add `timeout` parameter to `requests.post()` as a tuple `(connect_timeout, read_timeout)` e.g. `timeout=(10, 90)`.

### 1.5 Channel Message Drain Blocks Main Loop
- **File:** `chat_loop.py:385-403`
- **Problem:** `_drain_channel_messages()` runs `agentic.run()` synchronously in a `while has_pending()` loop. Each call takes minutes. CLI is completely unresponsive.
- **Fix:** Process channel messages in a background thread, or limit to 1 message per drain cycle with a timeout.

### 1.6 `bg_manager` None Crash
- **File:** `chat_loop.py:545`
- **Problem:** When BackgroundManager import fails, `bg_manager` is None. Using `&` prefix calls `bg_manager.submit()` → `AttributeError`.
- **Fix:** Add `if not bg_manager: console.print("Background tasks unavailable"); continue`

### 1.7 Debate Inner Pool Deadlock
- **File:** `debate_mode.py:229-238`
- **Problem:** On timeout, `with ThreadPoolExecutor` `__exit__` calls `shutdown(wait=True)` which blocks waiting for `brain.think()` which is blocked on `_think_lock`.
- **Fix:** Use `shutdown(wait=False, cancel_futures=True)` or fix `_think_lock` (1.1).

### 1.8 Orchestrator Missing TimeoutError Handler
- **File:** `orchestrator.py:182`
- **Problem:** `as_completed(futures, timeout=60)` raises `concurrent.futures.TimeoutError` which isn't caught.
- **Fix:** Wrap in `try/except concurrent.futures.TimeoutError`.

### 1.9 Fallback Chain Cascades Timeouts
- **File:** `brain.py:602-623`
- **Problem:** Each model in fallback chain gets 120s timeout. 5 models × 120s = 10 min block on network outage.
- **Fix:** Decrease per-model timeout in fallback (e.g. 30s for fallbacks), or set total budget.

---

## Part 2: Security Fixes

### 2.1 SAFE_SHELL_COMMANDS Contains Dangerous Commands
- **File:** `permissions.py:33-38`
- **Remove:** `rm`, `python`, `node`, `curl`, `wget` from `SAFE_SHELL_COMMANDS`
- These auto-approve without user confirmation. `rm` can delete files, `python`/`node` can execute arbitrary code, `curl`/`wget` can exfiltrate data.

### 2.2 Explicitly Block Process Killers
- **File:** `shell_executor.py:59-64` and `sandbox/executor.py:53-71`
- **Add to BLOCKED_COMMANDS:** `taskkill`, `pkill`, `killall`, `kill`
- Currently only blocked by omission from allowlist. User's explicit rule: NEVER run `taskkill //F //IM node.exe`.

### 2.3 Gemini API Key Leaked in Logs
- **File:** `gemini_provider.py:113,166`
- **Fix:** Redact API key from logged URLs. Use `url.split('?')[0]` in error messages.

### 2.4 `execute_raw()` Bypass
- **File:** `code_executor.py:411-426`
- **Fix:** Add a `require_validation=True` param that defaults to running AST validation. Only skip when explicitly called from trusted contexts.

### 2.5 `PermissionManager.current_mode` Doesn't Exist
- **File:** `narrative.py:301`
- **Fix:** Add `current_mode` property to `PermissionManager`, or fix the reference to use `pm.trust_mode`.

### 2.6 Unify Permission Systems
- **Files:** `permissions.py` vs `permissions_ui.py`
- **Fix:** Give `PermissionManager` awareness of the 5-mode system (PLAN, PLAN_APPROVE, CAREFUL, AUTO_EDIT, FULL_AUTO). The CLI modes should set state on `PermissionManager`, not just live in a callback.

---

## Part 3: UI/UX Overhaul

> Reference: See "Part 7: Competitive CLI Research" for specific patterns from Claude Code, Gemini CLI, Codex, and OpenCode.

### 3.1 Persistent Status Bar (Priority 1)
**Current:** Status bar is `console.print()` — scrolls away, reprinted as new line each cycle.
**Target:** Persistent bottom bar like Claude Code / Aider.

- **Files to change:** `display.py:68-86`, `input.py:285-297`, `status_bar.py`
- **Approach:** Use `prompt_toolkit` `bottom_toolbar` on `PromptSession` for a persistent bar that updates without printing new lines.
- **Content to show:** `mode | model | tokens used/limit | cost | git branch | bg tasks | session`

### 3.2 Wire Up Ignored Status Bar Data (Priority 1)
**Current:** `mood_indicator`, `bg_indicator`, `research_indicator`, `session_title`, `message_count` are computed every prompt cycle then caught by `**_ignored`.
- **Files:** `status_bar.py:14`, `display.py:74-76`, `chat_loop.py:328-360`
- **Fix:** Remove `**_ignored`, accept and display all indicators.

### 3.3 Banner with Visual Identity (Priority 2)
**Current:** `get_banner()` returns empty `Text()`. Single line: `AURA v4.6.0 -- / commands . Alt+M model . ? help`.
**Target:** Gradient-colored "AURA" using theme's `banner_gradient` (already defined on all 6 themes, never used).

- **Files:** `banner.py:7-33`, `themes.py:15`
- **Approach:** Apply character-by-character gradient from theme. Show version, project name, git branch, model.

### 3.4 Git Branch + Project Name in Prompt (Priority 2)
**Current:** `\n  > ` in cyan.
**Target:** `project-name (branch) > ` like Claude Code.
- **File:** `input.py:307`

### 3.5 Cost and Time Tracking (Priority 2)
**Current:** `cost_usd` never passed to status bar, always 0.0.
- **File:** `chat_loop.py:362-367`
- **Fix:** Track cumulative cost from LLM responses, pass to status bar. Show per-response time.

### 3.6 Fix Permission Prompt (Priority 3)
**Current:** Raw `input("  [y/n/always] ")` — literal brackets, no styling.
- **File:** `chat_loop.py:170`
- **Fix:** Use `console.input()` with Rich markup, or use prompt_toolkit with styled prompt.

### 3.7 Reduce IPC Heartbeat Latency (Priority 3)
**Current:** TCP socket connection on EVERY message — 100ms latency when IPC server not running.
- **File:** `chat_loop.py:552-565`
- **Fix:** Throttle to once per 30s, or make async/non-blocking.

### 3.8 Use Rich Live for Fleet Dashboard (Priority 3)
**Current:** Prints N complete dashboard renders (wall of duplicate panels).
- **File:** `agent_commands.py:145`
- **Fix:** Call `run_fleet_live()` from `fleet.py:137` which already uses `Rich.Live`.

### 3.9 Better Tool Output Display (Priority 3)
**Current:** Only 10 lines of shell output, 5+5 for files. Too aggressive truncation.
- **File:** `tool_output.py:9-14`
- **Fix:** Show 30 lines by default, collapsible with full output on expand.

### 3.10 Wire Diff Viewer to Post-Edit Display (Priority 3)
**Current:** `show_diff()` only shows preview BEFORE edit. After edit, diff is JSON to LLM only.
- **File:** `agentic_loop.py:431-438`
- **Fix:** After `code_edit.edit()`, render the actual diff with `diff_viewer.render_diff()`.

---

## Part 4: Dead Code & Stub Cleanup

### 4.1 Remove or Wire These

| Item | File | Action |
|------|------|--------|
| `get_banner()` returns empty Text | `banner.py:7-9` | Implement proper banner |
| `banner_gradient` on all themes | `themes.py:15` | Wire to banner or remove |
| `status_bg` on all themes | `themes.py:17` | Wire to status bar or remove |
| `response_border`, `response_header` | `themes.py:21-22` | Remove (unused by design) |
| `KeybindingsRegistry` | `keybindings.py:1-67` | Wire to input system or remove |
| `DisclosureManager` | `disclosure.py` | Wire `create_section_from_tool_call()` into agentic loop |
| `_typewriter_print` | `agentic_loop.py:120-132` | Remove (fake streaming, unused) |
| `_ERROR_SENTINELS` defined twice | `chat_loop.py:706,804` | Move to module-level constant |
| `args.resume == "pick"` dead check | `main.py:284` | Simplify to `else` |
| Activity log (CLI) never called | `activity_log.py` | Wire `log()` into chat loop or remove |
| Creative agent worldsim/theater dead code | `creative.py:81-82,105-119` | Remove dead branches |
| Analyst agent fluxmind dead code | `analyst.py:101-109` | Remove dead branch |
| Voice presence stub | `services/voice_presence.py` | Keep (safe no-op) |

### 4.2 Fix Autocomplete Lies

| Command | Autocomplete Shows | Handler Actually Has | File |
|---------|-------------------|---------------------|------|
| `/mcp` | `connect`, `list`, `disconnect` | No subcommand parsing at all | `input.py:149-153`, `system_commands.py:35-48` |
| `/hand` | `list`, `spawn`, `kill` | `list`, `activate`, `deactivate`, `run`, `status` | `input.py:159-163`, `agent_commands.py:365-431` |
| `/sessions` | `list`, `delete`, `new`, `switch` | `list`, `delete`, `new` (no `switch`) | `input.py:105`, `session_commands.py:8-76` |

### 4.3 Fix Hooks System
**Current:** Only `SESSION_END` fires. 7 other events never triggered.
- **File:** `hooks.py` + agentic loop
- **Fix:** Add `hook_mgr.fire(PRE_TOOL_CALL, ...)` and `POST_TOOL_CALL` in `agentic_loop.py` tool execution. Add `POST_EDIT` after code edits. Add `SESSION_START` in chat_loop startup.

---

## Part 5: Architecture Fixes

### 5.1 Session System Unification
**Problem:** Two separate session stores — AgenticSession (JSON in `data/agentic_sessions/`) and Brain conversations (JSON in `data/conversations/`). Listed together but incompatible formats.

- **Fix:** Make AgenticSession the single source of truth. Brain conversations should feed from AgenticSession, not maintain a parallel store.

### 5.2 Session Isolation
**Problem:** All sessions share global singletons. Switching sessions doesn't switch brain's active conversation. Memory has no session-scoping.

- **Fix:** Tag memories with `session_id`. When switching sessions, update `brain._current_conversation_id`.

### 5.3 100-Message Silent Truncation
- **File:** `agentic_loop.py:1082-1083`
- **Fix:** Show a warning when truncation happens. Consider using LLM summarization (brain.py has this) instead of hard truncation.

### 5.4 GatewayDaemon Not Stopped on Shutdown
- **File:** `agent.py:2495-2604`
- **Fix:** Add `self.gateway_daemon.stop()` to `shutdown()` method.

### 5.5 BackgroundManager No Cleanup
- **File:** `background.py`
- **Fix:** Add `shutdown()` method with `cancel_all()`. Register via `atexit`. Add task timeout (300s default).

### 5.6 Core-CLI Coupling
- **File:** `agentic_loop.py:32`
- **Problem:** `from aura.cli.display import console` at top level couples core to CLI.
- **Fix:** Dependency-inject a display interface, or lazy-import console only when needed.

### 5.7 `run_chat_mode` is 800 Lines
- **File:** `chat_loop.py:95-902`
- **Fix:** Extract into: `_handle_plan_approve()`, `_handle_normal_execution()`, `_handle_signals()`, `_setup_closures()`.

### 5.8 Two Compaction Strategies
- **Files:** `token_manager.py:135-188` (mechanical), `brain.py:1705-1707` (LLM)
- **Fix:** Unify. Use LLM summarization when model is available, mechanical as fallback.

### 5.9 Token Estimation Accuracy
- **File:** `core/token_manager.py:46-57`
- **Problem:** Character-count heuristic has ~20-30% error.
- **Fix:** Use `tiktoken` for OpenAI models, or Ollama's `/api/tokenize` endpoint for local models.

### 5.10 Context Window Map Incomplete
- **File:** `core/token_manager.py:13-41`
- **Problem:** Only 21 models listed. Provider models fall back to 128K regardless of actual limit.
- **Fix:** Add context windows for all provider models. Query providers' model info APIs.

---

## Part 6: Provider & LLM Fixes

### 6.1 Add Tool Calling to Providers
- **File:** `providers/base.py:18-21`
- **Fix:** Add `tools: list[dict] | None = None` to `BaseProvider.chat()`. Implement native tool calling for OpenAI-compat providers (they all support it). Add prompt-based fallback for providers that don't.

### 6.2 Provider Streaming Should Handle Tool Call Deltas
- **Files:** `providers/openai_compat.py:161-187`, `anthropic_provider.py:161-216`, `gemini_provider.py:154-209`
- **Fix:** Parse `delta.tool_calls` from streaming responses and yield as tool call chunks.

### 6.3 Model Routing Should Include Providers
- **File:** `core/router.py:20-66`
- **Fix:** Add direct API providers to `ROUTING_TABLE`. Include cost/latency data for routing decisions.

### 6.4 Error Messages Should Propagate API Response
- **Files:** `providers/openai_compat.py:114`, `anthropic_provider.py:143`
- **Fix:** Include response body in error: `raise ConnectionError(f"... {resp.status_code}: {error_text[:200]}")`

### 6.5 Unify Retry Logic
- **Files:** `brain.py:196-202`, `brain.py:1774-1785`
- **Fix:** Consolidate `_llm_retry` (tenacity) and `_retry_on_rate_limit` (manual) into one retry strategy. Use tenacity consistently.

---

## Part 7: Competitive CLI Research

> Full research documents saved to Desktop:
> - `C:\Users\asus\Desktop\Claude_Code_CLI_UI_Patterns.md`
> - `C:\Users\asus\Desktop\Gemini_CLI_UI_Patterns.md`
> - `C:\Users\asus\Desktop\Codex_CLI_UI_Patterns_for_Aura.md`
> - `C:\Users\asus\Desktop\OpenCode_TUI_Research.md`

### 7.1 Framework Comparison

| CLI | Framework | Language | Themes | Rendering |
|-----|-----------|----------|--------|-----------|
| **Claude Code** | React Ink | TypeScript | 6 (dark, light, daltonized, ANSI) | React re-render cycle |
| **Gemini CLI** | React Ink | TypeScript | 16 (8 dark, 6 light, 2 ANSI) | React re-render cycle |
| **Codex** | Ratatui | Rust | Auto dark/light detection | Inline viewport (scrollback) |
| **OpenCode** | @opentui/solid (custom) | TypeScript/SolidJS | 33 built-in + custom JSON | 60fps native rendering |
| **Aura** | Rich + prompt_toolkit | Python | 6 (defined, unused) | console.print() per update |

### 7.2 What to Copy — Best-of-Breed Patterns

#### Status Bar (Copy from: Gemini CLI + Codex)
- **Persistent bottom bar** — all 4 competitors have this, Aura doesn't
- **Configurable columns** (Gemini): workspace, git-branch, model, context-used, token-count, code-changes, quota
- **Smart width collapse** (Codex): progressively drop items when terminal is narrow
- **Right-aligned status line** (Codex): items joined with ` | `, adaptive fallback

#### Banner (Copy from: Gemini CLI)
- **Gradient-colored icon** with title + version + user info
- **Multiple size variants** adapted to terminal width (long/short/tiny)
- **Aura already has `banner_gradient` on all themes** — just needs wiring

#### Streaming (Copy from: Codex)
- **Newline-gated streaming** — buffer tokens, commit on `\n`, animate line-by-line via FIFO queue
- **Adaptive batch drain** — catch up when queue pressure builds
- Prevents jittery character-by-character rendering

#### Tool Call Display (Copy from: OpenCode + Gemini)
- **Per-tool icons** (OpenCode): `→` Read, `←` Write/Edit, `✱` Glob/Grep, `$` Shell, `%` WebFetch, `⚙` Generic
- **Status indicators** (Gemini): `o` pending, spinner executing, `✓` success, `?` confirming, `x` error
- **Braille spinner** (OpenCode): `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` at 80ms
- **Compact output** (Codex): max 5 lines with middle truncation (head + tail + "N lines omitted")

#### Diff Display (Copy from: Codex + Claude Code)
- **Theme-aware colors** with 3-tier fallback: TrueColor → 256-color → ANSI-16
- **Dark diff colors**: added `#213A2B`, removed `#4A221D` (Codex) or `rgb(34,92,43)`/`rgb(122,41,54)` (Claude Code)
- **Light diff colors**: added `#dafbe1`, removed `#ffebe9` (Codex GitHub-style)
- **Word-level diff** (Claude Code): `diffAddedWord rgb(56,166,96)`, `diffRemovedWord rgb(179,89,107)`
- **Line numbers** right-aligned in gutter
- **Syntax highlighting within diffs** (all competitors do this)

#### Permission Prompts (Copy from: Gemini + OpenCode)
- **Bordered panel** with "Action Required" header (Gemini)
- **Numbered radio options** — press 1-5 to select directly (Gemini + Codex)
- **Diff/command preview** inside the panel
- **Three-stage flow** (OpenCode): prompt → "always" confirmation → rejection with feedback textarea
- **Expandable to fullscreen** with Ctrl+F (OpenCode)

#### Spinner / Loading (Copy from: Claude Code + Codex)
- **Custom Unicode frames** (Claude Code): `['·', '✢', '*', '✶', '✻', '✽']` bounced forward/reverse
- **Fun verbs** (Claude Code): 80+ random words like "Clauding", "Cogitating", "Brewing" → replace with Aura-themed verbs
- **Shimmer effect** (Codex): cosine sweep highlight across text, 2-second period
- **Elapsed timer** (Gemini): `(esc to cancel, 5s)` shown next to spinner
- **Stall detection** (Claude Code): color shifts toward red when response is slow

#### Theming (Copy from: OpenCode)
- **Semantic color layer** — `text.primary`, `status.error`, `diff.added_bg` etc.
- **Dark/light variants** for every theme
- **Custom theme JSON files** loadable from `.aura/themes/`
- **Color categories**: UI (14), Diff (12), Markdown (13), Syntax (9)
- **Terminal background auto-detection** via escape sequence

#### Input (Copy from: Gemini + OpenCode)
- **Placeholder text** like "Type your message or @path/to/file" (Gemini)
- **File path autocomplete** with `@` prefix (Gemini)
- **Prompt history** with arrow up/down
- **Prompt stash** — save/restore prompts like git stash (OpenCode)
- **Large paste detection** — collapse >20 lines (Gemini)
- **Two-stage Ctrl+C** — first warns, second actually aborts (OpenCode + Codex)

### 7.3 Aura-Specific Advantages to Keep

Things Aura has that competitors DON'T:
- **Chain mode** (`->` syntax for prompt chaining) — unique
- **Debate mode** (multi-model debate) — unique when parallel is fixed
- **Proactive system** (curiosity, theory of mind) — unique
- **Mood/emotion system** — unique personality feature
- **Strategy Bandit** — unique adaptive learning
- **Knowledge graph integration** — unique
- **Hands (autonomous tasks)** — unique
- **Multiple permission modes** (5 levels) — more granular than competitors

### 7.4 Recommended Theme Colors for Aura

Based on research, here are recommended default colors:

```python
AURA_DARK_THEME = {
    # Core identity
    "aura":              "#D777AF",  # Aura's signature pink-purple
    "aura_shimmer":      "#E797CF",  # Lighter for shimmer

    # Text
    "text_primary":      "#FFFFFF",
    "text_secondary":    "#999999",
    "text_accent":       "#D7AFFF",  # Purple accent
    "text_muted":        "#505050",

    # Status
    "success":           "#4EBA65",  # Green
    "error":             "#FF6B80",  # Red-pink
    "warning":           "#FFC107",  # Yellow

    # Diff (Codex-style muted)
    "diff_added_bg":     "#213A2B",
    "diff_removed_bg":   "#4A221D",
    "diff_added_word":   "#38A660",
    "diff_removed_word": "#B3596B",

    # UI
    "border":            "#505050",
    "border_active":     "#87AFFF",
    "input_bg":          "#3F3F3F",
    "panel_bg":          "#1A1A1A",
    "focus_bg":          "#005F00",

    # Permission
    "permission":        "#B1B9F9",
    "permission_border": "#FFC107",  # Yellow border like OpenCode

    # Tool status
    "tool_pending":      "#4EBA65",
    "tool_running":      "#87AFFF",
    "tool_error":        "#FF6B80",

    # Gradient (for banner)
    "gradient":          ["#D777AF", "#B1B9F9", "#87D7D7"],
}
```

### 7.5 Tool Icons (from OpenCode, adapted for Aura)

```python
TOOL_ICONS = {
    "read":         "→",
    "write":        "←",
    "edit":         "←",
    "glob":         "✱",
    "grep":         "✱",
    "list_dir":     "→",
    "shell":        "$",
    "web_search":   "◈",
    "web_fetch":    "%",
    "code_search":  "◇",
    "git":          "⎇",
    "image_gen":    "◎",
    "inner_monologue": "💭",
    "generic":      "⚙",
}
```

### 7.6 Spinner Verbs (Aura-themed, inspired by Claude Code)

```python
AURA_SPINNER_VERBS = [
    "Thinking", "Perceiving", "Resonating", "Channeling", "Sensing",
    "Contemplating", "Manifesting", "Harmonizing", "Illuminating", "Attuning",
    "Synthesizing", "Crystallizing", "Weaving", "Flowing", "Pulsing",
    "Dreaming", "Evolving", "Awakening", "Transmuting", "Radiating",
    "Calibrating", "Decoding", "Unraveling", "Orchestrating", "Conjuring",
]
```

---

## Part 8: Phased Implementation Plan

### Phase 1: Stop the Bleeding — Critical Bugs & Security
> Goal: Make Aura not crash, not hang, not leak secrets. No UI changes.
> Estimated scope: ~15 files, ~200 lines changed

#### 1A. Fix `_think_lock` (ROOT CAUSE of fake parallelism)
- **File:** `brain.py:2393`
- **What:** Hold lock only for history read/write, release before LLM call
- **Steps:**
  1. Snapshot `conversation_history` under lock
  2. Release lock
  3. Call `client.chat()` with snapshot (no lock held)
  4. Re-acquire lock, append response to history
  5. For `use_history=False` calls (fleet/debate/orchestrator), skip lock entirely
- **Validates:** Run `/fleet`, `/debate`, `/agent parallel` and confirm multiple LLM calls happen concurrently
- **Unblocks:** Fleet mode, debate mode, multi-agent orchestrator

#### 1B. Add stale-stream timeouts
- **Files:** `brain.py:1015-1047`, `providers/openai_compat.py:148`, `providers/anthropic_provider.py:177`, `providers/gemini_provider.py:173`
- **What:**
  1. Add `_STREAM_STALE_TIMEOUT = 90` to `think_with_tools_stream` (copy from `think_stream:2578`)
  2. Add `timeout=(10, 90)` tuple to all provider `requests.post()` calls (10s connect, 90s read)
- **Validates:** Kill Ollama mid-stream → CLI recovers within 90s instead of hanging forever

#### 1C. Fix crash bugs
- **Files:** `chat_loop.py:545`, `orchestrator.py:182`, `debate_mode.py:229-238`
- **What:**
  1. Add `if not bg_manager:` guard before `bg_manager.submit()` (chat_loop.py:545)
  2. Wrap `as_completed()` in `try/except concurrent.futures.TimeoutError` (orchestrator.py:182)
  3. Fix debate inner pool: use `shutdown(wait=False, cancel_futures=True)` (debate_mode.py:235)
- **Validates:** Use `&` prefix when BackgroundManager unavailable → graceful error. Orchestrator timeout → handled.

#### 1D. Security hardening
- **Files:** `permissions.py:33-38`, `shell_executor.py:59-64`, `sandbox/executor.py:53-71`, `gemini_provider.py:113,166`, `narrative.py:301`
- **What:**
  1. Remove `rm`, `python`, `node`, `curl`, `wget` from `SAFE_SHELL_COMMANDS`
  2. Add `taskkill`, `pkill`, `killall`, `kill` to `BLOCKED_COMMANDS` in both `shell_executor.py` and `sandbox/executor.py`
  3. Redact API key from Gemini provider logs: `url.split('?')[0]` in error messages
  4. Fix `narrative.py:301`: replace `pm.current_mode` with `pm.trust_mode`
- **Validates:** `rm important.py` now requires user confirmation. `taskkill` is explicitly blocked.

#### 1E. Fix channel drain blocking
- **File:** `chat_loop.py:385-403`
- **What:** Process channel messages in background thread, limit to 1 per drain cycle
- **Validates:** Telegram/WhatsApp messages don't freeze the CLI

#### 1F. Reduce fallback chain timeout cascade
- **File:** `brain.py:602-623`
- **What:** Set per-model fallback timeout to 30s (not 120s). Total budget: 90s max.
- **Validates:** Network outage → error in ~30s, not 10 minutes

---

### Phase 2: UI Overhaul — State-of-the-Art Visual Polish
> Goal: Match Claude Code / Gemini CLI / Codex visual quality. Aura should look professional.
> Estimated scope: ~12 files, ~800 lines changed
> Reference: Part 7 competitive research

#### 2A. Persistent status bar (HIGHEST UI PRIORITY)
- **Files:** `display.py:68-86`, `status_bar.py`, `input.py:285-297`, `chat_loop.py:328-360`
- **What:**
  1. Use `prompt_toolkit` `bottom_toolbar` on `PromptSession` for persistent bar
  2. Remove `**_ignored` from `status_bar.py:14` and `display.py:74-76`
  3. Wire ALL computed indicators: mood, bg tasks, research, session title, message count
  4. Add smart width collapse (Codex pattern): drop low-priority items when narrow
- **Content format:** `mode | model | tokens/limit | $cost | branch | bg:N | session`
- **Validates:** Status bar stays fixed at bottom, updates without reprinting

#### 2B. Banner with gradient identity
- **Files:** `banner.py:7-33`, `themes.py:15`
- **What:**
  1. Apply character-by-character gradient from theme's `banner_gradient` to "AURA" text
  2. Show: gradient "AURA" + `v4.6.0` (dim) + model name + project name
  3. Below: `/ commands . Alt+M model . ? help . Shift+Tab perms`
  4. Multiple size variants: full (>80 cols), compact (>60), minimal (<60)
- **Validates:** Startup shows colored gradient banner, adapts to terminal width

#### 2C. Git branch + project in prompt
- **File:** `input.py:307`
- **What:** Change `\n  > ` to `\n  project-name (branch) > ` with git branch in dim, project in white
- **Validates:** Prompt shows `myapp (main) > ` like Claude Code

#### 2D. Tool call display overhaul
- **Files:** `display.py`, `tool_output.py:9-14`, new `aura/cli/tool_icons.py`
- **What:**
  1. Add per-tool Unicode icons (§7.5): `→` Read, `←` Edit, `$` Shell, `✱` Glob/Grep, etc.
  2. Add status indicators: `○` pending, braille spinner `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` executing, `✓` success, `✗` error
  3. Middle truncation for long output (Codex): head + "N lines omitted" + tail
  4. Increase defaults: 30 lines shell, 15 lines file (up from 10/5)
- **Validates:** Tool calls show icons, spinners animate while running, output is nicely truncated

#### 2E. Diff display upgrade
- **Files:** `diff_viewer.py`, `agentic_loop.py:431-438`, `display.py:138-152`
- **What:**
  1. Add theme-aware diff colors: dark `#213A2B`/`#4A221D`, light `#dafbe1`/`#ffebe9`
  2. Add line numbers in right-aligned gutter
  3. Wire `render_diff()` to show AFTER edits in agentic loop (not just preview before)
  4. Add 3-tier color fallback: TrueColor → 256 → ANSI-16
- **Validates:** File edits show colored diffs with line numbers after every edit

#### 2F. Permission prompt overhaul
- **File:** `chat_loop.py:170`, new permission prompt module
- **What:**
  1. Rich Panel with `box.ROUNDED`, "Action Required" header in yellow
  2. Show diff/command preview inside panel
  3. Numbered options: `1. Allow once`, `2. Allow for session`, `3. Always`, `4. Reject`
  4. Press 1-4 to select directly (not just y/n/always text)
  5. Show keyboard hint at bottom in dim
- **Validates:** Permission prompts look professional with bordered panel and number shortcuts

#### 2G. Spinner / loading overhaul
- **Files:** `display.py:89-96`, new `aura/cli/spinner.py`
- **What:**
  1. Custom bounce animation: `['·', '✢', '*', '✶', '✻', '✽']` forward then reverse
  2. Random verb from `AURA_SPINNER_VERBS` list (§7.6): "Thinking...", "Resonating...", etc.
  3. Elapsed timer: `(esc to cancel, 5s)` in dim
  4. Stall detection: shift color toward red after 30s with no chunks
  5. Step counter when in agentic loop: `Step 3 · Resonating... (12s)`
- **Validates:** Loading shows animated spinner with verb, timer, and step count

#### 2H. Cost and time tracking
- **Files:** `chat_loop.py:362-367`, `status_bar.py`, `display.py`
- **What:**
  1. Track per-response time with `time.monotonic()`
  2. Track cumulative cost from LLM token counts × model pricing
  3. Pass `cost_usd` and `elapsed_ms` to status bar
  4. Show per-response: `model-name · 2.3s` below response (like Claude Code)
- **Validates:** Status bar shows `$0.12` cost, responses show `devstral · 1.8s`

#### 2I. Streaming overhaul (Codex newline-gated pattern)
- **Files:** `display.py:187-207`, new `aura/cli/stream_controller.py`
- **What:**
  1. Implement `StreamController` from Codex: buffer → newline gate → FIFO queue → animated drain
  2. Adaptive batch drain when queue pressure builds
  3. Replace `_typewriter_print` (fake streaming) with real newline-gated rendering
  4. Use `Rich.Live` for streaming area, re-render accumulated markdown per committed line
- **Validates:** Streaming looks smooth line-by-line, no jitter, catches up on fast bursts

---

### Phase 3: Theming & Customization
> Goal: Professional theme system matching OpenCode's 33-theme standard
> Estimated scope: ~5 files, ~400 lines changed

#### 3A. Semantic color layer
- **File:** `themes.py` (rewrite)
- **What:**
  1. Define `SemanticTheme` dataclass with all color categories:
     - UI: primary, secondary, accent, error, warning, success, border, border_active, input_bg, panel_bg
     - Diff: added_bg, removed_bg, added_word, removed_word, context, line_number
     - Tool: pending, running, error, success
     - Text: primary, secondary, muted, accent, link
     - Gradient: list of colors for banner
  2. Implement Aura dark theme with colors from §7.4
  3. Add light theme variant
  4. Wire ALL display code to use semantic tokens instead of hardcoded colors
- **Validates:** `/theme dark` and `/theme light` change ALL colors consistently

#### 3B. Custom theme JSON files
- **What:**
  1. Load themes from `~/.aura/themes/*.json`
  2. JSON format with `defs` references (like OpenCode)
  3. `/theme load path/to/theme.json` command
  4. Validate required color keys on load
- **Validates:** User can create custom theme JSON, load it via `/theme load`

#### 3C. Terminal background auto-detection
- **What:**
  1. Query terminal with `\x1b]11;?\x07` escape sequence
  2. Parse response to detect dark/light
  3. Auto-select appropriate theme variant
  4. Fallback to dark after 1s timeout
- **Validates:** Aura auto-detects dark/light terminal and picks matching theme

#### 3D. Port 6+ popular themes
- **What:** Port from OpenCode/Gemini research: Dracula, Nord, Catppuccin, Solarized, Gruvbox, Tokyo Night
- Each theme defines all semantic color tokens for both dark and light modes
- **Validates:** `/theme list` shows 8+ themes, all look correct

---

### Phase 4: Dead Code Cleanup & Command Fixes
> Goal: Remove lies, wire stubs, fix inconsistencies
> Estimated scope: ~20 files, ~300 lines changed/removed

#### 4A. Fix autocomplete lies
- **Files:** `input.py:149-153,159-163,105`
- **What:**
  1. `/mcp`: Remove fake subcommands OR implement `connect`/`list`/`disconnect` in handler
  2. `/hand`: Fix autocomplete to match actual: `list`, `activate`, `deactivate`, `run`, `status`
  3. `/sessions`: Remove `switch` from autocomplete
- **Validates:** Every autocomplete suggestion corresponds to a working handler

#### 4B. Wire hooks into agentic loop
- **Files:** `hooks.py`, `agentic_loop.py`
- **What:**
  1. Fire `PRE_TOOL_CALL` before tool dispatch
  2. Fire `POST_TOOL_CALL` after tool completes
  3. Fire `POST_EDIT` after file edits
  4. Fire `SESSION_START` in chat_loop startup
  5. Fire `PRE_RESPONSE` / `POST_RESPONSE` around LLM calls
- **Validates:** `/hook add post_edit "echo edited"` → fires after every file edit

#### 4C. Wire or remove dead modules
- **What:**
  1. `KeybindingsRegistry` (`keybindings.py`): Wire to input system — load from `~/.aura/keybindings.json`, apply to `PromptSession`
  2. `DisclosureManager` (`disclosure.py`): Wire `create_section_from_tool_call()` into agentic loop for collapsible tool output
  3. `ActivityLog` (`activity_log.py`): Call `log()` from chat loop after each interaction, or remove in favor of proactive persistence
- **Validates:** Custom keybindings work. Tool output is collapsible. Activity is logged.

#### 4D. Remove dead code
- **Files:** `creative.py:81-82,105-119`, `analyst.py:101-109`, `agentic_loop.py:120-132`, `main.py:284`
- **What:**
  1. Remove `worldsim_result`/`theater_result` dead branches in creative.py
  2. Remove `fluxmind_reasoning` dead branch in analyst.py
  3. Remove `_typewriter_print` (replaced by StreamController in Phase 2I)
  4. Simplify `args.resume` dead condition in main.py
  5. Deduplicate `_ERROR_SENTINELS` → module-level constant
- **Validates:** No dead code paths remain. `grep -r "worldsim_result\|theater_result\|fluxmind_reasoning" aura/` returns nothing.

#### 4E. Fix misleading command names/help
- **What:**
  1. Rename `/edit` to `/read` (it reads files, doesn't edit them) or change its behavior
  2. Fix `/merge` to actually use its branch argument (currently ignored)
  3. Add `/help <command>` support for per-command help
  4. Add confirmation to `/clear` (currently wipes history with no "are you sure?")

---

### Phase 5: Architecture Fixes
> Goal: Unify fragmented systems, proper cleanup, decouple layers
> Estimated scope: ~15 files, ~500 lines changed

#### 5A. Unify session stores
- **Files:** `session.py`, `brain.py` conversation system, `session_commands.py`
- **What:**
  1. Make `AgenticSession` the single source of truth
  2. Remove brain's parallel `data/conversations/` persistence
  3. `/sessions list` shows only AgenticSession sessions
  4. Session switching updates `brain._current_conversation_id`
  5. Tag memories with `session_id` for isolation
- **Validates:** `/sessions list` shows one consistent list. Switching sessions changes context.

#### 5B. Proper shutdown cleanup
- **Files:** `agent.py:2495-2604`, `background.py`
- **What:**
  1. Add `self.gateway_daemon.stop()` to `agent.shutdown()`
  2. Add `BackgroundManager.shutdown()` with `cancel_all()` + task timeout (300s)
  3. Add `EventBus.stop()` to shutdown chain
  4. Register all cleanup via `atexit`
  5. Add `TheoryOfMind._save_state()` to shutdown
- **Validates:** Clean exit with no orphaned threads. `ps aux | grep aura` shows nothing after exit.

#### 5C. Decouple core from CLI
- **File:** `agentic_loop.py:32`
- **What:**
  1. Remove top-level `from aura.cli.display import console`
  2. Inject display interface as parameter to `AgenticLoop.__init__`
  3. Define `DisplayProtocol` with `print()`, `show_diff()`, `show_tool_call()` methods
  4. CLI provides `RichDisplay` implementation, API/Telegram can provide `NullDisplay`
- **Validates:** `from aura.core.agentic_loop import AgenticLoop` works without importing CLI

#### 5D. Unify permission systems
- **Files:** `permissions.py`, `permissions_ui.py`
- **What:**
  1. Add `mode: PermissionMode` property to `PermissionManager`
  2. PLAN mode enforced at `PermissionManager` level (not just callback)
  3. Remove separate `PermissionMode` enum from `permissions_ui.py`
  4. Single source of truth for permission state
- **Validates:** PLAN mode blocks mutations even from programmatic callers

#### 5E. Fix 100-message truncation
- **File:** `agentic_loop.py:1082-1083`
- **What:**
  1. Show warning when truncation happens: `console.print("[dim]Context compacted: oldest N messages summarized[/]")`
  2. Use LLM summarization (brain.py) instead of hard truncation when model is available
  3. Fall back to mechanical truncation only when LLM unavailable
- **Validates:** User sees notification when messages are compacted

---

### Phase 6: Provider & LLM Fixes
> Goal: Make all providers work in the agentic loop, not just Ollama
> Estimated scope: ~10 files, ~400 lines changed

#### 6A. Add tool calling to providers
- **File:** `providers/base.py`, `providers/openai_compat.py`, `providers/anthropic_provider.py`, `providers/gemini_provider.py`
- **What:**
  1. Add `tools: list[dict] | None = None` to `BaseProvider.chat()`
  2. Implement native tool calling for OpenAI-compat providers (they all support the `tools` param)
  3. Implement tool calling for Anthropic provider (different format)
  4. Implement tool calling for Gemini provider (different format)
  5. Add prompt-based fallback for providers that don't support tools
- **Validates:** `anthropic:claude-sonnet-4-20250514` works in the agentic loop with tool calling

#### 6B. Provider streaming tool call deltas
- **Files:** `providers/openai_compat.py:161-187`, `providers/anthropic_provider.py:161-216`, `providers/gemini_provider.py:154-209`
- **What:** Parse `delta.tool_calls` from streaming responses, yield as tool call chunks
- **Validates:** Streaming with provider models shows tool calls in real-time

#### 6C. Add providers to model routing
- **File:** `core/router.py:20-66`
- **What:**
  1. Add direct API providers to `ROUTING_TABLE` with cost/latency tiers
  2. Support `provider:model` syntax in routing decisions
  3. Include provider models in `/model` picker
- **Validates:** Model router can auto-select `anthropic:haiku` for simple tasks

#### 6D. Unify retry logic
- **Files:** `brain.py:196-202`, `brain.py:1774-1785`
- **What:**
  1. Remove manual `_retry_on_rate_limit` loop
  2. Use `tenacity` consistently: 3 attempts, exponential backoff, retry on `ConnectionError | TimeoutError | 429`
  3. Apply to both streaming and non-streaming paths
- **Validates:** Rate limit → 3 retries with backoff → clean error if all fail

#### 6E. Propagate error details
- **Files:** `providers/openai_compat.py:114`, `providers/anthropic_provider.py:143`, `brain.py:230-279`
- **What:**
  1. Include API response body in error messages: `f"{status_code}: {error_text[:200]}"`
  2. Stop `call_with_timeout` from swallowing all exceptions silently
  3. Propagate error type (rate limit vs invalid model vs auth failure) to user
- **Validates:** "Model not found" shows actual API error, not generic "LLM Error"

#### 6F. Complete context window map
- **File:** `core/token_manager.py:13-41`
- **What:**
  1. Add context windows for all 16 provider model registries
  2. Query Ollama `/api/show` for local model context sizes
  3. Fall back to conservative 32K (not 128K) for unknown models
- **Validates:** `moonshot-v1-8k` correctly reports 8K context, not 128K

---

### Phase 7: Input & Interaction Polish
> Goal: Input experience matches competitors
> Estimated scope: ~5 files, ~300 lines changed

#### 7A. Two-stage Ctrl+C
- **File:** `chat_loop.py:774-789`
- **What:**
  1. First Ctrl+C: show `"press Ctrl+C again to abort"` in dim, set 5s timer
  2. Second Ctrl+C within 5s: actually abort
  3. Timer expires: clear warning, resume normal
- **Validates:** Single Ctrl+C doesn't immediately kill mid-turn work

#### 7B. Placeholder text
- **File:** `input.py:307`
- **What:** Add placeholder: `Type your message or /help for commands` in dim gray
- **Validates:** Empty prompt shows dim hint text

#### 7C. File path autocomplete
- **File:** `input.py` autocomplete system
- **What:** Support `@path/to/file` syntax for adding files to context
- **Validates:** Typing `@src/` shows file completions

#### 7D. Reduce IPC heartbeat latency
- **File:** `chat_loop.py:552-565`
- **What:** Throttle to once per 30s with timestamp check, or make async
- **Validates:** No 100ms delay on every message when IPC server is down

#### 7E. Fleet live dashboard
- **File:** `agent_commands.py:145`
- **What:** Call `run_fleet_live()` from `fleet.py:137` (already uses `Rich.Live`)
- **Validates:** `/fleet` shows live-updating dashboard, not stacked static panels

---

### Phase Summary

| Phase | Focus | Files | Lines | Depends On |
|-------|-------|-------|-------|------------|
| **1** | Critical bugs + security | ~15 | ~200 | Nothing |
| **2** | UI overhaul | ~12 | ~800 | Phase 1 (streaming needs timeout fix) |
| **3** | Theming | ~5 | ~400 | Phase 2 (themes wire into new UI) |
| **4** | Dead code + commands | ~20 | ~300 | Phase 2 (cleanup after UI rewrite) |
| **5** | Architecture | ~15 | ~500 | Phase 1 + 2 (needs stable base) |
| **6** | Providers | ~10 | ~400 | Phase 1 (needs timeout fixes) |
| **7** | Input polish | ~5 | ~300 | Phase 2 (builds on new input system) |
| **Total** | | ~82 | ~2,900 | |

### Parallel Execution Strategy

Phases that can run in parallel:
- **Phase 1** → must be first (unblocks everything)
- **Phase 2 + Phase 6** → can run in parallel (UI vs provider, no overlap)
- **Phase 3 + Phase 4** → can run in parallel (theming vs cleanup, no overlap)
- **Phase 5 + Phase 7** → can run in parallel (architecture vs input, minimal overlap)

---

## Appendix A: File-Level Issue Index

| File | Issues |
|------|--------|
| `brain.py:2393` | _think_lock serializes all parallel |
| `brain.py:1015-1047` | No stale-stream timeout |
| `brain.py:999-1008` | Provider tools kwarg crash |
| `brain.py:230-279` | call_with_timeout swallows all exceptions |
| `brain.py:602-623` | Fallback chain cascades timeouts |
| `chat_loop.py:170` | Raw input() permission prompt |
| `chat_loop.py:385-403` | Channel drain blocks main loop |
| `chat_loop.py:545` | bg_manager None crash |
| `chat_loop.py:552-565` | IPC heartbeat every message |
| `chat_loop.py:706,804` | _ERROR_SENTINELS duplicated |
| `chat_loop.py:95-902` | 800-line function |
| `display.py:68-86` | Status bar not persistent |
| `display.py:74-76` | **_ignored drops all indicators |
| `banner.py:7-9` | get_banner() returns empty |
| `themes.py:15,17,21-22` | Unused theme fields |
| `status_bar.py:14` | **_ignored drops indicators |
| `input.py:149-153` | /mcp autocomplete lies |
| `input.py:159-163` | /hand autocomplete wrong |
| `input.py:105` | /sessions switch doesn't exist |
| `input.py:307` | No git branch in prompt |
| `keybindings.py:1-67` | Dead code, never integrated |
| `hooks.py` | 7/8 events never fired |
| `disclosure.py` | Complete but never wired |
| `permissions.py:33-38` | rm/python/node auto-approved |
| `shell_executor.py:59-64` | taskkill/pkill not blocked |
| `narrative.py:301` | pm.current_mode doesn't exist |
| `permissions_ui.py` | Disconnected from PermissionManager |
| `orchestrator.py:182` | Missing TimeoutError handler |
| `debate_mode.py:229-238` | Inner pool deadlock |
| `agentic_loop.py:1082` | 100-msg silent truncation |
| `agentic_loop.py:32` | Core imports CLI display |
| `agent_commands.py:145` | Fleet uses static dashboard |
| `agent.py:2495-2604` | GatewayDaemon not stopped |
| `background.py` | No cleanup/shutdown |
| `activity_log.py` | Never called |
| `gemini_provider.py:113,166` | API key in logs |
| `code_executor.py:411-426` | execute_raw bypasses checks |
| `providers/base.py:18-21` | No tool calling interface |
| `providers/*.py` streaming | No read timeout |
| `tool_output.py:9-14` | Too aggressive truncation |
| `research_mode.py` | Stub only, no execution |
| `creative.py:81-82` | Dead code (worldsim) |
| `analyst.py:101-109` | Dead code (fluxmind) |
| `token_manager.py:46-57` | ~20-30% token estimation error |
| `token_manager.py:13-41` | Incomplete context window map |
| `conversation_manager.py:332` | Deprecated asyncio.get_event_loop() |
| `theory_of_mind.py:749` | Non-atomic file write |
