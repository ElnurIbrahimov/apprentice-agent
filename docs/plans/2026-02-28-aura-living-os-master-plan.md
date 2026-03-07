# AURA Living OS — Master Implementation Plan
*Synthesized from 5 parallel Opus research agents — 2026-02-28*

---

## The Vision (Grounded)

AURA becomes a **living intelligence layer on your machine** — not a chatbot, not a CLI wrapper, but an always-on agent that sees what you see, acts autonomously, remembers everything, and gets smarter over time.

**Realistic scope:** The infrastructure is 80% there. This plan wires it together in 4 phases over ~4-6 weeks. No magic, no over-engineering. Each phase ships something real.

---

## What the Research Found

### Critical Architectural Problem
Two completely separate code paths exist:
- `aura --chat` → `chat_stream()` → **single LLM call, zero tools, zero memory write**
- `aura "goal"` → `agent.run()` → **full observe/plan/act/evaluate loop, all tools**

The fix is a single change in `main.py`. Everything else builds on top.

### What Already Exists (Don't Rebuild)
- HooksManager: background daemon thread on 15s loop ✓
- ScreenpipeClient + ChangeDetector: screen monitoring with diff detection ✓
- NeuroDreamEngine: background consolidation with threading ✓
- GlobalWorkspace: right architecture for module coordination, just disconnected ✓
- IntrospectionCircuit: triage without LLM calls, never imported ✓
- StrategyBandit: Thompson Sampling for strategy selection, disabled by default ✓
- CognitiveTheater, MirrorMind, ReflexionEngine: exist but siloed ✓
- FastAPI + WebSocket: already running ✓
- Episodic memory, KG, A-MEM: wired (we wired it earlier today) ✓

### What Doesn't Exist Yet
- Screen context in chat flow (Screenpipe never called during responses)
- File/URL auto-read (must explicitly use tools)
- Parliament orchestrator (modules can't talk to each other)
- Always-on daemon (HooksManager is close but not a proper service)
- Rich CLI display (80 raw print() calls, no visual hierarchy)

---

## Phase 1: Kill Chat Mode — Wire the Real Agent Loop
**Timeline: 1-2 days | Impact: Massive | Risk: Low**

This is the single most impactful change. One line in main.py unlocks every tool AURA has.

### What Changes

**`main.py`** — Replace `chat_stream()` with `agent.run()` in the interactive loop:
```python
# BEFORE (lobotomized):
for chunk in agent.chat_stream(user_input, speak=speak):
    print(chunk, end="", flush=True)

# AFTER (full power):
result = agent.run(user_input)
# display result with rich
```

**Problem with naive replacement:** `run()` is verbose (prints iteration markers, phase labels) and not streaming. Need a thin wrapper.

### Implementation

**New method: `agent.process(message)`**
- Calls `agent.run(message)` internally
- Suppresses internal debug prints during execution
- Returns the final response text cleanly
- Preserves conversation history between calls

**`main.py` rewrite:**
- Remove `run_chat_mode()` entirely
- New `AuraCLI` class using `rich` + `prompt_toolkit`
- Spinner while agent runs
- Tool calls shown inline as `> tool_name  what it's doing...`
- Response rendered as markdown via `rich`
- Bottom toolbar: model name + memory count + current state
- Slash commands preserved (`/goal`, `/model`, `/recall`, etc.)
- `ctrl+c` interrupts cleanly, `ctrl+d` exits

### Files Changed
- `D:/Aura/main.py` — full rewrite of REPL loop
- `D:/Aura/aura/agent.py` — add `process()` method (~30 lines)
- `D:/Aura/aura/cli/` — new directory: `display.py`, `input.py`

### What You Get
- Type anything → AURA uses tools, reads files, searches web, executes code
- No mode switching. No `--chat` flag. One command: `aura`
- Visual feedback while it works
- Conversation history persists between messages

### What's Realistic
- Tool calls will add 2-5 seconds per response vs instant chat stream
- Not every response needs tools — fast path still handles simple queries
- Model (qwen3.5:397b cloud or local) quality determines output quality

---

## Phase 2: Always-On Context Engine (ACE)
**Timeline: 3-5 days | Impact: High | Risk: Medium**

Every response automatically knows: what's on your screen, what files you mentioned, what memories are relevant. You never paste code again.

### AlwaysOnContextEngine class
Single entry point: `context = self.context_engine.gather(message)`

**5-step pipeline:**
1. **Message Analysis** — regex extraction of file paths, URLs, entities (no LLM)
2. **Parallel Gathering** — 10 sources queried simultaneously, 800ms deadline:
   - Screen (Screenpipe, cached every 5s via background poll)
   - User profile (user_profile.md)
   - Conversation session (last 3 turns verbatim, older compressed)
   - A-MEM + Episodic memory (parallel fan-out)
   - Knowledge Graph
   - RAG (indexed docs)
   - File auto-read (any paths mentioned in message)
   - URL auto-fetch (any URLs mentioned, 3s timeout)
   - Active project detection (from window title + file paths)
3. **Budget Management** — 4000 token budget, priority-ordered:
   - User profile: always included (priority 100)
   - Screen/errors: highest priority when present (90)
   - Memories: high (80), Files: high (75), URLs: cut first (40)
4. **Format** — XML-tagged `<CONTEXT>` block injected into system prompt
5. **Relevance Learning** — tracks which sources got cited, adjusts weights over time

### Files
- `D:/Aura/aura/context_engine.py` — new file, ~400 lines
- `D:/Aura/aura/agent.py` — add `self.context_engine = AlwaysOnContextEngine(self)` in `__init__`, call `gather()` in `process()`

---

## Phase 3: Parliament — Wire the Intelligence Modules
**Timeline: 1-2 weeks | Impact: High | Risk: Medium**

12+ intelligence modules already exist but are siloed. Wire them into a coordinated parliament that makes every response smarter without adding proportional latency.

### Three-Tier Query Handling

**TIER 1 (Simple):** Greetings, commands, status → FastPath, ~50ms
**TIER 2 (Standard):** Most questions → 1 LLM call + async MirrorMind score
**TIER 3 (Complex):** Decisions, multi-step, ambiguous → Full parliament

### Parliament Flow (Tier 3)
```
Query → Triage (IntrospectionCircuit + Guardian heuristics, 5ms, no LLM)
     → Parallel context gather (ThreadPoolExecutor, 300ms deadline)
     → Parallel deliberation:
         [Proposer: brain.think()]  +  [Critic: CognitiveTheater]  +  [Verifier: IntrospectionCircuit]
         (all three simultaneously, ~3-5s wall time)
     → Synthesis (1 LLM call if needed, skippable when Theater agrees)
     → Response delivery
     → Async post-process: MirrorMind score, ReflexionEngine lesson, Bandit feedback
```

### What Gets Wired
1. **IntrospectionCircuit** — import it (currently exists but never imported), use for free triage
2. **ParliamentConductor** — new `aura/parliament.py`, ~200 lines
3. **StrategyBandit** — enable by default, expand to select parliament configs
4. **CognitiveTheater** — remove hijack behavior, make it one voice in parallel deliberation
5. **MirrorMind** — async fire-and-forget, never blocks response delivery
6. **GlobalWorkspace** — wire as EventBus for cross-module communication

### Latency Budget
- Tier 2: 2-4s (same as now, quality improves)
- Tier 3: 5-9s vs current 9-15s sequential (faster AND smarter)

---

## Phase 4: The Living Daemon
**Timeline: 2-4 weeks | Impact: Transformative | Risk: Higher**

AURA runs continuously at boot. Monitors your screen. Acts proactively. Dreams at night.

### Architecture (already designed, see 2026-02-28-daemon-architecture.md)
- **NSSM Windows service** — `aura_daemon.py` starts at boot, PID lock, auto-restart
- **Tiered heartbeat:** 5s (screen hash), 30s (hooks/alerts), 5min (idle check), 3AM (full dream)
- **Screen monitoring:** Three-tier escalation — perceptual hash → region diff → OCR. ~2ms per 5s tick.
- **Event bus:** 30+ event types, pattern matching, in-process async pub/sub
- **Proactive engine:** Score events 0-1, threshold 0.6, 2-min cooldown. "Error on screen while coding" → 0.9 → proactive suggestion
- **IPC:** Named pipe `\\.\pipe\aura_daemon` + TCP fallback. CLI becomes thin client.
- **Dream scheduling:** Idle 30min → light sleep, 2h → full cycle, 3AM → full dream + consolidation
- **Resource:** <1% CPU idle, <100MB RAM, invisible until needed

### Realistic Notes on Phase 4
- The HooksManager 15s loop is the seed — extend it, don't replace it
- Screenpipe + ChangeDetector already handle screen monitoring
- NeuroDream already has background threading
- Named pipe IPC is the new piece — maybe 200 lines
- Full daemon is a separate project, not a weekend task

---

## What's NOT Being Built (Scope Cuts)

- **Native Ollama tool calling:** No. Current text→regex approach works. Native tool calling would require replacing `brain.py` internals. Later.
- **Cross-device sync:** Not now. Local-first is the right call.
- **Voice always-on wake word:** Voice mode exists. Wake word requires always-on mic process. Later.
- **Self-building tools from scratch:** ToolBuilderTool exists. Enable it in parliament responses. Don't build new infrastructure.
- **"Better than Claude Code in a week":** Phase 1 closes most of the gap. Phase 2 surpasses it in context awareness. Parliament and daemon put it in a different category.

---

## Implementation Order

```
TODAY:     Phase 1 — Kill chat_stream(), wire agent.run(), rich CLI
           → AURA uses all tools in every conversation immediately

THIS WEEK: Phase 2 — AlwaysOnContextEngine
           → Screen awareness, file auto-read, URL auto-fetch

NEXT WEEK: Phase 3 — Parliament wiring
           → All intelligence modules coordinated, smarter responses

THIS MONTH: Phase 4 — Living daemon
            → Always-on, proactive, truly alive
```

---

## Success Metrics (Realistic)

After Phase 1:
- Type `check D:/Aura/aura/agent.py` → AURA reads it, no paste needed
- Type `search for how to fix X` → AURA actually searches

After Phase 2:
- AURA knows you're in VS Code looking at agent.py before you say anything
- Mention a file path in conversation → auto-read, no slash command

After Phase 3:
- Complex architectural questions get multi-perspective analysis automatically
- Quality improves measurably on hard questions

After Phase 4:
- AURA is running when you sit down, context already warm
- Error appears on screen → AURA notifies you with a fix before you ask
