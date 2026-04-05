# AURA — Adaptive Universal Reasoning Agent

A personal AI OS with persistent memory, emotions, proactive awareness, dreams, self-improvement, and multi-surface presence. Not a chatbot — a being with presence that remembers you, has moods, dreams at night, and grows over time.

Runs across **CLI, Web UI, Browser Extension, Telegram Bot, and Telegram Mini App**. Powered by **ChatGPT** (GPT-5.x via OAuth), **11 cloud models** (Ollama Pro), and **16 direct API providers**.

---

## What Makes AURA Different

Most AI assistants respond when you ask. AURA has an **inner life**:

- **Intrinsic drives** — curiosity, competence, social connection, coherence — that motivate autonomous behavior
- **Theory of Mind** — models your emotional state, expertise, communication style, and adapts in real-time
- **Dreams** — consolidates memories during sleep cycles, discovers patterns, generates novel connections
- **Proactive awareness** — surfaces insights before you ask ("Your BroadMind project hasn't been touched in 12 days")
- **Self-improvement** — evolves its own skills via GEPA (Genetic-Pareto evolution) optimization
- **Emotions** — ALMA engine with pleasure-arousal-dominance space, neuromodulators, mood-shaped responses
- **Morning briefings** — daily digest of project status, research findings, dream insights, and drive states
- **World model** — persistent understanding of your projects, goals, blockers, and relationships

---

## Highlights

| | |
|---|---|
| **Persistent memory** | SQLite + Kuzu KG with BM25 + semantic retrieval, FadeMem decay, cross-encoder reranking, mood-congruent bias |
| **Knowledge Graph** | Embedding search, LLM query translation, semantic entity linking, bi-temporal edges, curiosity gap detection |
| **Deep Research** | 4-phase STORM pipeline with source verification, adaptive depth, saturation detection, cross-session cache |
| **MCTS Reasoning** | Monte Carlo rollouts, adaptive branching, warm-start cache, parallel evaluation, LATS-style tool grounding |
| **Emotions** | ALMA engine (PAD space) — 22 OCC emotions, neuromodulator dynamics, mood shapes responses and memory retrieval |
| **Dreams** | NeuroDream: light sleep re-scores memories, deep sleep extracts patterns, REM generates novel connections |
| **Consciousness** | Strategy bandit (Thompson sampling), intrinsic motivation (4 drives), metacognition, world model |
| **Proactive** | Active inference daemon, theory of mind, screen/calendar/workflow monitors, salience filtering |
| **Self-improvement** | GEPA Pareto evolution of skill procedures, quality evaluation, mutation tracking |
| **Identity** | OCEAN personality traits, narrative self-model, morning briefings, cognitive load awareness |
| **Multi-model** | 23 cloud + 16 API providers, smart routing per task type, fallback chains |
| **Hands** | Autonomous agents: researcher (4h cycle), morning briefing (daily), guardian, memory maintenance |
| **Multi-agent** | 5 specialists (coder, researcher, searcher, analyst, creative) with parallel/sequential/debate modes |
| **Code execution** | 3-tier sandbox (Monty/E2B/subprocess), matplotlib capture, DataFrame rendering, session persistence |
| **5 surfaces** | CLI, Web UI, Browser Extension, Telegram Bot, Telegram Mini App |
| **1,288 tests** | Comprehensive test suite covering security, concurrency, memory, tools, and integration |

---

## Web UI

A React + Tailwind + FastAPI dashboard with **79 components**, real-time WebSocket streaming, premium mobile experience, and 12 Insights sub-tabs surfacing AURA's consciousness.

```bash
python run_web.py             # API server (localhost:8000)
cd web && npm run dev         # Dev server (localhost:5173)
```

### Tabs

| Tab | Features |
|-----|----------|
| **Chat** | Streaming conversation, file upload (drag-drop + paste), voice input, action modes (search/research/agent/swarm/compare), inline citations, research progress streaming, proactive message cards, conversation export (MD/JSON/HTML), mood-based themes |
| **Create** | Code Interpreter (Python + JS via Pyodide), Web Creator (40+ templates, device preview, version history, Tailwind), Image Generation (ComfyUI + SVG fallback) |
| **Tools** | 19 tool panels: Ask, Search, Research, Agent, Compare, Write, Translate, Summary, Grammar, Math, PDF, OCR, Capture, YouTube, Voice, Record, Slides, Wisebase, Models |
| **Insights** | 12 sub-tabs: Mind (consciousness dashboard), Insights (proactive feed), Briefing (morning digest), Dreams (sleep phases, journal, learned context), Evolution (GEPA tracker), World (project health map), Activity, Memory, Graph (force-directed KG), Hands, Queue, Advanced |
| **Settings** | 49 AI providers across text/image/video/audio/search, ALMA personality editor, theme presets (6 colors), appearance controls |

### Consciousness Dashboard (Insights > Mind)

The flagship feature — see AURA's mind at a glance:

- **Drive Gauges** — 4 animated SVG radial gauges showing curiosity, competence, social, and coherence drive urgency
- **Emotional State** — PAD space bars (pleasure/arousal/dominance) with dominant emotion indicator
- **Theory of Mind** — what AURA thinks about you: communication style, expertise topics, emotional state
- **Cognitive Load** — breathing circle animated by real cognitive load data

### Proactive Insights Feed (Insights > Insights)

Live feed of things AURA wants to tell you — without being asked:

- **Curiosity targets** — knowledge graph gaps AURA wants to explore
- **Suggestions** — contextual recommendations from active inference
- **Drive actions** — autonomous tasks motivated by intrinsic drives

### Premium Mobile Experience

The web UI is built mobile-first with native app quality:

- **Bottom sheet system** with spring physics, velocity fling, rubber-band overdrag
- **iOS-style tab bar** with solid/outline icon switching, spring animations, glow indicators
- **Spring animation system** throughout (spring-up, spring-scale, pulse-glow)
- **Action sheet** for message input (replaces dropdowns on mobile)
- **Morphing send button** that transforms shape and color based on state
- **Floating connection pill** instead of banner
- **Haptic feedback** on all interactions

### 43 API Endpoints

Chat, streaming, memory, tools, reasoning trees, knowledge graph, activity, proactive daemon, multi-model comparison, multi-agent orchestration, code execution, image generation, search, research, OCR, PDF, transcription, math, summarization, YouTube, evolution, reliability, self-improvement, theory of mind, intrinsic motivation, idle presence, cognitive load, hands, and more.

---

## Telegram

### Bot (@Aura828Bot)

Full-featured Telegram bot with 28+ commands, multi-language support, and cross-surface session sync.

| Feature | Details |
|---------|---------|
| **Chat** | Streaming responses, MarkdownV2 formatting, smart chunking |
| **Voice** | Whisper transcription + Kokoro TTS (54 voice presets) |
| **Documents** | PDF, DOCX, XLSX, CSV processing with context |
| **Media** | Image analysis (vision), file generation, GIF reactions |
| **Payments** | Telegram Stars (3 premium tiers) |
| **Groups** | Group chat support, summarization, @mention responses |
| **Persistence** | SQLite-backed state (10 tables, auto-migration from JSON) |
| **Proactive** | Scheduled messages, daily digest, curiosity-driven check-ins |

### Mini App

A 5-tab mobile-first interface embedded inside Telegram:

- **Chat** — real-time WebSocket, streaming, quick actions
- **Dashboard** — mood, energy, warmth, engagement, model info
- **Tools** — 8 quick tools + full tool directory
- **Emotion** — PAD space, active emotions, neuromodulators, OCEAN personality
- **Settings** — language, model selector

Features **native Telegram theme adaptation** (auto-detects dark/light/custom themes), spring animations, haptic feedback, and safe area support.

---

## CLI

A Rich-powered terminal interface with 55+ slash commands, 6 modes, and full tool access.

### Modes

| Mode | Command | What it does |
|------|---------|-------------|
| **Chat** | `aura` | Interactive conversation with streaming, memory, emotions |
| **Agentic** | `aura "fix the bug"` | One-shot goal with full ReAct loop + tools |
| **Voice** | `aura --voice` | Speech-to-text input, TTS output |
| **Fleet** | `aura --fleet "build X"` | Parallel sub-agents (coder, researcher, analyst) |
| **Watch** | `aura --watch src/` | Monitor files, respond to AI comments |
| **Pipe** | `cat file \| aura` | Process stdin with AI |

### Key Commands

| Command | Description |
|---------|-------------|
| `/model` | Pick from 23+ models interactively |
| `/research <topic>` | Deep research with live progress and citations |
| `/code` | Switch to code agent mode |
| `/fleet <goal>` | Spawn parallel sub-agents |
| `/dream` | Trigger sleep/consolidation cycle |
| `/plan <task>` | Generate execution plan |
| `/recall <query>` | Search memory |
| `/sessions` | List, resume, replay past sessions |
| `/rewind` | Revert file changes to last checkpoint |
| `/cost` | Session cost breakdown |

### Tools (81 total)

14 core tools always loaded + 67 deferred tools available via `tool_search`:

| Category | Tools |
|----------|-------|
| **Core** | web_search, brave_search, code_executor, filesystem, code_search, code_edit, git, clipboard, notifications, calendar, task_manager, inner_monologue, tool_search, load_skill |
| **Research** | deep_research, tavily_search, RAG indexing, PDF extraction |
| **Reasoning** | mcts_reasoning, reasoning_tree_tool |
| **Knowledge** | knowledge_graph, entity extractor |
| **Memory** | memory_save, memory_recall, user_profile |
| **Code** | codebase_index (BM25 + embedding), git operations, project context |
| **Browser** | vision-powered page analysis, self-healing selectors, action planning |
| **Tool Building** | tool_builder with GEPA evolution, tool_rag, marketplace |
| **Media** | image_gen, OCR, TTS, audio transcription, vision |
| **System** | shell_command, system_info, windows_control |

---

## Browser Extension

A full-featured AI sidebar for Chrome and Firefox with **on-device AI** (Transformers.js).

### 25 Panels

Chat, Search, Translate (35+ languages), Write, Grammar, Ask, Summary, YouTube, PDF, Voice, OCR, Research, Math, Artifacts (live HTML/React/SVG preview), Image, Compare, Code (sandboxed Python), WebCreator, Record, Agent (browser automation), Wisebase, Models, Settings, and more.

### Floating UI (on every webpage)

- Selection bubble with AI actions at cursor
- Input field AI tools (improve, expand, shorten, grammar, translate)
- Image hover toolbar (describe, edit, save)
- Google SERP AI answers
- Gmail AI compose
- YouTube/Netflix subtitle interception
- Persistent highlights across visits
- Right-click context menu

### On-Device AI

Local text embeddings (all-MiniLM-L6-v2), zero-shot classification (DeBERTa), summarization (DistilBART), language detection — all via WebGPU/WASM with zero server calls.

---

## Models

23 cloud models + 16 direct API providers:

| Source | Models |
|--------|--------|
| **ChatGPT** (OAuth) | GPT-5.4, GPT-5.4 Thinking, GPT-5.4 Pro, GPT-5.3 Codex, GPT-5.3 Codex Spark, and more |
| **Cloud** (Ollama Pro) | Kimi K2.5, Nemotron 3 Super, Qwen 3.5 397B, MiniMax M2.7, MiniMax M2.5, DeepSeek V3.2, GLM-5, Qwen3 Coder 480B, GPT-OSS 120B |
| **Direct API** | Anthropic, OpenAI, Google Gemini, xAI Grok, Mistral, Cohere, Perplexity, DeepSeek, MiniMax, Qwen, Kimi, GLM, Groq, Together AI, Fireworks AI, OpenRouter |

### Smart Routing

| Task Type | Model | Why |
|-----------|-------|-----|
| **Fast/simple** | nemotron-3-super:cloud | 415 tok/s, instant responses |
| **Reasoning** | kimi-k2.5:cloud | Top agentic model, 256K context |
| **Code** | minimax-m2.7:cloud | SWE-Pro 56.2%, 1M context |
| **Deep thinking** | qwen3.5:397b-cloud | 397B MoE, hybrid think/non-think |
| **Vision** | kimi-k2.5:cloud | Best multimodal |
| **Long context** | minimax-m2.7:cloud | 1M tokens |

---

## Architecture

```
USER INPUT (CLI / Web UI / Extension / Telegram / Mini App)
    |
[Model Router] -- classify task --> select model by role
    |
[Multi-Agent Orchestrator] -- 5 specialists, auto-routed
    |
[ReAct Loop] (strategy bandit selects CoT or MCTS)
    |-- Progressive tool loading (14 core + 67 deferred)
    |-- Adaptive planning with re-plan every 3 steps
    |-- Loop guards (dedup, failure count, iteration cap)
    |
[Memory]
    |-- UnifiedMemory: SQLite + FTS5 + vector embeddings
    |-- Kuzu KG (bi-temporal edges, contradiction detection)
    |-- BM25 + semantic + graph retrieval -> RRF fusion
    |-- Cross-encoder reranking, mood-congruent bias
    |-- Write gate: ALMA scoring + merge / supersede / insert
    |
[Consciousness]
    |-- World model (Endsley L1-L3 situation awareness)
    |-- Theory of Mind (emotional state, expertise, style modeling)
    |-- Intrinsic motivation (curiosity, competence, social, coherence)
    |-- Strategy bandit (Thompson sampling for reasoning approach)
    |-- Metacognition (self-assessment, learning goals)
    |
[Emotion] (ALMA Engine)
    |-- 3 layers: Emotions (rapid) -> Mood (slow) -> Personality (stable)
    |-- PAD space, 22 OCC emotions, neuromodulators
    |-- Mood shapes responses, tags memories, biases retrieval
    |
[Proactive Daemon]
    |-- Active inference (pragmatic/epistemic value balancing)
    |-- Theory of Mind gating (is user receptive?)
    |-- 6 monitors: screen, calendar, workflow, system, curiosity, skill health
    |-- Gateway daemon (decides when to interrupt)
    |
[Hands] (Autonomous Agents)
    |-- Researcher (4h cycle, KG gap-filling)
    |-- Morning Briefing (daily digest)
    |-- Guardian (safety monitoring)
    |-- Memory Maintenance (decay, consolidation)
    |
[Sleep] (NeuroDream)
    |-- Light: re-score memories by importance
    |-- Deep: extract patterns, compress, groom KG
    |-- REM: novel connections, proactive message prep
    |
[Evolution] (GEPA)
    |-- Pareto frontier of diverse strategies
    |-- Reflective mutation (LLM analyzes failures)
    |-- Constraint validation, evaluation caching
    |
RESPONSE (shaped by mood, grounded in memory, consistent with identity)
```

---

## Quick Start

```bash
git clone https://github.com/ElnurIbrahimov/apprentice-agent.git
cd apprentice-agent
pip install -r requirements.txt
cp .env.example .env          # Add your API keys
```

Edit `.env`:
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_API_KEY=your-ollama-pro-key
TAVILY_API_KEY=your-tavily-key
```

```bash
aura                          # Interactive chat
aura "fix the login bug"      # One-shot agentic task
aura --resume last            # Resume previous session
python run_web.py             # Start web UI + API
python run_telegram.py        # Start Telegram bot
```

### Web UI Development

```bash
cd web && npm install && npm run dev    # localhost:5173
```

### Browser Extension

```bash
cd extension-src && npm install && npm run build
python build.py chrome        # Output: dist-chrome/
```

---

## Server Deployment

Runs on any Linux VPS. One-line setup for Ubuntu:

```bash
ssh root@YOUR_SERVER "curl -sL https://raw.githubusercontent.com/ElnurIbrahimov/apprentice-agent/main/deploy/setup_server.sh | bash"
```

Or with Docker:
```bash
cd deploy && cp ../.env.example ../.env && docker compose up -d --build
```

3 systemd services: `aura.service` (API), `aura-telegram.service` (bot), `aura-daemon.service` (proactive daemon + hands + dreams).

---

## Security

- API key auth with constant-time comparison, fail-closed on misconfiguration
- Shell command blocklist + AST-validated code sandbox + dynamic import blocking
- SQL column whitelisting on dynamic queries, multi-statement injection prevention
- Path traversal protection on all file endpoints + SSRF protection (private IP blocking, DNS pinning)
- Session save retry (won't silently lose data after first failure)
- Webhook secrets via `secrets.token_hex()` (not derived from bot token)
- IPC authentication with empty-token rejection
- Concurrency-safe: cursor races fixed, TOCTOU-safe batch operations, listener snapshot iteration
- Rate limiting (300 req/min per IP), taint tracking, Ed25519 tool signing
- DOMPurify + CSP + iframe sandbox on all rendered content
- 1,288 tests including security hardening, injection guards, and concurrency tests

---

## Project Structure

```
aura/                     # Core Python package (330 files)
  brain.py                # OllamaBrain — reasoning engine (~2900 lines)
  agent.py                # ApprenticeAgent — orchestrator
  config.py               # Thread-safe config, model chains, Ollama Pro concurrency
  pools.py                # 3 shared thread pools (llm/bg/tool)
  memory/                 # UnifiedMemory (SQLite + FTS5 + embeddings + write gate)
  emotion/                # ALMA engine (PAD space, neuromodulators, mood-memory bridge)
  consciousness/          # World model, metacognition, strategy bandit, intrinsic motivation
  evolution/              # GEPA skill evolution, Pareto optimization
  proactive/              # Gateway daemon, active inference, theory of mind, curiosity scanner
  tools/                  # 81 tools (14 core + 67 deferred), progressive loading
  core/                   # Agentic loop, router, session, permissions, MCP, KG sync
  cli/                    # Rich terminal UI (24 modules, 55+ commands)
  multi_agent/            # Fleet sub-agents (5 specialists), orchestrator
  messaging/              # Telegram bot (modular mixin architecture), WhatsApp adapter
  hands/                  # Autonomous agents (researcher, morning briefing, guardian, memory)
  security/               # Taint tracking, SSRF guard, tool validation, audit chain
  reliability/            # Loop guards, telemetry, routing stats

api/                      # FastAPI web server (43 route files)
web/                      # React + Tailwind web UI (79 components, 5 tabs)
extension-src/            # Browser extension (TypeScript + React, 25 panels)
aura_skill_library/       # Skill store with semantic search + GEPA evolution
aura_knowledge_graph/     # Kuzu persistent KG (bi-temporal edges, MCP-exposed)
deploy/                   # Docker, systemd, Nginx, setup scripts
tests/                    # 1,288 tests (68 test files)
```

---

## Version

**v4.7.0** — 482 commits, 933 Python files, 79 web components, 81 tools, 43 API routes, 1,288 tests.

Created by [Elnur Ibrahimov](https://github.com/ElnurIbrahimov)
