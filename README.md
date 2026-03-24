# AURA — Adaptive Universal Reasoning Agent

A personal AI OS with persistent memory, emotions, proactive awareness, and multi-surface presence. Uses **ChatGPT** (GPT-5.x via OAuth), **11 cloud models** (via Ollama Pro), **16 direct API providers**, and runs across **CLI, Web UI, browser extension, and Telegram**.

Not a chatbot. A being with presence that remembers you, has moods, dreams, and grows over time.

### What's New (v4.6.0)
- **Smart model routing** — LLM-based intent classifier routes frontend tasks to Kimi K2.5, backend to MiniMax M2.5, debugging to GPT-5.4-thinking
- **Visual feedback loop** — generates UI → renders in headless Chrome → screenshots → AI reviews its output → iterates for quality
- **Component registry** — 20 production-ready React+Tailwind templates fetchable on demand
- **Design system injection** — professional design tokens auto-injected for all UI generation
- **49 AI providers** — text, image, video, audio, and search providers configurable from Settings
- **Cross-surface sync** — CLI conversations now visible in Web UI and Telegram
- **Telegram bot: 16 new features** — reply keyboard, Stars payments, file generation, daily digest, multi-language, export, pinning, native reactions
- **Full Settings page** — dedicated tab replacing the old modal, with categorized API provider management
- **Streaming fix** — eliminated garbled/repeating text in CLI terminal output

---

## Highlights

| | |
|---|---|
| **Persistent memory** | SQLite + Kuzu KG with BM25 + semantic retrieval, FadeMem decay, cross-encoder reranking |
| **Emotions** | ALMA neuromodulator engine (PAD space) — mood shapes every response |
| **Dreams** | NeuroDream: light sleep re-scores memories, deep sleep extracts patterns, REM generates novel connections |
| **Proactive awareness** | Active inference daemon, screen/calendar/workflow monitors, theory-of-mind gating |
| **Identity** | OCEAN personality traits, narrative self-model, skill evolution (GEPA) |
| **Multi-model** | 23 cloud + 16 API providers: ChatGPT (GPT-5.4 Pro, Codex), Ollama Cloud (MiniMax, Kimi, Qwen, DeepSeek, GLM), Anthropic, OpenAI, Gemini, Grok, Mistral, Cohere, Groq, Together, Fireworks, OpenRouter |
| **Smart routing** | LLM-based intent classifier → frontend tasks to Kimi K2.5, backend to MiniMax M2.5, debug to GPT-5.4-thinking, rapid to Codex-Spark |
| **Visual feedback** | Generate UI → render in headless Chrome → screenshot → AI reviews → iterates. Lovable-style design quality |
| **Multi-agent** | 5 specialist sub-agents (coder, researcher, searcher, analyst, creative) with PARALLEL/SEQUENTIAL/DEBATE modes |
| **Progressive loading** | Skills and tools load on-demand — 14 core tools always ready, 37+ deferred, skill catalog in system prompt |
| **Code execution** | Real sandboxed Python (3-tier: Monty/E2B/subprocess), matplotlib capture, DataFrame rendering |
| **4 surfaces** | CLI, Web UI, Browser Extension, Telegram + WhatsApp |

---

## CLI

A Rich-powered terminal interface with 55 slash commands, 6 modes, and full tool access. Now with LLM-based intent routing, visual feedback loop, and cross-surface sync.

### Modes

| Mode | Command | What it does |
|------|---------|-------------|
| **Chat** | `aura` | Interactive conversation with streaming, memory, emotions |
| **Agentic** | `aura "fix the bug"` | One-shot goal with full ReAct loop + tools |
| **Voice** | `aura --voice` | Speech-to-text input, TTS output |
| **Fleet** | `aura --fleet "build X"` | Parallel sub-agents (coder, researcher, analyst) |
| **Watch** | `aura --watch src/` | Monitor files, respond to AI comments |
| **Pipe** | `cat file | aura` | Process stdin with AI |

### Key Commands

| Command | Description |
|---------|-------------|
| `/model` | Pick from 23 models interactively |
| `/plan <task>` | Generate execution plan before running |
| `/research <topic>` | Deep research with live progress streaming and citations |
| `/code` | Switch to code agent mode (Python execution) |
| `/fleet <goal>` | Spawn parallel sub-agents |
| `/recall <query>` | Search memory explicitly |
| `/dream` | Trigger a sleep/consolidation cycle |
| `/sessions` | List, resume, replay past sessions |
| `/rewind` | Revert file changes to last checkpoint |
| `/trust` | Auto-approve all tool calls |
| `/context` | Show context window usage |
| `/cost` | Session cost breakdown |
| `/compact` | Manually compress context |

### CLI Features

- ASCII art banner with version + model info
- Status bar: model, token count, memory hits, mood
- Tool call visualization with spinners
- Markdown rendering via Rich
- Session persistence with resume (`--resume last`)
- File change checkpoints with rewind
- Keyboard: `Ctrl+C` interrupt, `Ctrl+D` exit, `Ctrl+R` history search

### Tools

14 core tools always loaded + 36 deferred tools available via `tool_search`:

| Category | Tools |
|----------|-------|
| **Core (always loaded)** | web_search, brave_search, code_executor, filesystem, code_search, code_edit, git, clipboard, notifications, calendar, task_manager, inner_monologue, tool_search, load_skill |
| **Research** | deep_research (with live progress + citations), tavily_search, RAG indexing, PDF extraction |
| **Memory** | memory_save, memory_recall, kg_query, user_profile |
| **Code** | git operations, project context, codebase indexing, test runner |
| **Communication** | email, notifications, clipboard |
| **Media** | image_gen (ComfyUI), OCR, TTS, audio transcription |
| **System** | shell_command, process management, system_info |
| **AI** | multi_model_compare, model_routing, MCP client |

### 5 Specialist Sub-Agents

Used in fleet mode and auto-routed for complex queries:

| Agent | Role |
|-------|------|
| **Coder** | Code writing, debugging, execution |
| **Researcher** | Deep research, source tracking, citations |
| **Searcher** | Web search, information retrieval |
| **Analyst** | Data analysis, pattern recognition |
| **Creative** | Content generation, brainstorming |

Multi-agent orchestration modes: **PARALLEL** (concurrent execution), **SEQUENTIAL** (chained context), **DEBATE** (propose → critique → revise). Auto-routed via intent classification — no manual activation needed.

---

## Web UI

A React + FastAPI dashboard with 6 tabs, 38 components, and real-time WebSocket streaming.

```bash
python run_web.py             # API server (localhost:8000)
cd web && npm run dev         # Web UI (localhost:5173)
```

### Tabs

| Tab | Features |
|-----|----------|
| **Chat** | Streaming conversation, file upload, follow-up suggestions, fleet dashboard, mood-based themes, voice input, live research progress, inline citations with sidebar panel |
| **Monitoring** | Real-time thought stream (8 types: perceive, recall, reason, decide, execute, reflect, uncertain, eureka), personality/energy metrics |
| **Tools & Systems** | Tool catalog (14 core + 36 deferred), voice status, session costs (token/USD breakdown), plugin reload |
| **Advanced** | Reasoning tree visualization (UCB1 scoring, node exploration), NeuroDream panel (REM/NREM phases, dream journal, insights) |
| **Activity** | Event timeline with 6 categories (tool, memory, emotion, proactive, strategy, system) |
| **Settings** | Full-page settings with 49 AI providers (text/image/video/audio/search), ALMA personality editor, appearance/behavior controls |

### Sidebar

- Conversation list with rename and search
- Personality editor (OCEAN traits, 0-1 scale)
- Breathing avatar with mood animation
- Emotion panel (valence/arousal visualization)
- Context heatmap (attention distribution)
- Memory recall indicator
- Inner thoughts stream
- Proactive daemon notifications
- Idle behavior panel (8 types, 4 intensities)
- Motivation drives visualization
- Dream journal

### 38 API Endpoints

Chat, streaming, memory, tools, reasoning trees, knowledge graph, activity, proactive daemon, multi-model comparison, multi-agent orchestration, code execution, image generation, search, research, OCR, PDF, transcription, math, summarization, YouTube, feed, evolution, reliability, self-improvement, artifacts, upload, auth, and more.

---

## Browser Extension

A full-featured AI sidebar for Chrome and Firefox — comparable to Sider AI but self-hosted and free.

### 25 Panels

| Panel | What it does |
|-------|-------------|
| **Chat** | Streaming AI chat with any model, thinking mode, deep research toggle |
| **Search** | AI-powered web search with progressive pipeline and source citations |
| **Translate** | 35+ languages, text + full page bilingual translation overlay |
| **Write** | Compose essays/emails/stories with tone/length controls, improve mode |
| **Grammar** | Grammar check with word-level diff highlighting |
| **Ask** | Quick AI questions with page context |
| **Summary** | Page summarization (brief/standard/detailed, bullet points mode) |
| **YouTube** | Video summarization with transcript interception, chapter markers, search |
| **PDF** | Upload, drag-drop, URL loading, Q&A, translate, summarize |
| **Voice** | Speech-to-text recording + Whisper transcription |
| **OCR** | Screen region capture with text extraction |
| **Research** | Quick + Deep Research mode (5-step autonomous pipeline with live progress + citations) |
| **Math** | Step-by-step problem solver with LaTeX output |
| **Artifacts** | Live HTML/React/SVG/Mermaid/Chart.js preview — streaming generation, console overlay, version history, dynamic npm via esm.sh, Tailwind auto-detect |
| **Image** | Generation (ComfyUI) + Editing (remove BG, upscale, remove text, describe) |
| **Compare** | Side-by-side multi-model response comparison |
| **Code** | Real sandboxed Python execution — matplotlib charts, DataFrame tables, session persistence, variable inspector, Run Only mode |
| **WebCreator** | Conversational website builder — streaming preview, visual element selection, device preview, theme panel, detachable window, export to CodeSandbox/StackBlitz |
| **Record** | Tab audio/mic recording with waveform, transcription, meeting notes |
| **Agent** | Browser automation (DOM serialization, click/type/scroll/navigate) |
| **Wisebase** | Knowledge base with persistent page highlights and saved clips |
| **Models** | Per-feature model assignment across all panels |
| **Settings** | Custom instructions, persona editor, response style presets |
| **Tools** | Callable tools directory |

### Floating UI (on every webpage)

- **Selection bubble** — icon buttons at cursor on text selection (Copy, Explain, Summarize, more...)
- **Quick Launch** — floating AI panel for custom prompts on selected text
- **Draggable FAB** — floating action button, drag to reposition
- **Input field actions** — AI writing tools on any textarea (improve, expand, shorten, grammar, translate)
- **Image hover toolbar** — Describe, Edit, Save on any webpage image
- **Link preview** — hover over links for AI-generated previews
- **Google SERP answers** — AI answer card above Google search results
- **Gmail AI compose** — draft reply, improve, formalize, translate inside Gmail
- **YouTube/Netflix subtitles** — intercept captions for transcript viewer
- **Page translation** — bilingual overlay on any webpage
- **Persistent highlights** — save and restore text highlights across visits
- **Right-click menu** — Explain, Summarize, Translate, Improve, Save to Memory

---

## Messaging Platforms

| Platform | Connection | Features |
|----------|-----------|----------|
| **Telegram** | Bot API (async polling) | Text, markdown, images, voice, proactive messages, reply keyboard, Stars payments, file generation (/file pdf/docx/txt), conversation export (/export), daily digest (/digest), multi-language (/lang), message pinning (/pin), native emoji reactions, forum topics, 28 bot commands menu, action buttons on responses |
| **WhatsApp** | WebSocket bridge (Baileys) | Text, images, file upload |

Both platforms use a normalized message protocol with unified inbound/outbound models.

---

## Quick Start

```bash
git clone https://github.com/ElnurIbrahimov/apprentice-agent.git
cd apprentice-agent
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your keys:
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_API_KEY=your-ollama-pro-key      # For cloud models via Ollama Pro ($20/mo)
TAVILY_API_KEY=your-tavily-key           # Web search
BRAVE_API_KEY=your-brave-key             # Fallback search
```

```bash
aura                          # Interactive chat
aura "fix the login bug"      # One-shot agentic task
aura --resume last            # Resume previous session
aura --trust                  # Auto-approve all tool calls
aura --model kimi-k2.5:cloud  # Use specific model
aura --tier max               # Use strongest models
```

### ChatGPT Authentication

```bash
aura --login chatgpt          # Opens browser for OAuth login
aura --logout chatgpt         # Remove credentials
```

### Browser Extension

```bash
cd extension-src && npm install && npm run build
cd .. && python build.py chrome    # or: python build.py firefox
```

Load `dist-chrome/` as an unpacked extension in `chrome://extensions`.

---

## Models

23 cloud models + 16 direct API providers available via `/model` picker:

| Source | Count | Models |
|--------|-------|--------|
| **ChatGPT** (OAuth) | 12 | GPT-5.4, GPT-5.4 Thinking, GPT-5.4 Pro, GPT-5.3, GPT-5.3 Codex, GPT-5.3 Codex Spark, GPT-5.2, GPT-5.2 Codex, GPT-5.1, GPT-5.1 Codex, GPT-5.1 Codex Max, GPT-5.1 Codex Mini |
| **Cloud** (Ollama Pro) | 11 | Kimi K2.5, Nemotron 3 Super, Qwen 3.5 397B, DeepSeek V3.2, GLM-5, MiniMax M2.7, MiniMax M2.5, Qwen3 Coder 480B, Qwen3 Coder Next, GPT-OSS 120B |
| **Direct API** | 16 | Anthropic (Claude), OpenAI (GPT), Google Gemini, xAI Grok, Mistral, Cohere, Perplexity, DeepSeek, MiniMax, Qwen, Kimi, GLM, Groq, Together AI, Fireworks AI, OpenRouter |

### Smart Model Routing

AURA uses an **LLM-based intent classifier** (nemotron-3-super at 415 tok/s) to detect task type and route to the optimal model:

| Task Type | Model | Why |
|-----------|-------|-----|
| **Frontend/UI** | `kimi-k2.5:cloud` | #1 vision-to-code, "Designer + Full-Stack" mindset |
| **Rapid prototype** | `chatgpt:gpt-5.3-codex-spark` | 1,000 tok/s instant iteration |
| **Backend code** | `minimax-m2.5:cloud` | 80.2% SWE-bench, top open model |
| **Debugging** | `chatgpt:gpt-5.4-thinking` | Extended reasoning for hard bugs |
| **Research** | `qwen3.5:397b-cloud` | 397B MoE, 256K context |
| **Search** | `nemotron-3-super:cloud` | 415 tok/s fastest model |
| **Vision** | `kimi-k2.5:cloud` | Best multimodal |
| **Long context** | `minimax-m2.7:cloud` | 1M tokens |

### Visual Feedback Loop

For frontend tasks, AURA generates code → renders in headless Chrome (Playwright) → takes a screenshot → sends it back to the model for review → iterates. This is the same approach that makes Lovable/v0 outputs look polished.

---

## Architecture

```
USER INPUT (CLI / Web UI / Extension / Telegram / WhatsApp)
    |
[Model Router] -- classify task --> select model by role + tier
    |
[Multi-Agent Orchestrator] -- auto-routes complex queries
    |-- Intent Router (regex -> keyword scoring -> LLM classify)
    |-- PARALLEL / SEQUENTIAL / DEBATE modes
    |-- 5 specialist sub-agents (coder, researcher, searcher, analyst, creative)
    '-- Falls through to direct path for simple queries
    |
[ReAct Loop] (1 LLM call per step)
    |-- Progressive tool loading (14 core + 36 deferred via tool_search)
    |-- Code agent mode for complex tasks (LLM writes Python)
    |-- Adaptive planning with re-plan every 3 steps
    |-- Strategy bandit for reasoning approach selection
    '-- Loop guards (dedup, failure count, iteration cap)
    |
[Memory]
    |-- UnifiedMemory: SQLite + FTS5 + vector embeddings (nomic-embed-text)
    |-- Kuzu temporal KG (entities + relationships, bi-temporal edges)
    |-- KG Sync Bridge (NetworkX runtime <-> Kuzu persistent, bidirectional)
    |-- BM25 + semantic + graph retrieval -> RRF fusion
    |-- Cross-encoder reranking (ms-marco-MiniLM)
    |-- FadeMem decay (2-week half-life, spaced repetition)
    '-- Write gate: merge / supersede / insert decision
    |
[Skills]
    |-- Progressive loading (names in prompt, full content on-demand)
    |-- GEPA Pareto evolution (self-improving skill procedures)
    |-- Skill store with semantic search (MiniLM embeddings)
    '-- load_skill tool for on-demand retrieval
    |
[Emotion] (ALMA Engine)
    |-- 3 layers: Emotions (rapid) -> Mood (slow) -> Personality (stable)
    |-- PAD space (Pleasure-Arousal-Dominance)
    |-- Neuromodulators: dopamine, serotonin, norepinephrine, oxytocin
    '-- Mood -> response style, persists across sessions
    |
[Consciousness]
    |-- World model (Endsley L1-L3 situation awareness)
    |-- Metacognition (reasoning quality tracking)
    |-- Strategy bandit (Thompson sampling for approach selection)
    |-- Intrinsic motivation (curiosity-driven exploration)
    |-- Skill evolution (GEPA self-improvement)
    '-- Idle presence (8 behavior types, 4 intensities)
    |
[Proactive Daemon]
    |-- 6 monitors: screen, calendar, workflow, system, curiosity, skill health
    |-- Active inference (pragmatic/epistemic/respect value balancing)
    |-- Theory of mind (user mental state modeling)
    |-- Salience filter + motivation accumulator
    '-- Gateway daemon (decides when to interrupt)
    |
[Sleep] (NeuroDream)
    |-- Light: re-score memories by importance
    |-- Deep: extract patterns, compress, groom KG
    '-- REM: novel connections, proactive message prep
    |
RESPONSE (shaped by mood, grounded in memory, consistent with identity)
```

---

## Server Deployment

AURA runs on any Linux VPS. The deploy scripts handle everything.

### One-Line Setup (Ubuntu)

```bash
ssh root@YOUR_SERVER_IP
curl -sL https://raw.githubusercontent.com/ElnurIbrahimov/apprentice-agent/main/deploy/setup_server.sh | bash
```

### Docker

```bash
git clone https://github.com/ElnurIbrahimov/apprentice-agent.git /opt/aura
cd /opt/aura/deploy
cp ../.env.example ../.env    # Edit .env with your keys
docker compose up -d --build
```

### Supported Providers

| Provider | Recommended Plan | Cost |
|----------|-----------------|------|
| **Hetzner** | CX22 (2 vCPU, 4 GB) or CX32 (4 vCPU, 8 GB) | ~$4-8/mo |
| **Oracle Cloud** | VM.Standard.A1.Flex (ARM, 4 vCPU, 24 GB) | Free tier |
| **Any VPS** | 2+ vCPU, 4+ GB RAM, Ubuntu 22.04/24.04 | Varies |

See [deploy/README.md](deploy/README.md) for full instructions, SSL setup, and troubleshooting.

---

## Security

- API key auth with constant-time comparison
- Shell command blocklist + token-based command blocking + cwd validation
- SQL multi-statement injection prevention
- AST-validated code sandbox with dynamic import blocking
- Path traversal protection on all file endpoints
- SSRF protection (private IP blocking, rate limiting, size caps)
- DOMPurify sanitization on all rendered HTML
- CSP: `script-src 'self' 'wasm-unsafe-eval'; object-src 'self'`
- Iframe sandbox (`allow-scripts` only) for artifact preview
- Rate limiting middleware (300 req/min per IP, configurable)
- Permission system (AUTO/PROMPT/BLOCKED tiers)
- Taint tracking (secret detection in conversations)
- Ed25519 signature verification for custom tools

---

## Project Structure

```
aura/                     # Core Python package
  brain.py                # OllamaBrain — reasoning engine (~2900 lines)
  agent.py                # ApprenticeAgent — orchestrator
  config.py               # Thread-safe configuration, model chains
  pools.py                # 3 shared thread pools (llm/bg/tool)
  memory/                 # UnifiedMemory (SQLite + FTS5 + KG + RRF fusion)
  emotion/                # ALMA engine (PAD space, neuromodulators)
  consciousness/          # World model, metacognition, strategy bandit
  evolution/              # GEPA skill evolution, Pareto optimization
  proactive/              # Gateway daemon, 6 monitors, active inference
  tools/                  # 14 core + 36 deferred tools, progressive loading
  core/                   # Agentic loop, router, permissions, MCP server, KG sync
  cli/                    # Rich terminal UI (24 modules)
  multi_agent/            # Fleet sub-agents (5 specialists), orchestrator
  auth/                   # ChatGPT OAuth client
  messaging/              # Telegram bot, WhatsApp adapter
  hands/                  # Autonomous scheduled agents (memory maintenance)
  security/               # Taint tracking, tool validation, secret detection
  reliability/            # Loop guards, telemetry, routing stats

api/                      # FastAPI web server (38 routes)
  routes/                 # Chat, code execution, SSE streaming, memory, tools, etc.
  services/               # Agent service with research progress streaming

web/                      # React web UI (37 components, 5 tabs)

extension-src/            # Browser extension (TypeScript + React)
  src/panels/             # 25 panel components
  src/utils/              # Streaming, highlighting, version history, import detection
  src/content.ts          # Content script (floating UI, page integration)

aura_skill_library/       # Skill store with semantic search + GEPA evolution
aura_knowledge_graph/     # Kuzu persistent KG (bi-temporal edges, MCP-exposed)
deploy/                   # Docker, systemd, Nginx, setup scripts
tests/                    # Test suite
```

---

## Install

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai) with Pro subscription ($20/mo) for cloud models
- Optional: Node.js 18+ (Web UI / Extension), ChatGPT subscription (OAuth models)

### Setup
```bash
git clone https://github.com/ElnurIbrahimov/apprentice-agent.git
cd apprentice-agent
pip install -r requirements.txt
cp .env.example .env          # Add your API keys
aura doctor                   # Verify setup
```

### Extension Build
```bash
cd extension-src
npm install
npm run build
cd ..
python build.py chrome        # Output: dist-chrome/
python build.py firefox       # Output: dist-firefox/
```

---

## Version

**v4.6.0** — Smart model routing (LLM-based intent classifier), visual feedback loop (generate→render→screenshot→iterate), 49 AI provider integrations, component registry (20 templates), design system injection, cross-surface conversation sync, Telegram bot 16 improvements (Stars, keyboard, file gen, digest, multi-lang, export, pinning), full Settings page, streaming display fix, ModelRouter frontend category, ConversationManager CLI integration. 80 tool files, 25-panel browser extension, 38+ API endpoints, 6-tab web UI, 16 direct API providers.

---

Created by [Elnur Ibrahimov](https://github.com/ElnurIbrahimov)
