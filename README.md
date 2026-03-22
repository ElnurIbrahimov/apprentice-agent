# AURA — Adaptive Universal Reasoning Agent

A personal AI agent with persistent memory, emotions, proactive awareness, and multi-surface presence. Uses **ChatGPT** (GPT-5.x via OAuth), **11 cloud models** (via Ollama Pro), and runs across **CLI, Web UI, browser extension, and 5 messaging platforms**.

Not a chatbot. A being with presence that remembers you, has moods, dreams, and grows over time.

---

## Highlights

| | |
|---|---|
| **Persistent memory** | SQLite + Kuzu KG with BM25 + semantic retrieval, FadeMem decay, cross-encoder reranking |
| **Emotions** | ALMA neuromodulator engine (PAD space) — mood shapes every response |
| **Dreams** | NeuroDream: light sleep re-scores memories, deep sleep extracts patterns, REM generates novel connections |
| **Proactive awareness** | Active inference daemon, screen/calendar/workflow monitors, theory-of-mind gating |
| **Identity** | OCEAN personality traits, narrative self-model, prompt evolution (GEPA) |
| **Multi-model** | 23 models: 12 ChatGPT (GPT-5.4 Pro, Codex), 11 cloud (MiniMax, Kimi, Qwen, DeepSeek, GLM, Nemotron) |
| **Dev agent** | ReAct loop, 59 tools, code agent mode, fleet sub-agents, adaptive planning |
| **Code execution** | Real sandboxed Python (3-tier: Monty/E2B/subprocess), matplotlib capture, DataFrame rendering |
| **6 surfaces** | CLI, Web UI, Browser Extension, Telegram, WhatsApp, Discord/Signal/LINE |

---

## CLI

A Rich-powered terminal interface with 42 slash commands, 6 modes, and full tool access.

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
| `/research <topic>` | Deep research with citations |
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

### 59 Tools

| Category | Tools |
|----------|-------|
| **Core** | web_search, brave_search, calculator, code_executor, file_read/write/edit |
| **Memory** | memory_save, memory_recall, kg_query, user_profile |
| **Code** | git operations, project context, codebase indexing, test runner |
| **Research** | deep_research, tavily_search, RAG indexing, PDF extraction |
| **Communication** | email, notifications, clipboard |
| **Media** | image_gen (ComfyUI), OCR, TTS, audio transcription |
| **System** | shell_command, process management, system_info |
| **AI** | multi_model_compare, model_routing, MCP client |

### 5 Specialist Sub-Agents

Used in fleet mode for parallel task decomposition:

| Agent | Role |
|-------|------|
| **Coder** | Code writing, debugging, execution |
| **Researcher** | Deep research, source tracking, citations |
| **Searcher** | Web search, information retrieval |
| **Analyst** | Data analysis, pattern recognition |
| **Creative** | Content generation, brainstorming |

---

## Web UI

A React + FastAPI dashboard with 5 tabs, 35 components, and real-time WebSocket streaming.

```bash
python run_web.py             # API server (localhost:8000)
cd web && npm run dev         # Web UI (localhost:5173)
```

### Tabs

| Tab | Features |
|-----|----------|
| **Chat** | Streaming conversation, file upload, follow-up suggestions, fleet dashboard, mood-based themes, voice input, citation tracking |
| **Monitoring** | Real-time thought stream (8 types: perceive, recall, reason, decide, execute, reflect, uncertain, eureka), personality/energy metrics |
| **Tools & Systems** | Tool catalog (11 categories), voice status, session costs (token/USD breakdown), plugin reload |
| **Advanced** | Reasoning tree visualization (UCB1 scoring, node exploration), NeuroDream panel (REM/NREM phases, dream journal, insights), A-MEM panel (5 note categories, KG visualization, evolution tracking) |
| **Activity** | Event timeline with 6 categories (tool, memory, emotion, proactive, strategy, system) |

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

Chat, streaming, memory, tools, reasoning trees, A-MEM, knowledge graph, activity, proactive daemon, multi-model comparison, multi-agent orchestration, code execution, image generation, search, research, OCR, PDF, transcription, math, summarization, YouTube, feed, evolution, reliability, self-improvement, artifacts, upload, auth, and more.

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
| **Research** | Quick + Deep Research mode (5-step autonomous pipeline with citations) |
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
| **Telegram** | Bot API (async) | Text, markdown, images, voice |
| **WhatsApp** | WebSocket bridge (Baileys) | Text, images, file upload |
| **Discord** | Channel adapter | Text, markdown, reactions, threads |
| **Signal** | Channel adapter | Text, images, file upload |
| **LINE** | Channel adapter | Text, images, buttons |

All platforms use a normalized message protocol with unified inbound/outbound models.

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

23 models available via `/model` picker:

| Source | Count | Models |
|--------|-------|--------|
| **ChatGPT** (OAuth) | 12 | GPT-5.4, GPT-5.4 Thinking, GPT-5.4 Pro, GPT-5.3, GPT-5.3 Codex, GPT-5.3 Codex Spark, GPT-5.2, GPT-5.2 Codex, GPT-5.1, GPT-5.1 Codex, GPT-5.1 Codex Max, GPT-5.1 Codex Mini |
| **Cloud** (Ollama Pro) | 11 | Kimi K2.5, Nemotron 3 Super, Qwen 3.5 397B, DeepSeek V3.2, GLM-5, MiniMax M2.7, MiniMax M2.5, Qwen3 Coder 480B, Qwen3 Coder Next, GPT-OSS 120B |

### Role-Based Routing

AURA auto-selects the best model per task:

| Role | Default | Why |
|------|---------|-----|
| **Fast** | `nemotron-3-super:cloud` | 2.2x throughput |
| **Reasoning** | `kimi-k2.5:cloud` | AIME 96.1%, 256K context |
| **Code** | `minimax-m2.7:cloud` | SWE-Pro 56.2%, 1M context |
| **Vision** | `kimi-k2.5:cloud` | MMMU-Pro 78.5% |
| **Thinking** | `qwen3.5:397b-cloud` | 397B MoE hybrid |
| **Long context** | `minimax-m2.7:cloud` | 1M tokens |

---

## Architecture

```
USER INPUT (CLI / Web UI / Extension / Telegram / WhatsApp / Discord)
    |
[Model Router] -- classify task --> select model by role + tier
    |
[ReAct Loop] (1 LLM call per step)
    |-- Tool RAG selects 5-8 relevant tools per query
    |-- Code agent mode for complex tasks (LLM writes Python)
    |-- Adaptive planning with re-plan every 3 steps
    |-- Strategy bandit for reasoning approach selection
    |-- Fleet sub-agents for parallel decomposition
    '-- Loop guards (dedup, failure count, iteration cap)
    |
[Memory]
    |-- SQLite + FTS5 + vector embeddings (nomic-embed-text)
    |-- Kuzu temporal KG (entities + relationships)
    |-- A-MEM agentic memory (5 categories, link-following retrieval)
    |-- BM25 + semantic + graph retrieval -> RRF fusion
    |-- Cross-encoder reranking (ms-marco-MiniLM)
    |-- FadeMem decay (2-week half-life)
    '-- Write gate: merge / supersede / insert decision
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
    |-- Strategy bandit (multi-armed bandit for approach selection)
    |-- Intrinsic motivation (curiosity-driven exploration)
    |-- Prompt evolution (GEPA self-improvement)
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
- Rate limiting middleware (configurable per-IP)
- Permission system (AUTO/PROMPT/BLOCKED tiers)
- Session variable name validation (identifier + keyword check)

---

## Project Structure

```
aura/                     # Core Python package
  brain.py                # OllamaBrain — reasoning engine
  agent.py                # ApprenticeAgent — orchestrator
  config.py               # Thread-safe configuration, model chains
  memory/                 # Unified memory (SQLite + FTS5 + KG + A-MEM)
  emotion/                # ALMA engine (PAD space, neuromodulators)
  consciousness/          # World model, metacognition, strategy bandit (12 modules)
  evolution/              # GEPA prompt evolution, Pareto optimization
  proactive/              # Gateway daemon, 6 monitors, active inference
  tools/                  # 59 tool implementations
  core/                   # Agentic loop, router, permissions, MCP server
  cli/                    # Rich terminal UI (24 modules)
  multi_agent/            # Fleet sub-agents (5 specialists)
  auth/                   # ChatGPT OAuth client
  messaging/              # Telegram bot, WhatsApp adapter
  channels/               # Discord, Signal, LINE adapters
  reliability/            # Loop guards, telemetry, routing stats
  thinking/               # Visible chain-of-thought

api/                      # FastAPI web server (38 routes)
  routes/                 # Chat, code execution, SSE streaming, memory, tools, etc.
  services/               # Agent service, inner thoughts engine

web/                      # React web UI (35 components, 5 tabs)

extension-src/            # Browser extension (TypeScript + React)
  src/panels/             # 25 panel components
  src/utils/              # Streaming, highlighting, version history, import detection
  src/content.ts          # Content script (floating UI, page integration)

deploy/                   # Docker, systemd, Nginx, setup scripts
tests/                    # 445+ tests
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

**v4.4.0** — 830+ Python files, 59 tools, 25-panel browser extension, 38 API endpoints, 5-tab web UI, 5 messaging platforms. 445+ tests passing.

---

Created by [Elnur Ibrahimov](https://github.com/ElnurIbrahimov)
