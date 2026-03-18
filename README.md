# AURA — Adaptive Universal Reasoning Agent

A personal AI agent with persistent memory, emotions, proactive awareness, and a full-featured browser extension. Uses **ChatGPT** (GPT-5.x via OAuth), **60+ Ollama models** (local + cloud), and a **Sider-class browser sidebar** with 24 panels.

Not a chatbot. A being with presence that remembers you, has moods, dreams, and grows over time.

---

## Highlights

| | |
|---|---|
| **Persistent memory** | SQLite + Kuzu KG with BM25 + semantic retrieval, FadeMem decay, cross-encoder reranking |
| **Emotions** | ALMA neuromodulator engine (PAD space) — mood shapes every response |
| **Dreams** | NeuroDream: light sleep re-scores memories, deep sleep extracts patterns, REM generates novel connections |
| **Proactive awareness** | KG gap-driven curiosity, motivation-threshold gating, screen/calendar/workflow monitors |
| **Identity** | Narrative self-model evolves across sessions, temporal grounding, personality persistence |
| **Multi-model** | 60+ models: ChatGPT (GPT-5.4 Pro, Codex), cloud (Qwen 397B, DeepSeek, Kimi, Gemini), local (8B on your GPU) |
| **Dev agent** | ReAct loop, 50+ tools, code agent mode, adaptive planning, session persistence |
| **4 surfaces** | CLI, Web UI (React + FastAPI), Browser Extension (Chrome/Firefox), Telegram Bot |

---

## Browser Extension

A full-featured AI sidebar for Chrome and Firefox — comparable to Sider AI but self-hosted and free.

### 24 Panels

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
| **Artifacts** | Live HTML/React/SVG/Mermaid/Chart.js preview with code editor |
| **Image** | Generation (ComfyUI) + Editing (remove BG, upscale, remove text, describe) |
| **Compare** | Side-by-side multi-model response comparison |
| **Code** | Python code interpreter with CSV analysis and chart generation |
| **Record** | Tab audio/mic recording with waveform, transcription, meeting notes |
| **Agent** | Browser automation (DOM serialization, click/type/scroll/navigate) |
| **Wisebase** | Knowledge base with persistent page highlights and saved clips |
| **Models** | Per-feature model assignment across all 24 panels |
| **Settings** | Custom instructions, persona editor, response style presets |
| **Tools** | Callable tools directory |

### Floating UI (on every webpage)

- **Selection bubble** — tiny icon buttons at cursor on text selection (Copy, Explain, Summarize, more...)
- **Quick Launch** — floating AI panel for custom prompts on selected text
- **Draggable FAB** — floating action button, drag to reposition, side-switchable
- **Input field actions** — AI writing tools on any textarea (improve, expand, shorten, grammar, translate)
- **Image hover toolbar** — Describe, Edit, Save on any webpage image
- **Link preview** — hover over links for AI-generated previews
- **Google SERP answers** — AI answer card above Google search results
- **Gmail AI compose** — draft reply, improve, formalize, translate inside Gmail
- **YouTube subtitles** — intercepts captions for transcript viewer
- **Netflix subtitles** — intercepts subtitles via JSON.parse patch
- **Page translation** — bilingual overlay on any webpage
- **Persistent highlights** — save and restore text highlights across page visits
- **Right-click menu** — Explain, Summarize, Translate, Improve, Save to Memory

### Extension Features

- Light/dark theme with smooth toggle
- Keyboard shortcuts (Ctrl+K, Ctrl+L, Ctrl+N, Ctrl+1-5, and more)
- Per-feature model routing (assign different models to different panels)
- TTS voice output on AI messages
- Inline mic button for voice-to-text in chat
- Chat/data export (Markdown, JSON, Text)
- Offline mode with visual status and retry
- Think Mode with Low/Medium/High reasoning levels
- DOMPurify sanitization on all rendered content

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
OLLAMA_API_KEY=your-ollama-bridge-key    # For cloud models via Ollama bridge
TAVILY_API_KEY=your-tavily-key           # Web search
BRAVE_API_KEY=your-brave-key             # Fallback search
```

```bash
aura                          # Interactive chat
aura "fix the login bug"      # One-shot agentic task
aura --resume last            # Resume previous session
aura --trust                  # Auto-approve all tool calls
aura --model qwen3:8b         # Use specific model
aura --tier max               # Use strongest models
```

### ChatGPT Authentication

Use your ChatGPT subscription (GPT-5.4, Codex, etc.) via OAuth:

```bash
aura --login chatgpt          # Opens browser for OAuth login
aura --logout chatgpt         # Remove credentials
```

### Web UI

```bash
python run_web.py             # API server (localhost:8000)
cd web && npm run dev         # Web UI (localhost:5173)
```

### Browser Extension

```bash
cd extension-src && npm install && npm run build
cd .. && python build.py chrome    # or: python build.py firefox
```

Load `dist-chrome/` as an unpacked extension in `chrome://extensions`.

### Telegram Bot

```bash
python run_telegram.py        # Requires TELEGRAM_BOT_TOKEN in .env
```

---

## Models

60+ models available via `/model` picker:

| Source | Models | Examples |
|--------|--------|---------|
| **ChatGPT** (OAuth) | 12 | GPT-5.4 Pro, GPT-5.3 Codex Spark, GPT-5.1 Codex Max |
| **Cloud** (Ollama bridge) | 20+ | Qwen 3.5 397B, DeepSeek V3.2, Kimi K2.5, Gemini 3 Flash, Devstral 2 123B, Cogito 2.1 671B |
| **Local** (Ollama) | 26+ | DeepSeek R1 8B, Qwen 3 8B, Qwen 2.5 Coder 7B, Gemma 3 4B, LLaVA (vision) |

**Role-based routing** — auto-selects the best model per task:

| Role | Default |
|------|---------|
| **Fast** | `gemini-3-flash-preview:cloud` |
| **Code** | `deepseek-v3.2:cloud` |
| **Reasoning** | `kimi-k2.5:cloud` |
| **Vision** | `qwen3-vl:235b-cloud` |
| **Long context** | `minimax-m2.5:cloud` (1M tokens) |

---

## Architecture

```
USER INPUT
    |
[Fast-path] -- simple? --> 8B local model instant response
    |
[ReAct Loop] (1 LLM call per step)
    |-- Tool RAG selects 5-8 relevant tools per query
    |-- Code agent mode for complex tasks (LLM writes Python)
    |-- Adaptive planning with re-plan every 3 steps
    |-- Strategy bandit for reasoning approach selection
    '-- Loop guards (dedup, failure count, iteration cap)
    |
[Memory]
    |-- SQLite + FTS5 + vector embeddings (nomic-embed-text)
    |-- Kuzu temporal KG (entities + relationships)
    |-- BM25 + semantic + graph retrieval -> RRF fusion
    |-- Cross-encoder reranking (ms-marco-MiniLM)
    |-- FadeMem decay (2-week half-life)
    '-- Write gate: merge / supersede / insert decision
    |
[Emotion] (ALMA Engine)
    |-- Neuromodulators: dopamine, serotonin, norepinephrine, oxytocin
    |-- PAD space (Pleasure-Arousal-Dominance)
    '-- Mood -> response style, persists across sessions
    |
[Consciousness]
    |-- World model (Endsley L1-L3 situation awareness)
    |-- Metacognition (reasoning quality tracking)
    |-- Strategy bandit (multi-armed bandit for approach selection)
    |-- Intrinsic motivation (curiosity-driven exploration)
    |-- Prompt evolution (GEPA self-improvement)
    '-- Idle presence (cognitive load, sleep scheduling)
    |
[Sleep] (NeuroDream)
    |-- Light: re-score memories by importance
    |-- Deep: extract patterns, compress, groom KG
    '-- REM: novel connections, proactive message prep
    |
RESPONSE (shaped by mood, grounded in memory, consistent with identity)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/model` | Pick model interactively (60+ models) |
| `/model auto` | Return to auto-routing |
| `/clear` | Clear conversation history |
| `/compact` | Manually compact context window |
| `/context` | Show context window usage |
| `/cost` | Show session cost breakdown |
| `/sessions` | List/manage sessions |
| `/trust` | Enable trust mode (auto-approve tools) |
| `/plan <task>` | Generate execution plan before running |
| `/help` | Show all commands |

---

## Security

- API key auth with constant-time comparison (AURA_API_AUTH_ENABLED)
- Shell command blocklist + token-based command blocking + cwd validation
- SQL multi-statement injection prevention
- AST-validated custom tool sandbox with dynamic import blocking
- Path traversal protection on all file endpoints
- SSRF protection (private IP blocking, rate limiting, size caps)
- DOMPurify sanitization on all rendered HTML in extension
- CSP: `script-src 'self'; object-src 'self'` on extension
- Iframe sandbox (`allow-scripts` only) for artifact preview
- URL validation on all navigation and fetch calls
- Rate limiting middleware (configurable per-IP)
- Permission system (AUTO/PROMPT/BLOCKED tiers)

---

## Project Structure

```
aura/                     # Core Python package
  brain.py                # OllamaBrain — reasoning engine (2600 lines)
  agent.py                # ApprenticeAgent — orchestrator (5200 lines)
  config.py               # Thread-safe configuration
  memory/                 # Unified memory system (SQLite + FTS5 + KG)
  emotion/                # ALMA engine (PAD space, neuromodulators)
  consciousness/          # World model, metacognition, strategy bandit
  evolution/              # GEPA prompt evolution engine
  proactive/              # Monitors, gateway daemon, persistence
  tools/                  # 50+ tool implementations
  core/                   # Agentic loop, router, permissions, MCP server
  cli/                    # Terminal UI (Rich-based)
  auth/                   # ChatGPT OAuth, client
  messaging/              # Telegram bot, WhatsApp adapter
  channels/               # Discord, Signal, LINE adapters

api/                      # FastAPI web server
  routes/                 # 30+ route files
  services/               # Agent service, inner thoughts engine

extension-src/            # Browser extension (TypeScript + React)
  src/panels/             # 24 panel components
  src/components/         # Shared UI components
  src/content.ts          # Content script (4700 lines)
  background.ts           # Service worker
  src/youtube-inject.ts   # YouTube subtitle interception
  src/netflix-inject.ts   # Netflix subtitle interception

web/                      # React web UI
tests/                    # 445+ tests
```

---

## Install

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai) running locally
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

**v4.3.0** — 833 Python files, 24-panel browser extension, 445+ tests passing.

---

Created by [Elnur Ibrahimov](https://github.com/ElnurIbrahimov)
