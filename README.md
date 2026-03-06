# AURA — Adaptive Universal Reasoning Agent

AURA is a personal AI assistant with a multi-layer cognitive architecture. It combines long-term memory, emotional modeling, proactive behavior, multi-model routing, and a full-featured browser extension into a unified agent that learns and adapts over time.

---

## What's Inside

| Layer | Description |
|-------|-------------|
| **Chrome / Firefox Extension** | 14-panel sidebar, floating dock, per-feature model selection |
| **FastAPI Backend** | WebSocket chat, 30+ REST endpoints, all AI features |
| **React Web UI** | Full web interface at `localhost:5173` |
| **Cognitive Core** | Brain, Parliament, ALMA emotions, consciousness, proactive daemon |
| **Memory Systems** | A-MEM, Episodic (Qdrant), Knowledge Graph (Kuzu), Unified fan-out |

---

## Chrome Extension

The extension gives you AURA on every page as a Chrome Side Panel with a floating action dock.

### Install

1. Go to `chrome://extensions` and enable **Developer mode**
2. Click **Load unpacked** and select the `extension/` folder
3. Click the AURA icon in the toolbar to open the sidebar

### Build for Chrome or Firefox

```bash
python build.py chrome    # -> dist-chrome/
python build.py firefox   # -> dist-firefox/  (uses manifest.firefox.json)
```

Load `dist-firefox/` in Firefox via `about:debugging` -> This Firefox -> Load Temporary Add-on.

### Floating Dock

A slim dock appears on the right edge of every page. Shows the AURA logo by default; expands on hover to reveal action buttons:

| Button | Action |
|--------|--------|
| Chat | Open AURA chat sidebar |
| Search | Open search panel |
| This Page | Send the full page text to Ask AURA |
| Translate | Open translate panel |
| Save | Save selected text (or page URL) to Wisebase memory |

### Panels (14 total)

| Panel | What it does |
|-------|-------------|
| **Chat** | Full streaming conversation. Supports file attachments, thinking mode, deep research. |
| **Ask AURA** | Activated on text selection. Quick actions: Explain, Define, Summarize, Translate, Search web. |
| **Search** | Real web search via Tavily — AI-synthesized answer + clickable source cards with snippets. |
| **Wisebase** | Saved knowledge clips. Browse, search, click to send context to chat, delete. |
| **Translate** | Translate any text between 20+ languages with streaming output. |
| **Grammar** | Three modes: Grammar fix, Grammar + Style, Full Rewrite. Shows changed words inline. |
| **Write** | AI writing assistant: Compose, Summarize, Expand, Improve. |
| **Voice Notes** | Record audio -> live transcript -> "Summarize as Notes" -> save to Wisebase. Firefox: upload audio for Whisper. |
| **OCR** | Select a region on the page -> extract text from screenshot. Send to chat or translate. |
| **Chat with PDF** | Auto-detects PDF tabs. Upload or load by URL. Ask questions via streaming chat. |
| **Image Generator** | Generates images via local ComfyUI (port 8188). Styles: Default, Photo, Anime, Abstract. |
| **Browser Agent** | Describe a task -> AURA reads DOM and performs click/type/scroll/navigate actions (max 15 steps). |
| **Tools** | All AURA tools listed with status. |
| **Models** | Per-feature model routing config. |

### Per-Feature Model Selection

Every panel has an inline model pill (e.g. `● gemini-3-flash-preview`) that opens a picker. Models load directly from Ollama (`localhost:11434`) — works even when the backend is offline. Saved per-feature in browser storage.

- Cloud and local models grouped separately
- Dropdown auto-flips upward when near the bottom of the panel
- "Auto" defers to the backend's chain-based routing

### Context Menu

Right-click any selected text -> **Ask AURA** to instantly open the Ask panel with that text pre-loaded.

---

## Architecture Overview

```
User Input
    |
    v
MetacognitionGuardian  <-- pre-flight risk assessment
    |
    v
FastPath  <-- handles simple/cached queries instantly
    |
    v
Brain  <-- orchestrates all subsystems
    |-- Parliament (multi-voice deliberation)
    |-- ALMA Emotion Engine (neuromodulators)
    |-- Context Engine (relevance scoring)
    |-- State Machine (mood/mode transitions)
    +-- Memory Systems
            |-- A-MEM (associative notes)
            |-- Episodic Memory (Qdrant)
            |-- Knowledge Graph (NetworkX + Kuzu)
            +-- Unified Memory (fan-out + merge)
```

### Key Subsystems

| Module | Purpose |
|--------|---------|
| `aura/brain.py` | Core LLM orchestration, budget forcing, history management |
| `aura/parliament.py` | Multi-voice deliberation for complex queries |
| `aura/emotion/alma_engine.py` | Neuromodulator simulation (dopamine, serotonin, etc.) |
| `aura/consciousness/` | Global workspace, reward signals, world model, intrinsic motivation |
| `aura/proactive/` | Salience filter, theory of mind, proactive message generation |
| `aura/multi_user/` | Per-user identity, privacy guard, knowledge abstraction |
| `aura/tools/` | 30+ tools: memory, browser, filesystem, git, audio, reasoning trees |
| `aura/dream.py` | NREM-like memory consolidation with reverse replay |
| `aura/proto_agi_core.py` | Drive-based autonomous action loop |
| `aura_episodic_memory/` | Timeline-aware episodic memory with Qdrant vector store |
| `aura_knowledge_graph/` | Persistent knowledge graph via Kuzu (MCP server) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.ai) running locally
- Optional: Qdrant (episodic memory), ComfyUI port 8188 (image gen), Tesseract (OCR), ffmpeg (Whisper)

### Install

```bash
git clone https://github.com/ElnurIbrahimov/aura.git
cd aura
pip install -r requirements.txt
pip install tavily-python pdfplumber openai-whisper pytesseract
```

### Configure

```bash
cp .env.example .env
```

Key `.env` settings:

```env
OLLAMA_BASE_URL=http://localhost:11434
TAVILY_API_KEY=your-tavily-key        # web search
AURA_API_KEY=your-secret-key          # optional auth
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Start the API server

```bash
python run_web.py
```

Server starts on `http://127.0.0.1:8000`. API docs at `/docs`.

### Start the Web UI

```bash
cd web && npm install && npm run dev
```

Web UI at `http://localhost:5173`.

---

## Model Routing

AURA uses chain-based model routing — each chain tries models in order, falling back on timeout or error. Per-feature overrides can be set via the extension model picker or the `/api/models/config` API.

| Chain | Purpose | Cloud-first | Local fallback |
|-------|---------|-------------|----------------|
| `MODEL_FAST_CHAIN` | Quick replies | `gemini-3-flash-preview:cloud` | `qwen3:8b`, `qwen2:1.5b` |
| `MODEL_REASON_CHAIN` | Deep reasoning | `qwen3.5:397b-cloud` | `deepseek-r1:8b`, `qwen3:8b` |
| `MODEL_CODE_CHAIN` | Code tasks | `qwen3-coder:480b-cloud` | `qwen2.5-coder:7b` |
| `MODEL_THINK_CHAIN` | Step-by-step thinking | `kimi-k2-thinking:cloud` | `deepseek-r1:8b` |
| `MODEL_VISION_CHAIN` | Image understanding | `qwen3-vl:235b-cloud` | `llava:latest` |
| `MODEL_LONGCTX_CHAIN` | Long documents | `minimax-m2.5:cloud` | `qwen3:8b` |

Cloud models are routed via Ollama at `http://localhost:11434`.

---

## API Endpoints

### Core

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ws/{session_id}` | WS | WebSocket streaming chat |
| `/api/chat` | POST | Single-shot chat |
| `/api/health` | GET | Liveness probe |

### Memory & Knowledge

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/search` | GET | Semantic memory search |
| `/api/knowledge/save` | POST | Save a clip to Wisebase |
| `/api/knowledge/search` | GET | Search saved knowledge |
| `/api/knowledge/list` | GET | List saved clips (paginated) |
| `/api/knowledge/{id}` | DELETE | Delete a clip |

### Models

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/available` | GET | List all Ollama models (cloud + local) |
| `/api/models/config` | GET | Current role -> model assignments |
| `/api/models/config` | PATCH | Set one role's model |
| `/api/models/config/bulk` | PATCH | Set multiple roles at once |

### Tools

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | GET | Web search via Tavily (answer + sources) |
| `/api/pdf/extract` | POST | Extract text from uploaded PDF |
| `/api/pdf/extract-url` | POST | Extract text from a PDF URL |
| `/api/transcribe` | POST | Transcribe audio via Whisper |
| `/api/ocr` | POST | Extract text from image via Tesseract |
| `/api/image/generate` | POST | Generate image via ComfyUI |
| `/api/agent/action` | POST | LLM action planner for browser agent |
| `/api/compare` | POST | Run a prompt on multiple models in parallel |

### Agent & Consciousness

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Current agent state |
| `/api/introspection` | GET | Agent self-report |
| `/api/reasoning-tree` | POST | Trigger explicit reasoning tree |
| `/api/consciousness/status` | GET | Consciousness module status |
| `/api/proactive/status` | GET | Proactive daemon status |

Full OpenAPI docs: `http://127.0.0.1:8000/docs`

---

## Memory Architecture

### Short-term
- Sliding window conversation history (configurable `MAX_HISTORY_LENGTH`)

### Mid-term (A-MEM)
- Associative memory notes with embeddings
- LRU-bounded embedding cache (10K entries)
- Hybrid semantic + BM25 search

### Long-term
- **Episodic memory**: Qdrant-backed, timeline retrieval, saves page clips from extension
- **Knowledge graph**: NetworkX runtime + Kuzu persistent MCP server
- **Unified memory**: fan-out queries across all backends

### Consolidation (Dream Mode)
- NREM-like reverse-chronological replay
- Importance decay and clustering
- Runs as background daemon

---

## Security

- API key auth via `X-API-Key` header (constant-time comparison)
- Path traversal protection on all file uploads
- Cryptographically random privacy salt
- PII scrubbing before knowledge abstraction
- Governor gating on autonomous drive actions
- Rate limiting middleware (configurable via `API_RATE_LIMIT`)

---

## Multi-User Support

- Per-user identity profiles with cryptographic ID hashing
- k-anonymity enforcement (k>=5) before knowledge abstraction
- PII detection and redaction
- Session-scoped conversation history and memory

---

## Optional Dependencies

| Feature | Requirement | Install |
|---------|-------------|---------|
| Web search | Tavily API key | `pip install tavily-python` |
| PDF chat | pdfplumber | `pip install pdfplumber` |
| Voice notes | Whisper + ffmpeg | `pip install openai-whisper` + `winget install ffmpeg` |
| OCR | Tesseract | `pip install pytesseract` + `winget install UB-Mannheim.TesseractOCR` |
| Image generation | ComfyUI | [github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) on port 8188 |

---

## Project Structure

```
extension/               # Chrome/Firefox browser extension
  manifest.json          # Chrome MV3 manifest
  manifest.firefox.json  # Firefox manifest
  background.js          # Service worker (context menu, OCR, agent relay)
  content.js             # Page injection (dock, OCR overlay, DOM agent)
  sidebar.html           # Full sidebar UI (14 panels)
  sidebar.js             # All panel logic, model pills, WebSocket client
  icons/                 # 16px, 48px, 128px extension icons

aura/                    # Core agent package
  brain.py               # LLM orchestration
  agent.py               # Main ApprenticeAgent class
  config.py              # All configuration + model chains
  dream.py               # NREM consolidation
  parliament.py          # Multi-voice deliberation
  proto_agi_core.py      # Autonomous drive loop
  consciousness/         # Global workspace, reward, world model
  emotion/               # ALMA neuromodulator engine
  memory/                # Unified + SQLite memory systems
  multi_user/            # Per-user isolation
  proactive/             # Salience, theory of mind, heartbeat daemon
  tools/                 # 30+ agent tools

aura_episodic_memory/    # Qdrant-backed episodic memory
aura_knowledge_graph/    # Kuzu-backed knowledge graph (MCP server)

api/                     # FastAPI backend
  main.py                # App entry point, lifespan, routers
  routes/                # All API routes
  middleware.py          # Auth + rate limiting

web/                     # React frontend (Vite + Tailwind)

build.py                 # Build extension for Chrome or Firefox
generate_icons.py        # Regenerate extension icons
run_web.py               # Start FastAPI server
```

---

## License

MIT
