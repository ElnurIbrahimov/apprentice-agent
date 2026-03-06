# AURA — Adaptive Universal Reasoning Agent

AURA is a personal AI assistant with a multi-layer cognitive architecture. It combines long-term memory, emotional modeling, proactive behavior, and multi-model routing into a unified agent that learns and adapts over time.

## Architecture Overview

```
User Input
    │
    ▼
MetacognitionGuardian  ←── pre-flight risk assessment
    │
    ▼
FastPath  ←── handles simple/cached queries instantly
    │
    ▼
Brain  ←── orchestrates all subsystems
    ├── Parliament (multi-voice deliberation)
    ├── ALMA Emotion Engine (neuromodulators)
    ├── Context Engine (relevance scoring)
    ├── State Machine (mood/mode transitions)
    └── Memory Systems
            ├── A-MEM (associative notes)
            ├── Episodic Memory (Qdrant)
            ├── Knowledge Graph (NetworkX + Kuzu)
            └── Unified Memory (fan-out + merge)
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
| `aura_skill_library/` | Learned skill storage and execution |
| `aura_life_modeling/` | Causal life state modeling and scenario simulation |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.ai) running locally (for local model fallbacks)
- Optional: Qdrant running locally for episodic memory

### Install

```bash
git clone https://github.com/ElnurIbrahimov/aura.git
cd aura
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Key settings in `.env`:

```env
AURA_API_KEY=your-secret-key
AURA_PRIVACY_SALT=          # auto-generated on first run if blank
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Start the API server

```bash
python run_web.py
```

Server starts on `http://127.0.0.1:8000`. API docs at `http://127.0.0.1:8000/docs`.

### Start the Web UI

```bash
cd web
npm install
npm run dev
```

Web UI at `http://localhost:5173`. It proxies `/api` to the backend automatically.

### CLI

```bash
python -m aura
```

## Model Routing

AURA uses chain-based model routing — each chain tries models in order, falling back to the next if the previous times out or fails. Local models serve as final fallbacks when cloud is unavailable.

| Chain | Purpose | Cloud-first | Local fallback |
|-------|---------|-------------|----------------|
| `MODEL_FAST_CHAIN` | Quick replies | `gemini-3-flash-preview:cloud` | `qwen3:8b`, `qwen2:1.5b` |
| `MODEL_REASON_CHAIN` | Deep reasoning | `qwen3.5:397b-cloud`, `cogito-2.1:671b-cloud` | `deepseek-r1:8b`, `qwen3:8b` |
| `MODEL_CODE_CHAIN` | Code tasks | `devstral-2:123b-cloud` | `qwen2.5-coder:7b`, `deepseek-r1:8b` |
| `MODEL_THINK_CHAIN` | Step-by-step thinking | `kimi-k2-thinking:cloud` | `deepseek-r1:8b` |
| `MODEL_VISION_CHAIN` | Image understanding | `gemini-3-flash-preview:cloud` | `llava:latest` |
| `MODEL_LONGCTX_CHAIN` | Long context | `qwen3.5:397b-cloud` | `qwen3:8b` |

Cloud models are routed via [Ollama Pro](https://ollama.com) at `http://localhost:11434`. Local models are used directly.

## Multi-User Support

AURA supports multiple isolated user sessions:

- Per-user identity profiles (`aura/multi_user/identity_core.py`)
- Cryptographic user ID hashing with environment-provided salt
- k-anonymity enforcement (k≥5) before knowledge abstraction
- PII detection and redaction (names, emails, phones, credit cards, IPs)
- Session-scoped conversation history and memory

## Memory Architecture

### Short-term
- Sliding window conversation history (configurable via `MAX_HISTORY_LENGTH`)
- Auto-compaction when history exceeds limit

### Mid-term (A-MEM)
- Associative memory notes with embeddings
- LRU-bounded embedding cache (10K entries max)
- Hybrid search: semantic + BM25 keyword

### Long-term
- **Episodic memory**: timestamped episodes in Qdrant with timeline retrieval
- **Knowledge graph**: NetworkX (runtime) + Kuzu (persistent MCP server)
- **Unified memory**: fan-out queries across all backends with retry-after recovery

### Consolidation (Dream Mode)
- NREM-like reverse-chronological replay on alternating cycles
- Importance decay and clustering
- Runs as background daemon

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send message, receive response |
| `/api/chat/stream` | POST | Streaming chat response |
| `/api/ws/{session_id}` | WS | WebSocket chat session |
| `/api/memory/search` | GET | Search memory |
| `/api/memory/recall` | GET | Recall recent memories |
| `/api/state` | GET | Current agent state |
| `/api/consciousness/status` | GET | Consciousness module status |
| `/api/proactive/status` | GET | Proactive daemon status |
| `/api/introspection` | GET | Agent self-report |
| `/api/reasoning-tree` | POST | Trigger explicit reasoning tree |

Full OpenAPI docs: `http://127.0.0.1:8000/docs`

## Security

- API key authentication via `X-API-Key` header (constant-time comparison)
- Path traversal protection on file uploads (atomic symlink resolution)
- Cryptographically random privacy salt (read from env var, generated once)
- PII scrubbing before knowledge abstraction
- Governor gating on autonomous drive-triggered actions

## Development

### Running tests

```bash
python -m pytest tests/ -v
python aura_episodic_memory/test_episodic_memory.py
python aura_knowledge_graph/test_knowledge_graph.py
python aura_skill_library/test_skill_library.py
```

### Smoke tests

```bash
python -c "from aura import ApprenticeAgent; a = ApprenticeAgent(); print('Agent OK')"
python -c "from aura.memory.unified_memory import UnifiedMemory; print('Memory OK')"
python -c "from aura.dream import NeuroDream; print('Dream OK')"
```

### Project structure

```
aura/                    # Core agent package
  brain.py               # LLM orchestration
  agent.py               # Main ApprenticeAgent class
  config.py              # All configuration + model chains
  dream.py               # NREM consolidation
  fast_path.py           # Cache + fast response
  parliament.py          # Multi-voice deliberation
  identity.py            # Runtime identity (mutable)
  proto_agi_core.py      # Autonomous drive loop
  consciousness/         # Global workspace, reward, world model
  emotion/               # ALMA neuromodulator engine
  memory/                # Unified + SQLite memory systems
  multi_user/            # Per-user isolation
  proactive/             # Salience, theory of mind, heartbeat
  soul/                  # Static character definition
  tools/                 # All agent tools (30+)
  thinking/              # Visible thinking / scratchpad

aura_episodic_memory/    # Qdrant-backed episodic memory
aura_knowledge_graph/    # Kuzu-backed knowledge graph (MCP server)
aura_skill_library/      # Learned skill storage
aura_life_modeling/      # Causal life state modeling

api/                     # FastAPI backend
  main.py
  auth.py
  routes/
  middleware.py

web/                     # React frontend (Vite + Tailwind)

run_web.py               # Entry point: starts FastAPI server
```

## License

MIT
