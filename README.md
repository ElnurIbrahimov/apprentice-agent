# AURA — Adaptive Universal Reasoning Agent

AURA is a personal AI agent with persistent memory, multi-model routing, and a full dev CLI. It combines long-term memory, emotional modeling, proactive behavior, 40+ model routing, and autonomous coding tools into a unified agent that learns and adapts over time.

**What makes AURA different:** Persistent memory across sessions. No other dev CLI has this — AURA actually remembers you, your codebase, and past conversations.

---

## Dev CLI (agentic coding tool)

A Claude Code / Codex CLI competitor powered by Ollama with 40+ models. Run `aura` to start.

### Features

| Feature | Description |
|---------|-------------|
| **Structured tool calling** | 11 tools: read, edit, write, grep, glob, shell, git, search, list_dir, project_structure, spawn_agent |
| **Session persistence** | Full tool-call history saved to disk. Resume with `--resume last` |
| **Context window management** | Auto-compaction at 70%/85% thresholds. 40+ model context windows tracked |
| **Repo map** | Regex symbol extraction (Python, JS/TS, Go, Rust, Java) injected into system prompt |
| **Multi-agent** | Spawn sub-agents (reader/researcher/coder) via tool call. Thread pool, mutex, anti-recursion |
| **MCP server** | Expose tools via MCP protocol. `aura mcp-serve` for IDE integration |
| **Permission system** | AUTO/PROMPT/BLOCKED tiers. Per-tool overrides via AURA.md |
| **Auto-test after edits** | Detects test runner, feeds failures back to LLM |
| **Model routing** | 40+ models across local (qwen, deepseek, gemma) and cloud (qwen3.5:397b, devstral, kimi, cogito) with fallback chains |
| **Diff preview** | Colored diff display before applying edits |
| **Web search** | Tavily + Brave fallback |
| **Cost tracking** | Per-session token and cost tracking (`/cost`) |
| **Trust mode** | Auto-approve all tool calls (`--trust` or `/trust`) |
| **Voice mode** | Push-to-talk with Whisper + Sesame TTS |
| **IDE integration** | `aura ide setup` generates VS Code tasks.json + MCP config |

### Quick Start

```bash
# Interactive chat mode (default)
aura

# One-shot agentic task
aura "fix the login bug"

# Resume last session
aura --resume last

# With trust mode (auto-approve all)
aura --trust

# Non-interactive (pipe-friendly)
aura -p "explain this codebase"

# MCP server for IDE integration
aura mcp-serve
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/model` | Pick model interactively (40+ models) |
| `/clear` | Clear conversation history |
| `/compact` | Manually compact context |
| `/context` | Show context window usage |
| `/cost` | Show session cost breakdown |
| `/sessions` | List/manage sessions |
| `/trust` | Enable trust mode |
| `/plan <task>` | Generate execution plan before running |
| `/grep <pattern>` | Quick code search |
| `/shell <cmd>` | Run shell command |

### AURA.md Project Config

Create an `AURA.md` in your project root (`aura init`):

```yaml
---
tier: balanced
model: qwen3.5:397b-cloud
test_cmd: pytest
auto_test: true
permissions:
  shell: auto
  edit_file: auto
max_iterations: 50
---
# My Project
Project-specific instructions for Aura go here.
```

### VS Code Integration

```bash
aura ide setup
```

This generates `.vscode/tasks.json` with Aura commands and prints the MCP server config for `settings.json`:

```json
{
  "mcp.servers": {
    "aura": {
      "command": "python",
      "args": ["-m", "aura.core.mcp_server"]
    }
  }
}
```

---

## Full Agent Platform

Beyond the dev CLI, AURA is a complete AI agent platform:

| Layer | Description |
|-------|-------------|
| **Chrome / Firefox Extension** | 14-panel sidebar, floating dock, per-feature model selection |
| **FastAPI Backend** | WebSocket chat, 30+ REST endpoints, all AI features |
| **React Web UI** | Full web interface at `localhost:5173` |
| **Cognitive Core** | Brain, Parliament, ALMA emotions, consciousness, proactive daemon |
| **Memory Systems** | A-MEM, Episodic (Qdrant), Knowledge Graph (Kuzu), Unified fan-out |
| **GEPA Evolution** | Self-improving skills via Pareto evolution |

---

## Architecture

```
User Input
    |
    ├── Dev CLI (agentic loop)
    │     ├── ToolExecutor (11 tools)
    │     ├── SessionPersistence (JSON on disk)
    │     ├── ContextWindowManager (auto-compact)
    │     ├── RepoMap (symbol extraction)
    │     ├── SubAgentManager (multi-agent)
    │     └── MCP Server (JSON-RPC stdio)
    │
    ├── Brain (LLM orchestration)
    │     ├── ModelRouter (40+ models, fallback chains)
    │     ├── Parliament (multi-voice deliberation)
    │     └── ALMA Emotion Engine (neuromodulators)
    │
    ├── Memory
    │     ├── A-MEM (associative notes)
    │     ├── Episodic Memory (Qdrant)
    │     ├── Knowledge Graph (Kuzu)
    │     └── Unified Memory (fan-out + merge)
    │
    └── Consciousness
          ├── World Model
          ├── Metacognition Guardian
          ├── Proactive Awareness
          └── Strategy Bandit
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `aura/core/agentic_loop.py` | Autonomous tool-calling loop with streaming display |
| `aura/core/session.py` | Full session persistence (tool_calls preserved) |
| `aura/core/token_manager.py` | Context window tracking and auto-compaction |
| `aura/core/repo_map.py` | Regex symbol extraction for codebase awareness |
| `aura/core/sub_agent.py` | Multi-agent orchestration (reader/researcher/coder) |
| `aura/core/mcp_server.py` | MCP server (JSON-RPC 2.0 over stdio) |
| `aura/core/permissions.py` | Three-tier permission system |
| `aura/brain.py` | Core LLM orchestration, budget forcing, history |
| `aura/agent.py` | Main ApprenticeAgent class |
| `aura/evolution/` | GEPA self-improving skills via Pareto evolution |
| `aura/consciousness/` | World model, reward signals, intrinsic motivation |
| `aura/emotion/alma_engine.py` | Neuromodulator simulation |
| `aura/memory/unified_memory.py` | Fan-out queries across all memory backends |

---

## Model Routing

AURA routes across 40+ models with automatic fallback chains:

| Chain | Purpose | Primary | Fallback |
|-------|---------|---------|----------|
| `fast` | Quick replies | `gemini-3-flash-preview:cloud` | `qwen3:8b`, `qwen2:1.5b` |
| `reason` | Deep reasoning | `qwen3.5:397b-cloud` | `deepseek-r1:8b`, `qwen3:8b` |
| `code` | Code/agentic tasks | `devstral-2:123b-cloud` | `qwen2.5-coder:7b` |
| `think` | Step-by-step | `kimi-k2-thinking:cloud` | `deepseek-r1:8b` |
| `vision` | Image understanding | `qwen3-vl:235b-cloud` | `llava:latest` |
| `longctx` | Long documents | `minimax-m2.5:cloud` (196K) | `qwen3:8b` |

Tier selection via `--tier local|balanced|max` or AURA.md frontmatter.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Optional: Node.js 18+ (web UI), Qdrant (episodic memory), ComfyUI (image gen)

### Install

```bash
git clone https://github.com/ElnurBDa/aura.git
cd aura
pip install -r requirements.txt
```

### Run

```bash
# Dev CLI (main use case)
python main.py              # interactive chat
python main.py "fix bug"    # one-shot task
python main.py --chat       # explicit chat mode

# API server (for extension/web UI)
python run_web.py

# Web UI
cd web && npm install && npm run dev

# Diagnostics
python main.py doctor
```

### Configure

```bash
cp .env.example .env
```

Key `.env` settings:

```env
OLLAMA_BASE_URL=http://localhost:11434
TAVILY_API_KEY=your-tavily-key        # web search
AURA_API_KEY=your-secret-key          # optional API auth
```

---

## Chrome Extension

14-panel sidebar with floating dock for every page.

### Install

1. `chrome://extensions` → enable **Developer mode**
2. **Load unpacked** → select `extension/` folder
3. Click AURA icon to open sidebar

### Panels

Chat, Ask, Search, Wisebase, Translate, Grammar, Write, Voice Notes, OCR, PDF, Image Gen, Browser Agent, Tools, Models

### Build

```bash
python build.py chrome    # -> dist-chrome/
python build.py firefox   # -> dist-firefox/
```

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
| `/api/knowledge/save` | POST | Save clip to Wisebase |
| `/api/knowledge/search` | GET | Search saved knowledge |

### Models

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/available` | GET | List all Ollama models |
| `/api/models/config` | GET/PATCH | Role → model assignments |

Full OpenAPI docs: `http://127.0.0.1:8000/docs`

---

## Memory Architecture

| Layer | Backend | Purpose |
|-------|---------|---------|
| Short-term | Sliding window | Current conversation context |
| Mid-term (A-MEM) | SQLite + embeddings | Associative notes with semantic search |
| Episodic | Qdrant | Timeline-aware event memory |
| Knowledge Graph | NetworkX + Kuzu | Entity relationships |
| Unified | Fan-out | Queries all backends, merges results |

### Dream Mode

NREM-like memory consolidation: reverse-chronological replay, importance decay, clustering. Runs as background daemon or manually via `--dream`.

---

## Security

- API key auth (constant-time comparison)
- Path traversal protection on file uploads
- Shell command sanitization
- PII scrubbing before knowledge abstraction
- Rate limiting middleware
- AST validation on marketplace/custom tool imports
- Permission system for all mutating operations

---

## Project Structure

```
main.py                  # Entry point — CLI + one-shot + voice
aura/
  core/                  # Dev CLI engine
    agentic_loop.py      # Autonomous tool-calling loop
    session.py           # Session persistence
    token_manager.py     # Context window management
    repo_map.py          # Symbol extraction
    sub_agent.py         # Multi-agent orchestration
    mcp_server.py        # MCP server (JSON-RPC stdio)
    permissions.py       # Permission system
    tool_schemas.py      # Ollama tool schemas (11 tools)
    context.py           # Project context gathering
    commands.py          # Subcommand handlers
    router.py            # Model routing
    diff_display.py      # Colored diff preview
  brain.py               # Core LLM orchestration
  agent.py               # ApprenticeAgent
  config.py              # Configuration + model chains
  auth/                  # ChatGPT OAuth
  consciousness/         # World model, metacognition
  emotion/               # ALMA neuromodulator engine
  evolution/             # GEPA self-improving skills
  memory/                # Unified + SQLite memory
  tools/                 # 30+ agent tools
  cli/                   # Display, input, model picker

api/                     # FastAPI backend
  routes/                # 30+ REST endpoints
web/                     # React frontend (Vite + Tailwind)
extension/               # Chrome/Firefox extension (14 panels)
```

---

## License

MIT
