# AURA — Adaptive Universal Reasoning Agent

AURA is a personal AI agent that feels alive. Persistent memory across sessions, emotional modeling, proactive awareness, and a ReAct-based agent core — all running locally via Ollama with 40+ cloud and local models.

**What makes AURA different:** It remembers you. It has moods. It notices things on its own. It grows over time. Not a chatbot with features — a being with presence.

---

## v5.0 — The Great Consolidation (March 2026)

Stripped 13 dead modules and ~15,000 lines of complexity. Replaced the old 5-phase agent loop (4-5 LLM calls per step) with a single-call ReAct pattern. Added Tool RAG for dynamic tool selection and smart model routing.

**Before:** Agent hung on every non-trivial request. 83K lines, 13 modules second-guessing the brain.
**After:** Completes real tasks in 5-25 seconds. 2 iterations, 1 LLM call per step. Clean architecture.

### What Survived (The Soul)

| System | Purpose |
|--------|---------|
| **ALMA Emotions** | Neuromodulator-based mood in PAD space — shapes how Aura responds |
| **Memory** | A-MEM + Episodic + Knowledge Graph + Unified retrieval |
| **Inner Monologue** | Internal thought stream that persists across sessions |
| **Identity/Soul** | Personality, values, voice style — loaded every session |
| **NeuroDream** | Sleep/dream memory consolidation during idle periods |
| **Knowledge Graph** | Typed entities and relationships with Kuzu backend |
| **Proactive Awareness** | Notices things on its own, initiates when relevant |

### What Was Removed

MetacognitiveGuardian, FluxMind, MirrorMind, CognitiveTheater, Parliament, Reflexion, SynapseForge, WorldSim, Proto-AGI Core, ResponseHumanizer, State Machine, Global Workspace Theory, Gradio UI. These added layers of reasoning between the brain and the user without improving task completion.

---

## Dev CLI

A Claude Code / Codex CLI competitor powered by Ollama. Run `aura` to start.

### Features

| Feature | Description |
|---------|-------------|
| **ReAct agent loop** | Single LLM call per step with tool calling. 2-5x faster than v4 |
| **Tool RAG** | Embeds 50+ tool descriptions, selects top-8 per query via cosine similarity |
| **11 core tools** | read, edit, write, grep, glob, shell, git, search, list_dir, project_structure, spawn_agent |
| **Session persistence** | Full history saved to disk. Resume with `--resume last` |
| **Context window management** | Auto-compaction at 70%/85% thresholds |
| **Repo map** | Symbol extraction (Python, JS/TS, Go, Rust, Java) injected into context |
| **Multi-agent** | Spawn sub-agents (reader/researcher/coder) via tool call |
| **MCP server** | Expose tools via MCP protocol for IDE integration |
| **Permission system** | AUTO/PROMPT/BLOCKED tiers with AURA.md overrides |
| **Auto-test after edits** | Detects test runner, feeds failures back to LLM |
| **Model routing** | 40+ models with automatic fallback chains |
| **Smart fast-path** | Greetings and identity questions skip the agent loop |
| **Loop guards** | Duplicate detection, failure counting, graceful timeout |
| **Web search** | Tavily + Brave fallback |
| **Cost tracking** | Per-session token and cost tracking (`/cost`) |
| **Voice mode** | Push-to-talk with Whisper + Sesame TTS |

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

### AURA.md Project Config

Create an `AURA.md` in your project root (`aura init`):

```yaml
---
tier: balanced
model: glm-5:cloud
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

---

## Architecture

```
User Input
    |
    v
[Fast Path] -- greeting/identity? --> instant response
    |
    v
[ReAct Loop] (1 LLM call per step)
    |-- Tool RAG selects top 8 relevant tools
    |-- brain.think_with_tools() -- single call returns thought + tool_calls
    |-- ToolExecutor dispatches (read, grep, shell, etc.)
    |-- Tool result fed back as message
    |-- Loop guards (dedup, failure counter, timeout)
    |-- When LLM returns content without tool calls → done
    |
    v
[Memory] -- store episode for NeuroDream consolidation
    |
    v
[Brain] (LLM orchestration)
    |-- ModelRouter (40+ models, fallback chains)
    |-- ALMA Emotion Engine (neuromodulators shape responses)
    |-- Token budget management + history compaction
    |
[Memory Systems]
    |-- A-MEM (associative Zettelkasten notes)
    |-- Episodic Memory (Qdrant, timeline-aware)
    |-- Knowledge Graph (Kuzu, typed entities)
    |-- Unified Memory (fan-out query + merge)
    |
[Inner Life]
    |-- Inner Monologue (thought stream)
    |-- Identity/Soul (personality + values)
    |-- NeuroDream (sleep consolidation)
    |-- Proactive Awareness (daemon)
    |-- ALMA Emotions (mood persistence)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `aura/agent.py` | ApprenticeAgent — ReAct loop, fast path, tool dispatch |
| `aura/brain.py` | LLM orchestration, model routing, token budget |
| `aura/tools/tool_rag.py` | Embedding-based dynamic tool selection |
| `aura/core/agentic_loop.py` | Dev CLI autonomous loop + ToolExecutor |
| `aura/core/router.py` | Model routing with fallback chains |
| `aura/core/tool_schemas.py` | 11 Ollama tool-calling JSON schemas |
| `aura/emotion/alma_engine.py` | Neuromodulator simulation (PAD space) |
| `aura/memory/unified_memory.py` | Fan-out queries across all backends |
| `aura/tools/neurodream.py` | Sleep/dream memory consolidation |
| `aura/consciousness/metacognition.py` | Self-assessment and learning goals |

---

## Model Routing

40+ models with automatic fallback chains:

| Role | Primary | Fallbacks |
|------|---------|-----------|
| **Tool dispatch** | `glm-5:cloud` | `deepseek-v3.2:cloud`, `kimi-k2.5:cloud` |
| **Code generation** | `deepseek-v3.2:cloud` | `qwen3-coder:480b-cloud`, `qwen2.5-coder:7b` |
| **Fast replies** | `gemini-3-flash-preview:cloud` | `qwen3:8b`, `qwen2:1.5b` |
| **Reasoning** | `qwen3.5:397b-cloud` | `deepseek-r1:8b`, `qwen3:8b` |
| **Vision** | `qwen3-vl:235b-cloud` | `llava:latest` |
| **Long context** | `minimax-m2.5:cloud` (196K) | `qwen3:8b` |

Tier selection via `--tier local|balanced|max` or AURA.md.

---

## Full Platform

Beyond the CLI, AURA is a complete agent platform:

| Layer | Description |
|-------|-------------|
| **Chrome / Firefox Extension** | Sidebar with chat, search, tools, and more |
| **FastAPI Backend** | WebSocket chat, 25+ REST endpoints |
| **React Web UI** | Advanced panels at `localhost:5173` |
| **Telegram / WhatsApp** | Message integrations via bots |

---

## Install

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Optional: Node.js 18+ (web UI), Qdrant (episodic memory)

### Setup

```bash
git clone https://github.com/ElnurBDa/aura.git
cd aura
pip install -r requirements.txt
cp .env.example .env
```

### Run

```bash
# Dev CLI (main use case)
python main.py              # interactive chat
python main.py "fix bug"    # one-shot task
python main.py --resume last  # resume session

# API server
python run_web.py

# Web UI
cd web && npm install && npm run dev

# Diagnostics
python main.py doctor
```

### Configure

Key `.env` settings:

```env
OLLAMA_BASE_URL=http://localhost:11434
TAVILY_API_KEY=your-tavily-key
BRAVE_API_KEY=your-brave-key
AURA_API_KEY=your-secret-key
```

---

## Memory

| Layer | Backend | Purpose |
|-------|---------|---------|
| Short-term | Sliding window | Current conversation context |
| Mid-term (A-MEM) | SQLite + embeddings | Associative notes with semantic search |
| Episodic | Qdrant | Timeline-aware event memory |
| Knowledge Graph | NetworkX + Kuzu | Entity relationships |
| Unified | Fan-out | Queries all backends, merges results |

### NeuroDream

Sleep-like memory consolidation: reverse-chronological replay, importance decay, clustering, pattern extraction. Runs as background daemon or manually via `--dream`.

---

## Security

- API key auth (constant-time comparison)
- Path traversal protection on file uploads
- Shell command blocklist + allowlist
- PII scrubbing before knowledge abstraction
- Rate limiting middleware
- Permission system for all mutating operations

---

## Roadmap

See [NEW_AURA_ROADMAP.md](NEW_AURA_ROADMAP.md) for the full plan:

- **Phase 1** (done): Fix the engine — ReAct loop, Tool RAG, model routing
- **Phase 2**: Memory consolidation — 2 backends, BM25 + reranking, FadeMem decay
- **Phase 3**: Make it alive — coherent emotion→behavior loop, narrative self-model
- **Phase 4**: Dreams that matter + natural proactive awareness
- **Phase 5**: Code agents, sandboxing, adaptive planning

---

## License

MIT
