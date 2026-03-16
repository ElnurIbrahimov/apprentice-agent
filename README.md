# AURA — Adaptive Universal Reasoning Agent

A personal AI agent with persistent memory, emotions, and proactive awareness. Uses **ChatGPT** (GPT-5.x via OAuth) and **60+ Ollama models** (local + cloud) — switch between them mid-conversation with `/model`.

Not a chatbot. A being with presence that remembers you, has moods, dreams, and grows over time.

---

## Features

| | |
|---|---|
| **Persistent memory** | SQLite + Kuzu KG with BM25 + semantic retrieval, FadeMem decay, cross-encoder reranking. Remembers across sessions. |
| **Emotions** | ALMA neuromodulator engine (PAD space) — mood shapes every response. Show, don't tell. |
| **Dreams** | NeuroDream: light sleep re-scores memories, deep sleep extracts patterns, REM generates novel connections. |
| **Proactive awareness** | Notices things on its own. Curiosity driven by KG gap detection. Motivation-threshold gating learns from your feedback. |
| **Identity** | Narrative self-model evolves across sessions. Temporal grounding — knows how long since you last spoke. |
| **Multi-model** | 56+ models: ChatGPT (GPT-5.4 Pro, Codex), cloud (Qwen, DeepSeek, Kimi, Gemini), local (8B models on your GPU). |
| **Dev agent** | ReAct loop, 50+ tools, code agent mode, adaptive planning, session persistence, auto-test after edits. |
| **3 surfaces** | CLI, Web UI (React + FastAPI), Browser Extension (Chrome/Firefox). |

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

AURA can use your ChatGPT subscription (GPT-5.4, Codex, etc.) via OAuth:

```bash
aura --login chatgpt          # Opens browser for OAuth login
aura --logout chatgpt         # Remove credentials
```

Once authenticated, ChatGPT models appear in `/model` and can be selected like any other model. Your ChatGPT subscription tier (Plus/Pro) determines which models are available.

---

## Models

56+ models available via `/model` picker:

| Source | Models | Examples |
|--------|--------|---------|
| **ChatGPT** (OAuth) | 12 | GPT-5.4 Pro, GPT-5.3 Codex, GPT-5.1 Codex Max |
| **Cloud** (Ollama bridge) | 17 | Qwen 3.5 397B, DeepSeek V3.2, Kimi K2.5, Gemini 3 Flash, Devstral 2, Cogito 2.1 671B |
| **Local** (Ollama) | 26 | DeepSeek R1 8B, Qwen 3 8B, Qwen 2.5 Coder 7B, Gemma 3, LLaVA (vision) |

**Role-based routing** — AURA auto-selects the best model per task:

| Role | Default |
|------|---------|
| **Tool dispatch** | `glm-5:cloud` |
| **Code** | `deepseek-v3.2:cloud` |
| **Reasoning** | `kimi-k2.5:cloud` |
| **Vision** | `qwen3-vl:235b-cloud` |
| **Long context** | `minimax-m2.5:cloud` (1M tokens) |
| **Fast** | `gemini-3-flash-preview:cloud` |

Or lock to any model: `/model chatgpt:gpt-5.4-pro`

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/model` | Pick model interactively (56+ models) |
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

## Architecture

```
USER INPUT
    │
[Fast-path] ── simple? ──> 8B local model instant response
    │
[ReAct Loop] (1 LLM call per step)
    ├── Tool RAG selects 5-8 relevant tools per query
    ├── Thought + Action in single call
    ├── Code agent mode for complex tasks (LLM writes Python)
    ├── Adaptive planning with re-plan every 3 steps
    └── Loop guards (dedup, failure count, iteration cap)
    │
[Memory]
    ├── SQLite + FTS5 + vector embeddings
    ├── Kuzu temporal KG (entities + relationships)
    ├── BM25 + semantic + graph retrieval → RRF fusion
    ├── Cross-encoder reranking (ms-marco-MiniLM)
    ├── FadeMem decay (2-week half-life)
    └── Emotional PAD tagging on every memory
    │
[Emotion] (ALMA Engine)
    ├── Neuromodulators: dopamine, serotonin, norepinephrine, oxytocin
    ├── Mood → response style (show, don't tell)
    └── Persists across sessions with time decay
    │
[Inner Life]
    ├── Narrative self-model (evolves, loaded every session)
    ├── Talker/Thinker split (async private reasoning)
    ├── Temporal grounding (time awareness on reconnection)
    └── Curiosity driven by KG gap detection
    │
[Sleep] (NeuroDream)
    ├── Light: re-score memories by importance
    ├── Deep: extract patterns, compress, groom KG
    └── REM: novel connections, proactive message prep
    │
RESPONSE (shaped by mood, grounded in memory, consistent with identity)
```

---

## Project Config

Create `AURA.md` in any project root (`aura init`):

```yaml
---
tier: balanced
model: qwen3.5:397b-cloud
test_cmd: pytest
auto_test: true
permissions:
  shell: auto
  edit_file: auto
---
# My Project
Project-specific instructions for AURA go here.
```

---

## Surfaces

### CLI
The primary interface. Interactive chat, one-shot tasks, session resume, voice mode.

### Web UI
React + FastAPI at `localhost:5173`. Panels for emotion state, memory heatmap, inner thoughts, proactive awareness, NeuroDream stats.

```bash
python run_web.py             # API server (localhost:8000)
cd web && npm run dev         # Web UI (localhost:5173)
```

### Browser Extension
Chrome/Firefox sidebar with chat, search, page summarization. Shadow DOM isolated.

---

## Security

- API key auth with constant-time comparison
- Shell command blocklist + allowlist + injection detection
- Cypher query sanitization (KG)
- AST-validated code sandbox with E2B fallback
- Path traversal protection on all file endpoints
- SSRF protection with IPv6 + redirect checks
- Rate limiting middleware
- Permission system (AUTO/PROMPT/BLOCKED tiers)

---

## Install

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai) running locally
- Optional: Node.js 18+ (Web UI), ChatGPT subscription (OAuth models)

### Setup
```bash
git clone https://github.com/ElnurIbrahimov/apprentice-agent.git
cd apprentice-agent
pip install -r requirements.txt
cp .env.example .env          # Add your API keys
aura doctor                   # Verify setup
```

---

## Roadmap

All core phases complete. See [NEW_AURA_ROADMAP.md](NEW_AURA_ROADMAP.md).

- **Phase 1**: Engine — ReAct loop, Tool RAG, model routing
- **Phase 2**: Memory — SQLite + Kuzu, BM25 + reranking, FadeMem
- **Phase 3**: Alive — emotion→behavior loop, narrative self-model
- **Phase 4**: Dreams — NeuroDream phases, proactive curiosity
- **Phase 5**: Polish — code agent, adaptive planning, ChatGPT auth

---

Created by [Elnur Ibrahimov](https://github.com/ElnurIbrahimov)
