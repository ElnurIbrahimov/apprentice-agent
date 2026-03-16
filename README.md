# AURA — Adaptive Universal Reasoning Agent

A personal AI agent that feels alive. Persistent memory, emotional modeling, proactive awareness, dream consolidation, and a full agent core — running locally via Ollama with 40+ model routing.

Not a chatbot with features — a being with presence.

---

## What Makes AURA Different

| Capability | How It Works |
|-----------|-------------|
| **Remembers you** | Consolidated SQLite + Kuzu KG memory with BM25 + semantic retrieval, FadeMem decay, cross-encoder reranking |
| **Has moods** | ALMA neuromodulator engine (dopamine, serotonin, norepinephrine, oxytocin) in PAD space — mood shapes every response |
| **Dreams** | NeuroDream sleep consolidation: light sleep re-scores memories, deep sleep extracts patterns, REM generates novel connections |
| **Notices things** | Proactive awareness daemon with motivation-threshold gating and curiosity driven by KG gap detection |
| **Grows over time** | Narrative self-model evolves across sessions, emotional continuity persists with time decay |
| **Thinks privately** | Talker/Thinker split — async inner monologue shapes responses without being shown to the user |
| **Works as a dev agent** | ReAct loop, 50+ tools, code agent mode, adaptive planning, session persistence |

---

## Surfaces

AURA runs on three surfaces with a shared brain:

- **CLI** — Interactive chat, one-shot tasks, session resume, voice mode
- **Web UI** — React + FastAPI at `localhost:5173` with emotion panels, memory heatmap, inner thoughts
- **Browser Extension** — Chrome/Firefox sidebar with chat, search, page summarization

---

## Quick Start

```bash
git clone https://github.com/ElnurIbrahimov/apprentice-agent.git
cd apprentice-agent
pip install -r requirements.txt
cp .env.example .env  # Configure Ollama URL + API keys
```

```bash
# CLI (main interface)
aura                          # Interactive chat
aura "fix the login bug"      # One-shot agentic task
aura --resume last            # Resume previous session
aura --trust                  # Auto-approve all tool calls

# Web
python run_web.py             # API server (localhost:8000)
cd web && npm run dev         # Web UI (localhost:5173)

# Diagnostics
aura doctor                   # Check Ollama, models, dependencies
aura models                   # List available models
```

---

## Architecture

```
USER INPUT
    │
[Fast-path check] ── simple? ──> 8B model instant response
    │
[ReAct Loop] (1 LLM call per step)
    ├── Tool RAG selects 5-8 relevant tools
    ├── Thought + Action in single call
    ├── Deterministic evaluation
    └── Loop guards (dedup, failure count, iteration cap)
    │
[Memory] (2 backends)
    ├── SQLite + FTS5 + vectors (all memories)
    ├── Kuzu temporal KG (entities + relationships + validity)
    ├── BM25 + semantic + graph retrieval → RRF fusion
    ├── Cross-encoder reranking
    └── FadeMem decay (2-week half-life, spaced repetition)
    │
[Emotion] (ALMA)
    ├── Chain-of-emotion appraisal (1 cheap LLM call)
    ├── Mood → response style (show, don't tell)
    ├── Emotional memory tagging + mood-congruent retrieval
    └── Session persistence with time decay
    │
[Inner Life]
    ├── Talker/Thinker split (async private reasoning)
    ├── Narrative self-model (loaded every session)
    ├── Temporal grounding (time awareness)
    └── User profile (always in context)
    │
[Sleep] (NeuroDream)
    ├── Light: memory re-scoring by importance
    ├── Deep: pattern extraction, compression, KG grooming
    ├── REM: creative connections, proactive prep
    └── Updates self-model + user profile
    │
[Proactive]
    ├── Motivation accumulator (5-factor scoring)
    ├── Curiosity = specific KG gaps
    └── Learned threshold from user feedback
    │
RESPONSE (shaped by mood, grounded in memory, consistent with identity)
```

---

## Configuration

### Environment Variables

```env
OLLAMA_HOST=http://localhost:11434
AURA_API_KEY=your-secret-key
TAVILY_API_KEY=your-tavily-key     # Optional: web search
BRAVE_API_KEY=your-brave-key       # Optional: fallback search
```

### Model Routing

40+ models with automatic fallback chains:

| Role | Primary | Fallbacks |
|------|---------|-----------|
| **Fast replies** | `gemini-3-flash-preview:cloud` | `qwen3:8b`, `qwen2:1.5b` |
| **Code** | `deepseek-v3.2:cloud` | `qwen3-coder:480b-cloud`, `qwen2.5-coder:7b` |
| **Reasoning** | `qwen3.5:397b-cloud` | `deepseek-r1:8b`, `qwen3:8b` |
| **Vision** | `qwen3-vl:235b-cloud` | `llava:latest` |

Tier selection: `aura --tier local|balanced|max` or via `AURA.md`.

### Project Config (AURA.md)

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
Project-specific instructions go here.
```

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `main.py` | CLI entry — argument parsing, chat loop, subcommands |
| `aura/agent.py` | ReAct agent loop, fast path, tool dispatch, code agent mode |
| `aura/brain.py` | OllamaBrain — LLM orchestration, model routing, neuromodulator scaling |
| `aura/core/agentic_loop.py` | Dev CLI loop — tool executor, context management |
| `aura/core/adaptive_planner.py` | Complexity detection, plan generation, re-planning |
| `aura/core/code_agent.py` | LLM writes Python as actions (smolagents-style) |
| `aura/emotion/alma_engine.py` | ALMA neuromodulator simulation in PAD space |
| `aura/emotion/integration.py` | Behavioral style prompts, chain-of-emotion appraisal |
| `aura/emotion/temporal_grounding.py` | Session continuity — time awareness, mood decay |
| `aura/emotion/memory_tagging.py` | PAD tagging on memories, mood-congruent retrieval |
| `aura/memory/store.py` | Unified SQLite + FTS5 + vector memory store |
| `aura/memory/retrieval.py` | Multi-channel retrieval with RRF fusion |
| `aura/memory/fade_mem.py` | Exponential memory decay with spaced repetition |
| `aura/narrative_self.py` | Evolving identity narrative |
| `aura/tools/neurodream.py` | Sleep/dream consolidation (light/deep/REM) |
| `aura/proactive/gateway_daemon.py` | Proactive decision-making center |
| `aura/proactive/curiosity_scanner.py` | KG gap detection for curiosity-driven outreach |
| `aura/proactive/motivation_accumulator.py` | 5-factor scored message gating with learned threshold |

---

## Security

- API key auth (constant-time comparison)
- Shell command blocklist + allowlist
- Path traversal protection
- AST-validated code sandbox + E2B fallback
- Rate limiting middleware
- Permission system (AUTO/PROMPT/BLOCKED tiers)
- PII scrubbing before knowledge abstraction

---

## Roadmap

All core phases complete. See [NEW_AURA_ROADMAP.md](NEW_AURA_ROADMAP.md) for details.

- **Phase 1**: Engine — ReAct loop, Tool RAG, model routing
- **Phase 2**: Memory — 2 backends, BM25 + reranking, FadeMem, UserProfile
- **Phase 3**: Alive — coherent emotion→behavior loop, narrative self-model, temporal grounding
- **Phase 4**: Dreams — NeuroDream phases, proactive curiosity, motivation threshold
- **Phase 5**: Polish — code agent mode, adaptive planning, emotional continuity, sandboxing

---

Created by [Elnur Ibrahimov](https://github.com/ElnurIbrahimov)
