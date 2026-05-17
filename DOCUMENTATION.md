# AURA Technical Documentation

This is the canonical reference for AURA's current architecture, subsystems, and operation.
For a quick health check, see `CURRENT_STATE.md`. For a project overview, see `README.md`.

## Architecture Overview

```
aura/           — Core runtime, brain, agent, memory, tools, CLI
api/            — FastAPI backend (middleware, routes, auth, services)
web/            — React frontend (chat, tools, insights, settings)
extension-src/  — Browser extension (Chrome, 25+ panels)
deploy/         — Deployment configs (Docker, systemd)
tests/          — Python test suite (1488 passing)
```

## Core Subsystems

### Brain (`aura/brain.py`, ~2000 lines)
OllamaBrain — the LLM calling engine. Handles:
- Cloud model calls via Ollama Pro (14 models)
- ChatGPT OAuth authentication
- History persistence, token tracking, caching
- Model routing delegation, circuit breakers
- Warmup, taint tracking

### Agent (`aura/agent.py`, ~1500 lines)
ApprenticeAgent — the orchestrator. Built with mixins:
- ChatMixin, ReactMixin, KGBrainMixin
- Coordinates tool execution, planning, session persistence

### Agentic Loop (`aura/core/agentic_loop.py`, ~1600 lines)
CLI dev agentic loop with:
- ModelStepController, ToolCallCoordinator, LoopOutcome, LoopEventEmitter
- Adaptive planning, Reflexion failure diagnosis
- Verification stage with checkpoint/atomic rollback

### Memory (`aura/memory/`)
UnifiedMemory — consolidated retrieval pipeline:
- BM25 + semantic + KG retrieval
- RRF fusion, cross-encoder reranking
- FadeMem decay (2-week half-life)

### Tools (`aura/tools/`, 85+ files)
Tool implementation across categories: file, search, code, git, web, vision, telegram

### Hands (`aura/hands/`)
Tool execution lifecycle management, approval, scheduling

### Security (`aura/security/`)
- SSRF guard (`ssrf_guard.py`)
- Audit chain with Merkle trees (`audit_chain.py`)
- Tool signing (`tool_signing.py`)
- Taint tracker (`taint_tracker.py`)

## Model Infrastructure

15 cloud models via Ollama Pro ($20/month): kimi-k2.6, qwen3.5, minimax-m2.7, deepseek-v3.2, etc.
12 ChatGPT models via OAuth (gpt-5.4 through gpt-5.1)
4 local utility models: embeddings, OCR, edge multimodal

Default assignments: Fast→nemotron-3-super, Reason→kimi-k2.6, Code→minimax-m2.7

## Configuration

All config via `.env` (copy from `.env.example`). Required variables:
- `OLLAMA_API_KEY` — Ollama Pro subscription key
- `AURA_API_KEY` — API authentication key (generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
- `AURA_API_AUTH_ENABLED=true` — Enable before network exposure

See `.env.example` for full list with documentation.

## Running

```bash
# API server
python run_web.py

# Web UI (separate terminal)
cd web && npm run dev

# Telegram bot
python run_telegram.py

# CLI
python main.py
```

## Testing

```bash
# Unit tests
pytest tests/ -x -q --ignore=tests/integration

# All tests
pytest tests/ -x -q

# Lint
ruff check aura/ api/ --select E,W,F
```

## Archived Documentation

The old `DOCUMENTATION.md` (describing Phase A/B/C, Gradio GUI, etc.) has been archived to `docs/archive/DOCUMENTATION_legacy.md`. This file replaces it.
