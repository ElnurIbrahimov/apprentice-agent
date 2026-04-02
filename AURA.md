---
tier: balanced
# model: kimi-k2.5:cloud
test_cmd: pytest
auto_test: true
# permissions:
#   shell: auto
#   edit_file: auto
# max_iterations: 50
# budget: 5.0
---

# Aura

Stack: python
Frameworks: FastAPI, PyTorch, Transformers

## Architecture

- **Brain** (`aura/brain.py`): OllamaBrain — 3100-line reasoning engine. Handles LLM calls via Ollama (cloud) and ChatGPT (OAuth). Thread-safe with shared executor pool.
- **Agent** (`aura/agent.py`): ApprenticeAgent — 2600-line orchestrator (mixin-based). ReAct loop, tool dispatch, adaptive planning, session persistence.
- **Memory** (`aura/memory/unified_memory.py`): Consolidated retrieval pipeline — BM25 + semantic + KG, RRF fusion, cross-encoder reranking, FadeMem decay.
- **Config** (`aura/config.py`): Thread-safe config with model chains and fallback logic. All models are cloud-only via Ollama Pro.
- **Router** (`aura/core/router.py`): Task-aware model routing with 3 tiers (local/balanced/max) and 8 task categories.

## Models

### Cloud (Ollama Pro — 11 models)
- kimi-k2.5:cloud, nemotron-3-super:cloud, qwen3.5:397b-cloud, qwen3.5:cloud
- deepseek-v3.2:cloud, glm-5:cloud, minimax-m2.7:cloud, minimax-m2.5:cloud
- qwen3-coder:480b-cloud, qwen3-coder-next:cloud, gpt-oss:120b-cloud

### ChatGPT (OAuth — 12 models)
- gpt-5.4, gpt-5.4-thinking, gpt-5.4-pro
- gpt-5.3, gpt-5.3-codex, gpt-5.3-codex-spark
- gpt-5.2, gpt-5.2-codex
- gpt-5.1, gpt-5.1-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini

### Local (utility only — 2 models)
- nomic-embed-text:latest (embeddings/RAG)
- glm-ocr:latest (OCR)

## Default Model Roles
- **Fast**: nemotron-3-super:cloud
- **Reasoning**: kimi-k2.5:cloud
- **Code**: minimax-m2.7:cloud
- **Vision**: kimi-k2.5:cloud
- **Thinking**: qwen3.5:397b-cloud
- **Long context**: minimax-m2.7:cloud

## Key Patterns

- Centralized thread pools in `aura/pools.py`: `llm_pool(12)`, `bg_pool(8)`, `tool_pool(4)`
- TTL-cached system prompt additions
- Circuit breaker on world model
- FadeMem decay (2-week half-life) for memory
- ALMA neuromodulator engine for emotions (PAD space)
- NeuroDream sleep cycles (light/deep/REM)
- GEPA prompt evolution in `aura/evolution/`

## Deployment

- Server: bare metal (systemd) or Docker
- Headless mode: `AURA_HEADLESS=true` for servers without display
- Three systemd services: aura (backend), aura-telegram (bot), aura-daemon (background)
- See `deploy/README.md` for full setup

## Instructions

- All LLM inference is cloud-only. Do not add local model dependencies.
- Config defaults in `config.py` must match `.env.example` defaults.
- When adding new cloud models, add them to `VERIFIED_CLOUD_MODELS` in `config.py`.
- ChatGPT models are listed in `aura/auth/chatgpt_client.py` — update `CHATGPT_MODELS` dict when new GPT versions ship.
