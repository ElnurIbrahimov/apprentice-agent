# Aura Hermes-Feature-Parity Plan (Items 1-7)

Date: 2026-06-21
Goal: Bring Aura's provider/config/toolset system to Hermes Agent parity.

## Current State

Aura has 17 direct API providers + ChatGPT OAuth + Ollama in `providers/registry.py`,
a credential pool, 80 tools, and model routing. But:
- Config is hardcoded Python class (`Config`) + `.env`, no YAML config file
- No `aura config set` CLI
- No auxiliary model roles (everything uses the same brain model)
- No provider-level failover
- No profiles
- No credential pool CLI
- No toolset grouping or per-platform enable/disable

## Plan

### Item 1: Config-driven provider registration
**Files:** `aura/config_loader.py` (new), `aura/providers/registry.py` (patch), `aura/providers/__init__.py` (patch)

- New `aura/config_loader.py`: loads `~/.aura/config.yaml` (YAML), merges with env vars
  - `providers:` section with base_url, api_key, models list per provider
  - Custom providers registered from YAML without code changes
  - Falls back to existing `registry.py` PROVIDER_CONFIGS as defaults
- `providers/__init__.py`: `_init_providers()` reads config.yaml providers first, then falls back to hardcoded
- `.env` still works for API keys (env vars take priority over config.yaml)

### Item 2: Auxiliary model roles
**Files:** `aura/config_loader.py` (extend), `aura/auxiliary.py` (new), `aura/brain.py` (patch), `aura/core/model_router_mixin.py` (patch)

- Config section:
  ```yaml
  auxiliary:
    vision:       { provider: ollama-cloud, model: kimi-k2.6:cloud }
    compression:  { provider: ollama-cloud, model: nemotron-3-super:cloud }
    world_model:  { provider: ollama-cloud, model: glm-5:cloud }
    memory:       { provider: ollama-cloud, model: nemotron-3-super:cloud }
  ```
- New `aura/auxiliary.py`: `get_auxiliary_model(role)` → returns (client, model) using the same `_get_client_for_model` routing
- Brain's `_quick_generate`, `compact_history`, `run_world_model_extraction` use auxiliary models instead of the main llm_pool
- Falls back to main model if auxiliary not configured

### Item 3: Provider-level failover
**Files:** `aura/config_loader.py` (extend), `aura/core/model_router_mixin.py` (patch)

- Config section:
  ```yaml
  fallback_providers: [deepseek, openrouter, ollama-cloud]
  ```
- When a provider fails (ConnectionError), `_get_client_for_model` tries the next provider in the fallback list
- Provider failover is separate from model fallback chains (which stay within a provider)

### Item 4: `aura config` CLI subcommand
**Files:** `aura/cli/commands/config_commands.py` (new), `main.py` (patch), `aura/config_loader.py` (extend)

- `aura config` — show current config (prints YAML)
- `aura config set key value` — set a config value (writes to ~/.aura/config.yaml)
- `aura config get key` — get a config value
- `aura config edit` — open config.yaml in $EDITOR
- `aura config path` — print config.yaml path
- Also `/config` slash command inside chat

### Item 5: Profiles system
**Files:** `aura/profiles.py` (new), `main.py` (patch), `aura/config_loader.py` (extend)

- `~/.aura/profiles/<name>/` with own config.yaml, .env, sessions, skills
- `aura profile list` — list profiles
- `aura profile create <name>` — create new profile
- `aura profile use <name>` — set active profile (sticky)
- `aura profile show <name>` — show details
- `aura profile delete <name>` — delete
- `--profile <name>` flag on main entry point
- Default profile: "default" at `~/.aura/`

### Item 6: Credential pool CLI
**Files:** `aura/cli/commands/auth_commands.py` (new), `aura/cli/commands/__init__.py` (patch)

- `/auth list [provider]` — list all keys with status (available / cooling / exhausted)
- `/auth add` — interactive: prompt for provider + key, append to .env
- `/auth remove <provider> <index>` — remove a key by index
- `/auth reset <provider>` — clear exhaustion status for all keys of a provider
- Also `aura auth list/add/remove/reset` as shell subcommands

### Item 7: Toolsets
**Files:** `aura/toolsets.py` (new), `aura/tools/loader.py` (patch), `aura/cli/commands/toolset_commands.py` (new)

- Define toolsets mapping to tool groups:
  - `core`: code_search, code_edit, filesystem, web_search, code_executor, git
  - `research`: arxiv_search, deep_research, research_tool, brave_search, tavily_tool
  - `media`: vision, image_gen, pdf_reader, audio_transcriber, document_generator
  - `system`: system_control, windows_control, clipboard, screenshot, screen_reader
  - `communication`: email_tool, notifications, calendar_tool
  - `knowledge`: knowledge_graph, local_rag, obsidian_tool, load_skill
  - `cognitive`: inner_monologue, neurodream, evoemo, mcts_reasoning
  - `productivity`: task_manager, task_scheduler, spaced_repetition, predictive_tasks
  - `deployment`: deploy_tool, scaffold, github_tool
  - `browser`: browser
  - `voice`: voice, voice_synth, voice_manager
  - `mcp`: mcp_client, mcp_server
- Config section:
  ```yaml
  toolsets:
    enabled: [core, research, media, system, communication, knowledge]
    disabled: [cognitive, deployment, voice]
  platform_toolsets:
    cli: [core, research, media, system, communication, knowledge, productivity, deployment, browser, cognitive]
    telegram: [core, research, knowledge, communication]
    api: [core, research, media, system]
  ```
- `/toolsets` slash command: list all toolsets with status
- `/toolset enable <name>` / `/toolset disable <name>`
- `aura tools list` / `aura tools enable <name>` / `aura tools disable <name>` shell commands
- Tool loading respects toolset config: disabled toolsets' tools are not registered

## Execution Order

1. Config loader (item 1+4 foundation) — needed by everything else
2. Auxiliary roles (item 2) — depends on config loader
3. Provider failover (item 3) — depends on config loader
4. Config CLI (item 4) — depends on config loader
5. Profiles (item 5) — depends on config loader
6. Credential pool CLI (item 6) — standalone
7. Toolsets (item 7) — depends on config loader

## Verification

- `ruff check aura/ --select E,F,W --ignore E501,E402,F401,E741`
- `pytest tests/ -x -q --ignore=tests/integration -k "config or provider or toolset or auth or profile"`
- `python -c "from aura.config_loader import load_config; print(load_config())"`
- `python -c "from aura.toolsets import TOOLSETS; print(list(TOOLSETS.keys()))"`
- `python -c "from aura.auxiliary import get_auxiliary_model; print(get_auxiliary_model('vision'))"`