# Aura Integration Roadmap — OpenClaw / Hermes / Codex

Date: 2026-04-19
Status: Phase 1 approved for implementation

## Context

Survey of three adjacent open-source agent projects to identify portable patterns:

- **OpenClaw** (MIT) — multi-channel personal AI gateway, 53 SKILL.md files, 20+ channels
- **Hermes Agent** (Nous Research, MIT) — Python agent with context compression, credential pool, rate limiting, ACP server
- **Codex CLI** (OpenAI, Apache-2.0) — Rust coding agent with tiered sandbox, bidirectional MCP, headless exec

Aura's existing strengths (not regressing):
- 46-module CLI (~12K LOC) with plan/research/debate/watch/voice/fleet modes
- 6-dim task-aware model router across 16 providers, 44 models
- `aura_skill_library` + `skill_md_importer.py` already bridges OpenClaw SKILL.md format
- UnifiedMemory + Kuzu, GEPA evolution, Strategy Bandit
- Telegram production stack (14k+ LOC)

Aura's confirmed gaps worth filling:
- No exponential backoff with jitter
- No structured error taxonomy
- No rate-limit header tracking
- No memory fence tags on recall
- No tiered permission model (per-tool only)
- No credential rotation pool
- No iterative context compression (truncation only)
- No ACP server surface
- No bidirectional MCP
- No `aura exec` headless split
- Channel coverage: 4 prod, 2 partial vs OpenClaw's 20+

---

## Phase 1 — Reliability & Permissions (1 day)

Low-risk, pure stability wins. Zero architectural change.

1. **`aura/reliability/retry_utils.py`** — jittered decorrelated exponential backoff decorator (Hermes `retry_utils.py`)
2. **`aura/reliability/error_classifier.py`** — taxonomy mapping status/message patterns → recovery actions (Hermes `error_classifier.py`)
3. **`aura/reliability/rate_limit_tracker.py`** — parse `x-ratelimit-*` headers into structured buckets (Hermes `rate_limit_tracker.py`)
4. **`aura/memory/unified_memory.py` + `aura/prompt_builder.py`** — wrap recalled context in `<memory-context>` fence tags with system note preamble (Hermes memory manager pattern)
5. **`aura/cli/permissions_dialog.py`** — refactor to tiered model: `read-only → workspace-write → unrestricted` (Codex sandbox pattern). Preserve existing per-tool granularity as override.

---

## Phase 2 — Context & Providers

### Phase 2a — DONE (2026-04-19)

- **Provider retry shim** (`aura/reliability/provider_shim.py`) — classifier-driven retries with jittered backoff, automatic rate-limit header capture per provider, `ProviderGiveUp` exception for exhausted retries
- **Credential pool** (`aura/providers/credential_pool.py`) — comma/semicolon-separated env var keys (`OPENROUTER_API_KEY=k1,k2,k3`), round-robin rotation, exhaustion tracking with 60s rate-limit / 3600s billing cooldowns, hot-reload on env change
- **Wired into `OpenAICompatProvider`** — both sync and stream paths now use the shim. Transient errors retry transparently; credential-scoped failures cool the specific key so next acquire rotates.
- **Tests** — 20 new (`test_provider_shim.py`, `test_credential_pool.py`); 173 total across reliability layer
- `aura exec` already exists as a subcommand — deferred enhancements to 2c

### Phase 2b — DONE (2026-04-19)

- **Anthropic provider** now uses the shim (sync + stream) and registers in the credential pool; cools `x-api-key` on rate/billing/auth failures
- **Gemini provider** same pattern via `x-goog-api-key`
- **Status bar** has new P2 segment showing tightest provider bucket (e.g. `RPM 10/60` in red when ≥80% used). Only shown at terminal widths ≥120 cols and when at least one bucket is ≥50% used.
- **Tests** — 6 new (`test_status_bar_rate_limit.py`); 1 fixed regression (`test_engineering_r8_fixes.py` mock updated for new call path). 231 total across affected modules.

### Phase 2b — DONE (2026-04-19)

- **Iterative context compression** — `aura/memory/context_compressor.py` (5-phase: tool-result pruning, head/tail protection, LLM-driven structured summary with Active Task / Completed / Decisions / Pending / Files / Critical Context template, anti-thrash guard, orphan tool-pair sanitization). Wired into `_compact_history` at `agentic_loop_support.py:156` with legacy fallback. Config: `CONTEXT_COMPRESSION_THRESHOLD=80000`, `CONTEXT_COMPRESSION_KEEP_LAST=10`. 10 tests passing.

### Phase 2c — DONE (2026-04-19)

- **`aura exec` ergonomics** — new flags: `--timeout N` (hard wall-clock via threading.Timer, exits 124), `--quiet` (suppress progress, print only final response), `--output-failures` (emit per-tool failure JSON to stderr). 7 tests passing.
- **Cost tracking beef-up** — new SQL aggregates `get_stats_by_model()` + `get_stats_by_provider()` in activity_log. `cost` subcommand gets `--by-model`, `--by-provider`, `--session <id>` flags. Rate-limit bucket panel auto-rendered when provider snapshots exist. 5 tests passing.

---

## Phase 3 — Agent Interop

### Phase 3a — DONE (2026-04-19)

- **Coding-agent delegation tool** (`aura/tools/coding_agent.py`) — spawns claude / codex / aider / opencode / goose as subprocess with correct non-interactive flags per CLI. Sandbox-tier aware (READ_ONLY blocks). Exposes LLM-facing schema via `tool_schema()`. All 5 CLIs discovered on PATH.
- **OpenClaw skill cherry-pick** — 5 skills imported via `skill_md_importer`: `coding-agent` + `canvas` (coding), `skill-creator` + `tmux` (automation), `xurl` (research). Idempotent.
- **Tests** — 16 new (`test_coding_agent_tool.py`); 245 total passing. Zero regression.

### Phase 3b — DONE (2026-04-19)

- **ACP server** — `aura/acp/` package (minimal functional scaffolding, ~320 LOC). Stdio JSON-RPC 2.0. Supports initialize, session/new, session/load, session/list, session/prompt (async worker thread), session/cancel, ping. Sandbox-tier aware (READ_ONLY blocks prompts). Entry point: `aura acp-serve` or `python -m aura.acp`. Deferred for later: tool_call event streaming, resource loading, fork. 10 tests passing.
- **MCP bidirectional + write opt-in** — `AURA_MCP_ALLOW_WRITES=true` exposes `write_file`/`edit_file`/`delete_file`; `AURA_MCP_WRITE_ALLOWLIST=/path1,/path2` restricts to directories. Always-excluded (shell, spawn_agent, git_push) remain blocked even with opt-in. New `resources/list` + `resources/read` handlers expose project files with `.mcpignore` support. 10 tests passing.

---

## Phase 4 — Channels

### Phase 4 — DONE (2026-04-19)

- **Slack** — `aura/channels/slack_channel.py` via `slack-sdk` Socket Mode. `ChannelSource.SLACK` already in enum. Env: `SLACK_BOT_TOKEN` (xoxb-), `SLACK_APP_TOKEN` (xapp-), optional `SLACK_ALLOWED_CHANNELS`. Wired into `main.py --channels slack`. Ignores bot-self messages, respects allowlist, threads replies to source message's thread_ts. 9 tests passing.

### Phase 4 — dropped from scope

- **Signal** — requires JVM `signal-cli` daemon as separate systemd service on server; deferred
- **iMessage/BlueBubbles** — Mac-only, Apple sandboxed, no public API; permanently infeasible

---

## Explicitly not porting

- OpenClaw TypeScript gateway core (stack mismatch)
- OpenClaw Mac-only skills (`apple-notes`, `things-mac`, `imsg`, `bluebubbles`)
- Codex Rust TUI (Aura uses Python rich/textual)
- Hermes skin_engine, pairing, Dingtalk/Copilot-specific auth
- Web UI servers (Aura already has one)
