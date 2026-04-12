# Hermes Agent Analysis — Lessons for Aura

**Source:** https://github.com/NousResearch/hermes-agent (44K stars, Nous Research, MIT)
**Date:** 2026-04-10
**Method:** 4 parallel deep-dive agents reading actual source code via GitHub API

---

## 1. Skill Self-Improvement Loop

### How Hermes Does It
**It's prompt engineering, not an algorithmic loop.** The system prompt tells the LLM:
- "After completing a complex task (5+ tool calls), save the approach as a skill"
- "When using a skill and finding it outdated, patch it immediately"

The agent calls `skill_manage(action='patch', old_string='...', new_string='...')` to update SKILL.md files on disk. No usage counters, no scoring, no RL. The "self-improvement" is just the LLM writing files that persist to the next session.

### What Aura Should Take

**Aura's GEPA is far more sophisticated** (scored trajectories, Pareto optimization, reflection-based mutation). But Hermes has two patterns worth combining with GEPA:

| Pattern | Value | Implementation |
|---------|-------|----------------|
| **In-session skill patching** | Real-time incremental fixes during use, not just offline batch optimization | Add `SKILLS_GUIDANCE` to brain.py system prompt: "If a skill you loaded was wrong, update it before finishing" |
| **Fuzzy patch instead of full rewrite** | Robust when LLM reformats slightly during read-back | Import fuzzy string matching into GEPA's mutation proposals — emit diffs, not full rewrites |
| **Conditional activation** | `requires_toolsets` / `fallback_for_toolsets` in frontmatter — skills auto-hide/show based on available capabilities | Add `requires: [telegram]` to skill metadata; suppress in CLI context |
| **3-tier progressive loading** | Catalog (names only) → Instructions (full SKILL.md) → Resources (reference files). Saves ~80% tokens | Don't inject full skill bodies into every prompt — just names + descriptions at tier 1 |

---

## 2. Gateway Architecture (Multi-Platform Messaging)

### How Hermes Does It
Single `GatewayRunner` process manages all platforms. Each platform is a `BasePlatformAdapter` subclass wired to one shared handler via `adapter.set_message_handler(self._handle_message)`.

**The key abstraction: `MessageEvent`** — one universal dataclass for all inbound messages:
```
text, message_type (TEXT/VOICE/PHOTO/...), source (platform+chat_id+user_id+thread_id),
media_urls (local paths, already downloaded), reply_to, timestamp
```

All concurrency (interrupt, queueing, typing indicators) lives in the **base class**, not per-platform. Platforms only implement: `connect()`, `disconnect()`, `send()`, and optional media methods.

### What Aura Should Take

| Pattern | Current Aura | Hermes Way | Priority |
|---------|-------------|------------|----------|
| **Shared message handler** | Each bot has its own agent call logic (telegram_bot.py, api/routes/chat.py) | All adapters → one `_handle_message()` | HIGH — eliminates duplicated logic |
| **MessageEvent dataclass** | Platform-specific objects passed around | Universal `MessageEvent` normalizes everything | HIGH — enables cross-platform features |
| **Concurrency in base class** | Mixin-based, split across files | `handle_message()` in base handles interrupt/queue/typing | MEDIUM |
| **Media cached locally first** | Inconsistent | `cache_audio_from_bytes()` before dispatch — URLs expire, local paths don't | MEDIUM |
| **`_send_with_retry()`** | No retry | Exponential backoff + plain-text fallback | LOW |
| **Active-session interrupt** | No interrupt mechanism | `_active_sessions` dict + `_pending_messages` queue — clean interrupt-and-replace | HIGH |
| **DeliveryRouter** | No proactive delivery routing | Separates "reply to user" from "deliver to home channel" for cron/proactive output | MEDIUM |
| **SessionSource** | Platform-specific session keys | Universal `(platform, chat_id, user_id, thread_id)` → consistent session key | MEDIUM |
| **Bridge pattern** | N/A | WhatsApp = Node.js subprocess, Signal = signal-cli daemon. Adapter does HTTP to local bridge. | For future platforms |

### Recommended Aura Refactor (if pursued)
```
aura/messaging/
  gateway.py              # GatewayRunner — single handler
  event.py                # MessageEvent dataclass
  delivery.py             # DeliveryRouter for proactive output
  session.py              # SessionSource + session key builder
  adapters/
    base.py               # BasePlatformAdapter (connect/disconnect/send + concurrency)
    telegram.py           # TelegramAdapter
    web.py                # WebSocketAdapter (current api/routes/chat.py WebSocket)
    extension.py          # ExtensionAdapter
```

---

## 3. Serverless Execution Backends

### How Hermes Does It
6 backends, all implementing one interface: `_run_bash(cmd, login, timeout, stdin) -> ProcessHandle` + `cleanup()`.

**The killer feature: session state persistence via env snapshots.**
After each command, the backend exports the shell environment (`export -p`, `declare -f`, `alias -p`) to a snapshot file. Before the next command, it sources that snapshot. Result: `cd`, `export`, `alias` all persist across separate subprocess calls without keeping a shell open.

### Backend Comparison

| Backend | Isolation | Persistence | Cost | Latency |
|---------|-----------|-------------|------|---------|
| Local | None (host filesystem) | Always | Free | 0ms |
| Docker | Container (cap-drop ALL, pids-limit 256, tmpfs) | Bind mount or ephemeral | Free | ~500ms cold |
| SSH | Remote machine | Remote filesystem | VPS cost | ~200ms |
| Daytona | Cloud VM | Stop/start hibernation (disk preserved) | Pay per use | ~2s resume |
| Modal | Cloud sandbox | Snapshots (JSON-tracked) | Pay per use | ~3s cold, <1s resume |
| Singularity | HPC container (.sif image) | Writable overlay | HPC allocation | ~1s |

### What Aura Should Take

| Pattern | Value | Effort |
|---------|-------|--------|
| **Session env snapshots** | `cd`, `export`, `alias` persist across calls without keeping shell open. Aura's executor is fully stateless — every call starts fresh. | LOW — add `_wrap_command()` to `SandboxExecutor._local_run_python/shell` |
| **task_id isolation** | Different agent tasks get separate sandboxes. Aura has no per-task isolation — one executor instance, shared state. | LOW — add `task_id` parameter to `SandboxExecutor`, dict of active envs |
| **Idle cleanup thread** | 300s TTL reaper. Auto-teardown inactive sandboxes. Aura leaks E2B sessions. | LOW |
| **Docker backend** | Real isolation without E2B cost. `cap-drop ALL + pids-limit 256 + tmpfs + no-new-privileges`. Docker is on the Hetzner server. | MEDIUM — new `DockerEnvironment` class |
| **Credential env filtering** | Substring-match block: `KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PASSWD|AUTH`. More comprehensive than Aura's safe-list approach. | LOW — regex in env sanitization |
| **`_ThreadedProcessHandle`** | Wraps blocking SDK calls in background thread with queue-backed stdout. Makes poll loop work identically across all backends. | MEDIUM — useful if adding Modal/Daytona |

---

## 4. agentskills.io Standard

### What It Is
An **open standard created by Anthropic** (Dec 2025) for portable AI agent skills. A skill = a directory with a `SKILL.md` file (YAML frontmatter + Markdown instructions). Adopted by **30+ agents** including Claude Code, Codex, Copilot, Cursor, Gemini CLI.

### Skill File Format
```yaml
---
name: skill-name           # required, lowercase + hyphens, max 64 chars
description: >             # required, max 1024 chars — WHAT it does + WHEN to use
  Extract PDF text and fill forms. Use when handling PDF files.
license: MIT               # optional
compatibility: Python 3.12+ # optional
metadata:                  # optional, arbitrary key-value
  author: my-org
  version: "2.0"
---
# Skill Title

## When to Use
[Trigger conditions]

## Procedure
[Step-by-step instructions]
```

### Aura's Current State
- **Already consuming:** 634 community skills in `D:\Aura\skills\community/` are agentskills.io format
- **Internal format is different:** `aura_skill_library/skill.py` uses custom dataclass with `trigger_patterns`, `success_count`, `total_uses` — not standard fields
- **Discovery is pattern-matching:** `skill_library.find_applicable(user_input)` — keyword matching, not LLM-driven

### What Aura Should Do

| Action | Effort | Value |
|--------|--------|-------|
| **Scan `~/.agents/skills/` at startup** | LOW | Compatible with Claude Code, Cursor, Codex skills out of the box |
| **Build tier-1 catalog in system prompt** | LOW | Names + descriptions only (~50 tokens/skill). LLM decides activation. |
| **Make internal skills exportable** | MEDIUM | Move `trigger_patterns`, usage stats into `metadata:` sub-keys. `procedure` → SKILL.md body. Skills become portable to other agents. |
| **Drop `trigger_patterns` for LLM-driven activation** | MEDIUM | Better matching, no brittle keyword maintenance. Description becomes the sole trigger. |
| **Publish GEPA-evolved skills to HermesHub** | LOW | Aura's evolved skills could be shared with the 30+ agent ecosystem |

---

## Priority Ranking — What to Build First

### Immediate (< 1 day each)
1. **Session env snapshots** in SandboxExecutor — `cd`/`export` persistence
2. **Credential env filtering** — substring-match blocklist
3. **In-session skill patching prompt** — add SKILLS_GUIDANCE to brain.py
4. **Scan `~/.agents/skills/`** — 3-tier progressive loading

### Short-term (1-3 days each)
5. **Docker execution backend** — real isolation on Hetzner
6. **MessageEvent dataclass** — universal inbound message format
7. **Shared message handler** — single `_handle_message()` for all platforms
8. **Internal skills → agentskills.io export** — make GEPA skills portable

### Medium-term (1-2 weeks)
9. **Full gateway refactor** — adapters/base.py pattern with interrupt/queue
10. **task_id isolation** + idle cleanup in SandboxExecutor
11. **Active-session interrupt** — clean cancel-and-redirect

---

## Key Takeaway

Hermes is broader (more platforms, more backends, more polish) but shallower (no consciousness, no emotion, no knowledge graph, no Hands, prompt-only skill "improvement"). Aura is deeper but narrower. The best move is cherry-picking Hermes's infrastructure patterns (gateway, sandbox, skills format) while keeping Aura's unique cognitive architecture.
