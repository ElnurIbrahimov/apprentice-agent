> SUPERSEDED 2026-04-13. Current source of truth: D:/Aura/CURRENT_STATE.md

# OpenFang → Aura Wiring Roadmap

> Stolen from [RightNow-AI/openfang](https://github.com/RightNow-AI/openfang) — an open-source Agent OS built in Rust.
> This roadmap adapts OpenFang's best engineering patterns into Aura's consciousness-first architecture.
> Created: 2026-03-22

---

## Philosophy

OpenFang has superior **engineering discipline** (Rust, 16 security layers, single binary, 180ms cold start).
Aura has superior **cognitive architecture** (consciousness, emotions, self-improvement, world model).

**Goal:** Steal OpenFang's engineering rigor without losing Aura's soul.

---

## Phase 1: Security Hardening (Priority: CRITICAL)

### 1.1 SSRF & DNS Rebinding Protection
**Status:** NOT STARTED
**Why:** Engineering review flagged this. Current `validate_url()` only checks `http(s)://` prefix. No private IP blocking, no DNS rebinding defense.

**What OpenFang does:** Resolves DNS at check time, pins resolved IP, blocks RFC1918/loopback/link-local, re-validates on redirect.

**Implementation:**
- New module: `aura/security/ssrf_guard.py`
- Functions: `validate_url_safe(url)` → resolves DNS, checks IP against blocklist, returns pinned IP
- Blocklist: 127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16, ::1, fc00::/7
- DNS pin: resolve once, pass IP to requests via `allow_redirects=False` + manual redirect handling
- Wire into: all tools that make HTTP requests (browser, web_search, api_tester, deep_research)

**Files to create:**
- `aura/security/__init__.py`
- `aura/security/ssrf_guard.py`

**Files to modify:**
- `aura/api/middleware.py` — add SSRF check on any user-provided URLs
- `aura/tools/` — wrap outbound HTTP in all tools

### 1.2 Information Flow Taint Tracking
**Status:** NOT STARTED
**Why:** If a user shares an API key in chat, Aura's memory could surface it in logs, external API calls, or unencrypted storage. No data classification exists.

**What OpenFang does:** Tags data as tainted (secrets/PII), tracks flow through the system, blocks tainted data from reaching untrusted sinks.

**Implementation:**
- New module: `aura/security/taint_tracker.py`
- TaintLabel enum: SECRET, PII, INTERNAL, PUBLIC
- TaintedString wrapper: carries label + origin metadata
- Detection: regex patterns for API keys, tokens, passwords, SSNs, emails, credit cards
- Sink guards: before logging, before external API calls, before memory write
- Integration with write_gate.py: auto-redact tainted content before storage

**Files to create:**
- `aura/security/taint_tracker.py`

**Files to modify:**
- `aura/memory/write_gate.py` — check taint before STORE_NEW/MERGE_INTO
- `aura/tools/inner_monologue.py` — redact tainted content in session logs
- `aura/brain.py` — scan user messages for secrets on ingestion

### 1.3 Ed25519 Tool Signing
**Status:** NOT STARTED
**Why:** Custom tools loaded from `custom/` directory could be tampered with. No integrity verification.

**What OpenFang does:** Every agent manifest is Ed25519-signed. Signature verified before loading.

**Implementation:**
- New module: `aura/security/tool_signing.py`
- `sign_tool(tool_path, private_key)` → writes `.sig` file alongside tool
- `verify_tool(tool_path, public_key)` → checks signature before dynamic import
- Wire into `aura/tools/custom_loader.py` — reject unsigned tools in non-trust-mode
- CLI command: `/sign-tool <path>` for Elnur to sign approved tools

**Files to create:**
- `aura/security/tool_signing.py`

**Files to modify:**
- `aura/tools/custom_loader.py` — add signature verification gate

---

## Phase 2: Autonomous Hands System (Priority: HIGH)

### 2.1 Hand Architecture
**Status:** NOT STARTED
**Why:** Aura is reactive (waits for messages). OpenFang's Hands run autonomously on schedules. Aura already has `intrinsic_motivation.py` and `idle_presence.py` pointing this direction, but they're not packaged into discrete, deployable units.

**What OpenFang does:** Each Hand is a self-contained package with:
- Manifest (HAND.toml): name, schedule, triggers, resources, guardrails
- Multi-phase system prompt: domain expertise baked in
- Lifecycle: activate → running → paused → deactivated
- Budget tracking: token/cost limits per Hand

**Implementation:**
- New module: `aura/hands/`
- Hand base class: manifest, system prompt, schedule, guardrails, budget
- HandManager: lifecycle control, scheduling (APScheduler), resource allocation
- Built-in Hands (ported from OpenFang concepts, adapted to Aura's strengths):
  - **Researcher Hand**: scheduled deep research on topics of interest (uses existing deep_research tool)
  - **Collector Hand**: OSINT monitoring with change detection (uses existing web tools)
  - **Memory Hand**: scheduled memory consolidation, KG pruning, contradiction resolution
  - **Guardian Hand**: security monitoring, anomaly detection in Aura's own behavior
- Integration with existing systems:
  - `idle_presence.py` → Hands activate during idle periods
  - `intrinsic_motivation.py` → drives influence which Hands get priority
  - `consciousness/metacognition.py` → Hands report performance for self-evaluation

**Files to create:**
- `aura/hands/__init__.py`
- `aura/hands/base.py` — Hand base class + HandManifest dataclass
- `aura/hands/manager.py` — HandManager (scheduling, lifecycle, budget)
- `aura/hands/researcher.py` — first Hand implementation
- `aura/hands/collector.py` — OSINT monitoring Hand
- `aura/hands/memory_hand.py` — memory maintenance Hand
- `aura/hands/guardian.py` — self-monitoring Hand

**Files to modify:**
- `aura/consciousness/idle_presence.py` — wire Hand activation during idle
- `aura/consciousness/intrinsic_motivation.py` — drives influence Hand priority
- `main.py` — add `/hand` CLI commands (activate, status, pause, list)

### 2.2 Hand Manifest Format (HAND.toml equivalent)
```toml
[hand]
name = "researcher"
version = "0.1.0"
description = "Autonomous deep research on topics of interest"

[schedule]
cron = "0 */4 * * *"  # every 4 hours
max_duration_minutes = 30
idle_only = true  # only run when user is idle

[resources]
max_tokens = 50000
max_cost_usd = 0.50
model_preference = "reasoning"  # uses MODEL_REASON_CHAIN

[guardrails]
require_approval_for = ["publish", "send_message", "write_file"]
blocked_tools = ["shell", "deploy_tool"]
max_iterations = 10
```

---

## Phase 3: Cryptographic Audit Trail (Priority: MEDIUM)

### 3.1 Merkle Hash-Chain Action Log
**Status:** NOT STARTED
**Why:** Aura logs to monologue/metacognition, but nothing is cryptographically verified. If Aura gains real autonomy (Hands), provable action history becomes essential.

**What OpenFang does:** Every agent action is hashed and linked to the previous hash, forming an append-only tamper-evident chain.

**Implementation:**
- New module: `aura/security/audit_chain.py`
- AuditEntry: {timestamp, action_type, action_data, agent_id, prev_hash, hash}
- hash = SHA-256(prev_hash + timestamp + action_type + action_data)
- Storage: append-only SQLite table (no UPDATE/DELETE allowed)
- Verification: `verify_chain()` — walk entire chain, check each hash
- Integration: hook into agent.py ReAct loop — every tool call gets an audit entry

**Files to create:**
- `aura/security/audit_chain.py`

**Files to modify:**
- `aura/agent.py` — emit audit entry on every tool execution
- `aura/hands/manager.py` — emit audit entry on Hand lifecycle events
- `main.py` — add `/audit verify` and `/audit tail` CLI commands

---

## Phase 4: Channel Expansion (Priority: LOW)

### 4.1 Discord Adapter (Revive)
**Status:** NOT STARTED (deprecated code exists in `aura/channels/`)
**Why:** OpenFang has 40 adapters. Aura has 4. Discord is the most requested.

### 4.2 Matrix Adapter
**Status:** NOT STARTED
**Why:** Open protocol, privacy-focused, good for self-hosted deployments.

### 4.3 Clean Adapter Interface
**Status:** NOT STARTED
**Why:** OpenFang's 40 adapters suggest a clean interface pattern. Aura's current messaging code is monolithic (telegram_bot.py = 22K lines).

**Implementation:**
- Define `ChannelAdapter` ABC: `send()`, `receive()`, `on_message()`, `authenticate()`
- Refactor telegram_bot.py and whatsapp_bot.py to implement the interface
- New adapters implement same interface → trivial to add

---

## Phase 5: Distribution (Priority: FUTURE)

### 5.1 Single-Command Install
**Why:** OpenFang: `curl | sh` → single binary. Aura: install Python, pip install 50+ packages, set up Ollama, Qdrant, Kuzu...

**Options:**
- Docker Compose (most practical short-term)
- PyInstaller/Nuitka single binary (ambitious)
- `install.sh` script that handles everything

---

## Implementation Order

```
Week 1: Phase 1.1 (SSRF guard) + Phase 3.1 (Merkle audit chain)
Week 2: Phase 1.2 (Taint tracking) + Phase 1.3 (Tool signing)
Week 3: Phase 2.1 (Hand base class + manager + first Hand)
Week 4: Phase 2.1 continued (remaining Hands) + wire to consciousness
```

---

## Comparison Summary

| What | OpenFang | Aura | After Wiring |
|------|----------|------|-------------|
| Security layers | 16 | ~5 | ~12 |
| Autonomous execution | Hands (7) | Reactive only | Hands (4+) |
| Audit trail | Merkle chain | JSON logs | Merkle chain |
| Secret protection | Taint tracking | None | Taint tracking |
| Tool integrity | Ed25519 signed | AST validation only | Ed25519 + AST |
| SSRF protection | Full (DNS pin) | Basic regex | Full (DNS pin) |
| Channels | 40 | 4 | 6+ |
| Consciousness | None | 11 modules | 11 modules |
| Self-improvement | None | GEPA + bandit | GEPA + bandit |
| Emotions | None | ALMA (PAD space) | ALMA (PAD space) |

**Result:** Aura keeps its cognitive advantage while closing the engineering gap.