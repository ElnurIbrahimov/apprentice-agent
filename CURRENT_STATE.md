# Aura — Current State (2026-04-13)

Single canonical "what is true right now" document.
For historical context see `docs/archive/ENGINEERING_REVIEW_*.md` (R1–R8, closed 2026-04-12) and
`docs/archive/ENGINEERING_REVIEW_2026-04-02.md`.

---

## Recent Work (last 5 commits)

| Commit | Description |
|--------|-------------|
| `7b99b18` | fix(ci): ruff F541 + SSRF test bs4 dependency |
| `80a03b3` | Close out R8 §9 remaining items: 7 fixes, 1488 tests passing |
| `58db480` | R9+ refactor: ChatSession/agentic_loop splits, bg_pool migration, archive stale CLI docs |
| `e625e46` | fix(ci): AuthMiddleware respects runtime AURA_API_AUTH_ENABLED override |
| `394b2b8` | Engineering review R8+R9: 27 bug fixes, architectural upgrades, 1355/1355 tests passing |

---

## Active Subsystems

| Subsystem | Status |
|-----------|--------|
| **CLI** (`aura/cli/`) | Healthy — Rich terminal UI, status_bar, 42+ slash commands, block-level streaming, bg_pool migration done |
| **Brain / Core** (`aura/core/brain.py`, `agentic_loop.py`) | Healthy — ChatSession split out, centroid race fixed (R8), hot_files fix, SSRF DNS rebinding fixed |
| **Memory** (`aura/memory/`) | Healthy — UnifiedMemory (SQLite+FTS5+Kuzu), 5→1 consolidation complete |
| **Hands / Tools** (`aura/hands/`, `aura/tools/`) | Healthy — approval IDs strengthened to 128-bit; code_agent AST sandbox deferred (see Open Issues) |
| **Security** (`aura/security/`) | Healthy — ssrf_guard.py, audit_chain.py, tool_signing.py, taint_tracker.py all present |
| **Messaging / Telegram** (`aura/messaging/telegram/`) | Healthy — mixin-split refactor complete, auth gap fixed (R8), TTS via Kokoro-ONNX, MarkdownV2 fixed |
| **Providers** (`aura/providers/`) | Healthy — Anthropic, OpenAI-compat, Gemini; stream resp.close() in try/finally (R8) |
| **API** (`api/`) | Healthy — error leakage fixed, filename sanitization, CSP hardened; tools_new.py has zero tests (see Open Issues) |
| **Web UI** (`web/`) | Healthy — JS executor sandboxed in iframe (R8), postMessage e.source guards added |
| **Browser Extension** (`extension/`) | Functional — 25 panels; command palette and category tabs still aspirational (not implemented) |
| **Evolution / GEPA** (`aura/evolution/`) | Healthy — EvalCache OrderedDict fix (R8), GEPA self-improvement active |
| **Consciousness** (`aura/consciousness/`) | Healthy — StrategyBandit SQLite lock fix (R8), PatternProphet thread-safe + capped at 1000 entries |
| **Multi-user** (`aura/multi_user/`) | Healthy — Laplace differential privacy noise (R8 fix, replaces broken uniform noise) |
| **Tests** | 1488 passing / 1 pre-existing WebSocket failure; integration tests excluded from CI |

---

## Known Open Issues

### Must-Do
1. **Rotate exposed API key** — `chatgpt_login.py` git history contains a real key. Requires git history rewrite or provider-side rotation.

### Should-Do (Deferred, not urgent)
2. **`tools_new.py` endpoint tests** — calendar, flashcard, shell, email routes have zero test coverage.
3. **`store.py search_semantic` OOM** — loads all embeddings into RAM for linear scan; needs sqlite-vss or FAISS for large databases (deferred since R3).
4. **Route background LLM calls to `bg_pool`** — `_quick_generate`, `compact_history`, world-model extraction still contend with user calls in `llm_pool(4)` (flagged since R6).
5. **`code_executor.run_math` subprocess timeout on Windows** — exponent guard helps but a true subprocess timeout is more robust.

### Architectural (Future / Deferred)
6. **`code_agent.py` AST sandbox** — cannot prevent all MRO-chain escapes. Real fix requires E2B or Docker isolation. Deferred since R1.
7. **WebSocket test infrastructure** — pre-existing 1-test failure; redesign needed for reliable async WS testing.
8. **Hands manager approval** — 60s blocking approval window; should become async/event-based.
9. **Memory consolidation follow-through** — 5→1 plan was executed (UnifiedMemory), but verify all old memory systems are fully dead.
10. **Integration tests in CI** — currently permanently excluded; needs a strategy.

---

## How to Verify the Project Is Healthy

```bash
# Unit tests
pytest tests/ -x -q --ignore=tests/integration

# Integration tests (run manually, not in CI)
pytest tests/integration/ -x -q

# Lint
ruff check aura/ api/ --select E,W,F
```
