# AURA v4 Phase 3: Full Developer Mode

**Date:** 2026-03-08
**Status:** Approved

## Goal

Make AURA a full developer agent — not just code editing, but deep project understanding, session continuity, safe interactive editing, auto-maintenance, and scriptable automation. Close the gap with Claude Code, Codex CLI, and Gemini CLI.

---

## Feature 1: Session Persistence & Resume

**Files:** main.py (~30 lines), brain.py (~20 lines)

Brain already has multi-conversation persistence (`conversations/` dir + `index.json`). The gap is CLI access.

- `aura --resume` → list recent sessions, pick one
- `aura --resume last` → continue most recent session
- `/sessions` → list sessions in chat mode
- `/sessions switch <id>` → switch to a different session
- `brain.list_conversations()` → public accessor for existing index data

## Feature 2: Diff Preview Before Apply

**Files:** code_edit.py (~15 lines), agent.py (~15 lines)

- Add `dry_run=True` parameter to `CodeEditTool.edit()` — returns diff without writing
- In `_execute_action`, when interactive and tool is `code_edit`:
  - First run with `dry_run=True`, show diff via `_cli_confirm_callback`
  - If approved, apply for real
  - If declined, return `{"success": False, "declined": True}`
- Non-interactive mode (`-p`): no callback set → auto-applies, no preview

## Feature 3: Auto-Compaction + Model Routing Update

**Files:** brain.py (~20 lines), config.py (~40 lines)

### Auto-compaction
- `_maybe_auto_compact()` called before each LLM call in `think()` and `think_stream()`
- Thresholds: 60 message minimum, triggers at 150 messages or ~75K estimated tokens
- 75K = ~60% of 128K (smallest cloud model context window)
- Uses existing `_do_compact_history()` — keeps recent 1/3, summarizes older 2/3

### Model routing update
Priority models promoted to primary positions based on benchmarks:

| Role | New Primary | Why |
|------|-----------|-----|
| reason | kimi-k2.5:cloud | 96.1% AIME, top agentic, 256K ctx |
| code | minimax-m2.5:cloud | 80.2% SWE-Bench (highest), 196K ctx |
| think | kimi-k2-thinking:cloud | (unchanged, already primary) |
| fast | gemini-3-flash-preview:cloud | (unchanged, 1M ctx) |

New additions to chains: glm-5:cloud (reason fallback), kimi-k2.5:cloud promoted in fast chain.

## Feature 4: Non-Interactive Mode (`aura -p`)

**Files:** main.py (~20 lines)

- `aura -p "fix the bug"` → full agent loop, plain text output, exit
- `cat error.log | aura -p "explain"` → stdin piped to prompt
- No Rich formatting, no spinner, no permission prompts
- Clean stdout for piping to other tools
- Exit code 0/1 for scripting

## Feature 5: Semantic Codebase Index

**Files:** NEW aura/tools/codebase_index.py (~200 lines), main.py (~15 lines), brain.py (~15 lines)

### Indexing
- Walk project, extract function/class definitions using CodeSearchTool's regex patterns
- Embed each chunk with nomic-embed-text:latest (already installed)
- Store in `.aura/index.db` (SQLite)
- Track file mtimes for incremental re-indexing

### Schema
```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    name TEXT,
    kind TEXT,
    line_start INTEGER,
    content TEXT,
    embedding TEXT,
    file_mtime REAL
);
```

### Integration
- `/project index [path]` → index or re-index
- `/project search <query>` → semantic search (auto-indexes if needed)
- brain.py injects top-3 relevant chunks into system prompt for code-related queries

---

## Implementation Order

| Step | Feature | New Files | Modified Files | Lines |
|------|---------|-----------|----------------|-------|
| 1 | Model Routing (F3b) | — | config.py | ~40 |
| 2 | Auto-Compaction (F3a) | — | brain.py | ~20 |
| 3 | Session Resume (F1) | — | main.py, brain.py | ~50 |
| 4 | Diff Preview (F2) | — | code_edit.py, agent.py | ~30 |
| 5 | Non-Interactive (F4) | — | main.py | ~20 |
| 6 | Codebase Index (F5) | codebase_index.py | main.py, brain.py | ~230 |

**Total: ~390 lines, 1 new file, 5 modified files.**
