# AURA v4 Phase 3: Full Developer Mode — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make AURA a full developer agent with session continuity, safe interactive editing, auto-maintenance, scriptable automation, semantic code understanding, and optimized model routing.

**Architecture:** 6 features added incrementally. Each builds on existing infrastructure — brain.py already has conversations, code_edit.py already has diffs, CodeSearchTool already has definition extraction. We're wiring, not rebuilding.

**Tech Stack:** Python 3.12, Ollama (cloud models via bridge), SQLite, nomic-embed-text embeddings

---

### Task 1: Model Routing Update (config.py)

**Files:**
- Modify: `D:/Aura/aura/config.py:141-191`

**Step 1: Update MODEL chains to promote priority models**

Replace lines 141-191 in `D:/Aura/aura/config.py` with:

```python
    # Model chains (first available is used as fallback)
    MODEL_FAST_CHAIN = [
        "gemini-3-flash-preview:cloud",   # Primary: 1M ctx, fastest
        "kimi-k2.5:cloud",                # Fallback: 256K, strong general
        "nemotron-3-nano:30b-cloud",       # Fallback: 1M ctx, efficient
        "qwen3:8b",                        # Local fallback: offline-capable
        "qwen2:1.5b",                      # Local fallback: tiny/fast
    ]
    MODEL_REASON_CHAIN = [
        "kimi-k2.5:cloud",                # Primary: 96% AIME, top agentic, 256K
        "glm-5:cloud",                     # Fallback: 86% GPQA, strong agentic
        "qwen3.5:397b-cloud",             # Fallback: hybrid thinking, 256K
        "cogito-2.1:671b-cloud",           # Fallback: extended reasoning
        "deepseek-v3.2:cloud",             # Fallback: strong all-rounder, 128K
        "deepseek-r1:8b",                  # Local fallback: reasoning-capable
        "qwen3:8b",                        # Local fallback: offline-capable
    ]
    MODEL_CODE_CHAIN = [
        "minimax-m2.5:cloud",             # Primary: 80.2% SWE-Bench (highest), 196K
        "qwen3-coder:480b-cloud",          # Fallback: 480B code specialist
        "devstral-2:123b-cloud",           # Fallback: agentic code/SWE
        "qwen3-coder-next:cloud",          # Fallback: efficient code MoE
        "deepseek-v3.2:cloud",             # Fallback: strong at code, 128K
        "qwen2.5-coder:7b",               # Local fallback: code-specialized
        "deepseek-r1:8b",                  # Local fallback: offline code
    ]
    MODEL_VISION_CHAIN = [
        "qwen3-vl:235b-cloud",             # Primary: best open vision, 256K
        "kimi-k2.5:cloud",                 # Fallback: native multimodal
        "gemini-3-flash-preview:cloud",    # Fallback: Gemini supports vision
        "llava:latest",                    # Local fallback: vision-capable
    ]
    MODEL_THINK_CHAIN = [
        "kimi-k2-thinking:cloud",          # Primary: dedicated thinking mode, 256K
        "qwen3.5:397b-cloud",             # Fallback: hybrid think/non-think, 256K
        "cogito-2.1:671b-cloud",           # Fallback: extended reasoning
        "deepseek-r1:8b",                  # Local fallback: reasoning chain-of-thought
    ]
    MODEL_LONGCTX_CHAIN = [
        "gemini-3-flash-preview:cloud",    # Primary: 1M tokens
        "nemotron-3-nano:30b-cloud",       # Fallback: 1M tokens
        "minimax-m2.5:cloud",              # Fallback: 196K
        "kimi-k2.5:cloud",                 # Fallback: 256K
        "qwen3.5:397b-cloud",             # Fallback: 256K
        "qwen3:8b",                        # Local fallback: best local context window
    ]

    # Primary defaults
    MODEL_FAST: str = os.getenv("MODEL_FAST", "gemini-3-flash-preview:cloud")
    MODEL_REASON: str = os.getenv("MODEL_REASON", "kimi-k2.5:cloud")
    MODEL_CODE: str = os.getenv("MODEL_CODE", "minimax-m2.5:cloud")
    MODEL_VISION: str = os.getenv("MODEL_VISION", "qwen3-vl:235b-cloud")
    MODEL_THINK: str = os.getenv("MODEL_THINK", "kimi-k2-thinking:cloud")
    MODEL_LONGCTX: str = os.getenv("MODEL_LONGCTX", "gemini-3-flash-preview:cloud")

    MODEL_NAME: str = MODEL_REASON  # Default model (backward compat)
```

**Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('aura/config.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add aura/config.py
git commit -m "feat: promote kimi-k2.5, minimax-m2.5, glm-5, deepseek-v3.2 in model routing"
```

---

### Task 2: Auto-Compaction Threshold Fix (brain.py)

**Files:**
- Modify: `D:/Aura/aura/brain.py:1232-1239` (think method)
- Modify: `D:/Aura/aura/brain.py:1470-1477` (think_stream method)

Both `think()` and `think_stream()` already have auto-compact at 40 messages. Update threshold for cloud models (128K-1M context).

**Step 1: Update think() auto-compact threshold**

In `D:/Aura/aura/brain.py`, replace the auto-compact block in `think()` (around line 1232):

```python
        # Auto-compact if conversation history is getting long
        if use_history and len(self.conversation_history) > 40:
            try:
                summary = self.compact_history()
                if summary:
                    logger.info(f"[BRAIN] Auto-compacted history → {len(self.conversation_history)} msgs remain")
            except Exception:
                pass
```

With:

```python
        # Auto-compact: cloud models have 128K-256K context, compact at ~60%
        # ~100 tokens/msg average, 75K token threshold ≈ 150 messages
        if use_history and len(self.conversation_history) > 150:
            try:
                summary = self.compact_history()
                if summary:
                    logger.info(f"[BRAIN] Auto-compacted history → {len(self.conversation_history)} msgs remain")
            except Exception:
                pass
```

**Step 2: Update think_stream() auto-compact threshold**

Same change in `think_stream()` (around line 1470) — replace `> 40` with `> 150`.

**Step 3: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('aura/brain.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add aura/brain.py
git commit -m "fix: raise auto-compact threshold from 40 to 150 msgs for cloud model context windows"
```

---

### Task 3: Session Resume CLI (main.py)

**Files:**
- Modify: `D:/Aura/main.py:11-63` (argparse section)
- Modify: `D:/Aura/main.py:75-88` (dispatch section)
- Modify: `D:/Aura/main.py:172-276` (handle_command function)

**Step 1: Add --resume argument to argparse**

In `main.py`, after the `--no-barge-in` argument (line 61), add:

```python
    parser.add_argument(
        "--resume",
        nargs="?",
        const="pick",
        default=None,
        help="Resume a previous session ('last' for most recent, or pick from list)"
    )
```

**Step 2: Add resume dispatch before chat mode**

In `main.py`, after `agent = ApprenticeAgent()` and before the mode dispatch (around line 78), add:

```python
    # Handle session resume
    if args.resume:
        conversations = agent.brain.list_conversations()
        if not conversations:
            print("No previous sessions found.")
        elif args.resume == "last":
            latest = conversations[0]  # Already sorted by updated_at desc
            agent.brain.switch_conversation(latest["id"])
            print(f"  Resumed: {latest.get('title', 'Untitled')} ({latest.get('message_count', 0)} messages)")
        else:
            # Show picker
            print("\n  Recent sessions:\n")
            for i, conv in enumerate(conversations[:10], 1):
                active = " *" if conv.get("is_active") else ""
                title = conv.get("title", "Untitled")[:50]
                msgs = conv.get("message_count", 0)
                print(f"    {i}. {title} ({msgs} msgs){active}")
            print()
            try:
                choice = input("  Pick a session (number): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(conversations[:10]):
                    picked = conversations[idx]
                    agent.brain.switch_conversation(picked["id"])
                    print(f"  Resumed: {picked.get('title', 'Untitled')}")
                else:
                    print("  Invalid choice, starting new session.")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("  Starting new session.")
```

**Step 3: Add /sessions command to handle_command**

In the `handle_command` function, before the `else: Unknown command` block, add:

```python
    elif cmd == "/sessions":
        conversations = agent.brain.list_conversations()
        if not conversations:
            print("  No sessions found.")
            return
        parts_arg = arg.split(maxsplit=1) if arg else []
        subcmd = parts_arg[0].lower() if parts_arg else "list"
        if subcmd == "switch" and len(parts_arg) > 1:
            target = parts_arg[1]
            # Try matching by index number
            try:
                idx = int(target) - 1
                if 0 <= idx < len(conversations):
                    conv = conversations[idx]
                    agent.brain.switch_conversation(conv["id"])
                    print(f"  Switched to: {conv.get('title', 'Untitled')}")
                    return
            except ValueError:
                pass
            # Try matching by ID
            for conv in conversations:
                if conv["id"] == target:
                    agent.brain.switch_conversation(conv["id"])
                    print(f"  Switched to: {conv.get('title', 'Untitled')}")
                    return
            print(f"  Session not found: {target}")
        elif subcmd == "new":
            agent.brain.new_conversation(arg.split(maxsplit=1)[1] if len(parts_arg) > 1 else None)
            print("  Started new session.")
        else:
            print("\n  Sessions:\n")
            for i, conv in enumerate(conversations[:15], 1):
                active = " *" if conv.get("is_active") else ""
                title = conv.get("title", "Untitled")[:50]
                msgs = conv.get("message_count", 0)
                print(f"    {i}. {title} ({msgs} msgs){active}")
            print(f"\n  Usage: /sessions switch <number> | /sessions new [title]")
```

**Step 4: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('main.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add main.py
git commit -m "feat: add --resume flag and /sessions command for session persistence"
```

---

### Task 4: Diff Preview Before Apply (code_edit.py + agent.py)

**Files:**
- Modify: `D:/Aura/aura/tools/code_edit.py:121-218` (edit method)
- Modify: `D:/Aura/aura/agent.py` (_execute_action, around line 2570)

**Step 1: Add dry_run parameter to CodeEditTool.edit()**

In `D:/Aura/aura/tools/code_edit.py`, change the method signature on line 121:

```python
    def edit(self, path: str, old_string: str, new_string: str,
             replace_all: bool = False, dry_run: bool = False) -> dict:
```

Then, right before the `# === Create backup ===` comment (line 190), add:

```python
            # === Dry run: return diff without writing ===
            if dry_run:
                diff = "".join(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"a/{file_path.name}",
                    tofile=f"b/{file_path.name}",
                ))
                return {
                    "success": True,
                    "diff": diff,
                    "path": str(file_path),
                    "preview": True,
                }

```

**Step 2: Add diff preview hook in agent.py _execute_action**

In `D:/Aura/aura/agent.py`, find the `_execute_action` method. Right after `tool = self.tools[tool_name]` and the existing CLI permission check block, add a new block before `# Parse the action into tool method and arguments WITH TIMEOUT`:

```python
        # Diff preview for code edits in interactive mode
        if tool_name == "code_edit" and self._cli_confirm_callback and "edit" in action.lower():
            try:
                # Attempt dry-run preview (best-effort, don't block on parse failure)
                preview_result = self._parse_and_execute_tool_action(tool, tool_name, action + " --dry-run")
                if preview_result and preview_result.get("preview") and preview_result.get("diff"):
                    approved = self._cli_confirm_callback("code_edit_preview", preview_result["diff"])
                    if not approved:
                        return {"success": False, "error": "Edit declined after preview", "declined": True}
            except Exception:
                pass  # If preview fails, fall through to normal execution
```

**Step 3: Update _cli_confirm in main.py to handle diff display**

In `D:/Aura/main.py`, update the `_cli_confirm` function to handle diff previews:

```python
    def _cli_confirm(tool_name: str, action: str) -> bool:
        if tool_name == "code_edit_preview":
            # action contains the diff
            print(f"\n  Proposed edit:\n")
            for line in action.split("\n")[:40]:
                if line.startswith("+") and not line.startswith("+++"):
                    print(f"  \033[32m{line}\033[0m")  # green
                elif line.startswith("-") and not line.startswith("---"):
                    print(f"  \033[31m{line}\033[0m")  # red
                else:
                    print(f"  {line}")
            if action.count("\n") > 40:
                print(f"  ... ({action.count(chr(10)) - 40} more lines)")
        else:
            print(f"\n  \u26a0 Permission required:")
            print(f"    Tool: {tool_name}")
            print(f"    Action: {action[:200]}")
        try:
            response = input("    Allow? (y/n/always): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response == "always":
            for word in action.lower().split()[:3]:
                if len(word) > 3:
                    agent._approved_patterns.add(word)
            return True
        return response in ("y", "yes")
```

**Step 4: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('aura/tools/code_edit.py', doraise=True); py_compile.compile('aura/agent.py', doraise=True); py_compile.compile('main.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add aura/tools/code_edit.py aura/agent.py main.py
git commit -m "feat: diff preview before apply — dry_run mode on code_edit with colored CLI output"
```

---

### Task 5: Non-Interactive Mode (main.py)

**Files:**
- Modify: `D:/Aura/main.py:11-63` (argparse)
- Modify: `D:/Aura/main.py:75-88` (dispatch)

**Step 1: Add -p/--prompt argument**

In `main.py`, after the `--resume` argument, add:

```python
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Non-interactive: run prompt and exit (supports stdin piping)"
    )
```

**Step 2: Add prompt dispatch**

In `main.py`, after the resume handling and before `if args.voice:`, add:

```python
    if args.prompt:
        # Non-interactive mode: run prompt, print response, exit
        prompt = args.prompt
        # Read stdin if piped
        if not sys.stdin.isatty():
            try:
                stdin_text = sys.stdin.read()[:50000]
                if stdin_text.strip():
                    prompt = f"{stdin_text}\n\n{prompt}"
            except Exception:
                pass
        result = agent.run(prompt)
        response = result.get("response", "")
        if response:
            print(response)
        sys.exit(0 if result.get("success", True) else 1)
```

**Step 3: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('main.py', doraise=True); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: non-interactive mode — aura -p 'prompt' with stdin piping support"
```

---

### Task 6: Semantic Codebase Index (new file + integration)

**Files:**
- Create: `D:/Aura/aura/tools/codebase_index.py`
- Modify: `D:/Aura/main.py` (add /project index and /project search)
- Modify: `D:/Aura/aura/brain.py` (inject relevant code into system prompt)

**Step 1: Create codebase_index.py**

Create `D:/Aura/aura/tools/codebase_index.py`:

```python
"""Semantic Codebase Index — embed code definitions for semantic search.

Uses CodeSearchTool's definition finder to extract functions/classes,
embeds them with nomic-embed-text, stores in SQLite for fast retrieval.
"""

import json
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Re-use MemorySystem's embedding approach
_EMBED_MODEL = "nomic-embed-text:latest"
_EMBED_URL = "http://localhost:11434/api/embeddings"


def _embed(text: str) -> Optional[list]:
    """Get embedding from nomic-embed-text via Ollama."""
    try:
        import requests
        r = requests.post(
            _EMBED_URL,
            json={"model": _EMBED_MODEL, "prompt": text[:500]},
            timeout=3,
        )
        if r.status_code == 200:
            return r.json().get("embedding")
    except Exception:
        pass
    return None


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 0.0 else 0.0


class CodebaseIndex:
    """SQLite-backed semantic index of code definitions in a project."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self._db_dir = self.project_path / ".aura"
        self._db_dir.mkdir(exist_ok=True)
        self._db_path = self._db_dir / "index.db"
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    name TEXT,
                    kind TEXT,
                    line_start INTEGER,
                    content TEXT,
                    embedding TEXT,
                    file_mtime REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file ON chunks(file_path)")
            conn.commit()

    def index(self, progress_callback=None) -> dict:
        """Index or re-index the project. Only re-embeds changed files.

        Args:
            progress_callback: Optional callable(current, total, file_path)

        Returns:
            {indexed, skipped, total_chunks, elapsed}
        """
        from .code_search import CodeSearchTool, _walk_files, SKIP_DIRS

        t0 = time.time()
        searcher = CodeSearchTool()
        search_path = self.project_path

        # Get existing mtimes for incremental indexing
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT DISTINCT file_path, MAX(file_mtime) FROM chunks GROUP BY file_path"
            ).fetchall()
        existing_mtimes = {r[0]: r[1] for r in rows}

        files = list(_walk_files(search_path))
        indexed = 0
        skipped = 0
        total_chunks = 0

        for fi, fpath in enumerate(files):
            rel_path = str(fpath.relative_to(search_path))
            try:
                mtime = fpath.stat().st_mtime
            except OSError:
                continue

            # Skip if file hasn't changed
            if rel_path in existing_mtimes and existing_mtimes[rel_path] >= mtime:
                skipped += 1
                continue

            if progress_callback:
                progress_callback(fi + 1, len(files), rel_path)

            # Extract definitions from this file
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue

            chunks = self._extract_chunks(rel_path, content, searcher)

            if not chunks:
                # Store file-level summary
                summary = content[:300].strip()
                if summary:
                    chunks = [{
                        "id": f"{rel_path}:module:0",
                        "file_path": rel_path,
                        "name": fpath.stem,
                        "kind": "module",
                        "line_start": 1,
                        "content": summary,
                    }]

            # Delete old chunks for this file and insert new ones
            with self._lock:
                conn = self._get_conn()
                conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))
                for chunk in chunks:
                    emb = _embed(f"{chunk['kind']} {chunk['name']}: {chunk['content']}")
                    conn.execute(
                        "INSERT OR REPLACE INTO chunks(id, file_path, name, kind, line_start, content, embedding, file_mtime) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            chunk["id"], chunk["file_path"], chunk["name"],
                            chunk["kind"], chunk["line_start"], chunk["content"],
                            json.dumps(emb) if emb else None, mtime,
                        )
                    )
                conn.commit()

            indexed += 1
            total_chunks += len(chunks)

        elapsed = round(time.time() - t0, 1)
        logger.info(f"[CodebaseIndex] Indexed {indexed} files ({total_chunks} chunks) in {elapsed}s, skipped {skipped}")
        return {"indexed": indexed, "skipped": skipped, "total_chunks": total_chunks, "elapsed": elapsed}

    def _extract_chunks(self, rel_path: str, content: str, searcher) -> list:
        """Extract function/class definitions as chunks from file content."""
        import re

        chunks = []
        lines = content.split("\n")

        # Definition patterns (same as CodeSearchTool.find_definition)
        patterns = [
            (r'^\s*(async\s+)?def\s+(\w+)\s*\(', "function"),
            (r'^\s*class\s+(\w+)[\s(:]', "class"),
            (r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s+(\w+)\s*[\(<]', "function"),
            (r'^\s*(export\s+)?(default\s+)?class\s+(\w+)[\s{<]', "class"),
            (r'^\s*(pub\s+)?(async\s+)?fn\s+(\w+)[\s<(]', "function"),
            (r'^\s*(pub\s+)?struct\s+(\w+)[\s{<]', "struct"),
            (r'^\s*func\s+(\([^)]*\)\s+)?(\w+)\s*\(', "function"),
            (r'^\s*type\s+(\w+)\s+(struct|interface)\b', "struct"),
        ]

        compiled = [(re.compile(p), kind) for p, kind in patterns]

        for i, line in enumerate(lines):
            for regex, kind in compiled:
                m = regex.search(line)
                if m:
                    # Get the name from the last capturing group
                    name = m.group(m.lastindex) if m.lastindex else "unknown"
                    # If name is a keyword, skip
                    if name in ("struct", "interface", "class", "function", "def", "pub", "async", "export"):
                        groups = [g for g in m.groups() if g and g.strip() not in ("", "struct", "interface", "pub ", "async ", "export ", "default ")]
                        name = groups[-1] if groups else "unknown"

                    # Grab snippet: definition + next 10 lines
                    snippet_end = min(i + 10, len(lines))
                    snippet = "\n".join(lines[i:snippet_end])

                    chunks.append({
                        "id": f"{rel_path}:{name}:{i+1}",
                        "file_path": rel_path,
                        "name": name,
                        "kind": kind,
                        "line_start": i + 1,
                        "content": snippet[:500],
                    })
                    break

        return chunks

    def search(self, query: str, top_k: int = 10) -> list:
        """Semantic search across indexed code.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of {file_path, name, kind, line_start, content, score}
        """
        query_vec = _embed(query)

        with self._lock:
            rows = self._get_conn().execute(
                "SELECT id, file_path, name, kind, line_start, content, embedding FROM chunks"
            ).fetchall()

        if not rows:
            return []

        scored = []
        for row_id, fpath, name, kind, line, content, emb_str in rows:
            if query_vec and emb_str:
                try:
                    emb = json.loads(emb_str)
                    score = _cosine(query_vec, emb)
                except (json.JSONDecodeError, ValueError):
                    score = 0.0
            else:
                # Fallback: keyword overlap
                q_words = set(query.lower().split())
                c_words = set((content or "").lower().split())
                n_words = set((name or "").lower().split())
                overlap = q_words & (c_words | n_words)
                score = len(overlap) / max(len(q_words), 1)

            scored.append({
                "file_path": fpath,
                "name": name,
                "kind": kind,
                "line_start": line,
                "content": content,
                "score": round(score, 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def stats(self) -> dict:
        """Return index statistics."""
        with self._lock:
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            files = conn.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()[0]
            kinds = conn.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall()
        return {
            "total_chunks": total,
            "files_indexed": files,
            "by_kind": {k: c for k, c in kinds},
            "db_path": str(self._db_path),
        }

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
```

**Step 2: Add /project index and /project search to main.py**

In `main.py`, in the `_handle_project_command` function, add two new subcmd branches before the `else:` block:

```python
    elif subcmd == "index":
        path = parts[1] if len(parts) > 1 else "."
        from aura.tools.codebase_index import CodebaseIndex
        idx = CodebaseIndex(path)
        def on_progress(current, total, fpath):
            if current % 20 == 0 or current == total:
                print(f"  [{current}/{total}] {fpath}")
        print("  Indexing codebase...")
        result = idx.index(progress_callback=on_progress)
        print(f"\n  Done: {result['indexed']} files indexed, {result['total_chunks']} chunks, "
              f"{result['skipped']} unchanged, {result['elapsed']}s")
        idx.close()

    elif subcmd == "search":
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            print("Usage: /project search <query>")
            return
        path = "."
        from aura.tools.codebase_index import CodebaseIndex
        idx = CodebaseIndex(path)
        # Auto-index if empty
        if idx.stats()["total_chunks"] == 0:
            print("  No index found, indexing first...")
            idx.index()
        results = idx.search(query, top_k=10)
        if results:
            print(f"\n  Results for '{query}':\n")
            for r in results:
                score_pct = f"{r['score']:.0%}"
                print(f"  [{score_pct}] {r['file_path']}:{r['line_start']} ({r['kind']}) {r['name']}")
                snippet = (r.get('content') or '')[:100].replace('\n', ' ')
                print(f"        {snippet}")
        else:
            print("  No results found.")
        idx.close()
```

**Step 3: Inject semantic context into brain.py system prompt**

In `D:/Aura/aura/brain.py`, in `_build_full_system_prompt`, after the project context injection block (the one we modified in Phase 2), add:

```python
        # === SEMANTIC CODEBASE CONTEXT ===
        try:
            from aura.tools.codebase_index import CodebaseIndex
            import os
            _cwd = os.getcwd()
            _idx_db = Path(_cwd) / ".aura" / "index.db"
            if _idx_db.exists():
                idx = CodebaseIndex(_cwd)
                if idx.stats()["total_chunks"] > 0:
                    relevant = idx.search(prompt, top_k=3)
                    if relevant and relevant[0]["score"] > 0.3:
                        ctx_parts = []
                        for r in relevant:
                            if r["score"] > 0.3:
                                ctx_parts.append(f"**{r['file_path']}:{r['line_start']}** ({r['kind']} `{r['name']}`):\n```\n{r['content'][:300]}\n```")
                        if ctx_parts:
                            full = f"{full}\n\n## Relevant Code\n" + "\n\n".join(ctx_parts)
                idx.close()
        except Exception:
            pass
```

**Step 4: Verify all imports**

Run: `python -c "from aura.tools.codebase_index import CodebaseIndex; print('OK')"`
Expected: `OK`

**Step 5: Verify syntax on all modified files**

Run: `python -c "import py_compile; py_compile.compile('aura/tools/codebase_index.py', doraise=True); py_compile.compile('aura/brain.py', doraise=True); py_compile.compile('main.py', doraise=True); print('ALL OK')"`
Expected: `ALL OK`

**Step 6: Commit**

```bash
git add aura/tools/codebase_index.py aura/brain.py main.py
git commit -m "feat: semantic codebase index — nomic-embed-text powered code search with auto-inject"
```

---

## Summary

| Task | Feature | Files | Est. Lines |
|------|---------|-------|-----------|
| 1 | Model routing update | config.py | ~50 |
| 2 | Auto-compact threshold | brain.py | ~6 |
| 3 | Session resume CLI | main.py | ~70 |
| 4 | Diff preview | code_edit.py, agent.py, main.py | ~40 |
| 5 | Non-interactive mode | main.py | ~20 |
| 6 | Codebase index | NEW codebase_index.py, main.py, brain.py | ~250 |

**Total: ~436 lines, 1 new file, 5 modified files, 6 commits.**
