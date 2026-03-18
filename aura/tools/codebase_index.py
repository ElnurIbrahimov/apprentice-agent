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
    except Exception as e:
        logger.debug("[CodebaseIndex] Embedding failed: %s", e)
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
        from .code_search import CodeSearchTool, _walk_files

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
