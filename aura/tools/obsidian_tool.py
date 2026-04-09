"""Obsidian Vault Integration — index and search your Obsidian notes inside AURA.

Scans a vault directory for .md files, embeds them via nomic-embed-text,
and stores in ChromaDB for semantic search. AURA's memory and your notes
become one unified knowledge base.

Configure vault path via env var OBSIDIAN_VAULT_PATH or pass path directly.

Features:
- Index all Markdown files in vault (recursive)
- Semantic search across all notes
- Extract action items / TODOs from notes
- Watch for file changes and auto-reindex (polling)
- Parse YAML frontmatter metadata
"""

import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)

CHROMA_COLLECTION = "obsidian_vault"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800        # chars per chunk
CHUNK_OVERLAP = 100     # overlap between chunks
WATCH_INTERVAL = 30.0   # seconds between vault re-scans
INDEX_CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "obsidian_index.json"


class ObsidianTool:
    """Search and index your Obsidian vault notes from within AURA."""

    name = "obsidian"
    description = "Search and index your Obsidian vault notes — 'search my notes about X', 'list TODOs', 'get note Y'"

    def __init__(self):
        self._vault_path: Optional[Path] = None
        self._chroma_client = None
        self._collection = None
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watch = threading.Event()
        self._indexed_files: Set[str] = set()   # path → mtime cache key
        self._index_cache: Dict[str, float] = {}  # path → last mtime
        self._lock = threading.Lock()
        self._load_vault_path()
        self._init_chroma()
        self._load_index_cache()

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def _load_vault_path(self):
        raw = os.getenv("OBSIDIAN_VAULT_PATH", "")
        if raw and Path(raw).exists():
            self._vault_path = Path(raw)
            logger.info(f"[Obsidian] Vault: {self._vault_path}")
        else:
            logger.info("[Obsidian] No OBSIDIAN_VAULT_PATH set — use index_vault(path) to set it")

    def _init_chroma(self):
        if not CHROMA_AVAILABLE:
            return
        try:
            chroma_path = os.getenv("CHROMADB_PATH", "./data/chromadb")
            self._chroma_client = chromadb.PersistentClient(path=chroma_path)
            self._collection = self._chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"[Obsidian] ChromaDB ready ({self._collection.count()} chunks)")
        except Exception as e:
            logger.warning(f"[Obsidian] ChromaDB init failed: {e}")

    def _load_index_cache(self):
        try:
            if INDEX_CACHE_FILE.exists():
                with open(INDEX_CACHE_FILE, "r") as f:
                    self._index_cache = json.load(f)
        except Exception:
            self._index_cache = {}

    def _save_index_cache(self):
        try:
            INDEX_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_CACHE_FILE, "w") as f:
                json.dump(self._index_cache, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Markdown parsing
    # ------------------------------------------------------------------ #

    def _parse_frontmatter(self, content: str) -> tuple[Dict, str]:
        """Extract YAML frontmatter and return (meta, body)."""
        meta = {}
        if content.startswith("---"):
            end = content.find("\n---", 3)
            if end != -1:
                fm_text = content[3:end]
                body = content[end + 4:].strip()
                for line in fm_text.splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                return meta, body
        return meta, content

    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": f"{source}::{chunk_idx}",
                    "text": chunk_text,
                    "chunk_index": chunk_idx,
                })
                chunk_idx += 1
            start = end - CHUNK_OVERLAP if end < len(text) else end
        return chunks

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #

    def _embed(self, text: str) -> Optional[List[float]]:
        if not OLLAMA_AVAILABLE:
            return None
        try:
            resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text[:2000])
            return resp.get("embedding") or resp.get("embeddings", [None])[0]
        except Exception as e:
            logger.debug(f"[Obsidian] Embed failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def _index_file(self, filepath: Path) -> int:
        """Index a single markdown file. Returns number of chunks added."""
        if not self._collection:
            return 0
        try:
            raw = filepath.read_text(encoding="utf-8", errors="ignore")
            frontmatter, body = self._parse_frontmatter(raw)
            if not body.strip():
                return 0
            rel_path = str(filepath.relative_to(self._vault_path)) if self._vault_path else str(filepath)
            note_name = filepath.stem
            chunks = self._chunk_text(body, rel_path)
            ids, embeddings, documents, metadatas = [], [], [], []
            for chunk in chunks:
                emb = self._embed(chunk["text"])
                if emb:
                    ids.append(chunk["id"])
                    embeddings.append(emb)
                    documents.append(chunk["text"])
                    metadatas.append({
                        "source": rel_path,
                        "note_name": note_name,
                        "chunk_index": chunk["chunk_index"],
                        "tags": frontmatter.get("tags", ""),
                        "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                    })
            if ids:
                self._collection.upsert(
                    ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
                )
            return len(ids)
        except Exception as e:
            logger.warning(f"[Obsidian] Failed to index {filepath}: {e}")
            return 0

    def _get_all_md_files(self) -> List[Path]:
        if not self._vault_path:
            return []
        return [
            p for p in self._vault_path.rglob("*.md")
            if not any(part.startswith(".") for part in p.parts)
        ]

    def index_vault(self, path: Optional[str] = None, force: bool = False) -> Dict:
        """Index all markdown files in the vault.

        Args:
            path: Path to Obsidian vault (uses OBSIDIAN_VAULT_PATH if not provided)
            force: Re-index all files even if unchanged
        """
        if path:
            vault = Path(path)
            if not vault.exists():
                return {"success": False, "error": f"Vault path not found: {path}"}
            self._vault_path = vault

        if not self._vault_path:
            return {
                "success": False,
                "error": "No vault path set. Provide path argument or set OBSIDIAN_VAULT_PATH in .env",
            }

        md_files = self._get_all_md_files()
        if not md_files:
            return {"success": False, "error": f"No .md files found in {self._vault_path}"}

        indexed = 0
        skipped = 0
        total_chunks = 0

        for f in md_files:
            mtime = str(f.stat().st_mtime)
            cache_key = str(f)
            if not force and self._index_cache.get(cache_key) == mtime:
                skipped += 1
                continue
            chunks = self._index_file(f)
            if chunks > 0:
                total_chunks += chunks
                self._index_cache[cache_key] = mtime
                indexed += 1
            else:
                skipped += 1

        self._save_index_cache()
        return {
            "success": True,
            "vault": str(self._vault_path),
            "files_indexed": indexed,
            "files_skipped": skipped,
            "total_files": len(md_files),
            "chunks_added": total_chunks,
            "total_chunks_in_db": self._collection.count() if self._collection else 0,
        }

    # ------------------------------------------------------------------ #
    # Background watcher
    # ------------------------------------------------------------------ #

    def _watch_loop(self):
        logger.info(f"[Obsidian] Watching vault for changes every {WATCH_INTERVAL}s")
        while not self._stop_watch.is_set():
            self._stop_watch.wait(WATCH_INTERVAL)
            if self._stop_watch.is_set():
                break
            try:
                if self._vault_path:
                    for f in self._get_all_md_files():
                        mtime = str(f.stat().st_mtime)
                        cache_key = str(f)
                        if self._index_cache.get(cache_key) != mtime:
                            chunks = self._index_file(f)
                            if chunks > 0:
                                self._index_cache[cache_key] = mtime
                                logger.debug(f"[Obsidian] Re-indexed {f.name} ({chunks} chunks)")
                    self._save_index_cache()
            except Exception as e:
                logger.debug(f"[Obsidian] Watch error: {e}")

    def start_watch(self) -> Dict:
        """Start background vault watcher (auto-reindex changed files)."""
        if not self._vault_path:
            return {"success": False, "error": "No vault path set — call index_vault(path) first"}
        if self._watch_thread and self._watch_thread.is_alive():
            return {"success": True, "message": "Watcher already running"}
        self._stop_watch.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True, name="obsidian-watcher")
        self._watch_thread.start()
        return {"success": True, "message": f"Watching vault for changes: {self._vault_path}"}

    def stop_watch(self) -> Dict:
        self._stop_watch.set()
        return {"success": True, "message": "Vault watcher stopped"}

    # ------------------------------------------------------------------ #
    # Search & Retrieval
    # ------------------------------------------------------------------ #

    def search_notes(self, query: str, top_k: int = 5) -> Dict:
        """Semantic search across all indexed notes.

        Args:
            query: What to search for
            top_k: Number of results
        """
        if not self._collection or self._collection.count() == 0:
            return {
                "success": False,
                "error": "No notes indexed yet. Call index_vault(path) first.",
            }
        emb = self._embed(query)
        if not emb:
            return {"success": False, "error": "Embedding model unavailable (is Ollama running?)"}
        try:
            results = self._collection.query(
                query_embeddings=[emb],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            hits = []
            seen_notes = set()
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                note = meta.get("note_name", "unknown")
                relevance = round(1 - results["distances"][0][i], 3)
                if note not in seen_notes:
                    seen_notes.add(note)
                hits.append({
                    "note": note,
                    "source": meta.get("source"),
                    "relevance": relevance,
                    "tags": meta.get("tags", ""),
                    "modified": meta.get("modified"),
                    "excerpt": doc[:400],
                })
            return {"success": True, "query": query, "results": hits, "count": len(hits)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_note(self, name: str) -> Dict:
        """Get the full content of a specific note by name."""
        if not self._vault_path:
            return {"success": False, "error": "No vault path set"}
        # Sanitize: reject names with path separators or glob wildcards to prevent traversal
        if not name or any(c in name for c in ('/', '\\', '*', '?', '[', ']')):
            return {"success": False, "error": "Invalid note name"}
        matches = list(self._vault_path.rglob(f"{name}.md"))
        if not matches:
            # Try case-insensitive
            matches = [f for f in self._vault_path.rglob("*.md") if f.stem.lower() == name.lower()]
        if not matches:
            return {"success": False, "error": f"Note '{name}' not found in vault"}
        filepath = matches[0]
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        frontmatter, body = self._parse_frontmatter(content)
        return {
            "success": True,
            "name": filepath.stem,
            "path": str(filepath.relative_to(self._vault_path)),
            "frontmatter": frontmatter,
            "content": body,
            "word_count": len(body.split()),
        }

    def list_notes(self, folder: Optional[str] = None) -> Dict:
        """List all notes in the vault (or a specific folder)."""
        if not self._vault_path:
            return {"success": False, "error": "No vault path set"}
        root = self._vault_path / folder if folder else self._vault_path
        files = [p for p in root.rglob("*.md") if not any(part.startswith(".") for part in p.parts)]
        notes = []
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
            notes.append({
                "name": f.stem,
                "path": str(f.relative_to(self._vault_path)),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        return {
            "success": True,
            "vault": str(self._vault_path),
            "notes": notes,
            "count": len(notes),
            "indexed_count": self._collection.count() if self._collection else 0,
        }

    def extract_action_items(self, note_name: Optional[str] = None) -> Dict:
        """Extract TODO/checkbox action items from notes.

        Args:
            note_name: Specific note to scan (scans all if None)
        """
        if not self._vault_path:
            return {"success": False, "error": "No vault path set"}
        if note_name and any(c in note_name for c in ('/', '\\', '*', '?', '[', ']')):
            return {"success": False, "error": "Invalid note name"}
        files = (
            list(self._vault_path.rglob(f"{note_name}.md")) if note_name
            else self._get_all_md_files()
        )
        todo_pattern = re.compile(r"[-*]\s+\[\s*[ x]?\s*\]\s+(.+)", re.IGNORECASE)
        re.compile(r"^#+\s+(.+)", re.MULTILINE)
        all_items = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                _, body = self._parse_frontmatter(content)
                todos = todo_pattern.findall(body)
                for todo in todos:
                    checked = re.match(r"\[x\]", todo, re.IGNORECASE) is not None
                    text = re.sub(r"^\[.?\]\s*", "", todo).strip()
                    all_items.append({
                        "note": f.stem,
                        "task": text,
                        "done": checked,
                        "path": str(f.relative_to(self._vault_path)),
                    })
            except Exception:
                continue
        open_items = [i for i in all_items if not i["done"]]
        return {
            "success": True,
            "total_tasks": len(all_items),
            "open_tasks": len(open_items),
            "completed_tasks": len(all_items) - len(open_items),
            "open": open_items,
            "completed": [i for i in all_items if i["done"]],
        }

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute an Obsidian vault action."""
        a = action.lower().strip()
        if "index" in a:
            return self.index_vault(kwargs.get("path"), kwargs.get("force", False))
        if "search" in a or "find" in a or "recall" in a:
            q = kwargs.get("query") or kwargs.get("text") or action
            return self.search_notes(q, kwargs.get("top_k", 5))
        if "get" in a or "read" in a or "open" in a:
            return self.get_note(kwargs.get("name") or kwargs.get("note") or "")
        if "list" in a:
            return self.list_notes(kwargs.get("folder"))
        if "todo" in a or "action" in a or "task" in a:
            return self.extract_action_items(kwargs.get("note"))
        if "watch" in a and "stop" not in a:
            return self.start_watch()
        if "stop" in a:
            return self.stop_watch()
        return self.list_notes()
