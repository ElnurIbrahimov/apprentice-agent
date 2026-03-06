"""Clipboard Memory Monitor — auto-indexes clipboard content into RAG for semantic recall.

Background thread polls clipboard every 2 seconds. When content changes it is:
  1. Saved to disk (JSON history, compatible with ClipboardHistoryTool)
  2. Embedded via nomic-embed-text and stored in ChromaDB for semantic search

Usage:
    tool = ClipboardMemoryTool()
    tool.start_monitor()            # begin background monitoring
    tool.recall("JSON config")      # semantic search past clipboard items
    tool.save_current("my label")   # manually save + embed current clipboard
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

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

MEMORY_FILE = Path(__file__).parent.parent.parent / "data" / "clipboard_memory.json"
CHROMA_COLLECTION = "clipboard_memory"
EMBEDDING_MODEL = "nomic-embed-text"
POLL_INTERVAL = 2.0     # seconds
MAX_ENTRIES = 500
MIN_CONTENT_LENGTH = 5  # ignore tiny copies (single chars etc.)


class ClipboardMemoryTool:
    """Clipboard monitor that auto-indexes copied content for semantic recall."""

    name = "clipboard_memory"
    description = "Auto-indexes clipboard history for semantic recall — ask 'what did I copy earlier about X?'"

    def __init__(self):
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_content: Optional[str] = None
        self._chroma_client = None
        self._collection = None
        self._lock = threading.Lock()
        self._ensure_data_file()
        self._init_chroma()

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def _ensure_data_file(self):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not MEMORY_FILE.exists():
            self._save_entries([])

    def _init_chroma(self):
        if not CHROMA_AVAILABLE:
            logger.warning("[ClipboardMemory] chromadb not available — semantic recall disabled")
            return
        try:
            import os
            chroma_path = os.getenv("CHROMADB_PATH", "./data/chromadb")
            self._chroma_client = chromadb.PersistentClient(path=chroma_path)
            self._collection = self._chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"[ClipboardMemory] ChromaDB collection ready ({self._collection.count()} entries)")
        except Exception as e:
            logger.warning(f"[ClipboardMemory] ChromaDB init failed: {e}")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load_entries(self) -> List[Dict]:
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_entries(self, entries: List[Dict]):
        if len(entries) > MAX_ENTRIES:
            entries = entries[-MAX_ENTRIES:]
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"[ClipboardMemory] Failed to save: {e}")

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
            logger.debug(f"[ClipboardMemory] Embedding failed: {e}")
            return None

    def _store_in_chroma(self, entry_id: str, content: str, metadata: Dict):
        if not self._collection:
            return
        embedding = self._embed(content)
        if not embedding:
            return
        try:
            self._collection.upsert(
                ids=[entry_id],
                embeddings=[embedding],
                documents=[content[:4000]],
                metadatas=[metadata],
            )
        except Exception as e:
            logger.debug(f"[ClipboardMemory] Chroma upsert failed: {e}")

    # ------------------------------------------------------------------ #
    # Core logic
    # ------------------------------------------------------------------ #

    def _save_entry(self, content: str, label: Optional[str] = None) -> Dict:
        entry_id = str(uuid.uuid4())[:8]
        ts = datetime.now().isoformat()
        content_type = self._detect_type(content)
        entry = {
            "id": entry_id,
            "timestamp": ts,
            "content": content,
            "label": label or content_type,
            "content_type": content_type,
            "length": len(content),
        }
        with self._lock:
            entries = self._load_entries()
            entries.append(entry)
            self._save_entries(entries)
        meta = {k: v for k, v in entry.items() if k != "content"}
        self._store_in_chroma(entry_id, content, meta)
        return entry

    def _detect_type(self, content: str) -> str:
        s = content.strip()
        if s.startswith(("Traceback", "Error:", "Exception")):
            return "error"
        if s.startswith(("http://", "https://")):
            return "url"
        try:
            import json as _json
            _json.loads(s)
            return "json"
        except Exception:
            pass
        code_hints = ["def ", "class ", "import ", "const ", "function ", "SELECT ", "INSERT "]
        if any(h in content for h in code_hints):
            return "code"
        if s.startswith(("/", "C:\\", "D:\\")):
            return "path"
        return "text"

    # ------------------------------------------------------------------ #
    # Background monitor
    # ------------------------------------------------------------------ #

    def _monitor_loop(self):
        logger.info("[ClipboardMemory] Monitor started")
        while not self._stop_event.is_set():
            try:
                if PYPERCLIP_AVAILABLE:
                    current = pyperclip.paste()
                    if (
                        current
                        and current != self._last_content
                        and len(current) >= MIN_CONTENT_LENGTH
                    ):
                        self._last_content = current
                        entry = self._save_entry(current)
                        logger.debug(f"[ClipboardMemory] Saved [{entry['content_type']}] {len(current)} chars")
            except Exception as e:
                logger.debug(f"[ClipboardMemory] Monitor error: {e}")
            self._stop_event.wait(POLL_INTERVAL)
        logger.info("[ClipboardMemory] Monitor stopped")

    def start_monitor(self) -> Dict:
        """Start background clipboard monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return {"success": True, "message": "Monitor already running"}
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="clipboard-memory-monitor"
        )
        self._monitor_thread.start()
        return {"success": True, "message": "Clipboard memory monitor started — all copies will be indexed"}

    def stop_monitor(self) -> Dict:
        """Stop background clipboard monitoring."""
        self._stop_event.set()
        return {"success": True, "message": "Clipboard memory monitor stopped"}

    def is_monitoring(self) -> bool:
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def save_current(self, label: Optional[str] = None) -> Dict:
        """Save current clipboard content to memory with optional label."""
        if not PYPERCLIP_AVAILABLE:
            return {"success": False, "error": "pyperclip not available"}
        content = pyperclip.paste()
        if not content or len(content) < MIN_CONTENT_LENGTH:
            return {"success": False, "error": "Clipboard is empty or too short"}
        entry = self._save_entry(content, label)
        return {
            "success": True,
            "id": entry["id"],
            "label": entry["label"],
            "content_type": entry["content_type"],
            "length": entry["length"],
            "preview": content[:200],
        }

    def recall(self, query: str, top_k: int = 5) -> Dict:
        """Semantic search over past clipboard entries.

        Args:
            query: What to search for (e.g. 'JSON config', 'error traceback', 'GitHub URL')
            top_k: Number of results to return
        """
        if not self._collection:
            return self._fallback_text_search(query, top_k)
        embedding = self._embed(query)
        if not embedding:
            return self._fallback_text_search(query, top_k)
        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, self._collection.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
            hits = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                hits.append({
                    "id": meta.get("id"),
                    "timestamp": meta.get("timestamp"),
                    "label": meta.get("label"),
                    "content_type": meta.get("content_type"),
                    "relevance": round(1 - distance, 3),
                    "preview": doc[:300],
                    "full_content": doc,
                })
            return {
                "success": True,
                "query": query,
                "results": hits,
                "count": len(hits),
            }
        except Exception as e:
            logger.warning(f"[ClipboardMemory] Recall failed: {e}")
            return self._fallback_text_search(query, top_k)

    def _fallback_text_search(self, query: str, top_k: int) -> Dict:
        """Simple keyword search fallback when ChromaDB unavailable."""
        entries = self._load_entries()
        query_lower = query.lower()
        matches = [
            e for e in reversed(entries)
            if query_lower in e.get("content", "").lower()
            or query_lower in e.get("label", "").lower()
        ][:top_k]
        return {
            "success": True,
            "query": query,
            "mode": "keyword_fallback",
            "results": [
                {
                    "id": e["id"],
                    "timestamp": e["timestamp"],
                    "label": e["label"],
                    "content_type": e["content_type"],
                    "preview": e["content"][:300],
                    "full_content": e["content"],
                }
                for e in matches
            ],
            "count": len(matches),
        }

    def list_recent(self, limit: int = 10) -> Dict:
        """List most recent clipboard entries."""
        entries = self._load_entries()
        recent = list(reversed(entries[-limit:]))
        return {
            "success": True,
            "entries": [
                {
                    "id": e["id"],
                    "timestamp": e["timestamp"],
                    "label": e["label"],
                    "content_type": e["content_type"],
                    "length": e.get("length", len(e.get("content", ""))),
                    "preview": e["content"][:100],
                }
                for e in recent
            ],
            "total_stored": len(entries),
            "monitoring": self.is_monitoring(),
        }

    def clear_history(self) -> Dict:
        """Clear all stored clipboard memory."""
        self._save_entries([])
        if self._collection:
            try:
                self._chroma_client.delete_collection(CHROMA_COLLECTION)
                self._collection = self._chroma_client.get_or_create_collection(
                    name=CHROMA_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                pass
        return {"success": True, "message": "Clipboard memory cleared"}

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a clipboard memory action.

        Actions: start_monitor, stop_monitor, save, recall, list, clear
        """
        a = action.lower().strip()
        if "start" in a or "monitor" in a and "stop" not in a:
            return self.start_monitor()
        if "stop" in a:
            return self.stop_monitor()
        if "recall" in a or "search" in a or "find" in a or "remember" in a:
            query = kwargs.get("query") or kwargs.get("text") or action
            return self.recall(query)
        if "save" in a or "store" in a:
            return self.save_current(kwargs.get("label"))
        if "list" in a or "recent" in a or "history" in a:
            return self.list_recent(kwargs.get("limit", 10))
        if "clear" in a:
            return self.clear_history()
        return self.list_recent()
