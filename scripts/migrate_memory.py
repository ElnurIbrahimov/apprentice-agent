#!/usr/bin/env python3
"""Migrate all existing memory data to the consolidated SQLite store.

Phases:
  A: MemorySystem SQLite (data/chromadb/agent_memory.db) → memories table
  B: A-MEM notes.jsonl (data/amem/notes.jsonl) → memories table
  C: Episodic Qdrant (aura_data/episodic_memory/) → memories table
  D: RAG chunks (data/chromadb/chroma.sqlite3) → memories table
  E: Dream insights (data/chromadb/dream_insights.db) → memories table
  F: Background re-embedding (nomic-embed-text, 768 dims)

Idempotent: uses INSERT OR IGNORE (ID-based dedup).
Non-destructive: source data is never modified.

Usage:
  python scripts/migrate_memory.py [--re-embed] [--dry-run]

Author: Aura Development Team
Created: 2026-03-16
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_memory")


def get_store():
    """Get the consolidated MemoryStore."""
    from aura.memory.store import get_memory_store
    return get_memory_store()


def get_embedding(text: str):
    """Get embedding via Ollama nomic-embed-text."""
    try:
        import numpy as np
        import requests
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text:latest", "prompt": text[:1000]},
            timeout=5,
        )
        if r.status_code == 200:
            emb = r.json().get("embedding")
            if emb:
                return np.array(emb, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Embedding failed: {e}")
    return None


# ==========================================================================
# Phase A: MemorySystem SQLite → memories table
# ==========================================================================

def migrate_memory_system(store, dry_run=False):
    """Migrate from data/chromadb/agent_memory.db."""
    db_path = PROJECT_ROOT / "data" / "chromadb" / "agent_memory.db"
    if not db_path.exists():
        logger.info("[Phase A] No MemorySystem DB found at %s — skipping", db_path)
        return 0

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT id, content, type, timestamp, metadata, embedding FROM memories").fetchall()
    conn.close()

    if not rows:
        logger.info("[Phase A] MemorySystem DB is empty — skipping")
        return 0

    logger.info("[Phase A] Migrating %d MemorySystem records...", len(rows))
    if dry_run:
        return len(rows)

    from aura.memory.store import MemoryRecord
    import numpy as np

    records = []
    embeddings = []
    for row_id, content, mem_type, ts, meta_str, emb_str in rows:
        meta = {}
        try:
            meta = json.loads(meta_str or "{}")
        except json.JSONDecodeError:
            pass

        row_id = str(row_id)
        if not content or not content.strip():
            continue
        record = MemoryRecord(
            id=f"ms_{row_id}" if not row_id.startswith("ms_") else row_id,
            content=content,
            title=content[:80],
            source="memory_system",
            memory_type=mem_type or "episodic",
            importance=meta.get("importance", 0.5),
            created_at=ts or datetime.now().isoformat(),
            updated_at=ts or datetime.now().isoformat(),
            last_accessed=ts or datetime.now().isoformat(),
            metadata=meta_str or "{}",
        )
        records.append(record)

        # Convert JSON embedding to numpy if present
        emb = None
        if emb_str:
            try:
                emb_list = json.loads(emb_str)
                emb = np.array(emb_list, dtype=np.float32)
            except (json.JSONDecodeError, ValueError):
                pass
        embeddings.append(emb)

    count = store.batch_insert(records, embeddings)
    logger.info("[Phase A] Inserted %d / %d records", count, len(records))
    return count


# ==========================================================================
# Phase B: A-MEM notes.jsonl → memories table
# ==========================================================================

def migrate_amem(store, dry_run=False):
    """Migrate from data/amem/notes.jsonl."""
    notes_path = PROJECT_ROOT / "data" / "amem" / "notes.jsonl"
    if not notes_path.exists():
        logger.info("[Phase B] No A-MEM notes found at %s — skipping", notes_path)
        return 0

    notes = []
    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    notes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not notes:
        logger.info("[Phase B] A-MEM notes file is empty — skipping")
        return 0

    logger.info("[Phase B] Migrating %d A-MEM notes...", len(notes))
    if dry_run:
        return len(notes)

    from aura.memory.store import MemoryRecord
    import numpy as np

    # Try to load embeddings
    emb_path = PROJECT_ROOT / "data" / "amem" / "embeddings.npz"
    saved_embeddings = {}
    if emb_path.exists():
        try:
            data = np.load(str(emb_path), allow_pickle=True)
            for key in data.files:
                saved_embeddings[key] = data[key]
            logger.info("[Phase B] Loaded %d embeddings from npz", len(saved_embeddings))
        except Exception as e:
            logger.warning("[Phase B] Failed to load embeddings: %s", e)

    records = []
    embeddings = []
    for note in notes:
        note_id = note.get("id", "")
        if not note_id:
            continue
        if not note.get("content", "").strip():
            continue

        keywords = note.get("keywords", [])
        tags = note.get("tags", [])
        ts = note.get("created_at", note.get("timestamp", datetime.now().isoformat()))

        record = MemoryRecord(
            id=f"amem_{note_id}" if not note_id.startswith("amem_") else note_id,
            content=note.get("content", ""),
            title=note.get("content", "")[:80],
            source="amem",
            memory_type=note.get("category", "episodic"),
            importance=note.get("importance", 0.5),
            keywords=",".join(keywords) if isinstance(keywords, list) else str(keywords),
            tags=",".join(tags) if isinstance(tags, list) else str(tags),
            category=note.get("category", ""),
            boxes=json.dumps(note.get("boxes", [])),
            links=json.dumps(note.get("links", [])),
            access_count=note.get("access_count", 0),
            created_at=ts,
            updated_at=ts,
            last_accessed=ts,
            lifecycle_state="stable",
            metadata=json.dumps({k: v for k, v in note.items()
                                  if k not in ("id", "content", "keywords", "tags",
                                               "category", "importance", "boxes", "links",
                                               "access_count", "created_at", "timestamp")}),
        )
        records.append(record)

        # Use saved embedding if available, else None (will re-embed later)
        emb = saved_embeddings.get(note_id)
        embeddings.append(emb)

    count = store.batch_insert(records, embeddings)
    logger.info("[Phase B] Inserted %d / %d records", count, len(records))
    return count


# ==========================================================================
# Phase C: Episodic Qdrant → memories table
# ==========================================================================

def migrate_episodic(store, dry_run=False):
    """Migrate from aura_data/episodic_memory/."""
    ep_path = PROJECT_ROOT / "aura_data" / "episodic_memory"
    if not ep_path.exists():
        logger.info("[Phase C] No episodic memory found at %s — skipping", ep_path)
        return 0

    # Try to load via the episodic store
    try:
        from aura_episodic_memory import EpisodicMemoryStore
        from aura_episodic_memory.episode import EpisodeQuery
        ep_store = EpisodicMemoryStore(str(ep_path))
    except Exception as e:
        logger.info("[Phase C] Cannot load episodic store: %s — skipping", e)
        return 0

    # Search broadly to get all episodes
    try:
        query = EpisodeQuery(query_text="*", limit=1000)
        results = ep_store.search(query)
    except Exception as e:
        logger.info("[Phase C] Episodic search failed: %s — trying recent", e)
        try:
            results = ep_store.get_recent(n=1000)
        except Exception:
            logger.info("[Phase C] Cannot retrieve episodic memories — skipping")
            return 0

    if not results:
        logger.info("[Phase C] Episodic store is empty — skipping")
        return 0

    logger.info("[Phase C] Migrating %d episodic memories...", len(results))
    if dry_run:
        return len(results)

    from aura.memory.store import MemoryRecord

    records = []
    for sr in results:
        ep = sr.episode if hasattr(sr, 'episode') else sr
        ep_id = getattr(ep, 'id', '')
        if not ep_id:
            continue

        ts = datetime.now().isoformat()
        try:
            tc = getattr(ep, 'temporal_context', None)
            if tc and hasattr(tc, 'timestamp'):
                ts = tc.timestamp.isoformat() if hasattr(tc.timestamp, 'isoformat') else str(tc.timestamp)
        except Exception:
            pass

        record = MemoryRecord(
            id=f"ep_{ep_id}" if not ep_id.startswith("ep_") else ep_id,
            content=getattr(ep, 'content', ''),
            title=(getattr(ep, 'title', '') or '')[:80],
            source="episodic",
            memory_type=str(getattr(ep, 'episode_type', 'conversation')),
            importance=getattr(ep, 'importance', 0.5),
            emotional_valence=str(getattr(ep, 'emotional_valence', 'neutral')),
            temporal_context=json.dumps({"timestamp": ts}),
            created_at=ts,
            updated_at=ts,
            last_accessed=ts,
            lifecycle_state="stable",
            metadata=json.dumps(getattr(ep, 'metadata', {}) or {}),
        )
        records.append(record)

    count = store.batch_insert(records)
    logger.info("[Phase C] Inserted %d / %d records", count, len(records))

    try:
        ep_store.close()
    except Exception:
        pass
    return count


# ==========================================================================
# Phase D: RAG chunks (ChromaDB) → memories table
# ==========================================================================

def migrate_rag(store, dry_run=False):
    """Migrate from data/chromadb/chroma.sqlite3 (ChromaDB)."""
    chroma_path = PROJECT_ROOT / "data" / "chromadb" / "chroma.sqlite3"
    if not chroma_path.exists():
        logger.info("[Phase D] No ChromaDB found at %s — skipping", chroma_path)
        return 0

    try:
        conn = sqlite3.connect(str(chroma_path))
        # ChromaDB stores documents in embeddings_queue (WAL) or embedding_metadata
        # Try embeddings_queue first (has document column directly)
        rows = []
        try:
            rows = conn.execute("""
                SELECT id, document, metadata
                FROM embeddings_queue
                WHERE document IS NOT NULL AND document != ''
                AND operation != 3
            """).fetchall()
        except sqlite3.OperationalError:
            pass
        # Fallback: try segments_metadata for _document key
        if not rows:
            try:
                rows = conn.execute("""
                    SELECT DISTINCT em.id, em.string_value, ''
                    FROM embedding_metadata em
                    WHERE em.key = 'chroma:document'
                    AND em.string_value IS NOT NULL AND em.string_value != ''
                """).fetchall()
            except sqlite3.OperationalError:
                pass
        conn.close()
    except Exception as e:
        logger.info("[Phase D] ChromaDB extraction failed: %s — skipping", e)
        return 0

    if not rows:
        logger.info("[Phase D] ChromaDB is empty — skipping")
        return 0

    logger.info("[Phase D] Migrating %d RAG chunks...", len(rows))
    if dry_run:
        return len(rows)

    from aura.memory.store import MemoryRecord

    records = []
    for row_id, document, meta_str in rows:
        meta = {}
        try:
            meta = json.loads(meta_str or "{}")
        except json.JSONDecodeError:
            pass

        record = MemoryRecord(
            id=f"rag_{row_id}" if not str(row_id).startswith("rag_") else str(row_id),
            content=document,
            title=document[:80],
            source="rag_chunk",
            memory_type="semantic",
            importance=0.4,
            created_at=meta.get("timestamp", datetime.now().isoformat()),
            updated_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            lifecycle_state="stable",
            metadata=meta_str or "{}",
        )
        records.append(record)

    count = store.batch_insert(records)
    logger.info("[Phase D] Inserted %d / %d records", count, len(records))
    return count


# ==========================================================================
# Phase E: Dream insights → memories table
# ==========================================================================

def migrate_dream_insights(store, dry_run=False):
    """Migrate from data/chromadb/dream_insights.db."""
    db_path = PROJECT_ROOT / "data" / "chromadb" / "dream_insights.db"
    if not db_path.exists():
        logger.info("[Phase E] No Dream insights DB found at %s — skipping", db_path)
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT id, content, type, timestamp, metadata FROM memories").fetchall()
    except sqlite3.OperationalError:
        logger.info("[Phase E] Dream insights DB has no memories table — skipping")
        conn.close()
        return 0
    conn.close()

    if not rows:
        logger.info("[Phase E] Dream insights DB is empty — skipping")
        return 0

    logger.info("[Phase E] Migrating %d dream insights...", len(rows))
    if dry_run:
        return len(rows)

    from aura.memory.store import MemoryRecord

    records = []
    for row_id, content, mem_type, ts, meta_str in rows:
        row_id = str(row_id)
        if not content or not content.strip():
            continue
        record = MemoryRecord(
            id=f"dream_{row_id}" if not row_id.startswith("dream_") else row_id,
            content=content,
            title=content[:80],
            source="dream_consolidation",
            memory_type="insight",
            importance=0.6,
            created_at=ts or datetime.now().isoformat(),
            updated_at=ts or datetime.now().isoformat(),
            last_accessed=ts or datetime.now().isoformat(),
            lifecycle_state="summary",
            metadata=meta_str or "{}",
        )
        records.append(record)

    count = store.batch_insert(records)
    logger.info("[Phase E] Inserted %d / %d records", count, len(records))
    return count


# ==========================================================================
# Phase F: Re-embedding
# ==========================================================================

def re_embed(store, batch_size=50):
    """Re-embed all memories that lack embeddings."""
    import numpy as np

    # Preflight: verify Ollama is reachable
    test_emb = get_embedding("test connection")
    if test_emb is None:
        logger.warning("[Phase F] Ollama not reachable — skipping re-embedding")
        return 0

    ids_without = store.get_all_ids_with_embeddings(has_embedding=False)
    if not ids_without:
        logger.info("[Phase F] All memories have embeddings — nothing to do")
        return 0

    logger.info("[Phase F] Re-embedding %d memories...", len(ids_without))
    embedded = 0

    for i in range(0, len(ids_without), batch_size):
        batch_ids = ids_without[i:i + batch_size]
        for mem_id, _ in batch_ids:
            record = store.get(mem_id)
            if not record:
                continue
            emb = get_embedding(record.content)
            if emb is not None:
                store.update_embedding(mem_id, emb)
                embedded += 1

        logger.info("[Phase F] Progress: %d / %d embedded", embedded, len(ids_without))
        time.sleep(0.1)  # Brief pause to avoid hammering Ollama

    logger.info("[Phase F] Re-embedded %d / %d memories", embedded, len(ids_without))
    return embedded


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Migrate memory data to consolidated SQLite store")
    parser.add_argument("--re-embed", action="store_true", help="Run re-embedding pass (Phase F)")
    parser.add_argument("--dry-run", action="store_true", help="Count records without writing")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MEMORY MIGRATION — Consolidated SQLite Store")
    logger.info("=" * 60)

    store = get_store()
    totals = {}

    # Phase A
    totals["memory_system"] = migrate_memory_system(store, dry_run=args.dry_run)

    # Phase B
    totals["amem"] = migrate_amem(store, dry_run=args.dry_run)

    # Phase C
    totals["episodic"] = migrate_episodic(store, dry_run=args.dry_run)

    # Phase D
    totals["rag"] = migrate_rag(store, dry_run=args.dry_run)

    # Phase E
    totals["dream"] = migrate_dream_insights(store, dry_run=args.dry_run)

    # Phase F
    if args.re_embed and not args.dry_run:
        totals["re_embedded"] = re_embed(store)
    else:
        totals["re_embedded"] = 0

    logger.info("=" * 60)
    logger.info("MIGRATION COMPLETE")
    for phase, count in totals.items():
        logger.info("  %-20s: %d records", phase, count)
    logger.info("  %-20s: %d total", "TOTAL", sum(totals.values()))

    # Print store stats
    if not args.dry_run:
        stats = store.get_stats()
        logger.info("\nStore stats: %s", json.dumps(stats, indent=2, default=str))

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
