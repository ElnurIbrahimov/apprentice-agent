"""Share & deploy endpoints — upload project files and get a shareable URL."""

from __future__ import annotations

import json
import mimetypes
import os
import secrets
import sqlite3
import string
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from api.auth import require_api_key

router = APIRouter(tags=["share"])

# ── Config ──
DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "data"
SHARED_DIR = DATA_DIR / "shared"
DB_PATH = DATA_DIR / "shares.db"
MAX_SHARES = 50
MAX_PROJECT_BYTES = 5 * 1024 * 1024  # 5MB
MAX_FILES = 100
DEFAULT_EXPIRY_DAYS = 7
BASE_URL = os.environ.get("AURA_BASE_URL", "")

SHARED_DIR.mkdir(parents=True, exist_ok=True)

# ── Database ──
_db_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shares (
            id TEXT PRIMARY KEY,
            project_name TEXT NOT NULL,
            entry_point TEXT NOT NULL DEFAULT 'index.html',
            file_count INTEGER NOT NULL DEFAULT 0,
            total_bytes INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _cleanup_expired() -> int:
    """Delete expired shares from disk and database."""
    now = time.time()
    with _db_lock:
        db = _get_db()
        try:
            rows = db.execute("SELECT id FROM shares WHERE expires_at < ?", (now,)).fetchall()
            count = 0
            for row in rows:
                share_dir = SHARED_DIR / row["id"]
                if share_dir.exists():
                    import shutil
                    shutil.rmtree(share_dir, ignore_errors=True)
                db.execute("DELETE FROM shares WHERE id = ?", (row["id"],))
                count += 1
            db.commit()
            return count
        finally:
            db.close()


def _generate_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _count_active_shares() -> int:
    with _db_lock:
        db = _get_db()
        try:
            row = db.execute("SELECT COUNT(*) as cnt FROM shares WHERE expires_at > ?", (time.time(),)).fetchone()
            return row["cnt"] if row else 0
        finally:
            db.close()


def _safe_path(base: Path, user_path: str) -> Path:
    """Resolve path safely — prevent directory traversal."""
    normalized = user_path.replace("\\", "/").lstrip("/")
    parts = normalized.split("/")
    if any(p in {".", ".."} or not p for p in parts):
        raise ValueError(f"Invalid path: {user_path}")
    resolved = (base / normalized).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError(f"Path traversal detected: {user_path}")
    return resolved


# ── Schemas ──
class ShareRequest(BaseModel):
    project_name: str = Field(default="Untitled Project", max_length=200)
    files: Dict[str, str] = Field(..., min_length=1)
    entry_point: str = Field(default="index.html", max_length=300)
    expires_days: int = Field(default=DEFAULT_EXPIRY_DAYS, ge=1, le=30)


class ShareResponse(BaseModel):
    url: str
    id: str
    project_name: str
    file_count: int
    expires_at: float


class ShareInfo(BaseModel):
    id: str
    project_name: str
    entry_point: str
    file_count: int
    total_bytes: int
    created_at: float
    expires_at: float
    url: str


# ── Endpoints ──

@router.post("/api/share", dependencies=[Depends(require_api_key)])
async def share_project(request: ShareRequest) -> ShareResponse:
    """Upload project files and get a shareable URL."""
    # Cleanup expired first
    _cleanup_expired()

    # Validate entry_point proactively (also validated on serve via _safe_path)
    try:
        _safe_path(SHARED_DIR / "test", request.entry_point)
    except ValueError:
        raise HTTPException(400, f"Invalid entry_point: {request.entry_point}")

    # Limits
    if len(request.files) > MAX_FILES:
        raise HTTPException(400, f"Too many files (max {MAX_FILES})")
    if _count_active_shares() >= MAX_SHARES:
        raise HTTPException(429, f"Too many active shares (max {MAX_SHARES}). Delete some first.")

    total_bytes = sum(len(v.encode("utf-8")) for v in request.files.values())
    if total_bytes > MAX_PROJECT_BYTES:
        raise HTTPException(400, f"Project too large ({total_bytes} bytes, max {MAX_PROJECT_BYTES})")

    # Generate unique ID
    share_id = _generate_id()
    while (SHARED_DIR / share_id).exists():
        share_id = _generate_id()

    share_dir = SHARED_DIR / share_id
    now = time.time()
    expires_at = now + (request.expires_days * 86400)

    # Write files
    try:
        for path, content in request.files.items():
            file_path = _safe_path(share_dir, path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
    except ValueError as e:
        # Clean up partial writes
        import shutil
        shutil.rmtree(share_dir, ignore_errors=True)
        raise HTTPException(400, str(e))

    # Record in DB
    with _db_lock:
        db = _get_db()
        try:
            db.execute(
                "INSERT INTO shares (id, project_name, entry_point, file_count, total_bytes, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (share_id, request.project_name, request.entry_point, len(request.files), total_bytes, now, expires_at),
            )
            db.commit()
        finally:
            db.close()

    base_url = BASE_URL.rstrip("/") if BASE_URL else ""
    url = f"{base_url}/shared/{share_id}/"

    return ShareResponse(
        url=url,
        id=share_id,
        project_name=request.project_name,
        file_count=len(request.files),
        expires_at=expires_at,
    )


@router.get("/api/shares", dependencies=[Depends(require_api_key)])
async def list_shares() -> List[ShareInfo]:
    """List all active (non-expired) shares."""
    _cleanup_expired()
    base_url = BASE_URL.rstrip("/") if BASE_URL else ""
    with _db_lock:
        db = _get_db()
        try:
            rows = db.execute(
                "SELECT * FROM shares WHERE expires_at > ? ORDER BY created_at DESC",
                (time.time(),),
            ).fetchall()
            return [
                ShareInfo(
                    id=row["id"],
                    project_name=row["project_name"],
                    entry_point=row["entry_point"],
                    file_count=row["file_count"],
                    total_bytes=row["total_bytes"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    url=f"{base_url}/shared/{row['id']}/",
                )
                for row in rows
            ]
        finally:
            db.close()


@router.delete("/api/shares/{share_id}", dependencies=[Depends(require_api_key)])
async def delete_share(share_id: str):
    """Delete a share by ID."""
    share_dir = SHARED_DIR / share_id
    if share_dir.exists():
        import shutil
        shutil.rmtree(share_dir, ignore_errors=True)
    with _db_lock:
        db = _get_db()
        try:
            db.execute("DELETE FROM shares WHERE id = ?", (share_id,))
            db.commit()
        finally:
            db.close()
    return {"deleted": share_id}


@router.get("/shared/{share_id}/{path:path}")
async def serve_shared_file(share_id: str, path: str = ""):
    """Serve shared project files as static content. No auth required."""
    share_dir = SHARED_DIR / share_id
    if not share_dir.exists():
        raise HTTPException(404, "Share not found")

    # Check expiry
    with _db_lock:
        db = _get_db()
        try:
            row = db.execute("SELECT * FROM shares WHERE id = ?", (share_id,)).fetchone()
        finally:
            db.close()

    if not row:
        raise HTTPException(404, "Share not found")
    if row["expires_at"] < time.time():
        raise HTTPException(410, "Share has expired")

    # Default to entry point
    if not path or path == "/":
        path = row["entry_point"]

    try:
        file_path = _safe_path(share_dir, path)
    except ValueError:
        raise HTTPException(400, "Invalid path")

    if not file_path.exists():
        raise HTTPException(404, "File not found")
    if not file_path.is_file():
        raise HTTPException(404, "Not a file")

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = "application/octet-stream"

    return FileResponse(file_path, media_type=mime_type)


@router.get("/shared/{share_id}")
async def serve_shared_root(share_id: str):
    """Redirect to the entry point (handles URLs without trailing slash)."""
    return await serve_shared_file(share_id, "")
