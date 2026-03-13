"""File upload endpoint for AURA Web API."""

import os
import re
import uuid
import logging
import unicodedata
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from api.auth import require_api_key
from api.utils import safe_error_detail

from api.models.schemas import UploadResponse, FileAttachment, AttachmentType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024    # 10MB for images/code/documents
MAX_ARCHIVE_SIZE = 50 * 1024 * 1024  # 50MB for zip archives
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"

# Supported file types
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".json"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php", ".sh", ".yaml", ".yml", ".toml", ".xml", ".sql"}
ARCHIVE_EXTENSIONS = {".zip"}

# MIME type mapping
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".tsx": "text/typescript-jsx",
    ".jsx": "text/javascript-jsx",
    ".html": "text/html",
    ".css": "text/css",
    ".java": "text/x-java",
    ".c": "text/x-c",
    ".cpp": "text/x-c++",
    ".h": "text/x-c",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".rb": "text/x-ruby",
    ".php": "text/x-php",
    ".sh": "text/x-shellscript",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/toml",
    ".xml": "text/xml",
    ".sql": "text/x-sql",
    ".zip": "application/zip",
}

ALL_EXTENSIONS = IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS | CODE_EXTENSIONS | ARCHIVE_EXTENSIONS


def get_attachment_type(ext: str) -> AttachmentType:
    """Determine attachment type from file extension."""
    ext = ext.lower()
    if ext in IMAGE_EXTENSIONS:
        return AttachmentType.IMAGE
    elif ext in DOCUMENT_EXTENSIONS:
        return AttachmentType.DOCUMENT
    elif ext in ARCHIVE_EXTENSIONS:
        return AttachmentType.ARCHIVE
    else:
        return AttachmentType.CODE


def sanitize_filename(name: str) -> str:
    """Sanitize an uploaded filename to prevent path traversal and encoding tricks.

    Strips directory components, null bytes, Unicode control characters,
    and collapses whitespace.  Returns a safe basename.
    """
    # Strip null bytes
    name = name.replace("\x00", "")
    # Normalize Unicode (NFC) to collapse look-alike chars
    name = unicodedata.normalize("NFC", name)
    # Remove any directory separators the client may have sent
    name = name.replace("/", "_").replace("\\", "_")
    # Take only the final path component (handles remaining tricks)
    name = os.path.basename(name)
    # Remove control characters (categories Cc, Cf) except normal space
    name = "".join(ch for ch in name if unicodedata.category(ch) not in ("Cc", "Cf"))
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Fallback if nothing useful remains
    if not name or name.startswith("."):
        name = "upload" + name
    return name


def ensure_upload_dir():
    """Ensure upload directory exists."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a file for chat context.

    Accepts images (PNG, JPG, etc.), documents (PDF, TXT, MD),
    and code files (PY, JS, TS, etc.).

    Args:
        file: The file to upload (multipart/form-data)

    Returns:
        UploadResponse with attachment metadata
    """
    try:
        # Validate filename
        if not file.filename:
            return UploadResponse(
                success=False,
                error="No filename provided"
            )

        # Sanitize filename (path separators, null bytes, Unicode tricks)
        original_name = sanitize_filename(file.filename)

        # Enforce file extension whitelist
        ext = Path(original_name).suffix.lower()

        if ext not in ALL_EXTENSIONS:
            return UploadResponse(
                success=False,
                error=f"Unsupported file type: {ext}. Supported: images (png, jpg, gif, webp, bmp), documents (pdf, txt, md, json), code files (py, js, ts, etc.)"
            )

        # Check Content-Length header BEFORE reading full body into memory
        # (FastAPI/Starlette expose the raw ASGI scope for this)
        size_limit = MAX_ARCHIVE_SIZE if ext in ARCHIVE_EXTENSIONS else MAX_FILE_SIZE
        declared_size = file.size  # populated by python-multipart from Content-Length
        if declared_size is not None and declared_size > size_limit:
            return UploadResponse(
                success=False,
                error=f"File too large. Maximum size: {size_limit // (1024*1024)}MB"
            )

        # Stream-read in chunks to enforce limit without buffering entire file
        chunks = []
        bytes_read = 0
        while True:
            chunk = await file.read(1024 * 256)  # 256KB chunks
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > size_limit:
                return UploadResponse(
                    success=False,
                    error=f"File too large. Maximum size: {size_limit // (1024*1024)}MB"
                )
            chunks.append(chunk)
        content = b"".join(chunks)

        # Generate unique filename
        file_id = str(uuid.uuid4())
        unique_filename = f"{file_id}{ext}"

        # Ensure upload directory exists
        ensure_upload_dir()

        # Save file
        file_path = UPLOAD_DIR / unique_filename
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"[Upload] Saved file: {original_name} -> {unique_filename} ({len(content)} bytes)")

        # Determine attachment type and MIME type
        attachment_type = get_attachment_type(ext)
        mime_type = MIME_TYPES.get(ext, "application/octet-stream")

        attachment = FileAttachment(
            id=file_id,
            filename=original_name,
            mime_type=mime_type,
            size=len(content),
            type=attachment_type,
            path=file_path.name  # Only expose filename, not absolute path
        )

        return UploadResponse(
            success=True,
            attachment=attachment
        )

    except Exception as e:
        logger.error(f"[Upload] Error: {e}")
        return UploadResponse(
            success=False,
            error=safe_error_detail(e, "Upload failed")
        )


@router.delete("/{file_id}", dependencies=[Depends(require_api_key)])
async def delete_file(file_id: str):
    """Delete an uploaded file.

    Args:
        file_id: The UUID of the file to delete

    Returns:
        Success status
    """
    # Validate file_id is a valid UUID format
    if not re.match(r'^[0-9a-f-]{36}$', file_id):
        raise HTTPException(status_code=400, detail="Invalid file_id format. Must be a valid UUID.")

    try:
        # Find file with matching ID prefix
        for file_path in UPLOAD_DIR.glob(f"{file_id}.*"):
            file_path.unlink()
            logger.info(f"[Upload] Deleted file: {file_path}")
            return {"success": True, "message": "File deleted"}

        return {"success": False, "error": "File not found"}

    except Exception as e:
        logger.error(f"[Upload] Delete error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))
