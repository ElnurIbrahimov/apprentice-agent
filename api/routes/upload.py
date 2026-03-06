"""File upload endpoint for AURA Web API."""

import os
import re
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from api.auth import require_api_key

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

        # Get file extension
        original_name = file.filename
        ext = Path(original_name).suffix.lower()

        if ext not in ALL_EXTENSIONS:
            return UploadResponse(
                success=False,
                error=f"Unsupported file type: {ext}. Supported: images (png, jpg, gif, webp, bmp), documents (pdf, txt, md, json), code files (py, js, ts, etc.)"
            )

        # Read file content
        content = await file.read()

        # Validate file size (archives get a higher limit)
        size_limit = MAX_ARCHIVE_SIZE if ext in ARCHIVE_EXTENSIONS else MAX_FILE_SIZE
        if len(content) > size_limit:
            return UploadResponse(
                success=False,
                error=f"File too large. Maximum size: {size_limit // (1024*1024)}MB"
            )

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
            path=str(file_path.absolute())
        )

        return UploadResponse(
            success=True,
            attachment=attachment
        )

    except Exception as e:
        logger.error(f"[Upload] Error: {e}")
        return UploadResponse(
            success=False,
            error=str(e)
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
        raise HTTPException(status_code=500, detail=str(e))
