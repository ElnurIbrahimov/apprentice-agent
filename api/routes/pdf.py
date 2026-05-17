"""
PDF text extraction via pdfplumber.
Supports file upload and URL extraction.
"""

import asyncio
import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.auth import require_api_key
from api.routes.upload import sanitize_filename
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pdf", tags=["pdf"], dependencies=[Depends(require_api_key)])


@router.post("/extract")
async def extract_upload(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF file."""
    try:
        import pdfplumber
    except ImportError:
        raise HTTPException(503, "pdfplumber not installed. Run: pip install pdfplumber") from None

    # 50 MB limit on uploaded PDFs — stream to avoid buffering entire file
    _MAX_UPLOAD_SIZE = 50 * 1024 * 1024
    chunks = []
    total = 0
    async for chunk in file.stream():
        total += len(chunk)
        if total > _MAX_UPLOAD_SIZE:
            raise HTTPException(413, f"PDF too large (>{_MAX_UPLOAD_SIZE} bytes)")
        chunks.append(chunk)
    data = b"".join(chunks)

    def _extract(raw: bytes):
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]

    try:
        loop = asyncio.get_running_loop()
        pages = await loop.run_in_executor(None, _extract, data)
    except Exception as e:
        raise HTTPException(500, safe_error_detail(e, "PDF extraction failed")) from e

    text = "\n\n".join(pages)
    return {
        "page_count": len(pages),
        "word_count": len(text.split()),
        "text": text[:80000],
        "filename": sanitize_filename(file.filename) if file.filename else "unknown.pdf",
    }


@router.post("/extract-url")
async def extract_url(body: dict):
    """Extract text from a PDF at a given URL."""
    try:
        import httpx
        import pdfplumber
    except ImportError:
            raise HTTPException(503, "Missing dependencies. Run: pip install pdfplumber httpx") from None

    url = body.get("url", "")
    if not url:
        raise HTTPException(400, "url is required")
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Only http:// and https:// URLs are allowed")

    # SSRF protection: resolve DNS once, validate IP, pin to resolved IP
    # This eliminates the TOCTOU gap between DNS check and HTTP request
    try:
        from aura.security.ssrf_guard import validate_url_safe
        pinned_url, original_hostname = validate_url_safe(url)
    except ValueError as e:
            raise HTTPException(400, f"Cannot fetch URL: {e}") from e

    _MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB
    try:
        headers = {}
        if original_hostname:
            headers["Host"] = original_hostname
        async with httpx.AsyncClient(timeout=30) as c:
            chunks = []
            total = 0
            async with c.stream("GET", pinned_url, follow_redirects=False, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_PDF_SIZE:
                        raise HTTPException(413, f"PDF too large (>{_MAX_PDF_SIZE} bytes)")
                    chunks.append(chunk)
            pdf_bytes = b"".join(chunks)
    except HTTPException:
        raise
    except Exception as e:
           raise HTTPException(500, safe_error_detail(e, "Failed to fetch PDF")) from e

    def _extract_bytes(raw: bytes):
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]

    try:
        loop = asyncio.get_running_loop()
        pages = await loop.run_in_executor(None, _extract_bytes, pdf_bytes)
    except Exception as e:
        raise HTTPException(500, safe_error_detail(e, "PDF extraction failed")) from e

    text = "\n\n".join(pages)
    return {
        "page_count": len(pages),
        "word_count": len(text.split()),
        "text": text[:80000],
        "url": url,
    }
