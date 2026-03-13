"""
PDF text extraction via pdfplumber.
Supports file upload and URL extraction.
"""

import asyncio
import io
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pdf", tags=["pdf"], dependencies=[Depends(require_api_key)])


@router.post("/extract")
async def extract_upload(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF file."""
    try:
        import pdfplumber
    except ImportError:
        raise HTTPException(503, "pdfplumber not installed. Run: pip install pdfplumber")

    data = await file.read()

    # 50 MB limit on uploaded PDFs
    _MAX_UPLOAD_SIZE = 50 * 1024 * 1024
    if len(data) > _MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"PDF too large ({len(data)} bytes, max {_MAX_UPLOAD_SIZE})")

    def _extract(raw: bytes):
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]

    try:
        loop = asyncio.get_running_loop()
        pages = await loop.run_in_executor(None, _extract, data)
    except Exception as e:
        raise HTTPException(500, f"PDF extraction failed: {e}")

    text = "\n\n".join(pages)
    return {
        "page_count": len(pages),
        "word_count": len(text.split()),
        "text": text[:80000],
        "filename": file.filename,
    }


@router.post("/extract-url")
async def extract_url(body: dict):
    """Extract text from a PDF at a given URL."""
    try:
        import pdfplumber
        import httpx
    except ImportError:
        raise HTTPException(503, "Missing dependencies. Run: pip install pdfplumber httpx")

    url = body.get("url", "")
    if not url:
        raise HTTPException(400, "url is required")
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Only http:// and https:// URLs are allowed")

    # Block private/loopback IPs (basic SSRF protection)
    from urllib.parse import urlparse
    _host = urlparse(url).hostname or ""
    _blocked = ("127.", "10.", "172.16.", "172.17.", "172.18.", "172.19.",
                "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
                "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
                "192.168.", "169.254.", "0.", "localhost", "[::1]")
    if any(_host.startswith(b) for b in _blocked) or _host == "localhost":
        raise HTTPException(400, "Cannot fetch from private/loopback addresses")

    _MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(url, follow_redirects=True)
        r.raise_for_status()
        if len(r.content) > _MAX_PDF_SIZE:
            raise HTTPException(413, f"PDF too large ({len(r.content)} bytes, max {_MAX_PDF_SIZE})")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch PDF: {e}")

    def _extract_bytes(raw: bytes):
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]

    try:
        loop = asyncio.get_running_loop()
        pages = await loop.run_in_executor(None, _extract_bytes, r.content)
    except Exception as e:
        raise HTTPException(500, f"PDF extraction failed: {e}")

    text = "\n\n".join(pages)
    return {
        "page_count": len(pages),
        "word_count": len(text.split()),
        "text": text[:80000],
    }
