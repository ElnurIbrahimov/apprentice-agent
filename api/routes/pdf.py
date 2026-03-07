"""
PDF text extraction via pdfplumber.
Supports file upload and URL extraction.
"""

import asyncio
import io
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pdf", tags=["pdf"])


@router.post("/extract")
async def extract_upload(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF file."""
    try:
        import pdfplumber
    except ImportError:
        raise HTTPException(503, "pdfplumber not installed. Run: pip install pdfplumber")

    data = await file.read()

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

    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(url, follow_redirects=True)
        r.raise_for_status()
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
