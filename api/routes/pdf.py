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

    # Block private/loopback IPs (SSRF protection including IPv6)
    from urllib.parse import urlparse
    import ipaddress
    _host = urlparse(url).hostname or ""

    # Reject obviously blocked hostnames first
    _blocked_prefixes = (
        "127.", "10.", "172.16.", "172.17.", "172.18.", "172.19.",
        "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
        "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
        "192.168.", "169.254.", "0.", "localhost",
    )
    if any(_host.startswith(b) for b in _blocked_prefixes) or _host == "localhost":
        raise HTTPException(400, "Cannot fetch from private/loopback addresses")

    # Resolve the hostname to an IP and check it — catches IPv6 loopback,
    # IPv4-mapped IPv6 (::ffff:127.0.0.1), link-local, and other bypasses
    try:
        import socket
        infos = socket.getaddrinfo(_host, None, type=socket.SOCK_STREAM)
        for _family, _type, _proto, _canon, sockaddr in infos:
            addr = ipaddress.ip_address(sockaddr[0])
            if addr.is_loopback or addr.is_private or addr.is_reserved or addr.is_link_local:
                raise HTTPException(400, "Cannot fetch from private/loopback addresses")
    except HTTPException:
        raise
    except Exception:
        # If DNS resolution fails, let httpx handle it below
        pass

    _MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            chunks = []
            total = 0
            async with c.stream("GET", url, follow_redirects=False) as resp:
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
        raise HTTPException(500, f"Failed to fetch PDF: {e}")

    def _extract_bytes(raw: bytes):
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]

    try:
        loop = asyncio.get_running_loop()
        pages = await loop.run_in_executor(None, _extract_bytes, pdf_bytes)
    except Exception as e:
        raise HTTPException(500, f"PDF extraction failed: {e}")

    text = "\n\n".join(pages)
    return {
        "page_count": len(pages),
        "word_count": len(text.split()),
        "text": text[:80000],
        "url": url,
    }
