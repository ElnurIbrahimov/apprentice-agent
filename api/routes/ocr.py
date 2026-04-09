"""
OCR via pytesseract + Pillow.
Requires: pip install pytesseract pillow
          + winget install UB-Mannheim.TesseractOCR  (binary)
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ocr"], dependencies=[Depends(require_api_key)])


@router.post("/ocr")
async def ocr_image(body: dict):
    """Extract text from a base64-encoded image using Tesseract OCR."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        raise HTTPException(
            503,
            "Tesseract not found. Install it: winget install UB-Mannheim.TesseractOCR  "
            "(then restart the server)",
        )

    image_b64 = body.get("image_b64", "")
    if not image_b64:
        raise HTTPException(400, "image_b64 is required")
    # 20 MB base64 ≈ 15 MB decoded image — sufficient for high-res OCR
    if len(image_b64) > 20 * 1024 * 1024:
        raise HTTPException(400, "image_b64 exceeds maximum size of 20 MB")

    try:
        import base64
        import io

        from PIL import Image

        def _decode_and_ocr():
            import re as _re
            img_data = base64.b64decode(image_b64.split(",")[-1])
            img = Image.open(io.BytesIO(img_data))
            lang = body.get("lang", "eng")
            # Validate lang to prevent path traversal via Tesseract's traineddata lookup
            if not _re.match(r'^[a-z]{2,4}(\+[a-z]{2,4})*$', lang):
                lang = "eng"
            return pytesseract.image_to_string(img, lang=lang).strip()

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _decode_and_ocr)
        return {"text": text}
    except Exception as e:
        raise HTTPException(500, safe_error_detail(e, "OCR failed"))
