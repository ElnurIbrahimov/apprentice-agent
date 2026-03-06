"""
OCR via pytesseract + Pillow.
Requires: pip install pytesseract pillow
          + winget install UB-Mannheim.TesseractOCR  (binary)
"""

import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ocr"])


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

    try:
        import base64
        import io
        from PIL import Image

        img_data = base64.b64decode(image_b64.split(",")[-1])
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img, lang=body.get("lang", "eng")).strip()
        return {"text": text}
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")
