"""
Image generation via ComfyUI (local, port 8188).
ComfyUI must be running: cd ComfyUI && python main.py --port 8188
"""

import asyncio
import base64
import logging
import os
import random
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/image", tags=["image"], dependencies=[Depends(require_api_key)])

COMFY = os.getenv("COMFY_BASE_URL", "http://localhost:8188")


class ImageGenRequest(BaseModel):
    prompt: str = Field(..., max_length=1000)
    negative_prompt: Optional[str] = Field("", max_length=500)
    steps: Optional[int] = Field(20, ge=1, le=150)


def build_sdxl_workflow(prompt: str, negative_prompt: str, steps: int) -> dict:
    """Minimal ComfyUI SD1.5 txt2img workflow."""
    neg = negative_prompt or "blurry, low quality, deformed"
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": random.randint(0, 2**32 - 1), "steps": steps, "cfg": 7,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
                "model": ["4", 0], "positive": ["6", 0],
                "negative": ["7", 0], "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": 512, "width": 512},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": neg, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "aura", "images": ["8", 0]},
        },
    }


@router.post("/generate")
async def generate_image(body: ImageGenRequest):
    """Generate an image using ComfyUI. Returns base64 PNG."""
    # Check ComfyUI is running
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(f"{COMFY}/system_stats")
            r.raise_for_status()
    except Exception:
        raise HTTPException(
            503,
            {
                "error": "ComfyUI not running",
                "install": "cd ComfyUI && python main.py --port 8188",
            },
        )

    workflow = build_sdxl_workflow(
        body.prompt,
        body.negative_prompt or "",
        body.steps or 20,
    )

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{COMFY}/prompt", json={"prompt": workflow})
            pid = r.json()["prompt_id"]
    except Exception as e:
        raise HTTPException(500, safe_error_detail(e, "Failed to queue prompt"))

    # Poll for completion (max 120s) — async so event loop is not blocked
    async with httpx.AsyncClient(timeout=10) as c:
        for _ in range(120):
            await asyncio.sleep(1)
            try:
                hist_r = await c.get(f"{COMFY}/history/{pid}")
                hist = hist_r.json()
                if pid in hist:
                    outputs = hist[pid].get("outputs", {})
                    if outputs:
                        from urllib.parse import quote
                        fname = list(outputs.values())[0]["images"][0]["filename"]
                        img_r = await c.get(f"{COMFY}/view?filename={quote(fname)}")
                        b64 = base64.b64encode(img_r.content).decode()
                        return {"image_b64": b64}
            except Exception as poll_err:
                logger.debug("[ImageGen] Poll iteration error: %s", poll_err)

    raise HTTPException(504, "Image generation timed out after 120s")
