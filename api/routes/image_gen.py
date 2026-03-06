"""
Image generation via ComfyUI (local, port 8188).
ComfyUI must be running: cd ComfyUI && python main.py --port 8188
"""

import asyncio
import base64
import logging
from typing import Optional

import requests as req
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/image", tags=["image"])

COMFY = "http://localhost:8188"


class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = 20


def build_sdxl_workflow(prompt: str, negative_prompt: str, steps: int) -> dict:
    """Minimal ComfyUI SD1.5 txt2img workflow."""
    neg = negative_prompt or "blurry, low quality, deformed"
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42, "steps": steps, "cfg": 7,
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
        req.get(f"{COMFY}/system_stats", timeout=2).raise_for_status()
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
        r = req.post(f"{COMFY}/prompt", json={"prompt": workflow})
        pid = r.json()["prompt_id"]
    except Exception as e:
        raise HTTPException(500, f"Failed to queue prompt: {e}")

    # Poll for completion (max 120s)
    for _ in range(120):
        await asyncio.sleep(1)
        try:
            hist = req.get(f"{COMFY}/history/{pid}").json()
            if pid in hist:
                outputs = hist[pid].get("outputs", {})
                if outputs:
                    fname = list(outputs.values())[0]["images"][0]["filename"]
                    img_bytes = req.get(f"{COMFY}/view?filename={fname}").content
                    b64 = base64.b64encode(img_bytes).decode()
                    return {"image_b64": b64}
        except Exception:
            pass

    raise HTTPException(504, "Image generation timed out after 120s")
