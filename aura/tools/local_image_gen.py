"""Local Image Generation Tool — generate images on your RTX 4060 using diffusers.

Supports multiple models with automatic VRAM management:
  - SD-XL Turbo (default): Fast, high quality, 8GB VRAM. Best for everyday use.
  - FLUX.1-schnell: Higher quality, uses CPU offload to fit in 8GB (slower).
  - SD 1.5: Lightweight fallback, 4GB VRAM.

Images are saved to Desktop/generated_images/ by default.
Models are downloaded and cached on first use (~4-10GB per model).

Config (optional .env):
    LOCAL_IMAGE_MODEL — default model: 'sdxl-turbo' | 'flux-schnell' | 'sd15'
    LOCAL_IMAGE_OUTPUT_DIR — output directory (default: Desktop/generated_images)
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lazy imports — only load torch/diffusers when actually generating
_pipeline = None
_current_model = None
_load_lock = threading.Lock()

DESKTOP = Path.home() / "Desktop"
DEFAULT_OUTPUT_DIR = DESKTOP / "generated_images"

MODEL_CONFIGS = {
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "description": "SDXL Turbo — fast high-quality images, 8GB VRAM",
        "default_steps": 1,
        "default_guidance": 0.0,
        "width": 512,
        "height": 512,
        "variant": "fp16",
        "cpu_offload": False,
    },
    "flux-schnell": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "description": "FLUX.1-schnell — highest quality, CPU offload for 8GB VRAM (slower first run)",
        "default_steps": 4,
        "default_guidance": 0.0,
        "width": 1024,
        "height": 1024,
        "variant": "bf16",
        "cpu_offload": True,
        "pipeline_class": "FluxPipeline",
    },
    "sd15": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "description": "SD 1.5 — lightweight, 4GB VRAM, wide LoRA support",
        "default_steps": 20,
        "default_guidance": 7.5,
        "width": 512,
        "height": 512,
        "variant": "fp16",
        "cpu_offload": False,
    },
}

DEFAULT_MODEL = os.getenv("LOCAL_IMAGE_MODEL", "sdxl-turbo")


def _get_output_dir() -> Path:
    out = Path(os.getenv("LOCAL_IMAGE_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_pipeline(model_key: str) -> tuple:
    """Load a diffusers pipeline. Returns (pipeline, error)."""
    global _pipeline, _current_model

    if _pipeline is not None and _current_model == model_key:
        return _pipeline, None

    config = MODEL_CONFIGS.get(model_key)
    if not config:
        return None, f"Unknown model '{model_key}'. Choose from: {', '.join(MODEL_CONFIGS.keys())}"

    try:
        import torch
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            AutoPipelineForText2Image,
        )

        if not torch.cuda.is_available():
            return None, "CUDA not available — GPU required for local image generation"

        logger.info(f"[LocalImageGen] Loading {model_key} ({config['model_id']}) ...")

        pipeline_class_name = config.get("pipeline_class", "")

        if pipeline_class_name == "FluxPipeline":
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained(
                config["model_id"],
                torch_dtype=torch.bfloat16,
            )
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                config["model_id"],
                torch_dtype=torch.float16,
                variant=config.get("variant", "fp16"),
                use_safetensors=True,
            )

        if config.get("cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            logger.info(f"[LocalImageGen] CPU offload enabled for {model_key}")
        else:
            pipe = pipe.to("cuda")

        # Optimization
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            if not config.get("cpu_offload"):
                pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        _pipeline = pipe
        _current_model = model_key
        logger.info(f"[LocalImageGen] {model_key} loaded and ready")
        return pipe, None

    except ImportError as e:
        return None, f"diffusers/torch not available: {e}"
    except Exception as e:
        return None, f"Failed to load model: {e}"


class LocalImageGenTool:
    """Generate images locally on GPU — SDXL Turbo (fast) or FLUX.1-schnell (quality)."""

    name = "local_image_gen"
    description = "Generate images locally on RTX 4060 — SDXL Turbo (fast) or FLUX.1-schnell (quality). No API cost, fully private."

    def __init__(self):
        self._default_model = DEFAULT_MODEL
        logger.info(f"[LocalImageGen] Tool initialized, default model: {self._default_model}")

    # ------------------------------------------------------------------ #

    def list_models(self) -> Dict:
        """List available image generation models."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda_available else 0
        except ImportError:
            cuda_available = False
            gpu_name = "unknown"
            vram_gb = 0

        return {
            "success": True,
            "gpu": gpu_name,
            "vram_gb": vram_gb,
            "cuda_available": cuda_available,
            "current_model": _current_model,
            "default_model": self._default_model,
            "models": {
                k: {
                    "description": v["description"],
                    "default_steps": v["default_steps"],
                    "output_size": f"{v['width']}x{v['height']}",
                    "cpu_offload": v.get("cpu_offload", False),
                }
                for k, v in MODEL_CONFIGS.items()
            },
        }

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict:
        """Generate an image from a text prompt.

        Args:
            prompt: Description of the image to generate
            model: Model to use ('sdxl-turbo', 'flux-schnell', 'sd15'). Uses default if None.
            negative_prompt: Things to avoid in the image (not supported by FLUX)
            steps: Number of inference steps (lower = faster, higher = more detail)
            guidance_scale: How closely to follow prompt (0-10, 0 for Turbo/FLUX)
            width: Output image width in pixels
            height: Output image height in pixels
            seed: Random seed for reproducibility
            output_path: Full output file path (auto-generated if None)
            filename: Just the filename (saved to default output dir)
        """
        with _load_lock:
            model_key = model or self._default_model
            config = MODEL_CONFIGS.get(model_key)
            if not config:
                return {"success": False, "error": f"Unknown model '{model_key}'"}

            pipe, err = _load_pipeline(model_key)
            if err:
                return {"success": False, "error": err}

            try:
                import torch

                # Build generation kwargs
                gen_kwargs: Dict[str, Any] = {
                    "prompt": prompt,
                    "num_inference_steps": steps or config["default_steps"],
                    "width": width or config["width"],
                    "height": height or config["height"],
                }

                guidance = guidance_scale if guidance_scale is not None else config["default_guidance"]
                if guidance > 0:
                    gen_kwargs["guidance_scale"] = guidance

                if negative_prompt and model_key != "flux-schnell":
                    gen_kwargs["negative_prompt"] = negative_prompt

                if seed is not None:
                    gen_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)

                logger.info(f"[LocalImageGen] Generating with {model_key}: '{prompt[:60]}'")
                result = pipe(**gen_kwargs)
                image = result.images[0]

                # Save
                if output_path:
                    save_path = Path(output_path)
                else:
                    out_dir = _get_output_dir()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = filename or f"{ts}_{model_key}.png"
                    save_path = out_dir / fname

                save_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(save_path))

                return {
                    "success": True,
                    "path": str(save_path),
                    "model": model_key,
                    "prompt": prompt,
                    "steps": gen_kwargs["num_inference_steps"],
                    "size": f"{gen_kwargs['width']}x{gen_kwargs['height']}",
                    "seed": seed,
                }

            except Exception as e:
                logger.error(f"[LocalImageGen] Generation failed: {e}")
                return {"success": False, "error": str(e), "model": model_key}
            finally:
                global _pipeline
                if _pipeline is not None:
                    del _pipeline
                    _pipeline = None
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def unload_model(self) -> Dict:
        """Free GPU memory by unloading the current model."""
        global _pipeline, _current_model
        with _load_lock:
            if _pipeline is None:
                return {"success": True, "message": "No model loaded"}
            prev = _current_model
            try:
                import torch
                del _pipeline
                _pipeline = None
                _current_model = None
                torch.cuda.empty_cache()
                return {"success": True, "unloaded": prev}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def set_default_model(self, model_key: str) -> Dict:
        """Set the default model for future generations."""
        if model_key not in MODEL_CONFIGS:
            return {"success": False, "error": f"Unknown model. Choose from: {', '.join(MODEL_CONFIGS.keys())}"}
        self._default_model = model_key
        return {"success": True, "default_model": model_key}

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a local image generation action."""
        a = action.lower().strip()

        if "list" in a or "model" in a and "show" in a:
            return self.list_models()
        if "unload" in a or "free" in a or "clear" in a:
            return self.unload_model()
        if "default" in a or "set" in a and "model" in a:
            return self.set_default_model(kwargs.get("model") or "")
        if "generate" in a or "create" in a or "make" in a or "draw" in a or "image" in a:
            return self.generate(
                prompt=kwargs.get("prompt") or kwargs.get("text") or action,
                model=kwargs.get("model"),
                negative_prompt=kwargs.get("negative_prompt"),
                steps=kwargs.get("steps"),
                guidance_scale=kwargs.get("guidance_scale"),
                width=kwargs.get("width"),
                height=kwargs.get("height"),
                seed=kwargs.get("seed"),
                output_path=kwargs.get("output_path"),
                filename=kwargs.get("filename"),
            )

        # Default: treat action as a prompt
        prompt = kwargs.get("prompt") or action
        if prompt and len(prompt) > 5:
            return self.generate(prompt=prompt, model=kwargs.get("model"))
        return self.list_models()
