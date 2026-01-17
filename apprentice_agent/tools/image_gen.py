"""Image generation tool using Stable Diffusion."""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Lazy imports for torch and related libraries
_torch = None
_torch_available = None


def _check_torch():
    """Check if torch is available and import it lazily."""
    global _torch, _torch_available
    if _torch_available is None:
        try:
            import torch
            _torch = torch
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def _get_torch():
    """Get the torch module, raising an error if not available."""
    if not _check_torch():
        raise ImportError(
            "torch is required for image generation. "
            "Install it with: pip install torch"
        )
    return _torch


class ImageGenerationTool:
    """Generate images using Stable Diffusion 1.5."""

    def __init__(
        self,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        output_dir: str = "generated_images",
        device: Optional[str] = None,
        use_float16: bool = True
    ):
        """Initialize the image generation tool.

        Args:
            model_id: HuggingFace model ID for Stable Diffusion
            output_dir: Directory to save generated images
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_float16: Use float16 for faster inference (GPU only)
        """
        self._model_id = model_id
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._device = device
        self._use_float16_requested = use_float16
        self._use_float16 = None  # Resolved lazily
        self._pipeline = None

    def _resolve_device(self):
        """Resolve device setting lazily when torch is needed."""
        if self._device is None:
            torch = _get_torch()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._use_float16 is None:
            self._use_float16 = self._use_float16_requested and self._device == "cuda"

    def _load_pipeline(self):
        """Lazy load the Stable Diffusion pipeline."""
        if self._pipeline is None:
            torch = _get_torch()
            self._resolve_device()

            print(f"Loading Stable Diffusion model '{self._model_id}'...")
            print(f"Device: {self._device}, Float16: {self._use_float16}")

            from diffusers import StableDiffusionPipeline

            dtype = torch.float16 if self._use_float16 else torch.float32

            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
                safety_checker=None,  # Disable safety checker for faster inference
                requires_safety_checker=False
            )
            self._pipeline = self._pipeline.to(self._device)

            # Enable memory optimizations
            if self._device == "cuda":
                try:
                    self._pipeline.enable_attention_slicing()
                except Exception:
                    pass  # Not all pipelines support this

            print("Model loaded successfully.")

        return self._pipeline

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        save: bool = True
    ) -> dict:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: What to avoid in the image
            width: Image width (should be multiple of 8)
            height: Image height (should be multiple of 8)
            num_inference_steps: Number of denoising steps (more = better quality, slower)
            guidance_scale: How closely to follow the prompt (higher = more literal)
            seed: Random seed for reproducibility
            save: Whether to save the image to disk

        Returns:
            dict with 'success', 'image_path', 'image', and optional 'error'
        """
        if not prompt or not prompt.strip():
            return {"success": False, "error": "No prompt provided"}

        try:
            torch = _get_torch()
            pipeline = self._load_pipeline()

            # Set up generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(seed)

            # Generate the image
            print(f"Generating image for: '{prompt[:50]}...'")

            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

            image = result.images[0]

            # Save the image
            image_path = None
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create a safe filename from the prompt
                safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:30])
                safe_prompt = safe_prompt.strip().replace(" ", "_")
                filename = f"{timestamp}_{safe_prompt}.png"
                image_path = self._output_dir / filename
                image.save(image_path)
                print(f"Image saved to: {image_path}")

            return {
                "success": True,
                "image": image,
                "image_path": str(image_path) if image_path else None,
                "prompt": prompt,
                "seed": seed
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "image": None,
                "image_path": None
            }

    def generate_variations(
        self,
        prompt: str,
        num_images: int = 4,
        **kwargs
    ) -> list:
        """Generate multiple variations of an image.

        Args:
            prompt: Text description of the image
            num_images: Number of variations to generate
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of result dicts from generate()
        """
        results = []
        for i in range(num_images):
            print(f"Generating variation {i + 1}/{num_images}...")
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results

    def get_device_info(self) -> dict:
        """Get information about the device being used.

        Returns:
            dict with device information
        """
        if not _check_torch():
            return {
                "device": self._device or "unknown",
                "cuda_available": False,
                "float16": self._use_float16,
                "torch_available": False
            }

        torch = _get_torch()
        self._resolve_device()

        info = {
            "device": self._device,
            "cuda_available": torch.cuda.is_available(),
            "float16": self._use_float16,
            "torch_available": True
        }

        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"

        return info


# Convenience function for quick generation
def generate_image(prompt: str, **kwargs) -> dict:
    """Quick function to generate an image.

    Args:
        prompt: Text description of the image
        **kwargs: Additional arguments passed to ImageGenerationTool.generate()

    Returns:
        dict with generation results
    """
    tool = ImageGenerationTool()
    return tool.generate(prompt, **kwargs)
