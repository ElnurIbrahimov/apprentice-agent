"""Vision tool for analyzing images using Ollama vision models with fallback chain.

Supports:
- Ollama cloud models (kimi-k2.5, qwen3.5) via multi-model fallback
- Florence-2 (microsoft/Florence-2-base) via HuggingFace transformers for fast OCR/UI analysis
- VRAM-aware model selection to avoid OOM on constrained GPUs
"""

import json
import base64
import logging
import ollama
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from ..config import Config

logger = logging.getLogger(__name__)

from aura.tools._shared_models import get_florence2


def _check_florence2_available() -> bool:
    """Check if Florence-2 dependencies are installed."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        logger.info("[Vision] Florence-2 unavailable (torch/transformers not installed)")
        return False


def _load_florence2():
    """Load Florence-2 via shared singleton."""
    if not Config.FLORENCE2_ENABLED:
        raise RuntimeError("Florence-2 disabled in config")
    model, processor = get_florence2()
    if model is None:
        raise RuntimeError("Florence-2 unavailable (disabled or failed to load)")
    return model, processor


def _get_available_vram_gb() -> Optional[float]:
    """Get available GPU VRAM in GB. Returns None if no GPU or torch unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
    except Exception:
        pass
    return None


def _can_fit_model(model_name: str) -> bool:
    """Check if a model can fit in available VRAM."""
    vram_free = _get_available_vram_gb()
    if vram_free is None:
        return True  # Can't check, assume it fits (CPU fallback)

    # Look up estimated size
    base_name = model_name.split("/")[-1].lower().replace("microsoft/", "")
    vram_needed = Config.VISION_MODEL_VRAM.get(base_name)

    # Also try the raw model name
    if vram_needed is None:
        vram_needed = Config.VISION_MODEL_VRAM.get(model_name)

    if vram_needed is None:
        return True  # Unknown model, let it try

    if vram_free < vram_needed:
        logger.info("[Vision] Skipping %s: needs %.1fGB VRAM, only %.1fGB free",
                     model_name, vram_needed, vram_free)
        return False
    return True


# Model-specific system prompts for better output quality
_MODEL_SYSTEM_PROMPTS = {
    "qwen2.5-vl": (
        "You are a precise vision analyst. Always respond with structured, "
        "well-organized output. Use clear section headers and bullet points."
    ),
}


def _get_model_system_prompt(model_name: str) -> Optional[str]:
    """Get model-specific system prompt if available."""
    for prefix, prompt in _MODEL_SYSTEM_PROMPTS.items():
        if prefix in model_name:
            return prompt
    return None


def _run_florence2(image_path: str, task: str = "<DETAILED_CAPTION>") -> Optional[str]:
    """Run a Florence-2 task on an image. Returns text result or None.

    Lightweight wrapper around _load_florence2() for use in fast paths
    (analyze_image generic queries, analyze_screen_context).
    """
    if not _check_florence2_available():
        return None
    try:
        from PIL import Image
        import torch

        model, processor = _load_florence2()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=task, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
            )
        result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(result, task=task, image_size=image.size)
        if isinstance(parsed, dict):
            val = parsed.get(task, parsed)
            if isinstance(val, dict):
                return val.get("text", str(val))
            return str(val)
        return str(parsed)
    except Exception as e:
        logger.debug(f"[Vision] Florence-2 inference failed: {e}")
        return None


class VisionTool:
    """Tool for analyzing images using vision LLM with model fallback chain.

    Supports Florence-2 (fast structured OCR/detection) as primary analyzer
    and Ollama cloud models (kimi-k2.5, qwen3.5) as fallback.
    """

    def __init__(self, model: str = None, brain=None):
        """Initialize vision tool.

        Args:
            model: Vision model to use (default: from Config)
            brain: Optional OllamaBrain reference for client reuse.
                   If None, creates a local ollama.Client.
        """
        self.model = model or Config.get_model("vision")
        self._brain = brain
        if brain is None:
            self._local_client = ollama.Client(host=Config.OLLAMA_HOST)
        else:
            self._local_client = None  # use brain's client

    def _get_client(self, model: str):
        """Get an ollama client and resolved model name.

        If brain is available, uses brain's client. Otherwise uses local client.
        Cloud-suffixed models (e.g. 'kimi-k2.5:cloud') are stripped to their
        base name for Ollama API calls.

        Returns:
            Tuple of (client, actual_model_name)
        """
        actual_model = model
        if model.endswith("-cloud"):
            local_name = model.replace("-cloud", "")
            logger.warning(
                "Cloud model %s requested but no cloud routing available. "
                "Falling back to local name: %s", model, local_name
            )
            actual_model = local_name

        if self._brain is not None:
            return self._brain.client, actual_model
        return self._local_client, actual_model

    def _analyze_with_fallback(self, img_data: str, question: str) -> tuple:
        """Try vision models from the fallback chain until one succeeds.

        Args:
            img_data: Base64-encoded image data
            question: Question to ask about the image

        Returns:
            Tuple of (response_content, model_used)

        Raises:
            RuntimeError: If all models in the chain fail
        """
        # Build ordered list: primary model first, then chain (deduped)
        chain = [self.model]
        for m in Config.MODEL_VISION_CHAIN:
            if m not in chain:
                chain.append(m)

        errors = []
        for model in chain:
            # VRAM check: skip models that won't fit
            if not _can_fit_model(model):
                errors.append(f"{model}: skipped (insufficient VRAM)")
                continue

            try:
                client, actual_model = self._get_client(model)
                logger.info("Trying vision model: %s", actual_model)

                # Build messages with model-specific system prompt
                messages = []
                sys_prompt = _get_model_system_prompt(actual_model)
                if sys_prompt:
                    messages.append({'role': 'system', 'content': sys_prompt})
                messages.append({
                    'role': 'user',
                    'content': question,
                    'images': [img_data]
                })

                response = client.chat(
                    model=actual_model,
                    messages=messages,
                )
                content = response['message']['content']
                logger.info("Vision analysis succeeded with model: %s", actual_model)
                return content, actual_model
            except Exception as e:
                logger.warning("Vision model %s failed: %s", model, e)
                errors.append(f"{model}: {e}")

        raise RuntimeError(
            f"All vision models failed. Tried: {', '.join(chain)}. "
            f"Errors: {'; '.join(errors)}"
        )

    def _analyze_with_florence2(
        self, image_path: str, task: str = "<OCR>"
    ) -> Optional[Dict[str, Any]]:
        """Run Florence-2 on an image for fast structured analysis.

        Args:
            image_path: Path to the image file
            task: Florence-2 task token. One of:
                  "<OCR>" - Extract text
                  "<OCR_WITH_REGION>" - OCR with bounding boxes
                  "<OD>" - Object detection
                  "<CAPTION>" - Short caption
                  "<DETAILED_CAPTION>" - Detailed caption

        Returns:
            Dict with parsed results, or None if Florence-2 unavailable
        """
        if not Config.FLORENCE2_ENABLED or not _check_florence2_available():
            return None

        try:
            model, processor = _load_florence2()
        except Exception as e:
            logger.debug("[Vision] Florence-2 not available: %s", e)
            return None

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            device = next(model.parameters()).device

            inputs = processor(
                text=task,
                images=image,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )

            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = processor.post_process_generation(
                generated_text, task=task, image_size=image.size
            )

            return {
                "success": True,
                "model": "florence-2",
                "task": task,
                "result": parsed.get(task, parsed),
            }
        except Exception as e:
            logger.warning("[Vision] Florence-2 analysis failed: %s", e)
            return None

    def analyze_image(
        self,
        image_path: str,
        question: str = "What is in this image? Describe what you see."
    ) -> dict:
        """Analyze an image and answer a question about it.

        Args:
            image_path: Path to the image file
            question: Question to ask about the image

        Returns:
            dict with success status, description/error, and model_used
        """
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        if not path.is_file():
            return {"success": False, "error": f"Path is not a file: {image_path}"}

        supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'}
        if path.suffix.lower() not in supported_formats:
            return {
                "success": False,
                "error": f"Unsupported image format: {path.suffix}. Supported: {supported_formats}"
            }

        try:
            # Florence-2 fast path: for generic "describe" queries, try Florence-2 first
            generic_keywords = {"describe", "what is", "what's in", "what do you see"}
            is_generic = any(kw in question.lower() for kw in generic_keywords)

            if is_generic:
                florence_result = _run_florence2(str(path), "<DETAILED_CAPTION>")
                if florence_result:
                    return {
                        "success": True,
                        "description": florence_result,
                        "image_path": str(path.absolute()),
                        "question": question,
                        "model": "florence-2"
                    }
            else:
                # For specific questions, get Florence-2 caption as context enrichment
                florence_context = _run_florence2(str(path), "<CAPTION>")
                if florence_context:
                    question = f"Image context: {florence_context}\n\nQuestion: {question}"

            # Read and encode image as base64
            with open(path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()

            description, model_used = self._analyze_with_fallback(img_data, question)

            return {
                "success": True,
                "description": description,
                "image_path": str(path.absolute()),
                "question": question,
                "model": model_used
            }

        except RuntimeError as e:
            return {"success": False, "error": str(e)}
        except ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to Ollama. Is it running? Try: ollama serve"
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to analyze image: {e}"}

    def describe_screen(self, screenshot_path: str) -> dict:
        """Describe what's on a screenshot."""
        return self.analyze_image(
            screenshot_path,
            question=(
                "Describe what you see on this screen. What application or content is visible? "
                "Be specific about any text, UI elements, or notable features."
            )
        )

    def read_text(self, image_path: str, language_hint: str = None) -> dict:
        """Extract and read text from an image (OCR-like).

        Tries Florence-2 first for fast structured OCR, falls back to Ollama models.

        Args:
            image_path: Path to the image
            language_hint: Optional language hint (e.g. 'japanese', 'arabic', 'chinese').
                          Helps the model preserve non-Latin scripts accurately.

        Returns:
            dict with success status and extracted text
        """
        # Try Florence-2 first for fast OCR
        florence_result = self._analyze_with_florence2(image_path, "<OCR_WITH_REGION>")
        if florence_result and florence_result.get("success"):
            result_data = florence_result["result"]
            # Extract text from Florence-2 OCR result
            if isinstance(result_data, dict):
                text_parts = result_data.get("labels", [])
                text = "\n".join(text_parts) if text_parts else str(result_data)
            else:
                text = str(result_data)
            return {
                "success": True,
                "description": text,
                "image_path": str(Path(image_path).absolute()),
                "model": "florence-2",
                "florence2_raw": florence_result["result"],
            }

        # Fallback to Ollama vision models
        prompt = "Read and transcribe all visible text in this image. List the text exactly as it appears."
        if language_hint:
            prompt += (
                f" The text may contain {language_hint} characters. "
                "Preserve all non-Latin scripts exactly as they appear — do not transliterate."
            )
        return self.analyze_image(image_path, question=prompt)

    def analyze_ui(self, image_path: str) -> dict:
        """Analyze UI elements in a screenshot for structured extraction.

        Tries Florence-2 first for fast object detection + OCR, falls back to Ollama.

        Args:
            image_path: Path to the screenshot

        Returns:
            dict with structured UI analysis including elements, text, state, errors
        """
        # Try Florence-2 for fast structured UI detection
        florence_od = self._analyze_with_florence2(image_path, "<OD>")
        florence_ocr = self._analyze_with_florence2(image_path, "<OCR>")

        if florence_od and florence_od.get("success"):
            od_result = florence_od["result"]
            ocr_text = ""
            if florence_ocr and florence_ocr.get("success"):
                ocr_data = florence_ocr["result"]
                if isinstance(ocr_data, str):
                    ocr_text = ocr_data
                elif isinstance(ocr_data, dict):
                    ocr_text = "\n".join(ocr_data.get("labels", [str(ocr_data)]))

            # Build structured UI analysis from Florence-2 detection
            elements = []
            if isinstance(od_result, dict):
                labels = od_result.get("labels", [])
                bboxes = od_result.get("bboxes", [])
                for i, label in enumerate(labels):
                    entry = label
                    if i < len(bboxes):
                        bbox = bboxes[i]
                        entry += f" @ [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
                    elements.append(entry)

            result = {
                "success": True,
                "description": f"Florence-2 detected {len(elements)} UI elements",
                "image_path": str(Path(image_path).absolute()),
                "model": "florence-2",
                "ui_analysis": {
                    "application": "",
                    "ui_elements": elements,
                    "text_content": ocr_text,
                    "active_state": "",
                    "errors": "",
                    "suggested_actions": "",
                },
            }
            return result

        # Fallback to Ollama models
        prompt = (
            "Analyze the UI in this screenshot. Provide a structured analysis with:\n"
            "APPLICATION: What application or webpage is shown\n"
            "UI_ELEMENTS: List each visible UI element with approximate position "
            "(top-left, center, bottom-right, etc.) and type (button, input, menu, tab, etc.)\n"
            "TEXT_CONTENT: All readable text in the interface\n"
            "ACTIVE_STATE: What appears to be focused/selected/active\n"
            "ERRORS: Any error messages or warnings visible\n"
            "SUGGESTED_ACTIONS: What actions appear available to the user"
        )

        result = self.analyze_image(image_path, question=prompt)
        if not result.get("success"):
            return result

        # Parse structured response into fields
        description = result["description"]
        parsed = {
            "application": "",
            "ui_elements": [],
            "text_content": "",
            "active_state": "",
            "errors": "",
            "suggested_actions": "",
        }

        current_section = None
        section_map = {
            "APPLICATION": "application",
            "UI_ELEMENTS": "ui_elements",
            "TEXT_CONTENT": "text_content",
            "ACTIVE_STATE": "active_state",
            "ERRORS": "errors",
            "SUGGESTED_ACTIONS": "suggested_actions",
        }

        for line in description.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this line starts a new section
            matched = False
            for label, key in section_map.items():
                if stripped.upper().startswith(label):
                    current_section = key
                    # Grab text after the label and colon
                    remainder = stripped[len(label):].lstrip(":").strip()
                    if remainder:
                        if key == "ui_elements":
                            parsed[key].append(remainder)
                        else:
                            parsed[key] = remainder
                    matched = True
                    break

            if not matched and current_section:
                if current_section == "ui_elements":
                    if stripped.startswith(("-", "*", "\u2022")) or stripped[0].isdigit():
                        parsed[current_section].append(stripped.lstrip("-*\u2022 ").strip())
                    elif parsed[current_section]:
                        # Continuation of previous element
                        parsed[current_section][-1] += " " + stripped
                    else:
                        parsed[current_section].append(stripped)
                else:
                    if parsed[current_section]:
                        parsed[current_section] += " " + stripped
                    else:
                        parsed[current_section] = stripped

        result["ui_analysis"] = parsed
        return result

    def analyze_screen_context(self, screenshot_path: str) -> dict:
        """Analyze a screenshot and return structured context.

        Returns categorized info: app_type, has_errors, error_text,
        main_content, language, suggested_help.

        Args:
            screenshot_path: Path to the screenshot

        Returns:
            Structured dict with screen analysis
        """
        # Florence-2 fast path: try CAPTION + OCR before expensive Ollama call
        florence_caption = _run_florence2(screenshot_path, "<CAPTION>")
        florence_ocr = _run_florence2(screenshot_path, "<OCR>")
        if florence_caption and florence_ocr:
            parsed = self._parse_florence2_screen_context(florence_caption, florence_ocr)
            if parsed.get("app_type") != "other" or parsed.get("has_errors"):
                return parsed

        structured_prompt = """Analyze this screenshot and respond in this EXACT format (one field per line):

APP_TYPE: <one of: code_editor, browser, terminal, file_manager, chat, email, document, media, settings, other>
HAS_ERROR: <yes or no>
ERROR_TEXT: <the error message if visible, or "none">
MAIN_CONTENT: <brief description of what the user is working on, 1 sentence>
LANGUAGE: <programming language if code is visible, or "none">
SUGGESTED_HELP: <one brief suggestion for how an AI assistant could help, or "none">

Be concise. Only report what you actually see."""

        result = self.analyze_image(screenshot_path, structured_prompt)

        if not result.get("success"):
            return {
                "success": False,
                "available": False,
                "error": result.get("error", "Vision analysis failed"),
            }

        # Parse structured response
        raw = result.get("description", "")
        parsed = {
            "success": True,
            "available": True,
            "app_type": "other",
            "has_errors": False,
            "error_text": None,
            "main_content": "",
            "language": None,
            "suggested_help": None,
            "raw_analysis": raw,
        }

        for line in raw.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            value = value.strip()
            key_lower = key.strip().lower().replace(" ", "_")

            if key_lower == "app_type":
                parsed["app_type"] = value.lower()
            elif key_lower == "has_error":
                parsed["has_errors"] = value.lower() in ("yes", "true", "1")
            elif key_lower == "error_text":
                if value.lower() not in ("none", "n/a", ""):
                    parsed["error_text"] = value
            elif key_lower == "main_content":
                parsed["main_content"] = value
            elif key_lower == "language":
                if value.lower() not in ("none", "n/a", ""):
                    parsed["language"] = value
            elif key_lower == "suggested_help":
                if value.lower() not in ("none", "n/a", ""):
                    parsed["suggested_help"] = value

        # Record thought about screen analysis
        try:
            from api.routes.thinking import record_thought
            content_desc = parsed["main_content"][:50] if parsed["main_content"] else parsed["app_type"]
            if parsed["has_errors"]:
                record_thought("observing", f"detected error on screen in {parsed['app_type']}: {parsed.get('error_text', '')[:40]}", 0.7, "tool")
            else:
                record_thought("observing", f"screen shows {parsed['app_type']}: {content_desc}", 0.3, "tool")
        except Exception:
            pass

        return parsed

    @staticmethod
    def _parse_florence2_screen_context(caption: str, ocr_text: str) -> Dict[str, Any]:
        """Parse Florence-2 caption + OCR into structured screen context."""
        import re as _re
        caption_lower = caption.lower()
        ocr_lower = ocr_text.lower()

        # Detect app type from keywords
        app_type = "other"
        app_keywords = {
            "code_editor": ["code", "editor", "ide", "visual studio", "vscode", "pycharm", "sublime"],
            "browser": ["browser", "chrome", "firefox", "safari", "edge", "webpage", "website"],
            "terminal": ["terminal", "console", "command", "shell", "powershell", "cmd"],
            "file_manager": ["file", "explorer", "finder", "directory", "folder"],
            "chat": ["chat", "message", "discord", "slack", "telegram", "whatsapp"],
            "email": ["email", "mail", "inbox", "outlook", "gmail"],
            "document": ["document", "word", "docs", "notepad", "text editor"],
            "media": ["video", "music", "player", "spotify", "youtube"],
        }
        for atype, keywords in app_keywords.items():
            if any(kw in caption_lower or kw in ocr_lower for kw in keywords):
                app_type = atype
                break

        # Detect errors
        error_keywords = ["error", "exception", "traceback", "failed", "fatal", "crash"]
        has_errors = any(kw in ocr_lower for kw in error_keywords)
        error_text = None
        if has_errors:
            for line in ocr_text.split("\n"):
                if any(kw in line.lower() for kw in error_keywords):
                    error_text = line.strip()[:200]
                    break

        # Detect language
        language = None
        lang_keywords = {
            "python": ["python", ".py", "def ", "import ", "class "],
            "javascript": ["javascript", ".js", "const ", "function ", "=>"],
            "java": [".java", "public class", "void main"],
            "rust": [".rs", "fn main", "let mut"],
            "c++": [".cpp", "#include", "std::"],
        }
        for lang, keywords in lang_keywords.items():
            if any(kw in ocr_lower for kw in keywords):
                language = lang
                break

        return {
            "success": True,
            "available": True,
            "app_type": app_type,
            "has_errors": has_errors,
            "error_text": error_text,
            "main_content": caption[:200],
            "language": language,
            "suggested_help": None,
            "raw_analysis": f"Florence-2 caption: {caption}\nOCR: {ocr_text[:300]}",
            "model": "florence-2",
        }

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a vision action.

        Args:
            action: Action to perform (analyze, describe, read, ui, dom, element)
            **kwargs: Additional arguments (image_path, question, language_hint)

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        image_path = kwargs.get("image_path")
        if not image_path:
            image_path = self._extract_path(action)

        if not image_path:
            return {
                "success": False,
                "error": "No image path provided. Specify the path to analyze."
            }

        if "read" in action_lower or "text" in action_lower or "ocr" in action_lower:
            return self.read_text(image_path, language_hint=kwargs.get("language_hint"))
        elif "ui" in action_lower or "dom" in action_lower or "element" in action_lower:
            return self.analyze_ui(image_path)
        elif "screen" in action_lower:
            return self.describe_screen(image_path)
        else:
            question = kwargs.get("question", "What is in this image? Describe what you see.")
            return self.analyze_image(image_path, question)

    def _extract_path(self, action: str) -> Optional[str]:
        """Extract image path from action string."""
        import re

        # Look for quoted paths
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]

        # Look for paths with image extensions
        path_pattern = r'[\w./\\:-]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|tif)'
        paths = re.findall(path_pattern, action, re.IGNORECASE)
        if paths:
            return paths[0]

        # Look for Windows paths
        win_paths = re.findall(r'[A-Za-z]:[/\\][\w./\\-]+', action)
        if win_paths:
            return win_paths[0]

        return None
