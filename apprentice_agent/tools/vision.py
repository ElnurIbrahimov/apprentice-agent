"""Vision tool for analyzing images using Ollama vision models with fallback chain."""

import json
import base64
import logging
import ollama
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import Config

logger = logging.getLogger(__name__)


class VisionTool:
    """Tool for analyzing images using vision LLM with model fallback chain."""

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
        Cloud-suffixed models (e.g. 'qwen3-vl:235b-cloud') are stripped to their
        local name since we only have local Ollama connectivity.

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
            try:
                client, actual_model = self._get_client(model)
                logger.info("Trying vision model: %s", actual_model)
                response = client.chat(
                    model=actual_model,
                    messages=[{
                        'role': 'user',
                        'content': question,
                        'images': [img_data]
                    }]
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

        Args:
            image_path: Path to the image
            language_hint: Optional language hint (e.g. 'japanese', 'arabic', 'chinese').
                          Helps the model preserve non-Latin scripts accurately.

        Returns:
            dict with success status and extracted text
        """
        prompt = "Read and transcribe all visible text in this image. List the text exactly as it appears."
        if language_hint:
            prompt += (
                f" The text may contain {language_hint} characters. "
                "Preserve all non-Latin scripts exactly as they appear — do not transliterate."
            )
        return self.analyze_image(image_path, question=prompt)

    def analyze_ui(self, image_path: str) -> dict:
        """Analyze UI elements in a screenshot for structured extraction.

        Args:
            image_path: Path to the screenshot

        Returns:
            dict with structured UI analysis including elements, text, state, errors
        """
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
                    if stripped.startswith(("-", "*", "•")) or stripped[0].isdigit():
                        parsed[current_section].append(stripped.lstrip("-*• ").strip())
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
