"""Vision tool for analyzing images using Ollama's LLaVA model."""

import json
import base64
import logging
import ollama
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import Config

logger = logging.getLogger(__name__)


class VisionTool:
    """Tool for analyzing images using vision LLM."""

    def __init__(self, model: str = "llava"):
        """Initialize vision tool.

        Args:
            model: Vision model to use (default: llava)
        """
        self.model = model
        self.client = ollama.Client(host=Config.OLLAMA_HOST)

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
            dict with success status and description/error
        """
        # Validate image path
        path = Path(image_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"Image not found: {image_path}"
            }

        if not path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {image_path}"
            }

        # Check for supported image formats
        supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
        if path.suffix.lower() not in supported_formats:
            return {
                "success": False,
                "error": f"Unsupported image format: {path.suffix}. Supported: {supported_formats}"
            }

        try:
            # Read and encode image as base64
            with open(path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()

            # Call ollama with the vision model
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': question,
                    'images': [img_data]
                }]
            )

            description = response['message']['content']

            return {
                "success": True,
                "description": description,
                "image_path": str(path.absolute()),
                "question": question,
                "model": self.model
            }

        except ollama.ResponseError as e:
            return {
                "success": False,
                "error": f"Ollama error: {str(e)}"
            }
        except ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to Ollama. Is it running? Try: ollama serve"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze image: {str(e)}"
            }

    def describe_screen(self, screenshot_path: str) -> dict:
        """Describe what's on a screenshot.

        Args:
            screenshot_path: Path to the screenshot

        Returns:
            dict with success status and description
        """
        return self.analyze_image(
            screenshot_path,
            question="Describe what you see on this screen. What application or content is visible? Be specific about any text, UI elements, or notable features."
        )

    def read_text(self, image_path: str) -> dict:
        """Extract and read text from an image (OCR-like).

        Args:
            image_path: Path to the image

        Returns:
            dict with success status and extracted text
        """
        return self.analyze_image(
            image_path,
            question="Read and transcribe all visible text in this image. List the text exactly as it appears."
        )

    def analyze_screen_context(self, screenshot_path: str) -> Dict[str, Any]:
        """
        Analyze a screenshot and return structured context.

        Instead of a free-form description, returns categorized info:
        - app_type: What kind of application (code_editor, browser, terminal, etc.)
        - has_errors: Whether errors/exceptions are visible
        - error_text: The error content if detected
        - main_content: What the user is working on
        - suggested_action: What AURA could proactively help with

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
            action: Action to perform (analyze, describe, read)
            **kwargs: Additional arguments (image_path, question)

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        # Extract image path from action or kwargs
        image_path = kwargs.get("image_path")
        if not image_path:
            # Try to extract path from action string
            image_path = self._extract_path(action)

        if not image_path:
            return {
                "success": False,
                "error": "No image path provided. Specify the path to analyze."
            }

        # Determine action type
        if "read" in action_lower or "text" in action_lower or "ocr" in action_lower:
            return self.read_text(image_path)
        elif "screen" in action_lower:
            return self.describe_screen(image_path)
        else:
            # Default: analyze with custom or default question
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
        path_pattern = r'[\w./\\:-]+\.(?:png|jpg|jpeg|gif|webp|bmp)'
        paths = re.findall(path_pattern, action, re.IGNORECASE)
        if paths:
            return paths[0]

        # Look for Windows paths
        win_paths = re.findall(r'[A-Za-z]:[/\\][\w./\\-]+', action)
        if win_paths:
            return win_paths[0]

        return None
