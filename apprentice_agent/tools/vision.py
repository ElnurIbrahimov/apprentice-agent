"""Vision tool for analyzing images using Ollama's LLaVA model."""

import base64
import ollama
from pathlib import Path
from typing import Optional

from ..config import Config


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
