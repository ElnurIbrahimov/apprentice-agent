"""Clipboard tool for reading, writing, and analyzing clipboard content."""

import re
import json
from typing import Optional

import pyperclip


class ClipboardTool:
    """Tool for interacting with the system clipboard."""

    def __init__(self):
        """Initialize clipboard tool."""
        pass

    def read(self) -> dict:
        """Get current clipboard content.

        Returns:
            dict with success status and clipboard content
        """
        try:
            content = pyperclip.paste()
            if content:
                return {
                    "success": True,
                    "content": content,
                    "length": len(content),
                    "lines": content.count('\n') + 1
                }
            else:
                return {
                    "success": True,
                    "content": "",
                    "message": "Clipboard is empty"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read clipboard: {str(e)}"
            }

    def write(self, text: str) -> dict:
        """Copy text to clipboard.

        Args:
            text: Text to copy to clipboard

        Returns:
            dict with success status
        """
        if not text:
            return {
                "success": False,
                "error": "No text provided to copy"
            }

        try:
            pyperclip.copy(text)
            return {
                "success": True,
                "message": f"Copied {len(text)} characters to clipboard",
                "preview": text[:100] + "..." if len(text) > 100 else text
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write to clipboard: {str(e)}"
            }

    def analyze(self) -> dict:
        """Analyze clipboard content and detect its type.

        Returns:
            dict with content type analysis
        """
        read_result = self.read()
        if not read_result.get("success"):
            return read_result

        content = read_result.get("content", "")
        if not content:
            return {
                "success": True,
                "content_type": "empty",
                "message": "Clipboard is empty"
            }

        # Detect content type
        content_type = self._detect_content_type(content)

        result = {
            "success": True,
            "content_type": content_type,
            "length": len(content),
            "lines": content.count('\n') + 1,
            "preview": content[:200] + "..." if len(content) > 200 else content
        }

        # Add type-specific details
        if content_type == "python_error":
            result["error_info"] = self._parse_python_error(content)
        elif content_type == "url":
            result["url"] = content.strip()
        elif content_type == "json":
            result["json_valid"] = self._is_valid_json(content)

        return result

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content.

        Args:
            content: The content to analyze

        Returns:
            Content type string
        """
        content_stripped = content.strip()

        # Check for Python error/traceback
        if self._is_python_error(content):
            return "python_error"

        # Check for URL
        if self._is_url(content_stripped):
            return "url"

        # Check for JSON
        if self._is_valid_json(content_stripped):
            return "json"

        # Check for code patterns
        if self._is_code(content):
            return "code"

        # Check for file path
        if self._is_file_path(content_stripped):
            return "file_path"

        # Default to plain text
        return "text"

    def _is_python_error(self, content: str) -> bool:
        """Check if content looks like a Python error/traceback."""
        error_indicators = [
            'Traceback (most recent call last):',
            'File "',
            'Error:',
            'Exception:',
            'raise ',
            'SyntaxError:',
            'TypeError:',
            'ValueError:',
            'KeyError:',
            'IndexError:',
            'AttributeError:',
            'ImportError:',
            'ModuleNotFoundError:',
            'NameError:',
            'ZeroDivisionError:',
            'FileNotFoundError:',
            'RuntimeError:',
        ]
        return any(indicator in content for indicator in error_indicators)

    def _is_url(self, content: str) -> bool:
        """Check if content is a URL."""
        url_pattern = r'^https?://[^\s]+$'
        return bool(re.match(url_pattern, content.strip()))

    def _is_valid_json(self, content: str) -> bool:
        """Check if content is valid JSON."""
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _is_code(self, content: str) -> bool:
        """Check if content looks like code."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'return ',
            'if ', 'for ', 'while ', 'try:', 'except:',
            'function ', 'const ', 'let ', 'var ',
            '#!/', '#include', 'public class', 'private ',
            '=>', '$(', '<?php', '<html', '<!DOCTYPE',
        ]
        # Check for multiple code indicators or common patterns
        indicator_count = sum(1 for ind in code_indicators if ind in content)
        has_braces = '{' in content and '}' in content
        has_semicolons = content.count(';') > 2

        return indicator_count >= 2 or (indicator_count >= 1 and (has_braces or has_semicolons))

    def _is_file_path(self, content: str) -> bool:
        """Check if content looks like a file path."""
        # Windows path
        if re.match(r'^[A-Za-z]:[/\\]', content):
            return True
        # Unix absolute path
        if content.startswith('/') and '/' in content[1:]:
            return True
        return False

    def _parse_python_error(self, content: str) -> dict:
        """Parse Python error to extract useful info."""
        info = {}

        # Extract error type
        error_match = re.search(r'(\w+Error|\w+Exception):', content)
        if error_match:
            info["error_type"] = error_match.group(1)

        # Extract error message
        lines = content.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if ':' in last_line:
                parts = last_line.split(':', 1)
                info["error_message"] = parts[1].strip() if len(parts) > 1 else last_line

        # Extract file and line number from last File reference
        file_matches = re.findall(r'File "([^"]+)", line (\d+)', content)
        if file_matches:
            last_file, last_line = file_matches[-1]
            info["file"] = last_file
            info["line"] = int(last_line)

        return info

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a clipboard action.

        Args:
            action: Action to perform (read, write, copy, paste, analyze)
            **kwargs: Additional arguments (text for write/copy)

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        if "analyze" in action_lower or "detect" in action_lower or "type" in action_lower:
            return self.analyze()

        if "write" in action_lower or "copy" in action_lower:
            # Extract text to copy
            text = kwargs.get("text")
            if not text:
                text = self._extract_text_to_copy(action)
            if text:
                return self.write(text)
            return {"success": False, "error": "No text specified to copy"}

        # Default: read clipboard
        return self.read()

    def _extract_text_to_copy(self, action: str) -> Optional[str]:
        """Extract text to copy from action string."""
        # Look for quoted text
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]

        # Look for text after "copy:" or "write:"
        patterns = [
            r'copy[:\s]+(.+)',
            r'write[:\s]+(.+)',
            r'clipboard[:\s]+(.+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Remove trailing quotes if present
                text = text.strip('"\'')
                return text

        return None


# Singleton instance for easy import
clipboard_tool = ClipboardTool()
