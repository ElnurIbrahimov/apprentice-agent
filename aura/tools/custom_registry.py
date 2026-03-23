"""Custom tool loading from registry and keyword generation.

Extracted from aura.agent (2026-03-23).

Contains:
- load_custom_tools_from_registry() — load active custom tools from data/custom_tools.json
- generate_default_keywords() — keyword generation for tool detection
"""

import importlib.util
import json
import logging
import re
from pathlib import Path

from aura.security.tool_validator import validate_custom_tool_code

logger = logging.getLogger(__name__)


def load_custom_tools_from_registry(tools: dict, custom_tool_keywords: dict) -> None:
    """Load active custom tools from the custom_tools.json registry.

    Validates each tool's source code for security before dynamic import.
    Populates *tools* and *custom_tool_keywords* in-place.

    Args:
        tools: Agent tool registry dict (mutated)
        custom_tool_keywords: Keyword->tool_name mapping (mutated)
    """
    registry_path = Path(__file__).parent.parent.parent / "data" / "custom_tools.json"
    if not registry_path.exists():
        return

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        for tool_entry in registry.get("tools", []):
            if tool_entry.get("status") != "active":
                continue

            tool_name = tool_entry["name"]
            # SECURITY: Resolve path and verify it stays within the project tools directory
            tools_base = (Path(__file__).parent).resolve()
            tool_file = Path(tool_entry.get("file", ""))
            try:
                tool_file_resolved = tool_file.resolve()
            except Exception:
                continue
            # SECURITY: Use Path.relative_to() instead of string startswith() to prevent
            # sibling-directory bypass (e.g., tools_evil/ matching tools/ prefix).
            try:
                tool_file_resolved.relative_to(tools_base)
            except ValueError:
                logger.warning(f"[SECURITY] Custom tool path outside project directory: {tool_file}")
                continue
            if not tool_file_resolved.exists():
                continue
            tool_file = tool_file_resolved

            try:
                # SECURITY: Validate tool code before dynamic import
                tool_code = tool_file.read_text()
                is_valid, validation_msg = validate_custom_tool_code(tool_code, str(tool_file))

                if not is_valid:
                    logger.debug(f"[SECURITY] Rejected custom tool {tool_name}: {validation_msg}")
                    logger.warning(f"Custom tool {tool_name} failed security validation: {validation_msg}")
                    continue

                # Dynamic import of validated custom tool
                spec = importlib.util.spec_from_file_location(
                    tool_name,
                    tool_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Get the tool class
                    class_name = tool_entry.get("class_name")
                    if class_name and hasattr(module, class_name):
                        tool_class = getattr(module, class_name)
                        tools[tool_name] = tool_class()
                        logger.debug(f"[LOADED] Custom tool: {tool_name} (validated)")

                        # Load keywords for this tool
                        keywords = tool_entry.get("keywords", [])
                        if not keywords:
                            keywords = generate_default_keywords(
                                tool_name,
                                tool_entry.get("description", ""),
                                tool_entry.get("functions", [])
                            )
                        for kw in keywords:
                            custom_tool_keywords[kw.lower()] = tool_name
            except Exception as e:
                logger.error(f"[ERROR] Failed to load custom tool {tool_name}: {e}")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"[ERROR] Failed to load custom tools registry: {e}")


def generate_default_keywords(name: str, description: str, functions: list) -> list[str]:
    """Generate default keywords for a custom tool if not provided.

    Args:
        name: Tool name
        description: Tool description
        functions: List of function names

    Returns:
        List of keywords for tool detection
    """
    keywords = set()

    # Add words from tool name
    for word in name.lower().split('_'):
        if len(word) > 2:
            keywords.add(word)
    keywords.add(name.lower().replace('_', ' '))

    # Add words from description
    stop_words = {'a', 'an', 'the', 'is', 'are', 'to', 'of', 'in', 'for', 'on',
                  'with', 'at', 'by', 'from', 'and', 'or', 'but', 'can', 'will'}
    desc_words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
    for word in desc_words:
        if word not in stop_words and len(word) > 2:
            keywords.add(word)

    # Add function names
    for func in functions:
        for word in func.lower().split('_'):
            if len(word) > 2:
                keywords.add(word)

    # Add common variations
    if 'bmi' in name.lower():
        keywords.update(['body mass index', 'height and weight'])
    if 'temperature' in name.lower():
        keywords.update(['celsius', 'fahrenheit', 'temp'])

    return list(keywords)
