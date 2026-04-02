"""Prompt-based tool calling adapter for ChatGPT models.

ChatGPT's API doesn't support the tools= parameter. This adapter:
1. Encodes tool schemas into the system prompt as structured text
2. Parses <tool_call> XML blocks from the LLM's text response
3. Returns tool calls in Ollama-compatible format

The agentic loop doesn't need to know whether tools came from Ollama's
structured calling or this text-based adapter.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Regex for extracting tool calls from LLM text
_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(.*?)\s*</tool_call>',
    re.DOTALL,
)

# Fallback: bare JSON object with "name" and "arguments" keys
_BARE_JSON_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)


def build_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Convert tool schemas to a system prompt section.

    Args:
        tools: List of tool schemas in OpenAI function-calling format:
            [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]

    Returns:
        A string to append to the system prompt describing available tools.
    """
    lines = [
        "\n## Available Tools",
        "",
        "You have access to tools. To use a tool, output a <tool_call> block with JSON inside:",
        "",
        "<tool_call>",
        '{"name": "tool_name", "arguments": {"param1": "value1"}}',
        "</tool_call>",
        "",
        "You can call multiple tools by outputting multiple <tool_call> blocks.",
        "After receiving tool results in <tool_result> blocks, continue reasoning or give your final answer.",
        "If you don't need any tools, just respond normally.",
        "",
        "### Tools",
        "",
    ]

    for i, tool in enumerate(tools, 1):
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        # Build parameter list
        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required else ""
            param_parts.append(f"  - `{pname}` ({ptype}{req}): {pdesc}")

        lines.append(f"**{i}. {name}** — {desc}")
        if param_parts:
            lines.extend(param_parts)
        lines.append("")

    return "\n".join(lines)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse tool calls from LLM text response.

    Extracts <tool_call> XML blocks and converts to Ollama format.

    Args:
        text: Raw LLM text response.

    Returns:
        (clean_content, tool_calls) where:
        - clean_content: text with <tool_call> blocks removed
        - tool_calls: list in Ollama format:
          [{"function": {"name": "...", "arguments": {...}}}, ...]
    """
    if not text:
        return "", []

    tool_calls = []

    # Primary: extract <tool_call>...</tool_call> blocks
    for match in _TOOL_CALL_RE.finditer(text):
        raw = match.group(1).strip()
        parsed = _try_parse_json(raw)
        if parsed and "name" in parsed:
            tool_calls.append({
                "function": {
                    "name": parsed["name"],
                    "arguments": parsed.get("arguments", {}),
                }
            })

    # Fallback: look for bare JSON with name+arguments (no XML tags)
    if not tool_calls:
        for match in _BARE_JSON_RE.finditer(text):
            name = match.group(1)
            args_raw = match.group(2)
            args = _try_parse_json(args_raw)
            if args is not None:
                tool_calls.append({
                    "function": {
                        "name": name,
                        "arguments": args,
                    }
                })

    # Strip tool_call blocks from content
    clean = _TOOL_CALL_RE.sub("", text).strip()
    # Clean up multiple blank lines
    clean = re.sub(r'\n{3,}', '\n\n', clean)

    if tool_calls:
        logger.info(
            f"[PromptToolAdapter] Parsed {len(tool_calls)} tool call(s): "
            + ", ".join(tc["function"]["name"] for tc in tool_calls)
        )

    return clean, tool_calls


def format_tool_result(tool_name: str, result: str) -> str:
    """Format a tool result for feeding back to the LLM.

    Args:
        tool_name: Name of the tool that was executed.
        result: The tool's output as a string.

    Returns:
        Formatted string with XML tags.
    """
    # Truncate very long results to avoid context overflow
    if len(result) > 15000:
        result = result[:15000] + "\n... (truncated)"
    return f'<tool_result name="{tool_name}">\n{result}\n</tool_result>'


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON with fallbacks for common LLM formatting errors."""
    from aura.core.json_utils import parse_llm_json

    return parse_llm_json(raw)
