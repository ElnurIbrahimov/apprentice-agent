"""Anthropic provider — custom Messages API format.

Anthropic uses a different message format than OpenAI:
- System prompt is a separate top-level field
- Content uses structured blocks [{type: "text", text: "..."}]
- Requires anthropic-version header
"""

import json
import logging
import os
from typing import Iterator

import requests

from .base import BaseProvider
from .registry import PROVIDER_CONFIGS

logger = logging.getLogger(__name__)

_CFG = PROVIDER_CONFIGS["anthropic"]


class AnthropicProvider(BaseProvider):
    """Direct API client for Anthropic's Messages API."""

    def __init__(self):
        self._session = requests.Session()

    @property
    def display_name(self) -> str:
        return _CFG["display_name"]

    @property
    def prefix(self) -> str:
        return "anthropic:"

    def _get_api_key(self) -> str:
        return os.getenv(_CFG["env_var"], "")

    def is_configured(self) -> bool:
        return bool(self._get_api_key())

    def list_models(self) -> list[str]:
        return [f"anthropic:{m}" for m in _CFG["default_models"]]

    def _strip_prefix(self, model: str) -> str:
        return model[len("anthropic:"):] if model.startswith("anthropic:") else model

    def _build_headers(self, stream: bool = False) -> dict:
        headers = {
            "x-api-key": self._get_api_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if stream:
            headers["accept"] = "text/event-stream"
        return headers

    def _build_body(self, model: str, messages: list[dict], stream: bool,
                    options: dict | None = None) -> dict:
        """Build Anthropic Messages API request body.

        Key differences from OpenAI:
        - system is a top-level field, not a message role
        - content is [{type: "text", text: "..."}] blocks
        """
        system_text = ""
        api_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_text = content
            elif role in ("user", "assistant"):
                api_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })
            elif role == "tool":
                # Map tool results to user messages
                api_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": content}],
                })

        # Anthropic requires at least one message
        if not api_messages:
            api_messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        # Ensure conversation doesn't start with assistant
        if api_messages[0]["role"] == "assistant":
            api_messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "Continue."}]})

        body = {
            "model": self._strip_prefix(model),
            "messages": api_messages,
            "max_tokens": 4096,
            "stream": stream,
        }

        if system_text:
            body["system"] = system_text

        if options:
            if "temperature" in options:
                body["temperature"] = options["temperature"]
            if "num_predict" in options:
                body["max_tokens"] = options["num_predict"]
            if "top_p" in options:
                body["top_p"] = options["top_p"]

        return body

    def chat(self, model: str, messages: list[dict], stream: bool = False,
             options: dict | None = None, tools: list | None = None) -> dict | Iterator[dict]:
        if not self.is_configured():
            raise ConnectionError(
                f"Anthropic API key not set. Set {_CFG['env_var']} in your .env file."
            )

        url = f"{_CFG['base_url']}/messages"
        headers = self._build_headers(stream)
        body = self._build_body(model, messages, stream, options)

        # Convert OpenAI-format tools to Anthropic format
        if tools:
            anthropic_tools = []
            for t in tools:
                func = t.get("function", t)
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            body["tools"] = anthropic_tools

        if stream:
            return self._stream_chat(url, headers, body)
        else:
            return self._sync_chat(url, headers, body)

    def _sync_chat(self, url: str, headers: dict, body: dict) -> dict:
        try:
            resp = self._session.post(url, headers=headers, json=body, timeout=120)
        except requests.exceptions.RequestException as e:
            logger.error(f"[ANTHROPIC] Request failed: {e}")
            raise ConnectionError(f"Anthropic request failed: {e}")

        if resp.status_code != 200:
            error_text = resp.text[:500] if resp.text else ""
            logger.error(f"[ANTHROPIC] API error {resp.status_code}: {error_text}")
            raise ConnectionError(f"Anthropic API error: {resp.status_code} - {resp.text[:200]}")

        data = resp.json()

        # Extract text and tool_use from content blocks
        text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                # Convert to OpenAI-compatible tool_calls format
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        usage = data.get("usage", {})
        result = {
            "message": {"role": "assistant", "content": text},
            "done": True,
            "prompt_eval_count": usage.get("input_tokens", 0),
            "eval_count": usage.get("output_tokens", 0),
        }

        if tool_calls:
            result["message"]["tool_calls"] = tool_calls

        return result

    def _stream_chat(self, url: str, headers: dict, body: dict) -> Iterator[dict]:
        try:
            resp = self._session.post(url, headers=headers, json=body, stream=True, timeout=(10, 90))
        except requests.exceptions.RequestException as e:
            logger.error(f"[ANTHROPIC] Stream request failed: {e}")
            raise ConnectionError(f"Anthropic request failed: {e}")

        if resp.status_code != 200:
            error_text = resp.text[:500] if resp.text else ""
            logger.error(f"[ANTHROPIC] API error {resp.status_code}: {error_text}")
            raise ConnectionError(f"Anthropic API error: {resp.status_code} - {resp.text[:200]}")

        input_tokens = 0
        output_tokens = 0
        tool_calls = []  # Accumulated tool_use blocks
        current_tool = None  # Tool block being streamed
        resp.encoding = "utf-8"

        try:
            for raw_line in resp.iter_lines():
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace")
                else:
                    line = raw_line

                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                try:
                    event = json.loads(data_str)
                    event_type = event.get("type", "")

                    if event_type == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            current_tool = {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": "",
                                },
                            }

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield {
                                "message": {"role": "assistant", "content": delta.get("text", "")},
                                "done": False,
                            }
                        elif delta.get("type") == "input_json_delta" and current_tool:
                            current_tool["function"]["arguments"] += delta.get("partial_json", "")

                    elif event_type == "content_block_stop":
                        if current_tool:
                            tool_calls.append(current_tool)
                            current_tool = None

                    elif event_type == "message_start":
                        msg = event.get("message", {})
                        usage = msg.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif event_type == "message_delta":
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", output_tokens)

                except json.JSONDecodeError:
                    continue

            final_msg = {"role": "assistant", "content": ""}
            if tool_calls:
                final_msg["tool_calls"] = tool_calls
            yield {
                "message": final_msg,
                "done": True,
                "prompt_eval_count": input_tokens,
                "eval_count": output_tokens,
            }
        finally:
            resp.close()
