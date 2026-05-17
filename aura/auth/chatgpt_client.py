"""ChatGPT API client with ollama.Client-compatible interface.

Translates between ollama chat format and the ChatGPT/Codex Responses API,
allowing AURA's brain.py to use ChatGPT models seamlessly.
"""

import json
import logging
from typing import Iterator

import requests

from .chatgpt_oauth import get_valid_token

logger = logging.getLogger(__name__)

# ChatGPT backend API
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"

# Available models via ChatGPT subscription (as of March 2026)
CHATGPT_MODELS = {
    # GPT-5.4 family (latest)
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-thinking": "gpt-5.4",          # thinking mode
    "gpt-5.4-pro": "gpt-5.4",               # pro tier
    # GPT-5.3 family
    "gpt-5.3": "gpt-5.3",
    "gpt-5.3-codex": "gpt-5.3-codex",
    "gpt-5.3-codex-spark": "gpt-5.3-codex-spark",  # near-instant coding
    # GPT-5.2 family
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-codex": "gpt-5.2-codex",
    # GPT-5.1 family
    "gpt-5.1": "gpt-5.1",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
}

# Models available per subscription tier
CHATGPT_PLUS_MODELS = [
    "chatgpt:gpt-5.4",
    "chatgpt:gpt-5.4-thinking",
    "chatgpt:gpt-5.3",
    "chatgpt:gpt-5.3-codex",
    "chatgpt:gpt-5.2",
    "chatgpt:gpt-5.2-codex",
    "chatgpt:gpt-5.1",
    "chatgpt:gpt-5.1-codex",
    "chatgpt:gpt-5.1-codex-mini",
]

CHATGPT_PRO_MODELS = [*CHATGPT_PLUS_MODELS, "chatgpt:gpt-5.4-pro", "chatgpt:gpt-5.3-codex-spark", "chatgpt:gpt-5.1-codex-max"]

# All ChatGPT models (for listing)
ALL_CHATGPT_MODELS = CHATGPT_PRO_MODELS


def _messages_to_input(messages: list[dict]) -> list[dict]:
    """Convert ollama/chat messages to Responses API input format."""
    input_items = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            input_items.append({
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": content}],
            })
        elif role == "user":
            input_items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            })
        elif role == "assistant":
            input_items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            })
        elif role == "tool":
            # Tool results sent as user messages (Responses API has no tool role)
            input_items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            })
    return input_items


def _normalize_model(model: str) -> str:
    """Normalize model name, stripping chatgpt: prefix."""
    if model.startswith("chatgpt:"):
        model = model[8:]
    return CHATGPT_MODELS.get(model, model)


class ChatGPTClient:
    """Client that uses ChatGPT OAuth tokens to call the Codex Responses API.

    Provides an ollama.Client-compatible .chat() interface so brain.py
    can use it as a drop-in replacement.
    """

    def __init__(self):
        self._session = requests.Session()

    def _get_headers(self, tokens: dict) -> dict:
        """Build request headers for Codex API."""
        return {
            "Authorization": f"Bearer {tokens['access']}",
            "Content-Type": "application/json",
            "chatgpt-account-id": tokens.get("account_id", ""),
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "accept": "text/event-stream",
        }

    def _build_body(self, model: str, messages: list[dict], options: dict | None = None) -> dict:
        """Build the Responses API request body."""
        # Extract system prompt from messages to use as instructions
        instructions = "You are a helpful AI assistant."
        input_items = []
        for msg in messages:
            if msg.get("role") == "system":
                instructions = msg.get("content", instructions)
            else:
                input_items.extend(_messages_to_input([msg]))

        body = {
            "model": _normalize_model(model),
            "instructions": instructions,
            "input": input_items,
            "store": False,
            "stream": True,
            "reasoning": {
                "effort": "medium",
                "summary": "auto",
            },
            "text": {
                "verbosity": "medium",
            },
            "include": ["reasoning.encrypted_content"],
        }

        # NOTE: The Codex Responses API does NOT support standard LLM options
        # like temperature, top_p, num_predict, etc. Ignore all options from
        # brain.py's Ollama-style llm_options dict.

        return body

    def chat(self, model: str, messages: list[dict], stream: bool = False,
             options: dict | None = None) -> dict | Iterator[dict]:
        """Call ChatGPT Codex API, returning ollama-compatible response.

        Args:
            model: Model name (with or without chatgpt: prefix)
            messages: List of message dicts with role and content
            stream: Whether to stream the response
            options: Optional LLM options (temperature, etc.)

        Returns:
            If stream=False: dict with message.content, done, eval counts
            If stream=True: Iterator yielding chunk dicts
        """
        tokens = get_valid_token()
        if not tokens:
            raise ConnectionError(
                "ChatGPT not authenticated. Run: aura --login chatgpt"
            )

        url = f"{CODEX_BASE_URL}{CODEX_RESPONSES_PATH}"
        headers = self._get_headers(tokens)
        body = self._build_body(model, messages, options)

        if stream:
            return self._stream_chat(url, headers, body)
        else:
            return self._sync_chat(url, headers, body)

    def _sync_chat(self, url: str, headers: dict, body: dict) -> dict:
        """Non-streaming chat — collect full SSE response."""
        resp = self._session.post(
            url, headers=headers, json=body, stream=True, timeout=120
        )

        if resp.status_code != 200:
            error_text = ""
            try:
                error_text = resp.text[:500]
            except Exception as e:
                logger.debug(f"[ChatGPTClient] Failed to read error response text: {e}")
            logger.error(f"[CHATGPT] API error: {resp.status_code} {error_text}")
            raise ConnectionError(f"ChatGPT API error: {resp.status_code}")

        full_text = ""
        input_tokens = 0
        output_tokens = 0

        resp.encoding = "utf-8"
        for raw_line in resp.iter_lines():
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = raw_line
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
                event_type = event.get("type", "")

                if event_type == "response.output_text.delta":
                    full_text += event.get("delta", "")

                elif event_type in ("response.done", "response.completed"):
                    # Extract usage stats
                    response_obj = event.get("response", event)
                    usage = response_obj.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

                    # Try to get complete text from the done event
                    output = response_obj.get("output", [])
                    for item in output:
                        if item.get("type") == "message":
                            for part in item.get("content", []):
                                if part.get("type") == "output_text" and part.get("text"):
                                    full_text = part["text"]
            except json.JSONDecodeError:
                continue

        return {
            "message": {"role": "assistant", "content": full_text},
            "done": True,
            "prompt_eval_count": input_tokens,
            "eval_count": output_tokens,
        }

    def _stream_chat(self, url: str, headers: dict, body: dict) -> Iterator[dict]:
        """Streaming chat — yield ollama-compatible chunks."""
        resp = self._session.post(
            url, headers=headers, json=body, stream=True, timeout=120
        )

        if resp.status_code != 200:
            logger.error(f"[CHATGPT] API error: {resp.status_code}")
            raise ConnectionError(f"ChatGPT API error: {resp.status_code}")

        input_tokens = 0
        output_tokens = 0
        resp.encoding = "utf-8"

        for raw_line in resp.iter_lines():
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = raw_line
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
                event_type = event.get("type", "")

                if event_type == "response.output_text.delta":
                    yield {
                        "message": {"role": "assistant", "content": event.get("delta", "")},
                        "done": False,
                    }
                elif event_type in ("response.done", "response.completed"):
                    response_obj = event.get("response", event)
                    usage = response_obj.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
            except json.JSONDecodeError:
                continue

        # Final done chunk with token stats
        yield {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": input_tokens,
            "eval_count": output_tokens,
        }
