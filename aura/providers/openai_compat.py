"""OpenAI-compatible provider — handles 8 of 10 providers with one class.

Uses the standard OpenAI Chat Completions API format. Works with:
OpenAI, xAI (Grok), Perplexity, DeepSeek, MiniMax, Qwen, Kimi, GLM.
"""

import json
import logging
import os
from typing import Iterator

import requests

from .base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseProvider):
    """Single provider class for all OpenAI-compatible APIs."""

    def __init__(self, provider_name: str, base_url: str, env_var: str,
                 display_name: str, default_models: list[str]):
        self._provider_name = provider_name
        self._base_url = base_url.rstrip("/")
        self._env_var = env_var
        self._display_name = display_name
        self._default_models = default_models
        self._session = requests.Session()

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def prefix(self) -> str:
        return f"{self._provider_name}:"

    def _get_api_key(self) -> str:
        return os.getenv(self._env_var, "")

    def is_configured(self) -> bool:
        return bool(self._get_api_key())

    def list_models(self) -> list[str]:
        return [f"{self._provider_name}:{m}" for m in self._default_models]

    def _strip_prefix(self, model: str) -> str:
        """Remove provider prefix from model name."""
        pfx = f"{self._provider_name}:"
        return model[len(pfx):] if model.startswith(pfx) else model

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    def _build_body(self, model: str, messages: list[dict], stream: bool,
                    options: dict = None) -> dict:
        """Build OpenAI Chat Completions request body."""
        # Convert ollama message format to OpenAI format
        oai_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            oai_messages.append({"role": role, "content": content})

        body = {
            "model": self._strip_prefix(model),
            "messages": oai_messages,
            "stream": stream,
        }

        # Apply standard options if provided
        if options:
            if "temperature" in options:
                body["temperature"] = options["temperature"]
            if "num_predict" in options:
                body["max_tokens"] = options["num_predict"]
            if "top_p" in options:
                body["top_p"] = options["top_p"]

        return body

    def chat(self, model: str, messages: list[dict], stream: bool = False,
             options: dict = None) -> dict | Iterator[dict]:
        if not self.is_configured():
            raise ConnectionError(
                f"{self._display_name} API key not set. "
                f"Set {self._env_var} in your .env file."
            )

        url = f"{self._base_url}/chat/completions"
        headers = self._build_headers()
        body = self._build_body(model, messages, stream, options)

        if stream:
            return self._stream_chat(url, headers, body)
        else:
            return self._sync_chat(url, headers, body)

    def _sync_chat(self, url: str, headers: dict, body: dict) -> dict:
        """Non-streaming chat — single JSON response."""
        try:
            resp = self._session.post(url, headers=headers, json=body, timeout=120)
        except requests.exceptions.RequestException as e:
            logger.error(f"[{self._provider_name.upper()}] Request failed: {e}")
            raise ConnectionError(f"{self._display_name} request failed: {e}")

        if resp.status_code != 200:
            error_text = resp.text[:500] if resp.text else ""
            logger.error(f"[{self._provider_name.upper()}] API error {resp.status_code}: {error_text}")
            raise ConnectionError(f"{self._display_name} API error: {resp.status_code}")

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return {
            "message": {
                "role": "assistant",
                "content": message.get("content", ""),
            },
            "done": True,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

    def _stream_chat(self, url: str, headers: dict, body: dict) -> Iterator[dict]:
        """Streaming chat — parse SSE events."""
        try:
            resp = self._session.post(url, headers=headers, json=body, stream=True, timeout=120)
        except requests.exceptions.RequestException as e:
            logger.error(f"[{self._provider_name.upper()}] Stream request failed: {e}")
            raise ConnectionError(f"{self._display_name} request failed: {e}")

        if resp.status_code != 200:
            error_text = resp.text[:500] if resp.text else ""
            logger.error(f"[{self._provider_name.upper()}] API error {resp.status_code}: {error_text}")
            raise ConnectionError(f"{self._display_name} API error: {resp.status_code}")

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
            if data_str.strip() == "[DONE]":
                break

            try:
                event = json.loads(data_str)
                choice = event.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")

                # Some providers include usage in the final chunk
                usage = event.get("usage")
                if usage:
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)

                if content:
                    yield {
                        "message": {"role": "assistant", "content": content},
                        "done": False,
                    }
            except json.JSONDecodeError:
                continue

        # Final done chunk
        yield {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": input_tokens,
            "eval_count": output_tokens,
        }
