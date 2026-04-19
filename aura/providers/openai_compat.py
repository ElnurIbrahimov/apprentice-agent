"""OpenAI-compatible provider — handles 8 of 10 providers with one class.

Uses the standard OpenAI Chat Completions API format. Works with:
OpenAI, xAI (Grok), Perplexity, DeepSeek, MiniMax, Qwen, Kimi, GLM.
"""

import json
import logging
import os
from typing import Iterator

import requests

from aura.reliability.error_classifier import FailoverReason
from aura.reliability.provider_shim import ProviderGiveUp, record_rate_limit, request_with_retry

from .base import BaseProvider
from .credential_pool import get_pool

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
        # Register with the credential pool so comma-separated keys rotate
        # automatically. Single-key env vars still work unchanged.
        get_pool().register(provider_name, env_var)

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def prefix(self) -> str:
        return f"{self._provider_name}:"

    def _get_api_key(self) -> str:
        # Prefer the pool (rotates on exhaustion); fall back to raw env var so
        # tests that monkeypatch os.environ keep working.
        pool_key = get_pool().acquire(self._provider_name)
        if pool_key:
            return pool_key
        return os.getenv(self._env_var, "")

    def is_configured(self) -> bool:
        # Pool-aware: configured if ANY key exists in env, even if currently cooling.
        if get_pool().pool_size(self._provider_name) > 0:
            return True
        return bool(os.getenv(self._env_var, ""))

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
                    options: dict | None = None) -> dict:
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
             options: dict | None = None, tools: list | None = None) -> dict | Iterator[dict]:
        if not self.is_configured():
            raise ConnectionError(
                f"{self._display_name} API key not set. "
                f"Set {self._env_var} in your .env file."
            )

        url = f"{self._base_url}/chat/completions"
        headers = self._build_headers()
        body = self._build_body(model, messages, stream, options)

        if tools:
            body["tools"] = tools

        if stream:
            return self._stream_chat(url, headers, body)
        else:
            return self._sync_chat(url, headers, body)

    def _sync_chat(self, url: str, headers: dict, body: dict) -> dict:
        """Non-streaming chat — single JSON response with classifier-driven retries."""
        model = body.get("model", "")
        try:
            resp = request_with_retry(
                session=self._session,
                method="POST",
                url=url,
                headers=headers,
                json_body=body,
                provider=self._provider_name,
                model=model,
                timeout=(10, 120),
            )
        except ProviderGiveUp as give_up:
            # Cooldown the current key if appropriate, then surface as ConnectionError
            # (matching the prior contract so upstream code keeps working).
            self._maybe_cooldown_key(give_up, headers)
            logger.error(
                f"[{self._provider_name.upper()}] gave up after {give_up.attempts} attempts: "
                f"{give_up.classified.reason.value}"
            )
            raise ConnectionError(str(give_up)) from give_up

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        result = {
            "message": {
                "role": "assistant",
                "content": message.get("content", ""),
            },
            "done": True,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

        # Include tool_calls if present
        if message.get("tool_calls"):
            result["message"]["tool_calls"] = message["tool_calls"]

        return result

    def _maybe_cooldown_key(self, give_up: ProviderGiveUp, headers: dict) -> None:
        """If the failure is credential-scoped (rate/billing), cool down the key used."""
        reason = give_up.classified.reason
        if reason not in (FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.auth):
            return
        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return
        key = auth_header[len("Bearer "):].strip()
        if not key:
            return
        reason_str = reason.value if reason != FailoverReason.auth else "rate_limit"
        get_pool().mark_exhausted(self._provider_name, key, reason=reason_str)

    def _stream_chat(self, url: str, headers: dict, body: dict) -> Iterator[dict]:
        """Streaming chat — parse SSE events with classifier-driven retries."""
        model = body.get("model", "")
        try:
            resp = request_with_retry(
                session=self._session,
                method="POST",
                url=url,
                headers=headers,
                json_body=body,
                provider=self._provider_name,
                model=model,
                timeout=(10, 90),
                stream=True,
            )
        except ProviderGiveUp as give_up:
            self._maybe_cooldown_key(give_up, headers)
            logger.error(
                f"[{self._provider_name.upper()}] stream gave up after {give_up.attempts} attempts: "
                f"{give_up.classified.reason.value}"
            )
            raise ConnectionError(str(give_up)) from give_up

        input_tokens = 0
        output_tokens = 0
        accumulated_tool_calls = {}  # index -> {id, type, function: {name, arguments}}
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
                if data_str.strip() == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                    choice = event.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")

                    # Accumulate tool_calls from streaming deltas
                    delta_tool_calls = delta.get("tool_calls")
                    if delta_tool_calls:
                        for tc in delta_tool_calls:
                            idx = tc.get("index", 0)
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tc.get("id", ""),
                                    "type": tc.get("type", "function"),
                                    "function": {"name": "", "arguments": ""},
                                }
                            entry = accumulated_tool_calls[idx]
                            if tc.get("id"):
                                entry["id"] = tc["id"]
                            func = tc.get("function", {})
                            if func.get("name"):
                                entry["function"]["name"] += func["name"]
                            if func.get("arguments"):
                                entry["function"]["arguments"] += func["arguments"]

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
            final_msg = {"role": "assistant", "content": ""}
            if accumulated_tool_calls:
                final_msg["tool_calls"] = [
                    accumulated_tool_calls[i]
                    for i in sorted(accumulated_tool_calls.keys())
                ]
            yield {
                "message": final_msg,
                "done": True,
                "prompt_eval_count": input_tokens,
                "eval_count": output_tokens,
            }
        finally:
            resp.close()
