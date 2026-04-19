"""Google Gemini provider — custom GenerateContent API format.

Gemini uses a different format than OpenAI:
- API key is passed as URL parameter, not header
- Messages use contents/parts structure
- System instructions are a separate field
"""

import json
import logging
import os
from typing import Iterator

import requests

from aura.reliability.error_classifier import FailoverReason
from aura.reliability.provider_shim import ProviderGiveUp, request_with_retry

from .base import BaseProvider
from .credential_pool import get_pool
from .registry import PROVIDER_CONFIGS

logger = logging.getLogger(__name__)

_CFG = PROVIDER_CONFIGS["gemini"]


class GeminiProvider(BaseProvider):
    """Direct API client for Google's Gemini GenerateContent API."""

    def __init__(self):
        self._session = requests.Session()
        get_pool().register("gemini", _CFG["env_var"])

    @property
    def display_name(self) -> str:
        return _CFG["display_name"]

    @property
    def prefix(self) -> str:
        return "gemini:"

    def _get_api_key(self) -> str:
        pool_key = get_pool().acquire("gemini")
        if pool_key:
            return pool_key
        return os.getenv(_CFG["env_var"], "")

    def is_configured(self) -> bool:
        if get_pool().pool_size("gemini") > 0:
            return True
        return bool(os.getenv(_CFG["env_var"], ""))

    def _maybe_cooldown_key(self, give_up: ProviderGiveUp, api_key: str) -> None:
        reason = give_up.classified.reason
        if reason not in (FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.auth):
            return
        if not api_key:
            return
        reason_str = reason.value if reason != FailoverReason.auth else "rate_limit"
        get_pool().mark_exhausted("gemini", api_key, reason=reason_str)

    def list_models(self) -> list[str]:
        return [f"gemini:{m}" for m in _CFG["default_models"]]

    def _strip_prefix(self, model: str) -> str:
        return model[len("gemini:"):] if model.startswith("gemini:") else model

    def _build_body(self, messages: list[dict], options: dict | None = None) -> dict:
        """Build Gemini GenerateContent request body.

        Key differences from OpenAI:
        - Uses contents[].parts[].text instead of messages[].content
        - System instruction is a separate field
        - Role names: "user" and "model" (not "assistant")
        """
        system_text = ""
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_text = content
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })
            elif role in ("user", "tool"):
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })

        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        body = {"contents": contents}

        if system_text:
            body["systemInstruction"] = {
                "parts": [{"text": system_text}],
            }

        # Generation config
        gen_config = {}
        if options:
            if "temperature" in options:
                gen_config["temperature"] = options["temperature"]
            if "num_predict" in options:
                gen_config["maxOutputTokens"] = options["num_predict"]
            if "top_p" in options:
                gen_config["topP"] = options["top_p"]
        if gen_config:
            body["generationConfig"] = gen_config

        return body

    def chat(self, model: str, messages: list[dict], stream: bool = False,
             options: dict | None = None, tools: list | None = None) -> dict | Iterator[dict]:
        if not self.is_configured():
            raise ConnectionError(
                f"Gemini API key not set. Set {_CFG['env_var']} in your .env file."
            )

        bare_model = self._strip_prefix(model)
        api_key = self._get_api_key()

        if stream:
            url = f"{_CFG['base_url']}/models/{bare_model}:streamGenerateContent?alt=sse"
            return self._stream_chat(url, messages, options, tools, api_key=api_key)
        else:
            url = f"{_CFG['base_url']}/models/{bare_model}:generateContent"
            return self._sync_chat(url, messages, options, tools, api_key=api_key)

    def _sync_chat(self, url: str, messages: list[dict], options: dict | None = None,
                   tools: list | None = None, api_key: str = "") -> dict:
        body = self._build_body(messages, options)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key

        # Convert OpenAI-format tools to Gemini format
        if tools:
            declarations = []
            for t in tools:
                func = t.get("function", t)
                declarations.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            body["tools"] = [{"function_declarations": declarations}]

        safe_url = url.split("?")[0]
        try:
            resp = request_with_retry(
                session=self._session,
                method="POST",
                url=url,
                headers=headers,
                json_body=body,
                provider="gemini",
                model=safe_url.rsplit("/", 1)[-1].split(":")[0],
                timeout=(10, 120),
            )
        except ProviderGiveUp as give_up:
            self._maybe_cooldown_key(give_up, api_key)
            logger.error(
                f"[GEMINI] gave up after {give_up.attempts} attempts ({safe_url}): "
                f"{give_up.classified.reason.value}"
            )
            raise ConnectionError(str(give_up)) from give_up

        data = resp.json()

        # Extract text and functionCall from candidates
        text = ""
        tool_calls = []
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    text += part["text"]
                if "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append({
                        "id": f"call_{fc.get('name', '')}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("args", {})),
                        },
                    })

        # Gemini usage metadata
        usage = data.get("usageMetadata", {})
        result = {
            "message": {"role": "assistant", "content": text},
            "done": True,
            "prompt_eval_count": usage.get("promptTokenCount", 0),
            "eval_count": usage.get("candidatesTokenCount", 0),
        }

        if tool_calls:
            result["message"]["tool_calls"] = tool_calls

        return result

    def _stream_chat(self, url: str, messages: list[dict], options: dict | None = None,
                     tools: list | None = None, api_key: str = "") -> Iterator[dict]:
        body = self._build_body(messages, options)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key

        # Convert OpenAI-format tools to Gemini format
        if tools:
            declarations = []
            for t in tools:
                func = t.get("function", t)
                declarations.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            body["tools"] = [{"function_declarations": declarations}]

        safe_url = url.split("?")[0]
        try:
            resp = request_with_retry(
                session=self._session,
                method="POST",
                url=url,
                headers=headers,
                json_body=body,
                provider="gemini",
                model=safe_url.rsplit("/", 1)[-1].split(":")[0],
                timeout=(10, 90),
                stream=True,
            )
        except ProviderGiveUp as give_up:
            self._maybe_cooldown_key(give_up, api_key)
            logger.error(
                f"[GEMINI] stream gave up after {give_up.attempts} attempts ({safe_url}): "
                f"{give_up.classified.reason.value}"
            )
            raise ConnectionError(str(give_up)) from give_up

        input_tokens = 0
        output_tokens = 0
        tool_calls = []
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
                    candidates = event.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield {
                                    "message": {"role": "assistant", "content": text},
                                    "done": False,
                                }
                            if "functionCall" in part:
                                fc = part["functionCall"]
                                tool_calls.append({
                                    "id": f"call_{fc.get('name', '')}",
                                    "type": "function",
                                    "function": {
                                        "name": fc.get("name", ""),
                                        "arguments": json.dumps(fc.get("args", {})),
                                    },
                                })

                    usage = event.get("usageMetadata")
                    if usage:
                        input_tokens = usage.get("promptTokenCount", input_tokens)
                        output_tokens = usage.get("candidatesTokenCount", output_tokens)

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
