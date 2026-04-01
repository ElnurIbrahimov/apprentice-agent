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

from .base import BaseProvider
from .registry import PROVIDER_CONFIGS

logger = logging.getLogger(__name__)

_CFG = PROVIDER_CONFIGS["gemini"]


class GeminiProvider(BaseProvider):
    """Direct API client for Google's Gemini GenerateContent API."""

    def __init__(self):
        self._session = requests.Session()

    @property
    def display_name(self) -> str:
        return _CFG["display_name"]

    @property
    def prefix(self) -> str:
        return "gemini:"

    def _get_api_key(self) -> str:
        return os.getenv(_CFG["env_var"], "")

    def is_configured(self) -> bool:
        return bool(self._get_api_key())

    def list_models(self) -> list[str]:
        return [f"gemini:{m}" for m in _CFG["default_models"]]

    def _strip_prefix(self, model: str) -> str:
        return model[len("gemini:"):] if model.startswith("gemini:") else model

    def _build_body(self, messages: list[dict], options: dict = None) -> dict:
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
             options: dict = None, tools: list | None = None) -> dict | Iterator[dict]:
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

    def _sync_chat(self, url: str, messages: list[dict], options: dict = None,
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
            resp = self._session.post(url, headers=headers, json=body, timeout=120)
        except requests.exceptions.RequestException as e:
            logger.error(f"[GEMINI] Request failed: {safe_url}: {type(e).__name__}")
            raise ConnectionError(f"Gemini request failed: {type(e).__name__}")

        if resp.status_code != 200:
            error_text = resp.text[:500] if resp.text else ""
            logger.error(f"[GEMINI] API error {resp.status_code} ({safe_url}): {error_text}")
            raise ConnectionError(f"Gemini API error: {resp.status_code}")

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

    def _stream_chat(self, url: str, messages: list[dict], options: dict = None,
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
            resp = self._session.post(url, headers=headers, json=body, stream=True, timeout=(10, 90))
        except requests.exceptions.RequestException as e:
            logger.error(f"[GEMINI] Stream request failed: {safe_url}: {type(e).__name__}")
            raise ConnectionError(f"Gemini request failed: {type(e).__name__}")

        if resp.status_code != 200:
            error_text = resp.text[:500] if resp.text else ""
            logger.error(f"[GEMINI] API error {resp.status_code} ({safe_url}): {error_text}")
            raise ConnectionError(f"Gemini API error: {resp.status_code}")

        input_tokens = 0
        output_tokens = 0
        tool_calls = []
        resp.encoding = "utf-8"

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
