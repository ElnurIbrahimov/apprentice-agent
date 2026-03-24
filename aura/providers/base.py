"""Base provider interface — all direct API providers implement this."""

from abc import ABC, abstractmethod
from typing import Iterator


class BaseProvider(ABC):
    """Ollama-compatible provider interface for direct API access.

    All providers must return responses in the same format as ollama.Client.chat():
      Sync:   {"message": {"role": "assistant", "content": "..."}, "done": True,
               "prompt_eval_count": N, "eval_count": M}
      Stream: yields {"message": {"role": "assistant", "content": delta}, "done": False}
              then final {"message": {"role": "assistant", "content": ""}, "done": True, ...}
    """

    @abstractmethod
    def chat(self, model: str, messages: list[dict], stream: bool = False,
             options: dict = None, tools: list | None = None) -> dict | Iterator[dict]:
        """Send a chat request, returning ollama-compatible response."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return list of available model names (with provider prefix)."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if the API key is set."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name."""
        ...

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Provider prefix (e.g. 'anthropic:')."""
        ...
