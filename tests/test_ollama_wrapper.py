"""Tests for aura.reliability.ollama_wrapper.ResilientOllamaClient."""
from __future__ import annotations

from unittest.mock import MagicMock

import ollama
import pytest

from aura.reliability.ollama_wrapper import ResilientOllamaClient


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Patch time.sleep so tests don't actually wait."""
    monkeypatch.setattr("aura.reliability.ollama_wrapper.time.sleep", lambda *_: None)


def _mk_wrapped():
    """Return a MagicMock standing in for ollama.Client."""
    return MagicMock(spec=ollama.Client)


def test_successful_chat_passes_through():
    wrapped = _mk_wrapped()
    wrapped.chat.return_value = {"message": {"role": "assistant", "content": "hi"}}
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    result = client.chat(model="kimi-k2.5:cloud", messages=[{"role": "user", "content": "hi"}])
    assert result["message"]["content"] == "hi"
    assert wrapped.chat.call_count == 1


def test_retries_on_429():
    wrapped = _mk_wrapped()
    err = ollama.ResponseError("rate limit exceeded", status_code=429)
    wrapped.chat.side_effect = [err, err, {"message": {"role": "assistant", "content": "ok"}}]
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud", max_retries=5)
    result = client.chat(model="kimi:cloud", messages=[])
    assert result["message"]["content"] == "ok"
    assert wrapped.chat.call_count == 3


def test_retries_on_500():
    wrapped = _mk_wrapped()
    err = ollama.ResponseError("internal server error", status_code=500)
    wrapped.chat.side_effect = [err, {"message": {"role": "assistant", "content": "ok"}}]
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    result = client.chat(model="glm-5:cloud", messages=[])
    assert result["message"]["content"] == "ok"
    assert wrapped.chat.call_count == 2


def test_does_not_retry_on_401():
    wrapped = _mk_wrapped()
    err = ollama.ResponseError("unauthorized", status_code=401)
    wrapped.chat.side_effect = err
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    with pytest.raises(ollama.ResponseError):
        client.chat(model="kimi:cloud", messages=[])
    assert wrapped.chat.call_count == 1


def test_does_not_retry_on_model_not_found():
    wrapped = _mk_wrapped()
    err = ollama.ResponseError("model not found", status_code=404)
    wrapped.chat.side_effect = err
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    with pytest.raises(ollama.ResponseError):
        client.chat(model="nonexistent:cloud", messages=[])
    assert wrapped.chat.call_count == 1


def test_gives_up_after_max_retries():
    wrapped = _mk_wrapped()
    err = ollama.ResponseError("internal error", status_code=503)
    wrapped.chat.side_effect = err
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud", max_retries=2)
    with pytest.raises(ollama.ResponseError):
        client.chat(model="kimi:cloud", messages=[])
    # 1 initial + 2 retries = 3 calls
    assert wrapped.chat.call_count == 3


def test_retries_on_timeout_exception():
    wrapped = _mk_wrapped()
    wrapped.chat.side_effect = [TimeoutError("read timeout"), {"message": {"role": "assistant", "content": "ok"}}]
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    result = client.chat(model="kimi:cloud", messages=[])
    assert result["message"]["content"] == "ok"


def test_passes_through_non_chat_methods():
    wrapped = _mk_wrapped()
    wrapped.list.return_value = {"models": ["a", "b"]}
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    # __getattr__ should forward .list()
    result = client.list()
    assert result == {"models": ["a", "b"]}


def test_streaming_yields_chunks():
    wrapped = _mk_wrapped()
    chunks = [
        {"message": {"role": "assistant", "content": "he"}, "done": False},
        {"message": {"role": "assistant", "content": "llo"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    wrapped.chat.return_value = iter(chunks)
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    result = list(client.chat(model="kimi:cloud", messages=[], stream=True))
    assert len(result) == 3
    assert result[0]["message"]["content"] == "he"


def test_streaming_retries_on_opening_failure():
    wrapped = _mk_wrapped()
    err = ollama.ResponseError("rate limit", status_code=429)
    good_chunks = [
        {"message": {"role": "assistant", "content": "hi"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    wrapped.chat.side_effect = [err, iter(good_chunks)]
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    result = list(client.chat(model="kimi:cloud", messages=[], stream=True))
    assert len(result) == 2
    assert wrapped.chat.call_count == 2


def test_streaming_midstream_failure_propagates():
    """Once a stream has opened, a mid-stream exception cannot be retried."""
    wrapped = _mk_wrapped()

    def _bad_stream():
        yield {"message": {"role": "assistant", "content": "part"}, "done": False}
        raise ollama.ResponseError("connection lost", status_code=500)

    wrapped.chat.return_value = _bad_stream()
    client = ResilientOllamaClient(wrapped, provider_label="ollama_cloud")
    with pytest.raises(ollama.ResponseError):
        list(client.chat(model="kimi:cloud", messages=[], stream=True))


def test_credential_rotation_on_rate_limit(monkeypatch):
    """When pool has multiple keys, rate_limit triggers rotation."""
    from aura.providers.credential_pool import CredentialPool

    pool = CredentialPool()
    monkeypatch.setenv("OLLAMA_API_KEY", "key_alpha,key_bravo")
    pool.register("ollama_cloud", "OLLAMA_API_KEY")
    # Patch global pool
    monkeypatch.setattr("aura.providers.credential_pool._GLOBAL_POOL", pool)

    rebuild_calls = []

    def _rebuild(new_key):
        rebuild_calls.append(new_key)
        m = _mk_wrapped()
        m.chat.return_value = {"message": {"role": "assistant", "content": "ok after rotate"}}
        return m

    wrapped = _mk_wrapped()
    err = ollama.ResponseError("rate limit exceeded", status_code=429)
    wrapped.chat.side_effect = [err]

    client = ResilientOllamaClient(
        wrapped,
        provider_label="ollama_cloud",
        api_key="key_alpha",
        rebuild_client=_rebuild,
    )
    # First call hits rate limit → cooldown on key_alpha → rotate to key_bravo → retry succeeds
    result = client.chat(model="kimi:cloud", messages=[])
    assert result["message"]["content"] == "ok after rotate"
    assert len(rebuild_calls) == 1
    assert rebuild_calls[0] == "key_bravo"


def test_no_rotation_without_rebuild_factory():
    """If no rebuild_client given, credential rotation is a no-op (no crash)."""
    wrapped = _mk_wrapped()
    err = ollama.ResponseError("rate limit", status_code=429)
    wrapped.chat.side_effect = [err, {"message": {"role": "assistant", "content": "ok"}}]
    client = ResilientOllamaClient(
        wrapped, provider_label="ollama_cloud",
        api_key="single_key", rebuild_client=None,
    )
    result = client.chat(model="kimi:cloud", messages=[])
    assert result["message"]["content"] == "ok"
