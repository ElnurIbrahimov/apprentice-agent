"""Tests for aura.memory.context_compressor."""
from __future__ import annotations

from aura.memory.context_compressor import (
    CompressionResult,
    _estimate_tokens,
    _is_tool_result,
    compress_history,
)


def _large_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_bail_below_threshold():
    messages = [_large_msg("user", "hi"), _large_msg("assistant", "hello")]
    result = compress_history(messages, threshold_tokens=10000)
    assert not result.was_compressed
    assert "below threshold" in (result.skipped_reason or "")


def test_bail_when_history_too_short():
    messages = [_large_msg("user", "x" * 100000)]  # lots of tokens but one message
    result = compress_history(messages, threshold_tokens=1000)
    assert not result.was_compressed
    assert "history too short" in (result.skipped_reason or "")


def test_tool_result_pruning_happens():
    big_tool_content = "x" * 5000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "use the tool"},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": big_tool_content},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "yw"},
        {"role": "user", "content": "another"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "ok"},
    ] + [_large_msg("user", "y" * 4000) for _ in range(5)]

    # No summarizer → only pruning + sanitization
    result = compress_history(
        messages,
        threshold_tokens=1000,
        keep_last=3,
        keep_first=1,
        tool_result_prune_threshold_chars=100,
        summarize_fn=None,
    )
    # Pruning happens in phase 1 but anti-thrash may skip if reduction < 20%
    assert "tool_result_pruning" in result.phases_applied


def test_anti_thrash_skip():
    """Compression that would only reduce tokens <20% should be skipped."""
    small_middle = "tiny" * 5
    messages = (
        [_large_msg("system", "s")]
        + [_large_msg("user", small_middle)] * 20
        + [_large_msg("assistant", "x" * 40000)]  # huge tail we keep
    )
    result = compress_history(
        messages,
        threshold_tokens=5000,
        keep_last=1,
        keep_first=1,
        tool_result_prune_threshold_chars=1000,
        summarize_fn=None,
        min_reduction_pct=0.5,  # demand 50% reduction — won't happen
    )
    assert not result.was_compressed
    assert "reduction" in (result.skipped_reason or "")


def test_summarize_fn_replaces_middle():
    def fake_summarize(prompt: str) -> str:
        return "## Active Task\nhelp user with X\n## Completed Actions\n- did Y"

    messages = [
        _large_msg("system", "s"),
        *[_large_msg("user", "x" * 500) for _ in range(40)],
        *[_large_msg("user", "tail") for _ in range(3)],
    ]
    result = compress_history(
        messages,
        threshold_tokens=1000,
        keep_last=3,
        keep_first=1,
        tool_result_prune_threshold_chars=600,
        summarize_fn=fake_summarize,
        min_reduction_pct=0.1,
    )
    assert result.was_compressed
    assert "llm_summarization" in result.phases_applied
    # Middle should be exactly one synthesized message
    summary_msg = next(
        m for m in result.compressed if "<compressed-history>" in m.get("content", "")
    )
    assert "Active Task" in summary_msg["content"]
    assert "System note" in summary_msg["content"]


def test_orphan_tool_call_sanitized_after_summary():
    """After summarization, tool_calls in the middle no longer have matching
    results — the orphan tool_calls field should be dropped."""
    def fake_summarize(prompt: str) -> str:
        return "summary"

    messages = [
        _large_msg("system", "s"),
        _large_msg("user", "x" * 40000),
        {"role": "assistant", "content": "", "tool_calls": [{"id": "orphan"}]},
        _large_msg("tool", "y" * 10000),  # will be pruned
        _large_msg("user", "tail"),
    ]
    result = compress_history(
        messages,
        threshold_tokens=5000,
        keep_last=1,
        keep_first=1,
        tool_result_prune_threshold_chars=100,
        summarize_fn=fake_summarize,
        min_reduction_pct=0.05,
    )
    # No message in compressed output should have orphan tool_calls
    for msg in result.compressed:
        if msg.get("tool_calls"):
            # if present, its id must have a matching tool-result in the output
            ids = {tc.get("id") for tc in msg["tool_calls"]}
            results = {m.get("tool_call_id") for m in result.compressed if m.get("tool_call_id")}
            assert ids.issubset(results), f"orphan tool_call survived: {ids - results}"


def test_is_tool_result_detection():
    assert _is_tool_result({"role": "tool", "content": "x"})
    assert _is_tool_result({"role": "user", "content": "x", "tool_call_id": "c1"})
    assert not _is_tool_result({"role": "user", "content": "x"})


def test_estimate_tokens_is_reasonable():
    messages = [{"role": "user", "content": "a" * 400}]
    assert 90 < _estimate_tokens(messages) < 110  # ~400/4 = 100


def test_compression_result_reduction_pct():
    r = CompressionResult(
        compressed=[], original_token_estimate=100, compressed_token_estimate=60,
    )
    assert 0.39 < r.reduction_pct < 0.41


def test_zero_original_no_divzero():
    r = CompressionResult(compressed=[], original_token_estimate=0, compressed_token_estimate=0)
    assert r.reduction_pct == 0.0
