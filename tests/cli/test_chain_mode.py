"""Tests for prompt chain mode."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aura.cli.chain_mode import (
    ChainStep,
    Chain,
    ChainResult,
    parse_chain,
    run_chain,
    save_chain,
    load_chain,
    list_chains,
    delete_chain,
    _inject_context,
    MAX_PREV_OUTPUT_CHARS,
    CHAINS_DIR,
)


# ── parse_chain ──────────────────────────────────────────────────────

def test_parse_chain_single_step():
    steps = parse_chain("summarize this article")
    assert len(steps) == 1
    assert steps[0].prompt_template == "summarize this article"
    assert steps[0].model is None


def test_parse_chain_multiple_steps():
    steps = parse_chain("research X -> summarize -> format as markdown")
    assert len(steps) == 3
    assert steps[0].prompt_template == "research X"
    assert steps[1].prompt_template == "summarize"
    assert steps[2].prompt_template == "format as markdown"


def test_parse_chain_with_model_overrides():
    steps = parse_chain("research X @kimi-k2.6:cloud -> summarize @nemotron-3-super:cloud")
    assert len(steps) == 2
    assert steps[0].prompt_template == "research X"
    assert steps[0].model == "kimi-k2.6:cloud"
    assert steps[1].prompt_template == "summarize"
    assert steps[1].model == "nemotron-3-super:cloud"


def test_parse_chain_empty_input():
    steps = parse_chain("")
    assert steps == []


def test_parse_chain_strips_quotes():
    steps = parse_chain("'step one' -> \"step two\"")
    assert len(steps) == 2
    assert steps[0].prompt_template == "step one"
    assert steps[1].prompt_template == "step two"


def test_parse_chain_empty_arrows():
    """Empty segments between arrows are skipped."""
    steps = parse_chain("step1 -> -> step2")
    assert len(steps) == 2


def test_parse_chain_model_with_spaces_not_treated_as_model():
    """'@some thing with spaces' should NOT be parsed as a model override."""
    steps = parse_chain("tell me about @the big concept")
    assert len(steps) == 1
    # The @the should not be treated as model because 'big concept' has spaces
    # Actually: rfind(" @") finds " @the", candidate is "the big concept" which has spaces
    # So it should not be treated as model
    assert steps[0].model is None


# ── _inject_context ──────────────────────────────────────────────────

def test_inject_context_no_prev():
    step = ChainStep(prompt_template="do something")
    result = _inject_context(step, "")
    assert result == "do something"


def test_inject_context_with_placeholder():
    step = ChainStep(prompt_template="Summarize: {prev}")
    result = _inject_context(step, "The original text here")
    assert result == "Summarize: The original text here"


def test_inject_context_auto_prepend():
    step = ChainStep(prompt_template="now format it")
    result = _inject_context(step, "some previous output")
    assert "Based on the following context" in result
    assert "some previous output" in result
    assert "now format it" in result


def test_inject_context_truncation():
    step = ChainStep(prompt_template="summarize {prev}")
    long_output = "x" * (MAX_PREV_OUTPUT_CHARS + 1000)
    result = _inject_context(step, long_output)
    # Should be truncated and have the ellipsis prefix
    assert "..." in result
    assert len(result) < len(long_output) + 100


# ── run_chain ────────────────────────────────────────────────────────

def test_run_chain_two_steps():
    brain = MagicMock()
    brain.think = MagicMock(side_effect=["First output", "Second output"])

    steps = [
        ChainStep(prompt_template="step 1"),
        ChainStep(prompt_template="step 2 with {prev}"),
    ]
    result = run_chain(brain, steps)

    assert isinstance(result, ChainResult)
    assert len(result.step_results) == 2
    assert result.step_results[0]["response"] == "First output"
    assert result.step_results[1]["response"] == "Second output"
    assert result.success is True

    # Verify {prev} substitution happened in the second call
    second_call_prompt = brain.think.call_args_list[1][0][0]
    assert "First output" in second_call_prompt


def test_run_chain_prev_substitution():
    brain = MagicMock()
    brain.think = MagicMock(side_effect=["alpha", "beta"])

    steps = [
        ChainStep(prompt_template="generate"),
        ChainStep(prompt_template="refine: {prev}"),
    ]
    run_chain(brain, steps)

    # The second call should have "alpha" injected
    second_prompt = brain.think.call_args_list[1][0][0]
    assert "refine: alpha" == second_prompt


def test_run_chain_model_override():
    brain = MagicMock()
    brain.think = MagicMock(return_value="done")

    steps = [ChainStep(prompt_template="test", model="custom-model:latest")]
    run_chain(brain, steps)

    _, kwargs = brain.think.call_args
    assert kwargs["model_override"] == "custom-model:latest"


def test_run_chain_dict_response():
    brain = MagicMock()
    brain.think = MagicMock(return_value={"response": "from dict"})

    steps = [ChainStep(prompt_template="test")]
    result = run_chain(brain, steps)

    assert result.step_results[0]["response"] == "from dict"


def test_run_chain_step_failure():
    brain = MagicMock()
    brain.think = MagicMock(side_effect=RuntimeError("LLM down"))

    steps = [ChainStep(prompt_template="will fail")]
    result = run_chain(brain, steps)

    assert result.success is False
    assert "[Error:" in result.step_results[0]["response"]


def test_run_chain_on_step_callback():
    brain = MagicMock()
    brain.think = MagicMock(return_value="ok")

    callback_log = []

    def on_step(step_num, total, result_dict):
        callback_log.append((step_num, total, result_dict["response"]))

    steps = [ChainStep(prompt_template="a"), ChainStep(prompt_template="b")]
    run_chain(brain, steps, on_step=on_step)

    assert len(callback_log) == 2
    assert callback_log[0] == (1, 2, "ok")
    assert callback_log[1] == (2, 2, "ok")


def test_run_chain_none_response():
    brain = MagicMock()
    brain.think = MagicMock(return_value=None)

    steps = [ChainStep(prompt_template="test")]
    result = run_chain(brain, steps)

    assert result.step_results[0]["response"] == ""


# ── save / load / list / delete ──────────────────────────────────────

def test_save_load_round_trip(tmp_path):
    with patch("aura.cli.chain_mode.CHAINS_DIR", tmp_path):
        steps = [
            ChainStep(prompt_template="step 1", model=None),
            ChainStep(prompt_template="step 2", model="custom:latest"),
        ]
        path = save_chain("my-chain", steps)
        assert path.exists()

        loaded = load_chain("my-chain")
        assert loaded is not None
        assert loaded.name == "my-chain"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].prompt_template == "step 1"
        assert loaded.steps[1].model == "custom:latest"


def test_load_chain_not_found(tmp_path):
    with patch("aura.cli.chain_mode.CHAINS_DIR", tmp_path):
        assert load_chain("nonexistent") is None


def test_load_chain_corrupt_json(tmp_path):
    with patch("aura.cli.chain_mode.CHAINS_DIR", tmp_path):
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        assert load_chain("bad") is None


def test_list_chains(tmp_path):
    with patch("aura.cli.chain_mode.CHAINS_DIR", tmp_path):
        save_chain("alpha", [ChainStep(prompt_template="a")])
        save_chain("beta", [ChainStep(prompt_template="b")])

        names = list_chains()
        assert "alpha" in names
        assert "beta" in names
        assert names == sorted(names)


def test_list_chains_empty(tmp_path):
    with patch("aura.cli.chain_mode.CHAINS_DIR", tmp_path):
        assert list_chains() == []


def test_list_chains_no_dir():
    with patch("aura.cli.chain_mode.CHAINS_DIR", Path("/nonexistent/path")):
        assert list_chains() == []


def test_delete_chain(tmp_path):
    with patch("aura.cli.chain_mode.CHAINS_DIR", tmp_path):
        save_chain("to-delete", [ChainStep(prompt_template="x")])
        assert delete_chain("to-delete") is True
        assert not (tmp_path / "to-delete.json").exists()


def test_delete_chain_not_found(tmp_path):
    with patch("aura.cli.chain_mode.CHAINS_DIR", tmp_path):
        assert delete_chain("nonexistent") is False
