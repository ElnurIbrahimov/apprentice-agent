"""Tests for multi-model debate mode."""
import time
import pytest
from unittest.mock import MagicMock, patch
from concurrent.futures import TimeoutError as FuturesTimeoutError

from aura.cli.debate_mode import (
    DebatePosition,
    DebateResult,
    parse_debate_args,
    _select_models,
    run_debate,
    DEFAULT_MODELS,
    FALLBACK_MODELS,
    DEBATER_TIMEOUT,
    ROLE_CONFIG,
)


# ── parse_debate_args ────────────────────────────────────────────────

def test_parse_debate_args_question_only():
    question, models = parse_debate_args("should I use SQLite or Postgres?")
    assert question == "should I use SQLite or Postgres?"
    assert models is None


def test_parse_debate_args_with_models():
    question, models = parse_debate_args(
        "--models kimi,deepseek,chatgpt should I use SQLite or Postgres?"
    )
    assert question == "should I use SQLite or Postgres?"
    assert models == "kimi,deepseek,chatgpt"


def test_parse_debate_args_empty_input():
    question, models = parse_debate_args("")
    assert question == ""
    assert models is None


def test_parse_debate_args_models_no_question():
    """--models flag with model list but no question."""
    question, models = parse_debate_args("--models kimi,deepseek")
    assert question == ""
    assert models == "kimi,deepseek"


def test_parse_debate_args_models_alone():
    """--models flag with nothing after it."""
    question, models = parse_debate_args("--models")
    assert question == ""
    assert models is None


def test_parse_debate_args_whitespace():
    question, models = parse_debate_args("   is Rust better than Go?   ")
    assert question == "is Rust better than Go?"
    assert models is None


# ── _select_models ───────────────────────────────────────────────────

def test_select_models_default_with_chatgpt():
    brain = MagicMock()
    brain._chatgpt_client = MagicMock()  # ChatGPT available
    models = _select_models(brain)
    assert models == DEFAULT_MODELS
    assert "analyst" in models


def test_select_models_fallback_no_chatgpt():
    brain = MagicMock()
    brain._chatgpt_client = None
    models = _select_models(brain)
    assert models == FALLBACK_MODELS
    assert "analyst" not in models
    assert len(models) == 2


def test_select_models_user_override():
    brain = MagicMock()
    brain._chatgpt_client = None
    models = _select_models(brain, user_models="kimi,deepseek,chatgpt")
    assert models["advocate"] == "kimi-k2.6:cloud"
    assert models["critic"] == "deepseek-v3.2:cloud"
    assert models["analyst"] == "chatgpt:gpt-5.4"


def test_select_models_user_override_two_models():
    brain = MagicMock()
    models = _select_models(brain, user_models="qwen,minimax")
    assert models["advocate"] == "qwen3.5:397b-cloud"
    assert models["critic"] == "minimax-m2.7:cloud"
    assert "analyst" not in models


def test_select_models_user_override_raw_model_name():
    brain = MagicMock()
    models = _select_models(brain, user_models="my-custom-model:latest")
    assert models["advocate"] == "my-custom-model:latest"


# ── run_debate (mock brain) ──────────────────────────────────────────

@patch("aura.cli.debate_mode.Console")
@patch("aura.cli.debate_mode.Live")
@patch("aura.config.Config")
def test_run_debate_basic(mock_config, mock_live_cls, mock_console_cls):
    """Basic debate with mock brain returning canned responses."""
    mock_config.MODEL_THINK = "judge-model"

    brain = MagicMock()
    brain._chatgpt_client = None  # 2 debaters only

    call_count = {"n": 0}
    def fake_think(prompt, system_prompt=None, use_history=False, model_override=None):
        call_count["n"] += 1
        if "synthesize" in (system_prompt or "").lower() or "synthesize" in prompt.lower():
            return "Synthesis: both sides made good points."
        return f"Response from {model_override or 'default'}"

    brain.think = fake_think

    # Make Live context manager work
    mock_live = MagicMock()
    mock_live.__enter__ = MagicMock(return_value=mock_live)
    mock_live.__exit__ = MagicMock(return_value=False)
    mock_live_cls.return_value = mock_live

    result = run_debate(brain, "SQLite or Postgres?")

    assert isinstance(result, DebateResult)
    assert result.question == "SQLite or Postgres?"
    assert len(result.positions) == 2  # fallback models (no chatgpt)
    assert all(p.done for p in result.positions)
    assert result.synthesis  # synthesis should be non-empty


@patch("aura.cli.debate_mode.Console")
@patch("aura.cli.debate_mode.Live")
@patch("aura.config.Config")
def test_run_debate_three_models(mock_config, mock_live_cls, mock_console_cls):
    """Debate with 3 models (ChatGPT available)."""
    mock_config.MODEL_THINK = "judge-model"

    brain = MagicMock()
    brain._chatgpt_client = MagicMock()  # ChatGPT available

    brain.think = MagicMock(return_value="A thoughtful response")

    mock_live = MagicMock()
    mock_live.__enter__ = MagicMock(return_value=mock_live)
    mock_live.__exit__ = MagicMock(return_value=False)
    mock_live_cls.return_value = mock_live

    result = run_debate(brain, "Tabs or spaces?")

    assert len(result.positions) == 3
    assert {p.role for p in result.positions} == {"advocate", "critic", "analyst"}


@patch("aura.cli.debate_mode.Console")
@patch("aura.cli.debate_mode.Live")
@patch("aura.config.Config")
def test_run_debate_dict_response(mock_config, mock_live_cls, mock_console_cls):
    """brain.think returns dict instead of string."""
    mock_config.MODEL_THINK = "judge-model"

    brain = MagicMock()
    brain._chatgpt_client = None

    brain.think = MagicMock(return_value={"response": "Dict-based answer"})

    mock_live = MagicMock()
    mock_live.__enter__ = MagicMock(return_value=mock_live)
    mock_live.__exit__ = MagicMock(return_value=False)
    mock_live_cls.return_value = mock_live

    result = run_debate(brain, "Test question")

    assert all(p.done for p in result.positions)
    # At least one position should have the dict-extracted response
    assert any("Dict-based answer" in p.argument for p in result.positions)


@patch("aura.cli.debate_mode.Console")
@patch("aura.cli.debate_mode.Live")
@patch("aura.config.Config")
def test_run_debate_timeout(mock_config, mock_live_cls, mock_console_cls):
    """Test that a slow brain.think triggers timeout handling."""
    mock_config.MODEL_THINK = "judge-model"

    brain = MagicMock()
    brain._chatgpt_client = None

    def slow_think(*args, **kwargs):
        # Simulate slow response — the debate runner wraps this in a future with timeout
        time.sleep(120)
        return "too late"

    brain.think = slow_think

    mock_live = MagicMock()
    mock_live.__enter__ = MagicMock(return_value=mock_live)
    mock_live.__exit__ = MagicMock(return_value=False)
    mock_live_cls.return_value = mock_live

    # Patch DEBATER_TIMEOUT to a very short value for test speed
    with patch("aura.cli.debate_mode.DEBATER_TIMEOUT", 0.1):
        result = run_debate(brain, "Will this time out?")

    # At least the debater positions should have timed out
    assert all(p.done for p in result.positions)
    assert any(p.error and "Timed out" in p.error for p in result.positions)


@patch("aura.cli.debate_mode.Console")
@patch("aura.cli.debate_mode.Live")
@patch("aura.config.Config")
def test_run_debate_exception_handling(mock_config, mock_live_cls, mock_console_cls):
    """brain.think raises an exception."""
    mock_config.MODEL_THINK = "judge-model"

    brain = MagicMock()
    brain._chatgpt_client = None

    brain.think = MagicMock(side_effect=RuntimeError("connection lost"))

    mock_live = MagicMock()
    mock_live.__enter__ = MagicMock(return_value=mock_live)
    mock_live.__exit__ = MagicMock(return_value=False)
    mock_live_cls.return_value = mock_live

    result = run_debate(brain, "Error test")

    assert all(p.done for p in result.positions)
    assert any("connection lost" in p.error for p in result.positions)


# ── Data class smoke tests ───────────────────────────────────────────

def test_debate_position_defaults():
    pos = DebatePosition(model="test", role="advocate")
    assert pos.argument == ""
    assert pos.elapsed == 0.0
    assert pos.done is False
    assert pos.error == ""


def test_debate_result_defaults():
    result = DebateResult(question="test")
    assert result.positions == []
    assert result.synthesis == ""
    assert result.total_elapsed == 0.0


def test_role_config_completeness():
    assert set(ROLE_CONFIG.keys()) == {"advocate", "critic", "analyst"}
    for role, cfg in ROLE_CONFIG.items():
        assert "emoji" in cfg
        assert "label" in cfg
        assert "color" in cfg
        assert "system" in cfg
