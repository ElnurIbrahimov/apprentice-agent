"""Tests for budget-aware routing in ModelRouterMixin."""

from types import SimpleNamespace
from unittest.mock import patch


class _Mixin:
    """Standalone test harness that mixes in just the budget helpers."""

    def __init__(self, chain, costs, override=None):
        from aura.core.model_router_mixin import ModelRouterMixin
        self._mixin = ModelRouterMixin()
        # Bind instance attributes that _apply_budget_mode reads
        self._mixin._model_override = override
        self._mixin._MODEL_COST_PER_1K = costs
        self._mixin._DEFAULT_COST_PER_1K = (0.003, 0.003)
        self._mixin._session_cost = 0.0

        # Make _get_fallback_chain return our chain
        self._mixin._get_fallback_chain = lambda _m: chain
        # Fake get_session_stats
        self._mixin.get_session_stats = lambda: {"cost_usd": self._session_cost}

    @property
    def _session_cost(self):
        return getattr(self.__dict__.setdefault("_cost", 0.0), "__float__", lambda: 0.0)()

    def set_cost(self, value):
        self.__dict__["_cost"] = value
        self._mixin.get_session_stats = lambda: {"cost_usd": value}

    def apply(self, model, task_type=None):
        return self._mixin._apply_budget_mode(model, task_type)


def _make_mixin(chain, costs, override=None, session_cost=0.0):
    """Build a ModelRouterMixin-bound dummy instance with controlled state."""
    from aura.core.model_router_mixin import ModelRouterMixin

    class _Dummy(ModelRouterMixin):
        pass

    d = _Dummy()
    d._model_override = override
    d._MODEL_COST_PER_1K = costs
    d._DEFAULT_COST_PER_1K = (0.003, 0.003)
    d._get_fallback_chain = lambda _m: chain
    d.get_session_stats = lambda: {"cost_usd": session_cost}
    return d


def test_budget_mode_off_returns_model_unchanged():
    chain = ["premium-model", "cheap-model"]
    costs = {"premium-model": (0.01, 0.01), "cheap-model": (0.0001, 0.0001)}
    d = _make_mixin(chain, costs, session_cost=100.0)  # way over any budget

    with patch("aura.config.Config.BUDGET_MODE", False):
        assert d._apply_budget_mode("premium-model") == "premium-model"


def test_budget_mode_downgrades_when_over_budget():
    chain = ["premium-model", "cheap-model"]
    costs = {"premium-model": (0.01, 0.01), "cheap-model": (0.0001, 0.0001)}
    d = _make_mixin(chain, costs, session_cost=1.0)

    with patch("aura.config.Config.BUDGET_MODE", True), \
         patch("aura.config.Config.BUDGET_MAX_USD_PER_SESSION", 0.5):
        assert d._apply_budget_mode("premium-model") == "cheap-model"


def test_budget_mode_keeps_model_under_budget():
    chain = ["premium-model", "cheap-model"]
    costs = {"premium-model": (0.01, 0.01), "cheap-model": (0.0001, 0.0001)}
    d = _make_mixin(chain, costs, session_cost=0.01)

    with patch("aura.config.Config.BUDGET_MODE", True), \
         patch("aura.config.Config.BUDGET_MAX_USD_PER_SESSION", 0.5):
        # Under budget and not SIMPLE → keep the selected model
        assert d._apply_budget_mode("premium-model") == "premium-model"


def test_simple_task_always_prefers_cheapest():
    from aura.brain import TaskType
    chain = ["premium-model", "cheap-model"]
    costs = {"premium-model": (0.01, 0.01), "cheap-model": (0.0001, 0.0001)}
    d = _make_mixin(chain, costs, session_cost=0.0)  # $0 spent, still SIMPLE

    with patch("aura.config.Config.BUDGET_MODE", True), \
         patch("aura.config.Config.BUDGET_MAX_USD_PER_SESSION", 0.5):
        assert d._apply_budget_mode("premium-model", task_type=TaskType.SIMPLE) == "cheap-model"


def test_manual_override_bypasses_budget_mode():
    chain = ["premium-model", "cheap-model"]
    costs = {"premium-model": (0.01, 0.01), "cheap-model": (0.0001, 0.0001)}
    d = _make_mixin(chain, costs, override="premium-model", session_cost=100.0)

    with patch("aura.config.Config.BUDGET_MODE", True), \
         patch("aura.config.Config.BUDGET_MAX_USD_PER_SESSION", 0.5):
        # User picked a specific model — respect that even over budget
        assert d._apply_budget_mode("premium-model") == "premium-model"


def test_cheapest_in_chain_ranks_by_combined_cost():
    costs = {
        "mid": (0.002, 0.002),
        "cheap": (0.0001, 0.0001),
        "premium": (0.01, 0.01),
    }
    d = _make_mixin(chain=["mid", "cheap", "premium"], costs=costs)
    assert d._cheapest_in_chain("mid") == "cheap"


def test_cheapest_in_chain_uses_default_cost_for_unknown_models():
    costs = {"known-cheap": (0.0001, 0.0001)}
    chain = ["unknown-model", "known-cheap"]
    d = _make_mixin(chain=chain, costs=costs)
    assert d._cheapest_in_chain("unknown-model") == "known-cheap"


def test_get_model_override_returns_current_override():
    """get_model_override is the public read counterpart of set_model_override.

    The CLI uses it in the quiet banner and the legacy brain-mirror re-sync
    path; it must return whatever set_model_override last set (or None).
    """
    from aura.core.model_router_mixin import ModelRouterMixin

    # _model_override is initialized lazily by the host class (the brain
    # initializes it in __init__). For a bare mixin we set it manually so
    # the public getter has something to read.
    mixin = ModelRouterMixin()
    mixin._model_override = None
    assert mixin.get_model_override() is None
    mixin.set_model_override("gpt-4")
    assert mixin.get_model_override() == "gpt-4"
    mixin.set_model_override(None)
    assert mixin.get_model_override() is None
