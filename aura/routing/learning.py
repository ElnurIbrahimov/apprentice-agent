"""Feedback signal processing that updates model profiles."""

from __future__ import annotations

from aura.routing.profiles import DIMENSIONS

LEARNING_RATE = 0.05
RELEVANCE_THRESHOLD = 0.3  # only update dimensions the task actually needed

SIGNAL_WEIGHTS: dict[str, float] = {
    "regeneration": -0.3,
    "model_switch": -0.2,
    "model_switch_to": 0.1,
    "response_gap_good": 0.05,
    "response_gap_bad": -0.05,
    "thumbs_up": 0.4,
    "thumbs_down": -0.4,
    "conversation_turn": 0.02,
    "abort": -0.15,
}


def process_feedback(
    signal: str,
    model: str,
    task_dimensions: dict[str, float],
    profile_store,
    switched_to: str | None = None,
) -> None:
    """Apply a feedback *signal* to *model*'s profile via *profile_store*.

    For ``model_switch`` with *switched_to*, the switched-to model also
    receives a positive boost using the ``model_switch_to`` weight.
    """
    weight = SIGNAL_WEIGHTS.get(signal)
    if weight is None:
        return

    _apply(weight, model, task_dimensions, profile_store)

    if signal == "model_switch" and switched_to is not None:
        boost_weight = SIGNAL_WEIGHTS["model_switch_to"]
        _apply(boost_weight, switched_to, task_dimensions, profile_store)


def _apply(
    weight: float,
    model: str,
    task_dimensions: dict[str, float],
    profile_store,
) -> None:
    deltas: dict[str, float] = {}
    for dim in DIMENSIONS:
        if dim == "vision":
            continue
        relevance = task_dimensions.get(dim, 0.0)
        if relevance >= RELEVANCE_THRESHOLD:
            deltas[dim] = LEARNING_RATE * weight * relevance
    if deltas:
        profile_store.update(model, deltas)
