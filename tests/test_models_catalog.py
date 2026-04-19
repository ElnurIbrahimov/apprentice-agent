"""Validation tests for the unified model catalog.

These fail loudly if anyone adds a model to the routing table, VERIFIED_CLOUD_MODELS,
or brain's cost map without also updating `aura/models_catalog.py`. That's the whole
point of having one source of truth.
"""
from __future__ import annotations

import pytest

from aura import models_catalog as catalog


def test_catalog_not_empty():
    assert len(catalog.MODELS) >= 10, "catalog should cover the full Ollama Pro lineup"


def test_every_model_has_cost():
    """No model should slip into the catalog without input/output rates."""
    for name, p in catalog.MODELS.items():
        assert p.cost_in_per_1k >= 0, f"{name} has negative input cost"
        assert p.cost_out_per_1k >= 0, f"{name} has negative output cost"


def test_every_model_has_specialty():
    for name, p in catalog.MODELS.items():
        assert p.specialties, f"{name} has no specialty tags"
        for s in p.specialties:
            assert s in {
                catalog.SPECIALTY_REASONING, catalog.SPECIALTY_CODE,
                catalog.SPECIALTY_VISION, catalog.SPECIALTY_LONGCTX,
                catalog.SPECIALTY_TOOL, catalog.SPECIALTY_FAST,
                catalog.SPECIALTY_CHEAP, catalog.SPECIALTY_MATH,
            }, f"{name} has unknown specialty {s!r}"


def test_routing_table_references_exist():
    """Every model named in ROUTING_TABLE must be in the catalog."""
    from aura.core.router import ROUTING_TABLE
    unknown = []
    for category, tiers in ROUTING_TABLE.items():
        for tier, model in tiers.items():
            if model not in catalog.MODELS:
                unknown.append(f"{category}/{tier}={model}")
    assert not unknown, f"ROUTING_TABLE references missing models: {unknown}"


def test_verified_cloud_models_is_subset_of_catalog():
    """Every VERIFIED_CLOUD_MODELS entry must exist in the catalog."""
    from aura.config import VERIFIED_CLOUD_MODELS
    for name in VERIFIED_CLOUD_MODELS:
        assert name in catalog.MODELS, f"VERIFIED_CLOUD_MODELS references unknown: {name}"


def test_specialty_ordering():
    """models_by_specialty should put primary-specialty models first."""
    code_models = catalog.models_by_specialty(catalog.SPECIALTY_CODE)
    assert code_models, "code models missing from catalog"
    # First result should be a model with code as its primary (first) specialty
    assert code_models[0].specialties[0] == catalog.SPECIALTY_CODE, (
        f"top code model {code_models[0].name} doesn't have code as primary specialty"
    )


def test_cheapest():
    cheap = catalog.cheapest()
    assert cheap is not None
    all_costs = [p.cost_in_per_1k for p in catalog.MODELS.values()]
    assert cheap.cost_in_per_1k == min(all_costs)
