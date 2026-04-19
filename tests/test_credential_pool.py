"""Tests for aura.providers.credential_pool."""
from __future__ import annotations

import time

import pytest

from aura.providers.credential_pool import CredentialPool


@pytest.fixture
def pool():
    return CredentialPool()


def test_register_single_key(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "key_abc123")
    pool.register("testprov", "TEST_PROV_KEY")
    assert pool.pool_size("testprov") == 1
    assert pool.acquire("testprov") == "key_abc123"


def test_register_comma_separated(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "k1,k2,k3")
    pool.register("testprov", "TEST_PROV_KEY")
    assert pool.pool_size("testprov") == 3


def test_semicolon_separator(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "k1;k2;k3")
    pool.register("testprov", "TEST_PROV_KEY")
    assert pool.pool_size("testprov") == 3


def test_round_robin_rotation(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "k1,k2,k3")
    pool.register("testprov", "TEST_PROV_KEY")
    seen = [pool.acquire("testprov") for _ in range(6)]
    # Each key should appear twice in a round-robin pattern
    assert seen.count("k1") == 2
    assert seen.count("k2") == 2
    assert seen.count("k3") == 2


def test_exhausted_key_skipped(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "k1,k2")
    pool.register("testprov", "TEST_PROV_KEY")
    # First acquire lands on k1 (or k2 depending on index); exhaust whichever
    first = pool.acquire("testprov")
    pool.mark_exhausted("testprov", first, reason="rate_limit")
    # Next two acquires should NOT return the exhausted key
    for _ in range(4):
        k = pool.acquire("testprov")
        assert k != first


def test_all_cooling_returns_earliest(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "k1,k2")
    pool.register("testprov", "TEST_PROV_KEY")
    pool.mark_exhausted("testprov", "k1", reason="rate_limit", reset_seconds=10.0)
    pool.mark_exhausted("testprov", "k2", reason="billing", reset_seconds=100.0)
    # All cooling — should force a try on k1 (earliest expiry)
    result = pool.acquire("testprov")
    assert result == "k1"


def test_unregistered_provider_returns_none(pool):
    assert pool.acquire("nonexistent") is None


def test_empty_env_returns_none(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "")
    pool.register("testprov", "TEST_PROV_KEY")
    assert pool.acquire("testprov") is None


def test_status_reports_availability(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "key_one,key_two")
    pool.register("testprov", "TEST_PROV_KEY")
    pool.mark_exhausted("testprov", "key_two", reason="rate_limit")
    status = pool.status("testprov")
    assert len(status) == 2
    by_suffix = {s["key_suffix"]: s for s in status}
    assert by_suffix["_one"]["available"] is True
    assert by_suffix["_two"]["available"] is False
    assert by_suffix["_two"]["failure_count"] == 1


def test_explicit_reset_seconds_overrides_default(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "k1")
    pool.register("testprov", "TEST_PROV_KEY")
    pool.mark_exhausted("testprov", "k1", reason="rate_limit", reset_seconds=5.0)
    status = pool.status("testprov")
    assert 4.0 <= status[0]["cooldown_remaining"] <= 5.5


def test_hot_rotation_picks_up_new_keys(monkeypatch, pool):
    monkeypatch.setenv("TEST_PROV_KEY", "k1")
    pool.register("testprov", "TEST_PROV_KEY")
    assert pool.pool_size("testprov") == 1
    # User rotates env mid-session
    monkeypatch.setenv("TEST_PROV_KEY", "k1,k2,k3")
    pool.acquire("testprov")  # triggers load_from_env
    assert pool.pool_size("testprov") == 3
