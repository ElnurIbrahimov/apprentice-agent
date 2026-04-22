"""Tests for HMAC-signed MCTS value-function pickle loads."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from aura.tools.mcts_value_fn import (
    ValuePredictor,
    _sign_model,
    _sig_path,
    _verify_model,
)


def _write_dummy_model(tmp_path: Path, *, sign: bool = True) -> Path:
    """Drop a fake pickle payload on disk, optionally signed."""
    model_path = tmp_path / "model.pkl"
    payload = pickle.dumps({
        "model": {"fake": True},
        "n_samples": 100,
        "rmse": 0.1,
        "trained_at": 0.0,
        "embedding_dim": 8,
    })
    model_path.write_bytes(payload)
    if sign:
        _sign_model(model_path, payload)
    return model_path


def test_verify_accepts_freshly_signed(tmp_path: Path):
    model_path = _write_dummy_model(tmp_path, sign=True)
    data = model_path.read_bytes()
    assert _verify_model(model_path, data) is True


def test_verify_rejects_missing_sig(tmp_path: Path):
    model_path = _write_dummy_model(tmp_path, sign=False)
    data = model_path.read_bytes()
    assert _verify_model(model_path, data) is False


def test_verify_rejects_tampered_payload(tmp_path: Path):
    model_path = _write_dummy_model(tmp_path, sign=True)
    # Replace payload after signing — sig no longer matches.
    model_path.write_bytes(b"\x80\x04K\x00.")  # short harmless pickle
    tampered = model_path.read_bytes()
    assert _verify_model(model_path, tampered) is False


def test_verify_rejects_tampered_sig(tmp_path: Path):
    model_path = _write_dummy_model(tmp_path, sign=True)
    data = model_path.read_bytes()
    # Flip one byte of the sig.
    sig_path = _sig_path(model_path)
    sig_path.write_text("0" * 64, encoding="ascii")
    assert _verify_model(model_path, data) is False


def test_predictor_refuses_unsigned_file(tmp_path: Path):
    """An attacker drops a pickle but doesn't know the HMAC key → refused."""
    malicious = tmp_path / "model.pkl"
    # The pickle below would be an RCE opportunity without HMAC gating.
    malicious.write_bytes(pickle.dumps({
        "model": {"bogus": True},
        "n_samples": 0,
        "rmse": 1.0,
        "trained_at": 0.0,
        "embedding_dim": 8,
    }))
    predictor = ValuePredictor(model_path=malicious)
    # _ensure_loaded should refuse because sig sidecar is missing.
    assert predictor._ensure_loaded() is False
    assert predictor._model is None


def test_predictor_accepts_own_written_file(tmp_path: Path, monkeypatch):
    """Round-trip: write + sign + load works."""
    model_path = _write_dummy_model(tmp_path, sign=True)
    # Stub the embedder import so _ensure_loaded can complete.
    monkeypatch.setattr(
        "aura.tools.mcts_value_fn._load_embedder",
        lambda: (lambda text: [0.0] * 8),
    )
    predictor = ValuePredictor(model_path=model_path)
    assert predictor._ensure_loaded() is True
    assert predictor._model == {"fake": True}


def test_key_file_not_reused_across_paths(tmp_path: Path):
    """Each model dir gets its own key so a leak of one doesn't compromise
    others."""
    path_a = tmp_path / "a" / "model.pkl"
    path_b = tmp_path / "b" / "model.pkl"
    path_a.parent.mkdir()
    path_b.parent.mkdir()
    data_a = pickle.dumps({"x": 1})
    data_b = pickle.dumps({"x": 2})
    path_a.write_bytes(data_a)
    path_b.write_bytes(data_b)
    _sign_model(path_a, data_a)
    _sign_model(path_b, data_b)
    # Swap sigs — each should reject the other's signature.
    sig_a = _sig_path(path_a).read_text()
    sig_b = _sig_path(path_b).read_text()
    assert sig_a != sig_b
    # Cross-signing doesn't validate.
    _sig_path(path_a).write_text(sig_b)
    assert _verify_model(path_a, data_a) is False
