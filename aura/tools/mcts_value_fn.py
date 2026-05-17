"""Learned value function for MCTS reasoning.

Extracts (text, value) training pairs from the MCTS TreeCache, fits a ridge
regression over nomic-embed-text embeddings, and exposes a predictor that
plugs into the main MCTS evaluation blend.

No torch. No per-node network. Just a shallow linear model on top of an
already-present embedding model. Good enough for a first bump that gets
smarter as the cache grows.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import pickle
import secrets
import stat
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("data/mcts_value")
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "model.pkl"
MIN_TRAINING_PAIRS = 50

# --- HMAC signing for pickle integrity ---------------------------------------
# `pickle.load` is arbitrary-code-exec on hostile input. We sign every model
# file we write and refuse to load anything without a valid HMAC. The key is
# generated on first save and stored next to the model with user-only perms.

_KEY_FILE_NAME = "key"
_SIG_SUFFIX = ".sig"


def _key_path(model_path: Path) -> Path:
    return model_path.parent / _KEY_FILE_NAME


def _sig_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + _SIG_SUFFIX)


def _get_or_create_key(model_path: Path) -> bytes:
    """Return the HMAC key for this model path, creating it on first use.

    Generates a 32-byte random key, writes it 0o600 (POSIX) next to the model.
    On Windows, permissions rely on filesystem ACLs — the data dir should
    already be user-scoped.
    """
    kp = _key_path(model_path)
    if kp.exists():
        raw = kp.read_bytes()
        if len(raw) >= 32:
            return raw
        logger.warning("[MCTSValueFn] key file at %s is too short — regenerating", kp)
    kp.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    kp.write_bytes(key)
    try:
        kp.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError:
        pass  # Windows — ACL-based
    return key


def _sign_model(model_path: Path, data_bytes: bytes) -> None:
    key = _get_or_create_key(model_path)
    sig = hmac.new(key, data_bytes, hashlib.sha256).hexdigest()
    _sig_path(model_path).write_text(sig, encoding="ascii")


def _verify_model(model_path: Path, data_bytes: bytes) -> bool:
    """Return True iff the signature file matches an HMAC we'd produce today.

    Missing key file, missing sig file, or mismatch all fail closed.
    """
    kp = _key_path(model_path)
    sp = _sig_path(model_path)
    if not (kp.exists() and sp.exists()):
        return False
    try:
        key = kp.read_bytes()
        stored = sp.read_text(encoding="ascii").strip()
    except OSError:
        return False
    expected = hmac.new(key, data_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(stored, expected)


def _walk_nodes(node: dict, out: List[dict], parent_content: str = "") -> None:
    """Depth-first walk of a serialized MCTS tree, flattening into records."""
    thought = node.get("thought") or {}
    content = thought.get("content", "") or ""
    visits = node.get("visits", 0)
    avg_value = node.get("avg_value", 0.0)
    combined = (parent_content + " | " + content).strip(" |")
    out.append({
        "text": combined[:2000],
        "avg_value": float(avg_value),
        "visits": int(visits),
    })
    for child in node.get("children", []):
        _walk_nodes(child, out, combined)


def extract_training_pairs(
    cache_dir: Optional[Path] = None,
    min_visits: int = 3,
) -> List[Tuple[str, float]]:
    """Walk every cached tree and collect (text, value) pairs.

    Nodes with visits >= min_visits are considered. Each record's text is the
    node's own content concatenated with the chain leading to it, which gives
    the predictor some path awareness without needing a sequence model.
    """
    if cache_dir is None:
        cache_dir = Path("data") / "mcts_cache"
    if not cache_dir.exists():
        return []

    pairs: List[Tuple[str, float]] = []
    for cache_file in cache_dir.glob("*.json"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        tree = data.get("tree")
        if not tree:
            continue
        flat: List[dict] = []
        _walk_nodes(tree, flat)
        for rec in flat:
            if rec["visits"] >= min_visits and rec["text"]:
                value = max(0.0, min(1.0, rec["avg_value"]))
                pairs.append((rec["text"], value))
    return pairs


def _load_embedder():
    """Reuse the _embed function from codebase_index which is already wired
    to nomic-embed-text via the local Ollama endpoint."""
    try:
        from aura.tools.codebase_index import _embed
        return _embed
    except Exception as e:
        logger.warning(f"[MCTSValueFn] cannot import _embed: {e}")
        return None


def train_value_fn(
    pairs: Optional[List[Tuple[str, float]]] = None,
    model_path: Path = DEFAULT_MODEL_FILE,
    cache_dir: Optional[Path] = None,
) -> dict:
    """Fit a Ridge regression on (text → value) pairs from the MCTS cache.

    Saves a pickled dict at `model_path`. Returns status/metadata. If the
    cache is too small or sklearn isn't available, skips gracefully.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
    except ImportError:
        return {"trained": False, "reason": "scikit-learn not installed"}

    import numpy as np

    if pairs is None:
        pairs = extract_training_pairs(cache_dir=cache_dir)

    if len(pairs) < MIN_TRAINING_PAIRS:
        return {"trained": False, "reason": f"not enough data ({len(pairs)} < {MIN_TRAINING_PAIRS})"}

    embed = _load_embedder()
    if embed is None:
        return {"trained": False, "reason": "embedder unavailable"}

    t0 = time.time()
    X_list: List[List[float]] = []
    y_list: List[float] = []
    for text, value in pairs:
        vec = embed(text)
        if not vec:
            continue
        X_list.append(vec)
        y_list.append(value)

    if len(X_list) < MIN_TRAINING_PAIRS:
        return {"trained": False, "reason": f"too few embeddings returned ({len(X_list)})"}

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, predictions)))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = pickle.dumps({
        "model": model,
        "n_samples": len(X_list),
        "rmse": rmse,
        "trained_at": time.time(),
        "embedding_dim": X.shape[1],
    })
    # Write model + HMAC sidecar so _ensure_loaded can refuse tampered files.
    model_path.write_bytes(payload_bytes)
    _sign_model(model_path, payload_bytes)

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[MCTSValueFn] trained on {len(X_list)} pairs, rmse={rmse:.4f}, {elapsed}s")
    return {
        "trained": True,
        "n_samples": len(X_list),
        "rmse": rmse,
        "elapsed": elapsed,
        "path": str(model_path),
    }


class ValuePredictor:
    """Lazy-loaded Ridge predictor. Returns (value, confidence) per call."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_FILE):
        self.model_path = Path(model_path)
        self._model = None
        self._n_samples = 0
        self._rmse = 1.0
        self._embed = None
        self._load_attempted = False

    def _ensure_loaded(self) -> bool:
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True

        if not self.model_path.exists():
            return False
        try:
            payload_bytes = self.model_path.read_bytes()
            # HMAC verify BEFORE unpickling — pickle.load on hostile input is RCE.
            if not _verify_model(self.model_path, payload_bytes):
                logger.warning(
                    "[ValuePredictor] refusing to load %s: HMAC signature missing or "
                    "invalid. Retrain to regenerate (the predictor will return the "
                    "no-signal default in the meantime).",
                    self.model_path,
                )
                return False
            data = pickle.loads(payload_bytes)
            self._model = data["model"]
            self._n_samples = int(data.get("n_samples", 0))
            self._rmse = float(data.get("rmse", 1.0))
            self._embed = _load_embedder()
            if self._embed is None:
                self._model = None
                return False
            return True
        except Exception as e:
            logger.warning(f"[ValuePredictor] load failed: {e}")
            return False

    def predict(self, thought_text: str, path_context: str = "") -> Tuple[float, float]:
        """Return (predicted_value, confidence) in [0, 1].

        Confidence grows with training-set size and shrinks with training RMSE.
        Returns (0.5, 0.0) when the model is absent — the caller should treat
        that as "no signal."
        """
        if not self._ensure_loaded():
            return (0.5, 0.0)
        try:
            combined = (path_context + " | " + thought_text).strip(" |")[:2000]
            vec = self._embed(combined)
            if not vec:
                return (0.5, 0.0)
            import numpy as np
            X = np.array([vec], dtype=np.float32)
            pred = float(self._model.predict(X)[0])
            pred = max(0.0, min(1.0, pred))

            # Confidence: saturates as n_samples grows past ~500; penalized by rmse.
            n_factor = min(1.0, self._n_samples / 500.0)
            rmse_factor = max(0.0, 1.0 - 2.0 * self._rmse)
            confidence = max(0.0, min(1.0, n_factor * rmse_factor))
            return (pred, confidence)
        except Exception as e:
            logger.debug(f"[ValuePredictor] predict failed: {e}")
            return (0.5, 0.0)
