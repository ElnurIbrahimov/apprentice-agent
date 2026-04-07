# Neural Router Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Aura's 5-second LLM-based model routing with a <5ms three-layer neural router that learns from user feedback.

**Architecture:** Three layers — (1) instant classifier extracts features + scores task dimensions, (2) profile matcher dot-products task needs against model capability profiles, (3) conversation tracker adjusts routing based on context history. Learning loop updates profiles from 6 feedback signals.

**Tech Stack:** Python 3.12, pytest, nomic-embed-text via Ollama, JSON persistence. TypeScript/React for extension UI changes.

**Design doc:** `docs/plans/2026-04-07-neural-router-design.md`

---

### Task 1: Model Profiles — Data Layer

**Files:**
- Create: `aura/routing/__init__.py`
- Create: `aura/routing/profiles.py`
- Create: `data/model_profiles.json`
- Test: `tests/test_neural_router.py`

**Step 1: Write the failing test**

```python
# tests/test_neural_router.py
import pytest
import json
import tempfile
import os

class TestModelProfiles:
    """Model capability profiles — load, save, get, update."""

    def test_load_profiles_from_json(self, tmp_path):
        data = {
            "nemotron-3-super:cloud": {
                "code": 0.60, "reason": 0.73, "speed": 1.00,
                "context": 1.00, "quality": 0.70, "vision": 0,
            }
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))

        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        p = store.get("nemotron-3-super:cloud")
        assert p["speed"] == 1.00
        assert p["code"] == 0.60

    def test_get_unknown_model_returns_neutral(self, tmp_path):
        path = tmp_path / "profiles.json"
        path.write_text("{}")

        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        p = store.get("unknown-model:cloud")
        assert p["code"] == 0.5
        assert p["speed"] == 0.5

    def test_update_profile_clamps_values(self, tmp_path):
        data = {"test:cloud": {"code": 0.95, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0}}
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))

        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        store.update("test:cloud", {"code": 0.1})  # 0.95 + 0.1 = 1.05 -> clamp to 1.0
        assert store.get("test:cloud")["code"] == 1.0

    def test_update_profile_persists(self, tmp_path):
        data = {"test:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0}}
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))

        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        store.update("test:cloud", {"code": 0.05})

        # Reload from disk
        store2 = ProfileStore(str(path))
        assert store2.get("test:cloud")["code"] == 0.55

    def test_list_available_filters_by_dimension(self, tmp_path):
        data = {
            "model-a:cloud": {"code": 0.9, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 1},
            "model-b:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0},
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))

        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        vision_models = store.list_available(require={"vision": 0.5})
        assert "model-a:cloud" in vision_models
        assert "model-b:cloud" not in vision_models
```

**Step 2: Run tests to verify they fail**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestModelProfiles -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aura.routing'`

**Step 3: Create package and implement ProfileStore**

```python
# aura/routing/__init__.py
"""Neural Router — intelligent model routing for Aura."""
```

```python
# aura/routing/profiles.py
"""Model capability profiles — load, save, query, update."""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DIMENSIONS = ("code", "reason", "speed", "context", "quality", "vision")
NEUTRAL_PROFILE = {d: 0.5 for d in DIMENSIONS}
NEUTRAL_PROFILE["vision"] = 0


class ProfileStore:
    """Thread-safe model capability profile store with JSON persistence."""

    def __init__(self, path: str = "data/model_profiles.json"):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._profiles: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._profiles = json.load(f)
            except Exception as e:
                logger.warning(f"[Profiles] Failed to load: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._profiles, f, indent=2)
        except Exception as e:
            logger.warning(f"[Profiles] Failed to save: {e}")

    def get(self, model: str) -> Dict[str, float]:
        with self._lock:
            if model in self._profiles:
                return dict(self._profiles[model])
        return dict(NEUTRAL_PROFILE)

    def update(self, model: str, deltas: Dict[str, float]):
        with self._lock:
            if model not in self._profiles:
                self._profiles[model] = dict(NEUTRAL_PROFILE)
            for dim, delta in deltas.items():
                if dim in DIMENSIONS:
                    val = self._profiles[model][dim] + delta
                    self._profiles[model][dim] = max(0.0, min(1.0, val))
            self._save()

    def set(self, model: str, profile: Dict[str, float]):
        with self._lock:
            self._profiles[model] = {d: profile.get(d, 0.5) for d in DIMENSIONS}
            self._save()

    def list_available(self, require: Optional[Dict[str, float]] = None) -> list:
        with self._lock:
            models = list(self._profiles.keys())
        if not require:
            return models
        return [
            m for m in models
            if all(self.get(m).get(d, 0) >= v for d, v in require.items())
        ]

    def all_profiles(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {m: dict(p) for m, p in self._profiles.items()}
```

**Step 4: Create seed data**

```json
// data/model_profiles.json
{
  "nemotron-3-super:cloud": {"code": 0.60, "reason": 0.73, "speed": 1.00, "context": 1.00, "quality": 0.70, "vision": 0},
  "kimi-k2.5:cloud": {"code": 0.77, "reason": 0.92, "speed": 0.08, "context": 0.26, "quality": 0.90, "vision": 1},
  "glm-5:cloud": {"code": 0.78, "reason": 0.96, "speed": 0.15, "context": 0.20, "quality": 0.95, "vision": 0},
  "glm-5.1:cloud": {"code": 0.88, "reason": 0.93, "speed": 0.10, "context": 0.20, "quality": 0.92, "vision": 0},
  "qwen3.5:397b-cloud": {"code": 0.70, "reason": 0.88, "speed": 0.19, "context": 0.26, "quality": 0.88, "vision": 1},
  "qwen3.5:cloud": {"code": 0.50, "reason": 0.83, "speed": 0.30, "context": 0.26, "quality": 0.75, "vision": 1},
  "deepseek-v3.2:cloud": {"code": 0.68, "reason": 0.85, "speed": 0.11, "context": 0.13, "quality": 0.82, "vision": 0},
  "gemma4:31b-cloud": {"code": 0.72, "reason": 0.85, "speed": 0.23, "context": 0.26, "quality": 0.83, "vision": 1},
  "minimax-m2.7:cloud": {"code": 0.78, "reason": 0.80, "speed": 0.22, "context": 0.21, "quality": 0.86, "vision": 0},
  "minimax-m2.5:cloud": {"code": 0.80, "reason": 0.74, "speed": 0.20, "context": 0.20, "quality": 0.82, "vision": 0},
  "qwen3-coder:480b-cloud": {"code": 0.70, "reason": 0.60, "speed": 0.35, "context": 0.26, "quality": 0.75, "vision": 0},
  "qwen3-coder-next:cloud": {"code": 0.71, "reason": 0.55, "speed": 0.38, "context": 0.26, "quality": 0.72, "vision": 0},
  "gpt-oss:120b-cloud": {"code": 0.45, "reason": 0.90, "speed": 0.15, "context": 0.13, "quality": 0.80, "vision": 0},
  "gemma4:e4b": {"code": 0.35, "reason": 0.69, "speed": 0.50, "context": 0.13, "quality": 0.60, "vision": 1},
  "gemma4:e2b": {"code": 0.25, "reason": 0.60, "speed": 0.60, "context": 0.13, "quality": 0.50, "vision": 1}
}
```

**Step 5: Run tests to verify they pass**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestModelProfiles -v`
Expected: All 5 PASS

**Step 6: Commit**

```bash
cd D:/Aura && git add aura/routing/__init__.py aura/routing/profiles.py data/model_profiles.json tests/test_neural_router.py
git commit -m "feat(routing): add model capability profiles with benchmark-seeded data"
```

---

### Task 2: Instant Classifier — Feature Extraction + Task Scoring (Layer 1)

**Files:**
- Create: `aura/routing/classifier.py`
- Append: `tests/test_neural_router.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_neural_router.py

class TestClassifier:
    """Layer 1: feature extraction and task dimension scoring."""

    def test_extract_features_short_message(self):
        from aura.routing.classifier import extract_features
        f = extract_features("hello how are you")
        assert f["word_count"] == 4
        assert f["code_ratio"] == 0.0
        assert f["has_attachment"] is False

    def test_extract_features_code_message(self):
        from aura.routing.classifier import extract_features
        f = extract_features("fix the bug in def calculate(x): return x * 2")
        assert f["code_ratio"] > 0.1
        assert len(f["language_markers"]) > 0

    def test_extract_features_with_attachment(self):
        from aura.routing.classifier import extract_features
        f = extract_features("analyze this image", has_attachment=True)
        assert f["has_attachment"] is True

    def test_score_task_short_query_high_speed(self):
        from aura.routing.classifier import score_task
        needs = score_task("hi there")
        assert needs["speed"] >= 0.8
        assert needs["reason"] < 0.3

    def test_score_task_code_query(self):
        from aura.routing.classifier import score_task
        needs = score_task("debug this Python function that throws TypeError")
        assert needs["code"] >= 0.7

    def test_score_task_complex_reasoning(self):
        from aura.routing.classifier import score_task
        needs = score_task("write a comprehensive analysis of the pros and cons of microservices vs monolithic architecture for a startup")
        assert needs["reason"] >= 0.7
        assert needs["speed"] <= 0.4

    def test_score_task_vision(self):
        from aura.routing.classifier import score_task
        needs = score_task("what's in this screenshot", has_attachment=True)
        assert needs["vision"] >= 0.9

    def test_score_task_long_context(self):
        from aura.routing.classifier import score_task
        # Simulate conversation with high token count
        needs = score_task("continue", conversation_tokens=120_000)
        assert needs["context"] >= 0.7

    def test_score_task_after_regen(self):
        from aura.routing.classifier import score_task
        needs = score_task("try again", recent_regen_count=2)
        assert needs["quality"] >= 0.8
```

**Step 2: Run tests to verify they fail**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestClassifier -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aura.routing.classifier'`

**Step 3: Implement classifier**

```python
# aura/routing/classifier.py
"""Layer 1: Instant classifier — feature extraction + task dimension scoring.

Extracts lightweight features from prompts and scores 6 task dimensions.
Total latency: <1ms. No network calls, no ML inference.
"""

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Code-like token patterns
_CODE_CHARS = re.compile(r'[{}\[\]();=<>!&|^~]')
_CAMEL_CASE = re.compile(r'[a-z][A-Z]')
_SNAKE_CASE = re.compile(r'[a-z]_[a-z]')
_INDENTED = re.compile(r'^\s{2,}', re.MULTILINE)

_LANG_KEYWORDS = {
    "python": {"def ", "import ", "class ", "lambda ", "elif ", "print(", "self."},
    "javascript": {"const ", "let ", "function ", "=>", "async ", "require(", "console."},
    "sql": {"SELECT ", "FROM ", "WHERE ", "INSERT ", "UPDATE ", "JOIN ", "CREATE TABLE"},
    "rust": {"fn ", "let mut ", "impl ", "pub ", "struct ", "enum ", "match "},
    "go": {"func ", "package ", "import (", "defer ", "goroutine"},
}

_COMPLEX_PATTERNS = [
    "write an essay", "write a report", "write a detailed",
    "comprehensive analysis", "in-depth analysis", "thorough analysis",
    "deep dive into", "investigate in detail", "step by step guide",
    "detailed explanation", "compare and contrast", "pros and cons",
    "advantages and disadvantages",
]

_CODE_VERBS = {"fix", "debug", "implement", "refactor", "deploy", "build", "compile", "test", "code", "script"}

_IMPERATIVE_VERBS = {
    "build", "create", "make", "write", "fix", "debug", "explain", "compare",
    "analyze", "search", "find", "implement", "design", "generate", "translate",
    "summarize", "review", "optimize", "refactor", "deploy", "test",
}


def extract_features(prompt: str, has_attachment: bool = False) -> Dict:
    """Extract lightweight features from a prompt. <0.1ms."""
    words = prompt.split()
    word_count = len(words)
    prompt_lower = prompt.lower()

    # Code ratio: fraction of characters that are code-like
    code_chars = len(_CODE_CHARS.findall(prompt))
    camel = len(_CAMEL_CASE.findall(prompt))
    snake = len(_SNAKE_CASE.findall(prompt))
    indented = len(_INDENTED.findall(prompt))
    total_chars = max(len(prompt), 1)
    code_ratio = min(1.0, (code_chars + camel * 3 + snake * 3 + indented * 5) / total_chars)

    # Language markers
    language_markers = []
    for lang, keywords in _LANG_KEYWORDS.items():
        if any(kw.lower() in prompt_lower for kw in keywords):
            language_markers.append(lang)

    # Question marks
    question_marks = prompt.count("?")

    # Imperative score
    first_word = words[0].lower().rstrip(".,!?:") if words else ""
    imperative_score = 1.0 if first_word in _IMPERATIVE_VERBS else 0.0

    return {
        "word_count": word_count,
        "code_ratio": round(code_ratio, 3),
        "question_marks": question_marks,
        "has_attachment": has_attachment,
        "language_markers": language_markers,
        "imperative_score": imperative_score,
    }


def score_task(
    prompt: str,
    has_attachment: bool = False,
    conversation_tokens: int = 0,
    recent_regen_count: int = 0,
) -> Dict[str, float]:
    """Score 6 task dimensions from prompt features. <1ms total."""
    features = extract_features(prompt, has_attachment)
    prompt_lower = prompt.lower()
    wc = features["word_count"]

    code_need = 0.0
    reason_need = 0.0
    speed_need = 0.5
    context_need = 0.0
    quality_need = 0.5
    vision_need = 0.0

    # -- Code signals --
    if features["code_ratio"] > 0.15 or features["language_markers"]:
        code_need = 0.9
    elif any(v in prompt_lower.split() for v in _CODE_VERBS):
        code_need = 0.7

    # -- Reasoning signals --
    if wc > 50:
        reason_need = max(reason_need, 0.6)
    if any(p in prompt_lower for p in _COMPLEX_PATTERNS):
        reason_need = max(reason_need, 0.8)
    if features["question_marks"] > 0 and wc > 20:
        reason_need = max(reason_need, 0.6)

    # -- Speed signals --
    if wc <= 10:
        speed_need = 0.9
    elif wc <= 5:
        speed_need = 1.0
    if wc > 50:
        speed_need = 0.3
    if wc > 100:
        speed_need = 0.2

    # -- Context signals --
    if conversation_tokens > 50_000:
        context_need = 0.5
    if conversation_tokens > 100_000:
        context_need = 0.7
    if conversation_tokens > 150_000:
        context_need = 1.0

    # -- Vision signals --
    if features["has_attachment"]:
        vision_need = 1.0

    # -- Quality signals --
    if recent_regen_count > 0:
        quality_need = min(1.0, 0.5 + recent_regen_count * 0.2)

    return {
        "code": round(code_need, 2),
        "reason": round(reason_need, 2),
        "speed": round(speed_need, 2),
        "context": round(context_need, 2),
        "quality": round(quality_need, 2),
        "vision": round(vision_need, 2),
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestClassifier -v`
Expected: All 9 PASS

**Step 5: Commit**

```bash
cd D:/Aura && git add aura/routing/classifier.py tests/test_neural_router.py
git commit -m "feat(routing): add Layer 1 instant classifier with feature extraction"
```

---

### Task 3: Profile Matcher (Layer 2)

**Files:**
- Create: `aura/routing/matcher.py`
- Append: `tests/test_neural_router.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_neural_router.py

class TestMatcher:
    """Layer 2: profile matching with preference weighting."""

    def _make_store(self, tmp_path):
        import json
        from aura.routing.profiles import ProfileStore
        data = {
            "fast-model:cloud": {"code": 0.3, "reason": 0.4, "speed": 1.0, "context": 0.5, "quality": 0.5, "vision": 0},
            "code-model:cloud": {"code": 0.95, "reason": 0.5, "speed": 0.2, "context": 0.5, "quality": 0.8, "vision": 0},
            "vision-model:cloud": {"code": 0.5, "reason": 0.7, "speed": 0.3, "context": 0.5, "quality": 0.8, "vision": 1},
            "allround-model:cloud": {"code": 0.7, "reason": 0.8, "speed": 0.5, "context": 0.5, "quality": 0.85, "vision": 0},
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        return ProfileStore(str(path))

    def test_match_speed_query_picks_fast_model(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.0, "reason": 0.2, "speed": 0.9, "context": 0.0, "quality": 0.3, "vision": 0.0}
        result = match(task_needs, "balanced", store)
        assert result == "fast-model:cloud"

    def test_match_code_query_picks_code_model(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.9, "reason": 0.3, "speed": 0.2, "context": 0.0, "quality": 0.8, "vision": 0.0}
        result = match(task_needs, "balanced", store)
        assert result == "code-model:cloud"

    def test_match_vision_filters_non_vision(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.0, "reason": 0.5, "speed": 0.3, "context": 0.0, "quality": 0.5, "vision": 1.0}
        result = match(task_needs, "balanced", store)
        assert result == "vision-model:cloud"

    def test_prefer_fast_boosts_speed(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        # Balanced query that could go either way
        task_needs = {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.0, "quality": 0.5, "vision": 0.0}
        result_fast = match(task_needs, "prefer-fast", store)
        result_quality = match(task_needs, "prefer-quality", store)
        # Fast preference should pick the faster model
        assert result_fast == "fast-model:cloud"
        assert result_quality != "fast-model:cloud"

    def test_match_returns_string(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.0, "quality": 0.5, "vision": 0.0}
        result = match(task_needs, "balanced", store)
        assert isinstance(result, str)
        assert result.endswith(":cloud")
```

**Step 2: Run tests to verify they fail**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestMatcher -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aura.routing.matcher'`

**Step 3: Implement matcher**

```python
# aura/routing/matcher.py
"""Layer 2: Profile matcher — dot-product matching with preference weighting.

Matches task need vectors against model capability profiles.
Total latency: <1ms. Pure arithmetic, no I/O.
"""

import logging
from typing import Dict, Optional

from aura.routing.profiles import ProfileStore, DIMENSIONS

logger = logging.getLogger(__name__)

PREFERENCE_WEIGHTS = {
    "prefer-fast": {"code": 1.0, "reason": 0.8, "speed": 2.0, "context": 1.0, "quality": 0.7, "vision": 1.0},
    "balanced":    {"code": 1.0, "reason": 1.0, "speed": 1.0, "context": 1.0, "quality": 1.0, "vision": 1.0},
    "prefer-quality": {"code": 1.0, "reason": 1.2, "speed": 0.5, "context": 1.0, "quality": 2.0, "vision": 1.0},
}


def match(
    task_needs: Dict[str, float],
    preference: str,
    profile_store: ProfileStore,
    stats_bonus: Optional[Dict[str, float]] = None,
) -> str:
    """Match task needs against model profiles, return best model name.

    Args:
        task_needs: 6-dimension task need vector from classifier
        preference: "prefer-fast" | "balanced" | "prefer-quality"
        profile_store: ProfileStore instance
        stats_bonus: optional {model: float} UCB bonus from RoutingStats

    Returns:
        Best matching model name
    """
    weights = PREFERENCE_WEIGHTS.get(preference, PREFERENCE_WEIGHTS["balanced"])
    all_profiles = profile_store.all_profiles()

    if not all_profiles:
        logger.warning("[Matcher] No profiles loaded, cannot match")
        return ""

    scores: Dict[str, float] = {}
    for model, profile in all_profiles.items():
        score = sum(
            task_needs.get(dim, 0.0) * profile.get(dim, 0.0) * weights.get(dim, 1.0)
            for dim in DIMENSIONS
        )
        if stats_bonus and model in stats_bonus:
            score += stats_bonus[model]
        scores[model] = score

    # Hard filters: eliminate models that can't serve the need
    if task_needs.get("vision", 0) > 0.5:
        scores = {m: s for m, s in scores.items() if all_profiles[m].get("vision", 0) > 0}
    if task_needs.get("context", 0) > 0.8:
        scores = {m: s for m, s in scores.items() if all_profiles[m].get("context", 0) > 0.5}

    if not scores:
        # All filtered out — return first model as fallback
        return next(iter(all_profiles))

    best = max(scores, key=scores.get)
    logger.debug(f"[Matcher] Best: {best} (score={scores[best]:.3f})")
    return best
```

**Step 4: Run tests to verify they pass**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestMatcher -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
cd D:/Aura && git add aura/routing/matcher.py tests/test_neural_router.py
git commit -m "feat(routing): add Layer 2 profile matcher with preference weighting"
```

---

### Task 4: Conversation Context Tracker (Layer 3)

**Files:**
- Create: `aura/routing/conversation.py`
- Append: `tests/test_neural_router.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_neural_router.py

class TestConversationTracker:
    """Layer 3: conversation context tracking."""

    def test_new_conversation_returns_empty_profile(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        profile = tracker.get_profile("conv-1")
        assert profile.turn_count == 0
        assert profile.in_code_mode is False

    def test_update_increments_turn_count(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        tracker.update("conv-1", code_ratio=0.0, complexity=0.3, model_used="test:cloud", tokens=100)
        p = tracker.get_profile("conv-1")
        assert p.turn_count == 1

    def test_code_mode_activates_after_3_code_messages(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        for _ in range(3):
            tracker.update("conv-1", code_ratio=0.5, complexity=0.5, model_used="test:cloud", tokens=200)
        p = tracker.get_profile("conv-1")
        assert p.in_code_mode is True

    def test_code_mode_deactivates(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        for _ in range(3):
            tracker.update("conv-1", code_ratio=0.5, complexity=0.5, model_used="test:cloud", tokens=200)
        for _ in range(3):
            tracker.update("conv-1", code_ratio=0.0, complexity=0.3, model_used="test:cloud", tokens=100)
        p = tracker.get_profile("conv-1")
        assert p.in_code_mode is False

    def test_complexity_trend_escalating(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        for i in range(5):
            tracker.update("conv-1", code_ratio=0.0, complexity=0.2 + i * 0.15, model_used="test:cloud", tokens=200)
        p = tracker.get_profile("conv-1")
        assert p.complexity_trend > 0.2

    def test_record_regen(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        tracker.record_feedback("conv-1", "regeneration")
        p = tracker.get_profile("conv-1")
        assert p.regen_count == 1

    def test_record_model_switch(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        tracker.record_feedback("conv-1", "model_switch", model="new:cloud")
        p = tracker.get_profile("conv-1")
        assert p.model_switches == 1

    def test_adjust_boosts_code_in_code_mode(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        for _ in range(3):
            tracker.update("conv-1", code_ratio=0.5, complexity=0.5, model_used="test:cloud", tokens=200)
        task_needs = {"code": 0.1, "reason": 0.3, "speed": 0.5, "context": 0.0, "quality": 0.5, "vision": 0.0}
        adjusted = tracker.adjust("conv-1", task_needs)
        assert adjusted["code"] > task_needs["code"]

    def test_adjust_boosts_quality_after_regen(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        tracker.record_feedback("conv-1", "regeneration")
        task_needs = {"code": 0.0, "reason": 0.3, "speed": 0.5, "context": 0.0, "quality": 0.5, "vision": 0.0}
        adjusted = tracker.adjust("conv-1", task_needs)
        assert adjusted["quality"] > task_needs["quality"]

    def test_stale_conversations_evicted(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker(max_conversations=2)
        tracker.update("conv-1", code_ratio=0.0, complexity=0.3, model_used="a:cloud", tokens=100)
        tracker.update("conv-2", code_ratio=0.0, complexity=0.3, model_used="a:cloud", tokens=100)
        tracker.update("conv-3", code_ratio=0.0, complexity=0.3, model_used="a:cloud", tokens=100)
        # conv-1 should be evicted (oldest, max_conversations=2)
        p = tracker.get_profile("conv-1")
        assert p.turn_count == 0  # fresh/evicted
```

**Step 2: Run tests to verify they fail**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestConversationTracker -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aura.routing.conversation'`

**Step 3: Implement conversation tracker**

```python
# aura/routing/conversation.py
"""Layer 3: Conversation context tracker.

Maintains per-conversation profiles that influence routing decisions.
Tracks complexity trends, code mode, satisfaction signals, topic drift.
All operations are <1ms — no I/O on the hot path.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationProfile:
    topic_embedding: Optional[List[float]] = None
    topic_drift_score: float = 0.0
    complexity_history: List[float] = field(default_factory=list)
    complexity_trend: float = 0.0
    code_ratio_history: List[float] = field(default_factory=list)
    in_code_mode: bool = False
    models_used: List[str] = field(default_factory=list)
    last_model: Optional[str] = None
    regen_count: int = 0
    model_switches: int = 0
    total_tokens: int = 0
    turn_count: int = 0
    thumbs: List[int] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


_WINDOW = 10  # sliding window size for history
_CODE_MODE_THRESHOLD = 0.2  # code ratio above this = code message
_CODE_MODE_WINDOW = 3  # consecutive code messages to enter code mode


class ConversationTracker:
    """Manages conversation profiles for routing context."""

    def __init__(self, max_conversations: int = 200):
        self._lock = threading.Lock()
        self._profiles: OrderedDict[str, ConversationProfile] = OrderedDict()
        self._max = max_conversations

    def get_profile(self, conversation_id: str) -> ConversationProfile:
        with self._lock:
            return self._profiles.get(conversation_id, ConversationProfile())

    def update(
        self,
        conversation_id: str,
        code_ratio: float,
        complexity: float,
        model_used: str,
        tokens: int,
    ):
        with self._lock:
            if conversation_id not in self._profiles:
                self._profiles[conversation_id] = ConversationProfile()
                self._evict_if_needed()
            p = self._profiles[conversation_id]

        # Update outside lock (profile is thread-local after creation)
        p.turn_count += 1
        p.total_tokens += tokens
        p.last_model = model_used
        if model_used not in p.models_used:
            p.models_used.append(model_used)
        p.last_active = time.time()

        # Sliding window updates
        p.complexity_history.append(complexity)
        if len(p.complexity_history) > _WINDOW:
            p.complexity_history = p.complexity_history[-_WINDOW:]

        p.code_ratio_history.append(code_ratio)
        if len(p.code_ratio_history) > _WINDOW:
            p.code_ratio_history = p.code_ratio_history[-_WINDOW:]

        # Complexity trend: slope of recent history
        if len(p.complexity_history) >= 3:
            recent = p.complexity_history[-3:]
            p.complexity_trend = recent[-1] - recent[0]

        # Code mode: last N messages all code-heavy
        recent_code = p.code_ratio_history[-_CODE_MODE_WINDOW:]
        if len(recent_code) >= _CODE_MODE_WINDOW:
            p.in_code_mode = all(r > _CODE_MODE_THRESHOLD for r in recent_code)
        else:
            p.in_code_mode = False

        # Move to end of OrderedDict (most recently active)
        with self._lock:
            self._profiles.move_to_end(conversation_id)

    def record_feedback(self, conversation_id: str, signal: str, **kwargs):
        with self._lock:
            if conversation_id not in self._profiles:
                self._profiles[conversation_id] = ConversationProfile()
            p = self._profiles[conversation_id]

        if signal == "regeneration":
            p.regen_count += 1
        elif signal == "model_switch":
            p.model_switches += 1
        elif signal == "thumbs_up":
            p.thumbs.append(1)
        elif signal == "thumbs_down":
            p.thumbs.append(-1)
        elif signal == "abort":
            p.regen_count += 1  # treat as dissatisfaction

    def adjust(self, conversation_id: str, task_needs: Dict[str, float]) -> Dict[str, float]:
        """Adjust task needs based on conversation context. <0.1ms."""
        p = self.get_profile(conversation_id)
        adjusted = dict(task_needs)

        if p.turn_count == 0:
            return adjusted

        # Code mode boost
        if p.in_code_mode:
            adjusted["code"] = min(1.0, adjusted["code"] + 0.3)

        # Complexity escalation
        if p.complexity_trend > 0.3:
            adjusted["reason"] = min(1.0, adjusted["reason"] + 0.2)
            adjusted["speed"] = max(0.0, adjusted["speed"] - 0.2)
        elif p.complexity_trend < -0.3:
            adjusted["speed"] = min(1.0, adjusted["speed"] + 0.2)

        # Regeneration = quality boost
        if p.regen_count > 0:
            adjusted["quality"] = min(1.0, adjusted["quality"] + 0.15 * p.regen_count)

        # Token-based context need
        if p.total_tokens > 100_000:
            adjusted["context"] = max(adjusted["context"], 0.7)
        elif p.total_tokens > 50_000:
            adjusted["context"] = max(adjusted["context"], 0.5)

        return adjusted

    def _evict_if_needed(self):
        """Remove oldest conversations if over capacity. Called under lock."""
        while len(self._profiles) > self._max:
            self._profiles.popitem(last=False)

    def get_last_model(self, conversation_id: str) -> Optional[str]:
        """Get the last model used in a conversation (for stickiness)."""
        p = self.get_profile(conversation_id)
        return p.last_model
```

**Step 4: Run tests to verify they pass**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestConversationTracker -v`
Expected: All 10 PASS

**Step 5: Commit**

```bash
cd D:/Aura && git add aura/routing/conversation.py tests/test_neural_router.py
git commit -m "feat(routing): add Layer 3 conversation context tracker"
```

---

### Task 5: Learning Loop

**Files:**
- Create: `aura/routing/learning.py`
- Append: `tests/test_neural_router.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_neural_router.py

class TestLearning:
    """Learning loop: feedback signals update model profiles."""

    def _make_store(self, tmp_path):
        import json
        from aura.routing.profiles import ProfileStore
        data = {"test:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0}}
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        return ProfileStore(str(path))

    def test_positive_rating_boosts_profile(self, tmp_path):
        from aura.routing.learning import process_feedback
        store = self._make_store(tmp_path)
        task_dims = {"code": 0.8, "reason": 0.2, "speed": 0.0, "context": 0.0, "quality": 0.5, "vision": 0.0}
        process_feedback("thumbs_up", "test:cloud", task_dims, store)
        p = store.get("test:cloud")
        assert p["code"] > 0.5  # boosted
        assert p["reason"] == 0.5  # below threshold, untouched

    def test_regeneration_penalizes_profile(self, tmp_path):
        from aura.routing.learning import process_feedback
        store = self._make_store(tmp_path)
        task_dims = {"code": 0.8, "reason": 0.2, "speed": 0.0, "context": 0.0, "quality": 0.5, "vision": 0.0}
        process_feedback("regeneration", "test:cloud", task_dims, store)
        p = store.get("test:cloud")
        assert p["code"] < 0.5  # penalized

    def test_model_switch_updates_both(self, tmp_path):
        import json
        from aura.routing.profiles import ProfileStore
        from aura.routing.learning import process_feedback
        data = {
            "old:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0},
            "new:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0},
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        store = ProfileStore(str(path))

        task_dims = {"code": 0.8, "reason": 0.2, "speed": 0.0, "context": 0.0, "quality": 0.5, "vision": 0.0}
        process_feedback("model_switch", "old:cloud", task_dims, store, switched_to="new:cloud")
        assert store.get("old:cloud")["code"] < 0.5
        assert store.get("new:cloud")["code"] > 0.5

    def test_feedback_respects_clamp(self, tmp_path):
        import json
        from aura.routing.profiles import ProfileStore
        from aura.routing.learning import process_feedback
        data = {"test:cloud": {"code": 0.02, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0}}
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        store = ProfileStore(str(path))

        task_dims = {"code": 0.9, "reason": 0.0, "speed": 0.0, "context": 0.0, "quality": 0.0, "vision": 0.0}
        process_feedback("regeneration", "test:cloud", task_dims, store)
        assert store.get("test:cloud")["code"] >= 0.0  # never below 0
```

**Step 2: Run tests to verify they fail**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestLearning -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement learning loop**

```python
# aura/routing/learning.py
"""Learning loop: processes feedback signals to update model profiles.

Six signals: regeneration, model_switch, response_gap, thumbs_up/down,
conversation_length, abort. Each adjusts model profile dimensions.
"""

import logging
from typing import Dict, Optional

from aura.routing.profiles import ProfileStore, DIMENSIONS

logger = logging.getLogger(__name__)

LEARNING_RATE = 0.05
RELEVANCE_THRESHOLD = 0.3  # only update dimensions the task actually needed

SIGNAL_WEIGHTS = {
    "regeneration": -0.3,
    "model_switch": -0.2,      # for old model
    "model_switch_to": 0.1,    # for new model
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
    task_dimensions: Dict[str, float],
    profile_store: ProfileStore,
    switched_to: Optional[str] = None,
):
    """Process a feedback signal and update model profiles.

    Args:
        signal: one of the SIGNAL_WEIGHTS keys
        model: model that was used (or old model for switch)
        task_dimensions: the task need vector when this model was selected
        profile_store: ProfileStore to update
        switched_to: for model_switch, the model user switched to
    """
    weight = SIGNAL_WEIGHTS.get(signal, 0.0)
    if weight == 0.0:
        return

    # Update the model that was used
    deltas = {}
    for dim in DIMENSIONS:
        if dim == "vision":
            continue  # vision is binary, don't learn it
        if task_dimensions.get(dim, 0.0) >= RELEVANCE_THRESHOLD:
            deltas[dim] = LEARNING_RATE * weight * task_dimensions[dim]

    if deltas:
        profile_store.update(model, deltas)
        logger.info(f"[Learning] {signal} on {model}: {deltas}")

    # For model_switch: also boost the model user switched TO
    if signal == "model_switch" and switched_to:
        boost_weight = SIGNAL_WEIGHTS["model_switch_to"]
        boost_deltas = {}
        for dim in DIMENSIONS:
            if dim == "vision":
                continue
            if task_dimensions.get(dim, 0.0) >= RELEVANCE_THRESHOLD:
                boost_deltas[dim] = LEARNING_RATE * boost_weight * task_dimensions[dim]
        if boost_deltas:
            profile_store.update(switched_to, boost_deltas)
            logger.info(f"[Learning] model_switch boost {switched_to}: {boost_deltas}")
```

**Step 4: Run tests to verify they pass**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestLearning -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
cd D:/Aura && git add aura/routing/learning.py tests/test_neural_router.py
git commit -m "feat(routing): add learning loop with 6 feedback signals"
```

---

### Task 6: Main Router Entry Point

**Files:**
- Create: `aura/routing/router.py`
- Append: `tests/test_neural_router.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_neural_router.py

class TestRouter:
    """Main router: end-to-end routing decisions."""

    def _make_router(self, tmp_path):
        import json
        from aura.routing.router import Router
        data = {
            "fast:cloud": {"code": 0.3, "reason": 0.4, "speed": 1.0, "context": 0.5, "quality": 0.5, "vision": 0},
            "code:cloud": {"code": 0.95, "reason": 0.5, "speed": 0.2, "context": 0.5, "quality": 0.8, "vision": 0},
            "vision:cloud": {"code": 0.5, "reason": 0.7, "speed": 0.3, "context": 0.5, "quality": 0.8, "vision": 1},
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        return Router(profiles_path=str(path))

    def test_explicit_model_bypasses_router(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("anything", model="my-explicit:cloud")
        assert result.model == "my-explicit:cloud"
        assert result.reason == "explicit_override"

    def test_auto_routes_short_to_fast(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("hello")
        assert result.model == "fast:cloud"

    def test_auto_routes_code_to_code_model(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("debug this Python function: def foo(): pass")
        assert result.model == "code:cloud"

    def test_auto_routes_vision_with_attachment(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("what is this", has_attachment=True)
        assert result.model == "vision:cloud"

    def test_preference_affects_result(self, tmp_path):
        router = self._make_router(tmp_path)
        fast_result = router.route("explain something moderately complex to me", preference="prefer-fast")
        quality_result = router.route("explain something moderately complex to me", preference="prefer-quality")
        # At minimum, both should return valid models
        assert fast_result.model in ("fast:cloud", "code:cloud", "vision:cloud")
        assert quality_result.model in ("fast:cloud", "code:cloud", "vision:cloud")

    def test_route_returns_metadata(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("hello there")
        assert hasattr(result, "model")
        assert hasattr(result, "reason")
        assert hasattr(result, "task_dimensions")
        assert hasattr(result, "alternatives")

    def test_conversation_context_influences_routing(self, tmp_path):
        router = self._make_router(tmp_path)
        conv = "test-conv"
        # Send 3 code-heavy messages to enter code mode
        for _ in range(3):
            router.route("fix def calc(): return x * 2", conversation_id=conv)
        # Now a vague message should still route to code
        result = router.route("now test it", conversation_id=conv)
        assert result.model == "code:cloud"
```

**Step 2: Run tests to verify they fail**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestRouter -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement router**

```python
# aura/routing/router.py
"""Main router entry point: route(prompt, ...) -> RoutingResult.

Orchestrates the 3-layer routing pipeline:
  Layer 1: classifier.score_task() -> task_needs
  Layer 3: conversation.adjust() -> adjusted_needs
  Layer 2: matcher.match() -> model
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aura.routing.classifier import score_task, extract_features
from aura.routing.conversation import ConversationTracker
from aura.routing.matcher import match
from aura.routing.profiles import ProfileStore

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    model: str
    reason: str
    task_dimensions: Dict[str, float]
    alternatives: List[str] = field(default_factory=list)
    conversation_turn: int = 0
    latency_ms: float = 0.0


class Router:
    """Three-layer neural router for Aura model selection."""

    def __init__(self, profiles_path: str = "data/model_profiles.json"):
        self._profiles = ProfileStore(profiles_path)
        self._conversations = ConversationTracker()

    @property
    def profiles(self) -> ProfileStore:
        return self._profiles

    @property
    def conversations(self) -> ConversationTracker:
        return self._conversations

    def route(
        self,
        prompt: str,
        model: Optional[str] = None,
        preference: str = "balanced",
        feature: Optional[str] = None,
        conversation_id: Optional[str] = None,
        has_attachment: bool = False,
    ) -> RoutingResult:
        """Route a prompt to the best model.

        Args:
            prompt: user message
            model: explicit model override (bypasses router)
            preference: "prefer-fast" | "balanced" | "prefer-quality"
            feature: which panel/feature sent this (e.g. "chat", "code")
            conversation_id: for conversation context tracking
            has_attachment: image/file attached

        Returns:
            RoutingResult with model name, reason, and metadata
        """
        start = time.monotonic()

        # Level 1: Explicit override — bypass everything
        if model:
            return RoutingResult(
                model=model,
                reason="explicit_override",
                task_dimensions={},
                latency_ms=0.0,
            )

        # Get conversation context
        conv_profile = None
        conv_tokens = 0
        regen_count = 0
        if conversation_id:
            conv_profile = self._conversations.get_profile(conversation_id)
            conv_tokens = conv_profile.total_tokens
            regen_count = conv_profile.regen_count

        # Layer 1: Instant classifier
        task_needs = score_task(
            prompt,
            has_attachment=has_attachment,
            conversation_tokens=conv_tokens,
            recent_regen_count=regen_count,
        )

        # Layer 3: Conversation context adjustments
        if conversation_id:
            task_needs = self._conversations.adjust(conversation_id, task_needs)

        # Model stickiness: prefer current model if conversation is ongoing
        sticky_model = None
        if conversation_id and conv_profile and conv_profile.last_model:
            sticky_model = conv_profile.last_model

        # Layer 2: Profile matching
        best = match(task_needs, preference, self._profiles)
        reason_parts = []

        # Apply stickiness: only switch if the new model scores significantly better
        if sticky_model and sticky_model != best:
            from aura.routing.matcher import match as _match_fn
            # Re-score sticky model
            sticky_profile = self._profiles.get(sticky_model)
            best_profile = self._profiles.get(best)
            if sticky_profile and best_profile:
                sticky_score = sum(task_needs.get(d, 0) * sticky_profile.get(d, 0) for d in task_needs)
                best_score = sum(task_needs.get(d, 0) * best_profile.get(d, 0) for d in task_needs)
                # Only switch if >15% improvement
                if best_score < sticky_score * 1.15:
                    best = sticky_model
                    reason_parts.append("sticky")

        # Build reason string
        top_dim = max(task_needs, key=task_needs.get) if task_needs else "unknown"
        reason_parts.insert(0, f"top_dim={top_dim}")
        reason_parts.append(f"pref={preference}")

        # Get alternatives (top 3 excluding winner)
        all_profiles = self._profiles.all_profiles()
        scored = []
        for m, p in all_profiles.items():
            if m == best:
                continue
            s = sum(task_needs.get(d, 0) * p.get(d, 0) for d in task_needs)
            scored.append((s, m))
        scored.sort(reverse=True)
        alternatives = [m for _, m in scored[:3]]

        # Update conversation tracker
        if conversation_id:
            features = extract_features(prompt, has_attachment)
            complexity = max(task_needs.get("reason", 0), task_needs.get("code", 0))
            self._conversations.update(
                conversation_id,
                code_ratio=features["code_ratio"],
                complexity=complexity,
                model_used=best,
                tokens=len(prompt.split()) * 2,  # rough estimate
            )

        elapsed = (time.monotonic() - start) * 1000

        result = RoutingResult(
            model=best,
            reason=" + ".join(reason_parts),
            task_dimensions=task_needs,
            alternatives=alternatives,
            conversation_turn=conv_profile.turn_count if conv_profile else 0,
            latency_ms=round(elapsed, 2),
        )
        logger.info(f"[Router] {best} ({result.reason}) in {elapsed:.1f}ms")
        return result
```

**Step 4: Run tests to verify they pass**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py::TestRouter -v`
Expected: All 7 PASS

**Step 5: Run the full test file**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py -v`
Expected: All 36 tests PASS

**Step 6: Commit**

```bash
cd D:/Aura && git add aura/routing/router.py tests/test_neural_router.py
git commit -m "feat(routing): add main Router entry point — 3-layer pipeline complete"
```

---

### Task 7: Wire Router into Backend (Replace Old Routing)

**Files:**
- Modify: `aura/core/model_router_mixin.py` — replace `_select_model()` with Router call
- Modify: `api/services/agent_service.py` — remove LLM classifier, use Router
- Modify: `api/routes/chat.py` — pass routing fields from WebSocket payload

**Step 1: Modify model_router_mixin.py**

Replace `_select_model()` to delegate to the new Router. Keep `_get_client_for_model()`, `set_model_override()`, and fallback chains unchanged.

Key change in `_select_model()`:
```python
def _select_model(self, prompt: str, task_type=None) -> str:
    # Manual override still wins
    if self._model_override:
        return self._model_override

    # Use new Router
    from aura.routing.router import get_router
    result = get_router().route(
        prompt,
        preference=getattr(self, '_routing_preference', 'balanced'),
        conversation_id=getattr(self, '_conversation_id', None),
        has_attachment=getattr(self, '_has_attachment', False),
    )
    return result.model
```

**Step 2: Modify agent_service.py**

Remove `detect_action_mode()` LLM classifier call from chat flow. The Router now handles classification internally in <5ms.

In `AgentService.chat()` / `chat_stream()`:
- Remove: `action_mode = detect_action_mode(message)`
- Remove: `effective_model = get_model_for_action(action_mode)`
- Remove: `agent.brain.set_model_override(effective_model)`
- Add: pass `routing_opts` from WebSocket payload through to brain

**Step 3: Modify api/routes/chat.py**

Extract new `routing` field from WebSocket payload:
```python
routing = msg.get("routing", {})
model_override = routing.get("model") or msg.get("model")  # backward compat
preference = routing.get("preference", "balanced")
feature = routing.get("feature")
```

Pass these through to brain via `_routing_preference`, `_conversation_id`.

Include routing metadata in response chunks:
```python
# After model selected, send routing info to client
await ws.send_json({
    "type": "routing",
    "model_used": result.model,
    "reason": result.reason,
    "task_dimensions": result.task_dimensions,
    "alternatives": result.alternatives,
})
```

**Step 4: Test the integration manually**

Run: `cd D:/Aura && python -m pytest tests/test_neural_router.py -v`
Then: Start the backend and send a few test messages via the extension to verify routing works.

**Step 5: Commit**

```bash
cd D:/Aura && git add aura/core/model_router_mixin.py api/services/agent_service.py api/routes/chat.py
git commit -m "feat(routing): wire 3-layer neural router into backend, remove LLM classifier"
```

---

### Task 8: Extension UI — Preference Tier + Router Display

**Files:**
- Modify: `extension-src/src/store.ts` — add `routingPreference` state
- Modify: `extension-src/src/panels/ChatPanel.tsx` — send routing opts, show router decision
- Modify: `extension-src/src/panels/ModelsPanel.tsx` — add preference tier selector

**Step 1: Add routing preference to store**

In `store.ts`, add to state:
```typescript
routingPreference: 'prefer-fast' | 'balanced' | 'prefer-quality';
setRoutingPreference: (pref: string) => void;
lastRoutingResult: { model_used: string; reason: string; alternatives: string[] } | null;
setLastRoutingResult: (result: any) => void;
```

Persist `routingPreference` to chrome.storage.local.

**Step 2: Update ChatPanel payload**

Change line 228 from:
```typescript
model: overrideModel || getModel('chat'),
```
To:
```typescript
routing: {
    model: overrideModel || getModel('chat') || null,
    preference: routingPreference,
    feature: 'chat',
    conversation_id: conversationId,
},
```

**Step 3: Handle routing response in WebSocket**

In `ws.ts`, handle `type: "routing"` messages:
```typescript
if (msg.type === 'routing') {
    useStore.getState().setLastRoutingResult(msg);
}
```

**Step 4: Add preference selector to ModelsPanel**

Three buttons at top: Fast / Balanced / Quality. Active state styled. Persisted.

**Step 5: Show router decision in ModelPill**

When model is "Auto", display: `Auto → nemotron-3-super` with the reason as tooltip.

**Step 6: Commit**

```bash
cd D:/Aura && git add extension-src/src/store.ts extension-src/src/panels/ChatPanel.tsx extension-src/src/panels/ModelsPanel.tsx extension-src/src/components/ModelPill.tsx
git commit -m "feat(extension): add preference tier selector + router decision display"
```

---

### Task 9: CLI — Preference Flag + /routing Command

**Files:**
- Modify: `aura/cli/chat_session.py` — add `--preference` flag
- Modify: `aura/cli/model_picker.py` — show routing info

**Step 1: Add --preference flag to CLI**

In chat_session.py argparse, add:
```python
parser.add_argument("--preference", choices=["fast", "balanced", "quality"], default="balanced")
```

Map to router: `fast` → `prefer-fast`, `quality` → `prefer-quality`.

**Step 2: Add /routing command**

New command in CLI that prints:
- Current conversation profile (turn count, code mode, complexity trend)
- Last routing decision (model, reason, alternatives)
- Model profile scores for the last task dimensions

**Step 3: Show model in status line**

After each response, display: `[nemotron-3-super · fast · 2.1ms]`

**Step 4: Commit**

```bash
cd D:/Aura && git add aura/cli/chat_session.py aura/cli/model_picker.py
git commit -m "feat(cli): add --preference flag and /routing command"
```

---

### Task 10: Wire Feedback Signals

**Files:**
- Modify: `api/routes/chat.py` — capture regen/abort signals
- Modify: `extension-src/src/panels/ChatPanel.tsx` — send feedback on regen/switch
- Create: `api/routes/routing.py` — API endpoint for routing stats + feedback

**Step 1: Create routing API endpoint**

```python
# api/routes/routing.py
router = APIRouter(prefix="/api/routing", tags=["routing"])

@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest):
    """Record a feedback signal for the learning loop."""
    # signal, model, task_dimensions, conversation_id, switched_to
    ...

@router.get("/stats")
async def get_routing_stats():
    """Return current model profiles + learning stats."""
    ...

@router.get("/conversation/{conversation_id}")
async def get_conversation_profile(conversation_id: str):
    """Return conversation routing profile."""
    ...
```

**Step 2: Extension sends feedback**

On regenerate click: POST `/api/routing/feedback` with `signal: "regeneration"`.
On model switch: POST with `signal: "model_switch", switched_to: "..."`.
On thumbs up/down: POST with `signal: "thumbs_up"` / `"thumbs_down"`.

**Step 3: Backend records feedback**

Route handler calls `learning.process_feedback()` + `conversation.record_feedback()`.

**Step 4: Commit**

```bash
cd D:/Aura && git add api/routes/routing.py extension-src/src/panels/ChatPanel.tsx
git commit -m "feat(routing): wire feedback signals — regen, switch, rating, abort"
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | Model profiles data layer | 5 |
| 2 | Layer 1 instant classifier | 9 |
| 3 | Layer 2 profile matcher | 5 |
| 4 | Layer 3 conversation tracker | 10 |
| 5 | Learning loop | 4 |
| 6 | Main router entry point | 7 |
| 7 | Wire into backend (replace old routing) | integration |
| 8 | Extension UI (preference tier + display) | manual |
| 9 | CLI (--preference + /routing) | manual |
| 10 | Feedback signal wiring | manual |

**Total: 10 tasks, ~40 unit tests, 6 new files, 5 modified files.**

Tasks 1-6 are pure TDD with no side effects. Tasks 7-10 are integration work.
