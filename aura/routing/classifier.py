"""Layer 1 Instant Classifier – feature extraction + task dimension scoring.

Pure Python, no external deps.  Target: < 0.1 ms per call.
"""

from __future__ import annotations

import re
from typing import Dict, List

# ── action verbs for imperative detection ────────────────────────────────
_ACTION_VERBS = frozenset(
    "build create make write fix debug explain compare analyze search find "
    "implement design generate translate summarize review optimize refactor "
    "deploy test".split()
)

# ── code-related verbs ──────────────────────────────────────────────────
_CODE_VERBS = frozenset(
    "fix debug implement refactor deploy build compile test code script".split()
)

# ── language keyword map  (lowercase) ────────────────────────────────────
_LANG_KEYWORDS: Dict[str, List[str]] = {
    "python":     ["def ", "import ", "class ", "lambda ", "print(", "self."],
    "javascript": ["const ", "let ", "function ", "=>", "async ", "require(", "console."],
    "sql":        ["SELECT", "FROM", "WHERE", "INSERT", "CREATE TABLE"],
    "rust":       ["fn ", "let mut", "impl ", "pub ", "struct "],
    "go":         ["func ", "package ", "import (", "defer "],
}

# ── complex-reasoning trigger phrases ────────────────────────────────────
_COMPLEX_PATTERNS = [
    "comprehensive analysis",
    "step by step guide",
    "compare and contrast",
    "pros and cons",
    "detailed explanation",
    "write an essay",
    "write a report",
    "write a detailed",
    "write an report",
    "deep dive into",
]

# ── code-like character set ──────────────────────────────────────────────
_CODE_CHARS = set("{}[]();=<>!&|^~")

# precompiled patterns for camelCase and snake_case
_CAMEL_RE = re.compile(r"[a-z][A-Z]")
_SNAKE_RE = re.compile(r"[a-z]+_[a-z]+")


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════

def extract_features(prompt: str, *, has_attachment: bool = False) -> dict:
    """Extract lightweight features from a raw prompt (< 0.1 ms)."""
    words = prompt.split()
    word_count = len(words)

    # language markers (computed early so code_ratio can use them)
    language_markers: List[str] = []
    _lang_kw_tokens: set = set()  # tokens that matched language keywords
    for lang, keywords in _LANG_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt:
                language_markers.append(lang)
                # collect the first word of the keyword for code_ratio boost
                _lang_kw_tokens.add(kw.strip().split()[0].lower().rstrip("("))
                break

    # code_ratio: fraction of tokens that look code-like
    code_token_count = 0
    for token in words:
        tok_lower = token.lower().rstrip(",:;.!?()")
        if any(ch in _CODE_CHARS for ch in token):
            code_token_count += 1
        elif _CAMEL_RE.search(token):
            code_token_count += 1
        elif _SNAKE_RE.search(token):
            code_token_count += 1
        elif tok_lower in _lang_kw_tokens:
            code_token_count += 1
    # also count leading whitespace (indentation) as a code signal
    lines = prompt.splitlines()
    for line in lines:
        if line and line[0] in (" ", "\t") and line.strip():
            code_token_count += 1

    code_ratio = code_token_count / max(word_count, 1)
    code_ratio = min(code_ratio, 1.0)

    # question marks
    question_marks = prompt.count("?")

    # imperative score
    first_word = words[0].lower().rstrip(",:;") if words else ""
    imperative_score = 1.0 if first_word in _ACTION_VERBS else 0.0

    return {
        "word_count": word_count,
        "code_ratio": code_ratio,
        "question_marks": question_marks,
        "has_attachment": has_attachment,
        "language_markers": language_markers,
        "imperative_score": imperative_score,
    }


def score_task(
    prompt: str,
    *,
    has_attachment: bool = False,
    conversation_tokens: int = 0,
    recent_regen_count: int = 0,
) -> dict:
    """Return a 6-dimension task-need vector (all floats 0.0–1.0)."""
    feats = extract_features(prompt, has_attachment=has_attachment)
    wc = feats["word_count"]
    prompt_lower = prompt.lower()

    # ── code ─────────────────────────────────────────────────────────
    code = 0.0
    if feats["code_ratio"] > 0.15 or feats["language_markers"]:
        code = max(code, 0.9)
    # check for code verbs anywhere in the prompt
    prompt_words_lower = {w.lower().rstrip(",:;.!?") for w in prompt.split()}
    if prompt_words_lower & _CODE_VERBS:
        code = max(code, 0.7)

    # ── reason ───────────────────────────────────────────────────────
    reason = 0.0
    if wc > 50:
        reason = max(reason, 0.6)
    if any(pat in prompt_lower for pat in _COMPLEX_PATTERNS):
        reason = max(reason, 0.8)
    if feats["question_marks"] > 0 and wc > 20:
        reason = max(reason, 0.6)

    # ── speed ────────────────────────────────────────────────────────
    if wc <= 5:
        speed = 1.0
    elif wc <= 10:
        speed = 0.9
    elif wc > 100:
        speed = 0.2
    elif wc > 50:
        speed = 0.3
    else:
        speed = 0.5
    # complex reasoning tasks are inherently slow — penalize speed
    if reason >= 0.7:
        speed = min(speed, 0.3)

    # ── context ──────────────────────────────────────────────────────
    context = 0.0
    if conversation_tokens > 150_000:
        context = 1.0
    elif conversation_tokens > 100_000:
        context = 0.7
    elif conversation_tokens > 50_000:
        context = 0.5

    # ── vision ───────────────────────────────────────────────────────
    vision = 1.0 if has_attachment else 0.0

    # ── quality ──────────────────────────────────────────────────────
    quality = 0.5
    if recent_regen_count > 0:
        quality = min(0.5 + recent_regen_count * 0.2, 1.0)

    return {
        "code": code,
        "reason": reason,
        "speed": speed,
        "context": context,
        "quality": quality,
        "vision": vision,
    }
