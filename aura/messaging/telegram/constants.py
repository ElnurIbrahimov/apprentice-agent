"""
Constants and simple data structures used across the telegram package.
No dependencies on bot.py or any mixin.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

# Max chars for code output before truncation in Telegram messages
_MAX_OUTPUT_CHARS = 3500

# Premium tier definitions for Telegram payments
PREMIUM_TIERS = {
    "supporter": {
        "title": "AURA Supporter",
        "description": "Support AURA development",
        "price": 500,  # $5.00 in cents
        "currency": "USD",
        "benefits": ["Priority responses", "Badge in status"],
    },
    "pro": {
        "title": "AURA Pro",
        "description": "Unlock advanced features",
        "price": 1500,  # $15.00
        "currency": "USD",
        "benefits": ["Unlimited research", "Fleet mode", "Priority model routing", "Custom personality"],
    },
    "patron": {
        "title": "AURA Patron",
        "description": "Maximum support + all features",
        "price": 5000,  # $50.00
        "currency": "USD",
        "benefits": ["Everything in Pro", "Direct feature requests", "Early access", "Custom training"],
    },
}

# ============================================================================
#  Emotion-to-sticker/GIF mapping for contextual reactions (Phase 5)
# ============================================================================
EMOTION_REACTIONS = {
    "joy": {"sticker_query": "happy", "gif_queries": ["celebration", "happy dance", "yay"]},
    "excited": {"sticker_query": "excited", "gif_queries": ["excited", "woohoo", "amazing"]},
    "curious": {"sticker_query": "thinking", "gif_queries": ["thinking", "hmm", "curious"]},
    "surprised": {"sticker_query": "surprised", "gif_queries": ["shocked", "wow", "surprised"]},
    "sad": {"sticker_query": "sad", "gif_queries": ["sad", "cry", "disappointed"]},
    "frustrated": {"sticker_query": "angry", "gif_queries": ["frustrated", "facepalm"]},
    "grateful": {"sticker_query": "thank you", "gif_queries": ["thank you", "grateful", "heart"]},
    "empathetic": {"sticker_query": "hug", "gif_queries": ["hug", "comfort", "care"]},
    "confident": {"sticker_query": "cool", "gif_queries": ["confident", "boss", "cool"]},
    "neutral": {"sticker_query": "ok", "gif_queries": ["ok", "thumbs up", "nod"]},
}

EMOTION_EMOJI = {
    "joy": "\U0001f60a",
    "excited": "\U0001f525",
    "curious": "\U0001f914",
    "surprised": "\U0001f62e",
    "sad": "\U0001f622",
    "frustrated": "\U0001f624",
    "grateful": "\u2764\ufe0f",
    "empathetic": "\U0001f917",
    "confident": "\U0001f4aa",
    "neutral": "\U0001f44d",
}


class _LRUCache:
    """Simple LRU cache using OrderedDict."""

    def __init__(self, maxsize: int = 50):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> "Optional[str]":
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: str):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
