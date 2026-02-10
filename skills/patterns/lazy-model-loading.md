# Skill: Lazy Model Loading Pattern

## When to Use
For heavy ML models (Florence-2, Whisper, sentence-transformers) that shouldn't load at startup.

## Pattern
```python
# Module-level singletons
_model = None
_available: Optional[bool] = None


def _check_available() -> bool:
    """Check if dependency is installed."""
    global _available
    if _available is not None:
        return _available
    try:
        import the_library
        _available = True
    except ImportError:
        _available = False
        logger.info("Library not installed (pip install the-library)")
    return _available


def _load_model(variant: str = "base"):
    """Lazy-load model on first use."""
    global _model
    if _model is not None:
        return _model

    if not _check_available():
        raise RuntimeError("Library not installed")

    import the_library
    logger.info(f"Loading model '{variant}'...")
    _model = the_library.load(variant)
    logger.info(f"Model '{variant}' loaded")
    return _model
```

## Examples in AURA
- `vision.py` — Florence-2 model
- `audio_transcriber.py` — Whisper model
- `screen_reader.py` — OCR model (reuses Florence-2)

## Benefits
- Startup time stays fast
- Memory only used when needed
- Singleton prevents duplicate loading
- Graceful degradation if dependency missing
