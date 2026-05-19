"""Shared ML model singletons to avoid loading duplicates into GPU/RAM."""
import logging

logger = logging.getLogger(__name__)

_st_models: dict = {}

def get_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    """Return a shared SentenceTransformer instance, loading once per model name."""
    global _st_models
    if model_name not in _st_models:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[SharedModels] Loading SentenceTransformer: {model_name}")
        _st_models[model_name] = SentenceTransformer(model_name)
    return _st_models[model_name]


_florence2_model = None
_florence2_processor = None

def get_florence2():
    """Return shared Florence-2 model and processor."""
    global _florence2_model, _florence2_processor
    # Sentinel: False means a previous load attempt failed — don't retry.
    if _florence2_model is False:
        return None, None
    if _florence2_model is None:
        import os
        if not os.getenv("FLORENCE2_ENABLED", "true").lower() == "true":
            return None, None
        try:
            import os as _os

            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            model_name = "microsoft/Florence-2-base"
            _trust_remote = _os.environ.get("AURA_TRUST_REMOTE_CODE", "0") == "1"
            if not _trust_remote:
                logger.warning(
                    "[SharedModels] Florence-2 requires trust_remote_code=True. "
                    "Set AURA_TRUST_REMOTE_CODE=1 to enable. Skipping load."
                )
                return None, None
            logger.info(f"[SharedModels] Loading Florence-2: {model_name}")
            _florence2_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            _florence2_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                _florence2_model = _florence2_model.cuda()
        except Exception as e:
            logger.warning(f"[SharedModels] Florence-2 load failed: {e}")
            _florence2_model = False  # Sentinel: prevent repeated load attempts
            return None, None
    return _florence2_model, _florence2_processor
