"""Shared ML model singletons to avoid loading duplicates into GPU/RAM."""
import logging

logger = logging.getLogger(__name__)

_st_model = None

def get_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    """Return a shared SentenceTransformer instance, loading once on first call."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[SharedModels] Loading SentenceTransformer: {model_name}")
        _st_model = SentenceTransformer(model_name)
    return _st_model


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
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            model_name = "microsoft/Florence-2-base"
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
