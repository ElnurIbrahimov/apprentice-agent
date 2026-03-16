"""Emotional Memory Tagging — Phase 3.3.

Tags every memory with the current ALMA PAD state at write time,
and biases retrieval toward mood-congruent memories (20% weight).

During NeuroDream, strengthens memories whose emotional valence
matches the persistent mood.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Weight for mood-congruent retrieval bias (0-1)
MOOD_CONGRUENCE_WEIGHT = 0.20


def get_current_pad() -> Dict[str, float]:
    """Get current ALMA PAD state for memory tagging."""
    try:
        from aura.emotion.alma_engine import get_emotional_state
        state = get_emotional_state()
        pad = state.get("pad", {})
        return {
            "pleasure": round(pad.get("pleasure", 0.0), 3),
            "arousal": round(pad.get("arousal", 0.0), 3),
            "dominance": round(pad.get("dominance", 0.0), 3),
        }
    except Exception:
        return {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}


def tag_memory_with_emotion(memory_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Add emotional PAD tag to memory metadata before storing.

    Call this before writing any memory to the store.
    """
    pad = get_current_pad()
    memory_metadata["emotion_pad"] = pad
    memory_metadata["emotion_valence"] = _compute_valence(pad)
    return memory_metadata


def compute_mood_congruence(memory_pad: Dict[str, float], current_pad: Optional[Dict[str, float]] = None) -> float:
    """Compute similarity between a memory's emotional state and current mood.

    Returns 0-1 where 1 = perfect match.
    Uses cosine-like similarity in PAD space.
    """
    if current_pad is None:
        current_pad = get_current_pad()

    if not memory_pad:
        return 0.5  # Neutral — no emotional tag

    # Euclidean distance in PAD space, normalized to 0-1
    dp = (memory_pad.get("pleasure", 0) - current_pad.get("pleasure", 0)) ** 2
    da = (memory_pad.get("arousal", 0) - current_pad.get("arousal", 0)) ** 2
    dd = (memory_pad.get("dominance", 0) - current_pad.get("dominance", 0)) ** 2

    # Max possible distance in PAD cube [-1,1]^3 is sqrt(12) ≈ 3.46
    distance = (dp + da + dd) ** 0.5
    similarity = max(0.0, 1.0 - distance / 3.46)

    return round(similarity, 3)


def rerank_with_mood_congruence(
    results: List[Tuple[Any, float]],
    weight: float = MOOD_CONGRUENCE_WEIGHT,
) -> List[Tuple[Any, float]]:
    """Rerank retrieval results by blending original score with mood congruence.

    Args:
        results: List of (memory, score) tuples from retrieval.
        weight: How much to weight mood congruence (0-1).

    Returns:
        Re-sorted list of (memory, adjusted_score) tuples.
    """
    if not results or weight <= 0:
        return results

    current_pad = get_current_pad()
    reranked = []

    for memory, score in results:
        # Extract stored PAD from memory metadata
        memory_pad = {}
        if hasattr(memory, "metadata"):
            memory_pad = memory.metadata.get("emotion_pad", {})
        elif isinstance(memory, dict):
            memory_pad = memory.get("metadata", {}).get("emotion_pad", {})

        congruence = compute_mood_congruence(memory_pad, current_pad)
        adjusted = score * (1 - weight) + congruence * weight
        reranked.append((memory, adjusted))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def strengthen_mood_congruent_memories(
    memories: List[Dict[str, Any]],
    persistent_pad: Dict[str, float],
    boost_factor: float = 0.1,
) -> List[str]:
    """During NeuroDream: strengthen memories matching persistent mood.

    Args:
        memories: List of memory dicts with 'id' and 'metadata'.
        persistent_pad: The average PAD state during the session.
        boost_factor: How much to boost strength (0-1).

    Returns:
        List of memory IDs that were strengthened.
    """
    strengthened = []

    for mem in memories:
        mem_pad = mem.get("metadata", {}).get("emotion_pad", {})
        if not mem_pad:
            continue

        congruence = compute_mood_congruence(mem_pad, persistent_pad)
        if congruence > 0.6:
            mem_id = mem.get("id", "")
            if mem_id:
                try:
                    from aura.memory.fade_mem import get_fade_mem
                    fm = get_fade_mem()
                    if hasattr(fm, "reinforce"):
                        fm.reinforce(mem_id, boost_factor * congruence)
                        strengthened.append(mem_id)
                except Exception as e:
                    logger.debug(f"[MemoryTagging] Reinforce error: {e}")

    if strengthened:
        logger.info(
            f"[MemoryTagging] Strengthened {len(strengthened)} mood-congruent memories"
        )

    return strengthened


def _compute_valence(pad: Dict[str, float]) -> float:
    """Compute overall emotional valence from PAD state (-1 to 1)."""
    pleasure = pad.get("pleasure", 0.0)
    arousal = pad.get("arousal", 0.0)
    return round(pleasure * 0.7 + arousal * 0.3, 3)
