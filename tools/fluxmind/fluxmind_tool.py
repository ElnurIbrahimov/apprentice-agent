"""
FluxMind Tool for Aura (Apprentice Agent)
=========================================
Tool #16: Calibrated reasoning engine with uncertainty awareness.

Integration with Aura's 4-model routing:
- Routes structured reasoning tasks to FluxMind
- Provides calibrated confidence (unlike LLM confidence)
- Enables "should I trust this?" decisions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import FluxMind (will be in same tools/ directory)
try:
    from .fluxmind_core import FluxMind, FluxMindConfig, load_fluxmind
    FLUXMIND_AVAILABLE = True
except ImportError:
    try:
        from fluxmind_core import FluxMind, FluxMindConfig, load_fluxmind
        FLUXMIND_AVAILABLE = True
    except ImportError:
        FLUXMIND_AVAILABLE = False


class FluxMindTool:
    """
    FluxMind Integration for Aura

    Capabilities:
    - Calibrated uncertainty (knows when it doesn't know)
    - Sub-millisecond inference
    - Compositional reasoning (can mix reasoning strategies)
    - OOD detection (flags unfamiliar inputs)
    """

    def __init__(self, model_path: str = None):
        self.name = "fluxmind"
        self.description = "Calibrated reasoning engine with uncertainty awareness"
        self.model = None
        self.model_path = model_path or "models/fluxmind_v0751.pt"

        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.low_confidence_threshold = 0.5

        # Load model if available
        if FLUXMIND_AVAILABLE and Path(self.model_path).exists():
            self._load_model()

    def _load_model(self):
        """Load the FluxMind model."""
        try:
            self.model = load_fluxmind(self.model_path)
            return True
        except Exception as e:
            print(f"Failed to load FluxMind: {e}")
            return False

    def is_available(self) -> bool:
        """Check if FluxMind is loaded and ready."""
        return self.model is not None

    # ========================================================================
    # CORE METHODS (for Aura tool detection)
    # ========================================================================

    def step(self, state: List[int], operation: int, context: int) -> Dict:
        """
        Execute single reasoning step with calibrated confidence.

        Args:
            state: Current state [x, y, z, w] (4 integers, 1-15 range)
            operation: Operation index (0-7)
            context: Context/DSL index (0=additive, 1=multiplicative)

        Returns:
            dict with next_state, confidence, should_trust

        Example:
            >>> fluxmind.step([5, 3, 7, 2], 0, 0)
            {'next_state': [6, 3, 7, 2], 'confidence': 0.97, 'should_trust': True}
        """
        if not self.is_available():
            return {"error": "FluxMind not loaded", "should_trust": False}

        result = self.model.step(state, operation, context)
        result["should_trust"] = result["confidence"] >= self.high_confidence_threshold
        result["uncertainty_flag"] = result["confidence"] < self.low_confidence_threshold

        return result

    def execute(self, initial_state: List[int], operations: List[int],
                contexts: List[int]) -> Dict:
        """
        Execute full reasoning program with trajectory and confidence tracking.

        Args:
            initial_state: Starting state [x, y, z, w]
            operations: List of operation indices
            contexts: List of context indices per step

        Returns:
            dict with trajectory, confidences, mean_confidence, should_trust

        Example:
            >>> fluxmind.execute([5, 3, 7, 2], [0, 2, 4], [0, 0, 1])
            {'trajectory': [[5,3,7,2], [6,3,7,2], ...], 'mean_confidence': 0.94, ...}
        """
        if not self.is_available():
            return {"error": "FluxMind not loaded", "should_trust": False}

        result = self.model.execute(initial_state, operations, contexts)
        result["should_trust"] = result["mean_confidence"] >= self.high_confidence_threshold
        result["low_confidence_steps"] = [
            i for i, c in enumerate(result["confidences"])
            if c < self.low_confidence_threshold
        ]

        return result

    def get_confidence(self, state: List[int], operation: int, context: int) -> Dict:
        """
        Get confidence score without full execution (faster).

        Use this for quick "should I proceed?" checks.

        Returns:
            dict with confidence, should_trust, interpretation
        """
        if not self.is_available():
            return {"error": "FluxMind not loaded", "confidence": 0.0}

        result = self.model.step(state, operation, context)
        conf = result["confidence"]

        if conf >= self.high_confidence_threshold:
            interpretation = "High confidence - proceed"
        elif conf >= self.low_confidence_threshold:
            interpretation = "Medium confidence - verify recommended"
        else:
            interpretation = "Low confidence - abstain or get human input"

        return {
            "confidence": conf,
            "should_trust": conf >= self.high_confidence_threshold,
            "should_abstain": conf < self.low_confidence_threshold,
            "interpretation": interpretation
        }

    def verify_sequence(self, actions: List[Dict]) -> Dict:
        """
        Verify if a sequence of actions is coherent.

        Args:
            actions: List of {"state": [...], "op": int, "context": int}

        Returns:
            dict with valid, confidence, issues

        Use this to verify LLM-generated action plans!
        """
        if not self.is_available():
            return {"error": "FluxMind not loaded", "valid": False}

        confidences = []
        issues = []

        for i, action in enumerate(actions):
            result = self.get_confidence(
                action["state"],
                action["op"],
                action["context"]
            )
            confidences.append(result["confidence"])

            if result["should_abstain"]:
                issues.append(f"Step {i}: Low confidence ({result['confidence']:.2f})")

        mean_conf = sum(confidences) / len(confidences) if confidences else 0

        return {
            "valid": len(issues) == 0,
            "mean_confidence": mean_conf,
            "confidences": confidences,
            "issues": issues,
            "recommendation": "Proceed" if mean_conf >= self.high_confidence_threshold else "Review"
        }

    def status(self) -> Dict:
        """Get FluxMind status and capabilities."""
        if not self.is_available():
            return {
                "available": False,
                "reason": "Model not loaded",
                "model_path": self.model_path
            }

        return {
            "available": True,
            "model_path": self.model_path,
            "version": "0.75.1",
            "capabilities": [
                "Calibrated uncertainty",
                "Compositional reasoning",
                "OOD detection",
                "Sub-ms inference"
            ],
            "thresholds": {
                "high_confidence": self.high_confidence_threshold,
                "low_confidence": self.low_confidence_threshold
            }
        }

    # ========================================================================
    # AURA INTEGRATION HELPERS
    # ========================================================================

    def should_use_fluxmind(self, query: str) -> bool:
        """
        Determine if FluxMind should handle this query.

        Used by Aura's router to decide model selection.
        """
        # Keywords that suggest structured reasoning
        structured_keywords = [
            "step by step", "sequence", "procedure", "algorithm",
            "confidence", "uncertain", "sure", "verify",
            "state", "transition", "operation"
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in structured_keywords)

    def format_for_aura(self, result: Dict) -> str:
        """Format FluxMind result for Aura's output."""
        if "error" in result:
            return f"FluxMind: {result['error']}"

        conf = result.get("confidence") or result.get("mean_confidence", 0)
        trust = "OK" if result.get("should_trust") else "WARN"

        output = f"{trust} Confidence: {conf:.1%}\n"

        if "next_state" in result:
            output += f"Result: {result['next_state']}\n"

        if "trajectory" in result:
            output += f"Trajectory: {len(result['trajectory'])} steps\n"

        if result.get("low_confidence_steps"):
            output += f"Low confidence at steps: {result['low_confidence_steps']}\n"

        if result.get("issues"):
            output += f"Issues: {', '.join(result['issues'])}\n"

        return output.strip()


# ============================================================================
# TOOL REGISTRATION (for Aura's tool detection)
# ============================================================================

TOOL_INFO = {
    "name": "fluxmind",
    "description": "Calibrated reasoning engine with uncertainty awareness. "
                   "Use for structured reasoning tasks where knowing confidence matters.",
    "keywords": ["reasoning", "confidence", "uncertainty", "verify", "calibrated",
                 "sequence", "state", "step"],
    "methods": {
        "step": "Execute single reasoning step with calibrated confidence",
        "execute": "Execute full reasoning program with trajectory tracking",
        "get_confidence": "Quick confidence check without full execution",
        "verify_sequence": "Verify if action sequence is coherent",
        "status": "Get FluxMind status and capabilities"
    }
}


def get_tool():
    """Factory function for Aura's tool loader."""
    return FluxMindTool()


if __name__ == "__main__":
    # Test the tool
    tool = FluxMindTool()
    print(f"Status: {json.dumps(tool.status(), indent=2)}")

    if tool.is_available():
        # Test step
        result = tool.step([5, 3, 7, 2], 0, 0)
        print(f"\nStep result: {json.dumps(result, indent=2)}")
        print(f"\nFormatted: {tool.format_for_aura(result)}")
