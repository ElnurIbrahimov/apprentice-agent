"""
Introspection Tool for AURA
===========================

Tool interface for the Introspection Circuit.
Integrates uncertainty detection into AURA's response pipeline.
"""

import logging
from typing import Dict, Any, Optional, Callable

from .introspection_circuit import (
    IntrospectionCircuit,
    IntrospectionConfig,
    IntrospectionResult,
    IntrospectionAction,
    ConfidenceLevel,
    QueryType,
)

logger = logging.getLogger(__name__)


class IntrospectionTool:
    """
    Tool interface for Introspection Circuit.

    This tool allows AURA to:
    - Assess confidence before responding
    - Automatically verify uncertain claims
    - Add appropriate epistemic markers
    - Track uncertainty patterns over time
    """

    name = "introspection"
    description = """Uncertainty detection tool that helps AURA know when it doesn't know.
    Use this to assess confidence in responses and trigger verification when needed."""

    def __init__(
        self,
        llm_func: Callable[[str, Optional[str]], str],
        search_func: Optional[Callable[[str], str]] = None,
        config: Optional[IntrospectionConfig] = None,
        fluxmind: Optional[Any] = None,
        guardian: Optional[Any] = None,
    ):
        """
        Initialize the Introspection Tool.

        Args:
            llm_func: LLM function (prompt, system) -> response
            search_func: Optional search function for verification
            config: Optional configuration
            fluxmind: Optional FluxMind instance
            guardian: Optional Guardian instance
        """
        self.circuit = IntrospectionCircuit(
            llm_func=llm_func,
            config=config or IntrospectionConfig(),
            search_func=search_func,
            fluxmind=fluxmind,
            guardian=guardian,
        )

        # Track recent introspections for UI
        self._recent_results: list = []
        self._max_recent = 50

    def analyze_query(
        self,
        query: str,
        response: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze a query for uncertainty.

        Args:
            query: The user's query
            response: Optional pre-generated response
            context: Additional context

        Returns:
            Dictionary with confidence analysis
        """
        result = self.circuit.analyze(query, response, context)
        self._add_recent(result)

        return {
            "success": True,
            **result.to_dict(),
            "recommendation": self._get_recommendation(result),
        }

    def pre_check(
        self,
        query: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Quick pre-check before generating a response.
        Determines if verification is needed.

        Args:
            query: The user's query
            context: Additional context

        Returns:
            Dictionary with pre-check results
        """
        result, verification_info = self.circuit.pre_response_check(query, context)
        self._add_recent(result)

        return {
            "success": True,
            "needs_verification": result.should_verify,
            "verification_query": result.verification_query,
            "verification_result": verification_info,
            "confidence": result.confidence,
            "confidence_level": result.confidence_level.value,
            "query_type": result.query_type.value,
            "action": result.action.value,
        }

    def wrap_response(
        self,
        response: str,
        query: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Wrap a response with appropriate epistemic markers.

        Args:
            response: The response to wrap
            query: The original query
            context: Additional context

        Returns:
            Dictionary with wrapped response
        """
        result = self.circuit.analyze(query, response, context)
        wrapped = self.circuit.wrap_response(response, result)

        return {
            "success": True,
            "original_response": response,
            "wrapped_response": wrapped,
            "was_modified": response != wrapped,
            "confidence": result.confidence,
            "confidence_level": result.confidence_level.value,
            "markers_used": result.epistemic_markers,
        }

    def should_verify(self, query: str) -> bool:
        """Quick check if a query needs verification"""
        result = self.circuit.analyze(query)
        return result.should_verify

    def get_confidence(self, query: str, response: str = "") -> float:
        """Get confidence score for a query/response"""
        result = self.circuit.analyze(query, response)
        return result.confidence

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit statistics"""
        stats = self.circuit.get_stats()
        return {
            "success": True,
            **stats,
            "recent_count": len(self._recent_results),
        }

    def get_recent(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent introspection results"""
        recent = self._recent_results[-limit:]
        return {
            "success": True,
            "results": [r.to_dict() for r in recent],
            "total": len(self._recent_results),
        }

    def status(self) -> Dict[str, Any]:
        """Get tool status"""
        config = self.circuit.config
        return {
            "success": True,
            "tool": "introspection",
            "description": "Uncertainty detection and confidence calibration",
            "config": {
                "high_threshold": config.high_confidence_threshold,
                "medium_threshold": config.medium_confidence_threshold,
                "low_threshold": config.low_confidence_threshold,
                "verify_factual_below": config.verify_factual_below,
                "consistency_check_enabled": config.enable_consistency_check,
                "auto_verification_enabled": config.enable_auto_verification,
                "epistemic_markers_enabled": config.enable_epistemic_markers,
            },
            "stats": self.circuit.get_stats(),
            "integrations": {
                "fluxmind": self.circuit.fluxmind is not None,
                "guardian": self.circuit.guardian is not None,
                "search": self.circuit.search_func is not None,
            },
        }

    def _add_recent(self, result: IntrospectionResult):
        """Add to recent results, maintaining max size"""
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent:]

    def _get_recommendation(self, result: IntrospectionResult) -> str:
        """Get human-readable recommendation"""
        action_recommendations = {
            IntrospectionAction.RESPOND: "Respond directly with high confidence.",
            IntrospectionAction.RESPOND_HEDGED: "Respond with uncertainty markers (e.g., 'I believe...', 'Based on my understanding...').",
            IntrospectionAction.VERIFY_THEN_RESPOND: f"Verify before responding. Suggested search: '{result.verification_query}'",
            IntrospectionAction.ABSTAIN: "Consider declining to answer or asking for clarification. Confidence is too low.",
            IntrospectionAction.ASK_CLARIFICATION: "Ask the user for more details before attempting to answer.",
        }
        return action_recommendations.get(result.action, "Proceed with caution.")

    # Main execution method for tool interface
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an introspection action.

        Args:
            action: The action to perform
            **kwargs: Additional arguments

        Returns:
            Result dictionary
        """
        action = action.lower().strip()

        if action in ["analyze", "check", "assess"]:
            return self.analyze_query(
                query=kwargs.get("query", ""),
                response=kwargs.get("response"),
                context=kwargs.get("context", ""),
            )

        elif action in ["pre_check", "precheck", "before"]:
            return self.pre_check(
                query=kwargs.get("query", ""),
                context=kwargs.get("context", ""),
            )

        elif action in ["wrap", "hedge", "modify"]:
            return self.wrap_response(
                response=kwargs.get("response", ""),
                query=kwargs.get("query", ""),
                context=kwargs.get("context", ""),
            )

        elif action in ["stats", "statistics"]:
            return self.get_stats()

        elif action in ["recent", "history"]:
            return self.get_recent(kwargs.get("limit", 10))

        elif action in ["status", "info"]:
            return self.status()

        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "available_actions": [
                    "analyze", "pre_check", "wrap", "stats", "recent", "status"
                ],
            }


# Convenience function
def get_introspection_tool(
    llm_func: Callable[[str, Optional[str]], str],
    **kwargs
) -> IntrospectionTool:
    """Create an Introspection Tool"""
    return IntrospectionTool(llm_func=llm_func, **kwargs)
