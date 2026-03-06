"""MirrorMind - Self-Critique System for AURA.

Tool #21: Evaluates and improves responses before showing to user.

Flow:
    User Query -> Generate Response -> Critique -> Score Too Low? -> Improve -> Repeat
                                           |
                                     Score Good? -> Return Final Response
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import ollama

from ..config import Config

_OLLAMA_CLOUD_HOST = "https://ollama.com"


logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of self-critique and improvement cycle."""
    original: str
    improved: str
    quality_score: float
    iterations: int
    improvements_made: List[str] = field(default_factory=list)

    def was_improved(self) -> bool:
        """Check if the response was actually improved."""
        return self.original != self.improved and self.iterations > 0


class MirrorMind:
    """Self-critique system that evaluates and improves responses."""

    # Critique dimensions with weights
    DIMENSIONS = {
        "accuracy": 0.25,      # Is it correct?
        "completeness": 0.25,  # Fully answers question?
        "clarity": 0.20,       # Easy to understand?
        "actionable": 0.15,    # Gives next steps?
        "tone": 0.15           # Appropriate for context?
    }

    CRITIQUE_PROMPT = """Evaluate this response on a scale of 0.0 to 1.0.

QUERY: {query}

RESPONSE: {response}

Score each dimension (0.0 = terrible, 1.0 = perfect):
- Accuracy: Is the information correct and factual?
- Completeness: Does it fully answer the question?
- Clarity: Is it easy to understand?
- Actionable: Does it provide clear next steps?
- Tone: Is the tone appropriate for the context?

Respond in EXACTLY this format:
ACCURACY: <score>
COMPLETENESS: <score>
CLARITY: <score>
ACTIONABLE: <score>
TONE: <score>
OVERALL_SCORE: <weighted_average>
ISSUES: <comma-separated list of problems, or "none">
SUGGESTIONS: <comma-separated list of improvements, or "none">"""

    IMPROVE_PROMPT = """Improve this response based on the feedback.

ORIGINAL QUERY: {query}

CURRENT RESPONSE: {response}

ISSUES IDENTIFIED:
{issues}

SUGGESTIONS FOR IMPROVEMENT:
{suggestions}

Write an improved response that addresses these issues. Be concise but complete.
Only output the improved response, nothing else."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = None,
        quality_threshold: float = 0.75,
        max_iterations: int = 2
    ):
        """Initialize MirrorMind.

        Args:
            ollama_url: Ollama API base URL
            model: Model to use for critique and improvement (default: Config.MODEL_FAST)
            quality_threshold: Minimum score to accept (0.0-1.0)
            max_iterations: Maximum improvement rounds
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model or Config.MODEL_FAST
        self.quality_threshold = max(0.0, min(1.0, quality_threshold))
        self.max_iterations = max(1, min(5, max_iterations))
        self.name = "mirrormind"
        self.description = "Self-critique and response improvement system"

        # Initialize ollama clients (mirrors CognitiveTheater pattern)
        self._local_client = ollama.Client(host=ollama_url)
        self._cloud_client = None
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            self._cloud_client = ollama.Client(
                host=_OLLAMA_CLOUD_HOST,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            logger.info("[MirrorMind] Cloud client initialized")

    def _get_client(self):
        """Return cloud client for cloud models, local client otherwise.
        Returns None if cloud model requested but OLLAMA_API_KEY not set."""
        model = self.model or Config.get_model("fast")
        if model.endswith(("-cloud", ":cloud")):
            if self._cloud_client:
                return self._cloud_client
            logger.warning(
                f"[MirrorMind] Cloud model '{model}' requires OLLAMA_API_KEY — critique disabled"
            )
            return None
        return self._local_client

    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama to generate a critique or improvement."""
        client = self._get_client()
        if client is None:
            return "__NO_CLIENT__"

        try:
            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": temperature}
            )
            return response.get("response", "").strip()
        except Exception as e:
            logger.error(f"[MirrorMind] LLM call failed: {e}")
            return ""

    def _parse_critique(self, critique_text: str) -> Tuple[float, List[str], List[str]]:
        """Parse critique response into score, issues, and suggestions.

        Args:
            critique_text: Raw critique from LLM

        Returns:
            Tuple of (overall_score, issues_list, suggestions_list)
        """
        score = 0.5  # Default if parsing fails
        issues = []
        suggestions = []

        try:
            # Extract individual dimension scores
            dimension_scores = {}
            for dim in self.DIMENSIONS:
                pattern = rf"{dim.upper()}:\s*([\d.]+)"
                match = re.search(pattern, critique_text, re.IGNORECASE)
                if match:
                    dim_score = float(match.group(1))
                    dimension_scores[dim] = max(0.0, min(1.0, dim_score))

            # Calculate weighted average if we have dimension scores
            if dimension_scores:
                total_weight = sum(self.DIMENSIONS[dim] for dim in dimension_scores if dim in self.DIMENSIONS)
                if total_weight > 0:
                    weighted_sum = sum(
                        dimension_scores[dim] * self.DIMENSIONS[dim]
                        for dim in dimension_scores
                        if dim in self.DIMENSIONS
                    ) / total_weight
                else:
                    weighted_sum = 0.5  # fallback if no dimensions parsed
                score = weighted_sum
            else:
                # Fall back to OVERALL_SCORE if present
                overall_match = re.search(r"OVERALL_SCORE:\s*([\d.]+)", critique_text, re.IGNORECASE)
                if overall_match:
                    score = float(overall_match.group(1))
                    score = max(0.0, min(1.0, score))

            # Extract issues
            issues_match = re.search(r"ISSUES:\s*(.+?)(?=SUGGESTIONS:|$)", critique_text, re.IGNORECASE | re.DOTALL)
            if issues_match:
                issues_text = issues_match.group(1).strip()
                if issues_text.lower() != "none":
                    issues = [i.strip() for i in re.split(r"[,\n]", issues_text) if i.strip()]

            # Extract suggestions
            suggestions_match = re.search(r"SUGGESTIONS:\s*(.+?)$", critique_text, re.IGNORECASE | re.DOTALL)
            if suggestions_match:
                suggestions_text = suggestions_match.group(1).strip()
                if suggestions_text.lower() != "none":
                    suggestions = [s.strip() for s in re.split(r"[,\n]", suggestions_text) if s.strip()]

        except (ValueError, AttributeError) as e:
            logger.warning(f"[MirrorMind] Parse error: {e}, using defaults")

        return score, issues, suggestions

    def _critique(self, query: str, response: str) -> Tuple[float, List[str], List[str]]:
        """Evaluate a response and identify issues.

        Args:
            query: Original user query
            response: Response to evaluate

        Returns:
            Tuple of (quality_score, issues, suggestions)
        """
        prompt = self.CRITIQUE_PROMPT.format(query=query, response=response)

        try:
            critique_text = self._call_llm(prompt, temperature=0.3)
            if critique_text == "__NO_CLIENT__":
                logger.warning("[MirrorMind] No cloud client — critique skipped")
                return None, [], []
            if not critique_text:
                return 0.5, ["LLM returned empty critique"], []
            return self._parse_critique(critique_text)
        except Exception as e:
            logger.warning(f"[MirrorMind] Critique error: {e}")
            return 0.5, ["error during evaluation"], []

    def _improve(self, query: str, response: str, issues: List[str], suggestions: List[str]) -> str:
        """Generate an improved version of the response.

        Args:
            query: Original user query
            response: Current response to improve
            issues: List of identified issues
            suggestions: List of improvement suggestions

        Returns:
            Improved response text
        """
        issues_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- No major issues"
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "- General improvements"

        prompt = self.IMPROVE_PROMPT.format(
            query=query,
            response=response,
            issues=issues_text,
            suggestions=suggestions_text
        )

        try:
            improved = self._call_llm(prompt, temperature=0.7)
            if improved == "__NO_CLIENT__":
                logger.warning("[MirrorMind] No cloud client — improvement skipped")
                return response
            # Sanity check: don't return empty or very short improvements
            if len(improved) < 10:
                return response
            return improved
        except Exception as e:
            logger.warning(f"[MirrorMind] Improvement error: {e}")
            return response

    def refine(self, query: str, initial_response: str) -> CritiqueResult:
        """Main method: critique and improve a response until quality threshold met.

        Args:
            query: The original user query
            initial_response: The initial response to evaluate and improve

        Returns:
            CritiqueResult with original, improved, score, and metadata
        """
        current = initial_response
        improvements_made = []
        best_score = 0.0
        best_response = initial_response

        try:
            for iteration in range(self.max_iterations):
                # 1. Critique current response
                score, issues, suggestions = self._critique(query, current)
                if score is None:
                    return CritiqueResult(
                        original=initial_response,
                        improved=initial_response,
                        quality_score=0.0,
                        iterations=0,
                        improvements_made=[]
                    )

                # Track best version seen
                if score > best_score:
                    best_score = score
                    best_response = current

                # 2. Good enough? Return it
                if score >= self.quality_threshold:
                    return CritiqueResult(
                        original=initial_response,
                        improved=current,
                        quality_score=score,
                        iterations=iteration + 1,
                        improvements_made=improvements_made
                    )

                # 3. Not good enough? Improve it
                if issues or suggestions:
                    improvement_note = f"Iteration {iteration + 1}: Fixed {', '.join(issues[:3]) if issues else 'general quality'}"
                    improvements_made.append(improvement_note)
                    current = self._improve(query, current, issues, suggestions)

            # Max iterations reached, return best we have
            final_score, _, _ = self._critique(query, current)
            if final_score is not None and final_score > best_score:
                best_score = final_score
                best_response = current

            return CritiqueResult(
                original=initial_response,
                improved=best_response,
                quality_score=best_score,
                iterations=self.max_iterations,
                improvements_made=improvements_made
            )

        except Exception as e:
            # Catch-all: never crash, return original with error note
            logger.error(f"[MirrorMind] Unexpected error: {e}")
            return CritiqueResult(
                original=initial_response,
                improved=initial_response,
                quality_score=0.5,
                iterations=0,
                improvements_made=[f"Error: {str(e)[:50]}"]
            )

    def quick_score(self, query: str, response: str) -> float:
        """Get just the quality score without improvement.

        Args:
            query: The original query
            response: The response to score

        Returns:
            Quality score 0.0-1.0
        """
        score, _, _ = self._critique(query, response)
        return score

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a MirrorMind action.

        Args:
            action: Action to perform
            **kwargs: Additional arguments

        Returns:
            Result dictionary
        """
        action_lower = action.lower()

        if "refine" in action_lower or "improve" in action_lower:
            query = kwargs.get("query", "")
            response = kwargs.get("response", "")
            if not query or not response:
                return {"success": False, "error": "Both 'query' and 'response' required"}

            result = self.refine(query, response)
            return {
                "success": True,
                "original": result.original,
                "improved": result.improved,
                "quality_score": result.quality_score,
                "iterations": result.iterations,
                "improvements_made": result.improvements_made,
                "was_improved": result.was_improved()
            }

        elif "score" in action_lower:
            query = kwargs.get("query", "")
            response = kwargs.get("response", "")
            if not query or not response:
                return {"success": False, "error": "Both 'query' and 'response' required"}

            score = self.quick_score(query, response)
            return {"success": True, "quality_score": score}

        elif "status" in action_lower:
            return {
                "success": True,
                "model": self.model,
                "quality_threshold": self.quality_threshold,
                "max_iterations": self.max_iterations,
                "dimensions": list(self.DIMENSIONS.keys())
            }

        return {
            "success": False,
            "error": f"Unknown action: {action}",
            "available_actions": ["refine", "score", "status"]
        }


if __name__ == "__main__":
    print("=" * 60)
    print("MirrorMind - Self-Critique System Test")
    print("=" * 60)

    mirror = MirrorMind()

    # Test with a deliberately weak response
    query = "How do I center a div in CSS?"
    bad_response = "Use margin auto."

    print(f"\nQuery: {query}")
    print(f"Original response: {bad_response}")
    print("\nRefining...")

    result = mirror.refine(query, bad_response)

    print(f"\n--- Results ---")
    print(f"Improved: {result.improved}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Iterations: {result.iterations}")
    print(f"Was Improved: {result.was_improved()}")
    print(f"Improvements Made:")
    for imp in result.improvements_made:
        print(f"  - {imp}")
