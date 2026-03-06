"""CognitiveTheater - Multi-Perspective Reasoning for AURA.

Tool #22: Analyzes questions from multiple perspectives in a single LLM call,
then synthesizes a balanced recommendation.

Flow:
    User Question -> Single LLM Call -> 4 Perspectives -> Synthesis -> Balanced Answer
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import ollama
import requests

from ..config import Config


logger = logging.getLogger(__name__)


@dataclass
class Deliberation:
    """Result of multi-perspective deliberation."""
    question: str
    perspectives: Dict[str, str] = field(default_factory=dict)
    synthesis: str = ""
    confidence: float = 0.65
    raw_response: str = ""

    def has_all_perspectives(self) -> bool:
        """Check if all 4 perspectives were generated."""
        required = {"advocate", "critic", "analyst", "integrator"}
        return required.issubset(self.perspectives.keys())

    def get_summary(self) -> str:
        """Get a brief summary of the deliberation."""
        if self.synthesis:
            return self.synthesis[:200] + "..." if len(self.synthesis) > 200 else self.synthesis
        return "No synthesis available"


class CognitiveTheater:
    """Multi-perspective reasoning through simulated deliberation."""

    PROMPT_TEMPLATE = """You are engaging in multi-perspective deliberation.

QUESTION: {question}
{context_section}
Analyze from these 4 perspectives (2-3 sentences each):

🔵 ADVOCATE: Argue enthusiastically IN FAVOR. What are the benefits? Why is this a good idea?

🔴 CRITIC: Argue AGAINST with skepticism. What are the risks? What could go wrong?

🟡 ANALYST: Neutral, data-driven analysis. What do the facts and evidence say?

🟢 INTEGRATOR: Synthesize all views into a balanced, actionable recommendation.

After all perspectives, add:
📊 CONFIDENCE: [High/Medium/Low]

Begin:"""

    # Patterns for parsing perspectives (handles with/without emojis, **bold** markdown)
    PERSPECTIVE_PATTERNS = {
        "advocate": r"\*{0,2}(?:🔵\s*)?ADVOCATE:?\*{0,2}:?\s*(.*?)(?=\*{0,2}(?:🔴\s*)?CRITIC|\*{0,2}(?:🔴)|$)",
        "critic": r"\*{0,2}(?:🔴\s*)?CRITIC:?\*{0,2}:?\s*(.*?)(?=\*{0,2}(?:🟡\s*)?ANALYST|\*{0,2}(?:🟡)|$)",
        "analyst": r"\*{0,2}(?:🟡\s*)?ANALYST:?\*{0,2}:?\s*(.*?)(?=\*{0,2}(?:🟢\s*)?INTEGRATOR|\*{0,2}(?:🟢)|$)",
        "integrator": r"\*{0,2}(?:🟢\s*)?INTEGRATOR:?\*{0,2}:?\s*(.*?)(?=\*{0,2}(?:📊\s*)?CONFIDENCE|📊|$)",
    }

    CONFIDENCE_MAP = {
        "high": 0.85,
        "medium": 0.65,
        "low": 0.40,
    }

    OLLAMA_CLOUD_HOST = "https://ollama.com"

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = None
    ):
        """Initialize CognitiveTheater.

        Args:
            ollama_url: Ollama API base URL
            model: Model to use for deliberation (default: Config.MODEL_FAST)
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model or Config.MODEL_FAST
        self.name = "cognitive_theater"

        # Initialize ollama clients
        self._local_client = ollama.Client(host=ollama_url)
        self._cloud_client = None

        # Set up cloud client if API key is available
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            self._cloud_client = ollama.Client(
                host=self.OLLAMA_CLOUD_HOST,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            logger.info("[CognitiveTheater] Cloud client initialized")

    def set_model(self, model: str) -> None:
        """Update the model used for deliberation.

        Args:
            model: Model name to use
        """
        self.model = model
        self.description = "Multi-perspective reasoning and deliberation"

    def _get_client(self) -> Optional[ollama.Client]:
        """Return cloud client for cloud models, local client otherwise.
        Returns None if cloud model requested but OLLAMA_API_KEY not set."""
        model = self.model or Config.get_model("fast")
        if model.endswith(("-cloud", ":cloud")):
            if self._cloud_client:
                return self._cloud_client
            logger.warning(
                f"[CognitiveTheater] Cloud model '{model}' requires OLLAMA_API_KEY — deliberation skipped"
            )
            return None
        return self._local_client

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama API to generate response.

        Args:
            prompt: The prompt to send

        Returns:
            Generated response text, or empty string if no client available

        Raises:
            Exception: On API errors
        """
        client = self._get_client()
        if client is None:
            return "__NO_CLIENT__"

        response = client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.7,
                "num_predict": 1000
            }
        )
        return response.get("response", "").strip()

    def _build_prompt(self, question: str, context: str = "") -> str:
        """Build the multi-perspective deliberation prompt.

        Args:
            question: The question to deliberate
            context: Optional additional context

        Returns:
            Formatted prompt string
        """
        context_section = ""
        if context:
            context_section = f"\nCONTEXT: {context}\n"

        return self.PROMPT_TEMPLATE.format(
            question=question,
            context_section=context_section
        )

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM response to extract perspectives.

        Args:
            raw: Raw LLM response text

        Returns:
            Dict with perspectives, synthesis, and confidence
        """
        perspectives = {}

        # Extract each perspective using regex
        for name, pattern in self.PERSPECTIVE_PATTERNS.items():
            match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Clean up any trailing emoji or markers
                text = re.sub(r"^[:\s]+", "", text)
                perspectives[name] = text

        # Parse confidence level (handles with/without emoji)
        conf_match = re.search(r"(?:📊\s*)?CONFIDENCE:?\s*\[?(\w+)\]?", raw, re.IGNORECASE)
        confidence = 0.65  # Default medium
        if conf_match:
            level = conf_match.group(1).lower()
            confidence = self.CONFIDENCE_MAP.get(level, 0.65)

        return {
            "perspectives": perspectives,
            "synthesis": perspectives.get("integrator", ""),
            "confidence": confidence
        }

    def deliberate(self, question: str, context: str = "") -> Deliberation:
        """Main method: generate multi-perspective analysis.

        Args:
            question: The question or decision to analyze
            context: Optional additional context

        Returns:
            Deliberation object with all perspectives and synthesis
        """
        try:
            # Build and send prompt
            prompt = self._build_prompt(question, context)
            raw = self._call_llm(prompt)
            if raw == "__NO_CLIENT__":
                logger.warning("[CognitiveTheater] No cloud client — deliberation skipped")
                return None
            raw_response = raw

            # Parse the response
            parsed = self._parse_response(raw_response)

            return Deliberation(
                question=question,
                perspectives=parsed["perspectives"],
                synthesis=parsed["synthesis"],
                confidence=parsed["confidence"],
                raw_response=raw_response
            )

        except ollama.ResponseError as e:
            logger.error(f"[CognitiveTheater] Ollama error: {e}")
            return Deliberation(
                question=question,
                synthesis=f"Model error: {str(e)[:100]}",
                confidence=0.0
            )

        except (requests.Timeout, TimeoutError) as e:
            logger.warning(f"[CognitiveTheater] Request timed out: {e}")
            return Deliberation(
                question=question,
                synthesis="I need more time to think about this complex question.",
                confidence=0.3
            )

        except (requests.RequestException, ConnectionError) as e:
            logger.error(f"[CognitiveTheater] Connection error: {e}")
            return Deliberation(
                question=question,
                synthesis="Unable to connect to AI service. Please check if Ollama is running.",
                confidence=0.0
            )

        except Exception as e:
            logger.error(f"[CognitiveTheater] Unexpected error: {e}")
            return Deliberation(
                question=question,
                synthesis=f"Error during deliberation: {str(e)[:100]}",
                confidence=0.0
            )

    def quick_debate(self, question: str, context: str = "") -> str:
        """Returns formatted string for direct chat output.

        Args:
            question: The question to deliberate
            context: Optional additional context

        Returns:
            Formatted multi-perspective analysis string
        """
        result = self.deliberate(question, context)

        if result is None:
            return "Multi-perspective analysis unavailable (no cloud client configured)."

        # If we got a raw response but no parsed perspectives, just return the raw response
        # This handles cases where the model doesn't follow the expected format
        if result.raw_response and not result.perspectives:
            return result.raw_response

        output_parts = ["**Multi-Perspective Analysis**\n"]

        has_content = False

        # Add each perspective if available
        if result.perspectives.get("advocate"):
            advocate = result.perspectives["advocate"]
            if len(advocate) > 500:
                advocate = advocate[:500] + "..."
            output_parts.append(f"**Pro:** {advocate}\n")
            has_content = True

        if result.perspectives.get("critic"):
            critic = result.perspectives["critic"]
            if len(critic) > 500:
                critic = critic[:500] + "..."
            output_parts.append(f"**Con:** {critic}\n")
            has_content = True

        if result.perspectives.get("analyst"):
            analyst = result.perspectives["analyst"]
            if len(analyst) > 500:
                analyst = analyst[:500] + "..."
            output_parts.append(f"**Analysis:** {analyst}\n")
            has_content = True

        # If no perspectives were parsed but we have raw response, show it
        if not has_content and result.raw_response:
            return result.raw_response

        # Always show synthesis/recommendation
        if result.synthesis:
            output_parts.append(f"**Recommendation:** {result.synthesis}\n")
        elif result.perspectives:
            output_parts.append("**Recommendation:** Consider the perspectives above.\n")

        # Add confidence
        output_parts.append(f"Confidence: {result.confidence:.0%}")

        return "\n".join(output_parts)

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a CognitiveTheater action.

        Args:
            action: Action to perform
            **kwargs: Additional arguments

        Returns:
            Result dictionary
        """
        action_lower = action.lower()

        if "deliberate" in action_lower or "analyze" in action_lower:
            question = kwargs.get("question", action)
            context = kwargs.get("context", "")
            result = self.deliberate(question, context)
            return {
                "success": True,
                "question": result.question,
                "perspectives": result.perspectives,
                "synthesis": result.synthesis,
                "confidence": result.confidence,
                "has_all_perspectives": result.has_all_perspectives()
            }

        elif "debate" in action_lower or "compare" in action_lower:
            question = kwargs.get("question", action)
            context = kwargs.get("context", "")
            output = self.quick_debate(question, context)
            return {
                "success": True,
                "output": output
            }

        elif "status" in action_lower:
            return {
                "success": True,
                "model": self.model,
                "perspectives": ["advocate", "critic", "analyst", "integrator"]
            }

        return {
            "success": False,
            "error": f"Unknown action: {action}",
            "available_actions": ["deliberate", "debate", "status"]
        }


# Decision detection keywords for agent integration
DECISION_KEYWORDS = [
    "should i",
    "should we",
    "compare",
    "pros and cons",
    "pros cons",
    "decide",
    "decision",
    " vs ",
    "better to",
    "worth it",
    "is it worth",
    "worth learning",
    "worth using",
    "good idea",
    "bad idea",
    "recommend",
    "trade-off",
    "tradeoff",
    "advantages",
    "disadvantages",
]


def is_decision_question(query: str) -> bool:
    """Check if a query is a decision-type question.

    Args:
        query: User query string

    Returns:
        True if this appears to be a decision question
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in DECISION_KEYWORDS)


if __name__ == "__main__":
    print("=" * 60)
    print("CognitiveTheater - Multi-Perspective Reasoning Test")
    print("=" * 60)

    theater = CognitiveTheater()

    questions = [
        "Should I use Python or Rust for my CLI tool?",
        "Should I quit my job to start a startup?",
        "Is it better to rent or buy a house?",
    ]

    for q in questions:
        print(f"\n{'=' * 50}")
        print(f"Q: {q}\n")
        print(theater.quick_debate(q))
        print()
