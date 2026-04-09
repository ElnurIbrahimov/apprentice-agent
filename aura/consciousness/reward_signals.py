"""
Reward Signal Evaluators — LLM-based proxy metrics for strategy evaluation.

Provides self-consistency, judge, and coherence evaluators that produce
[0, 1] reward signals for the Strategy Bandit's composite reward computation.

These evaluators are optional (STRATEGY_BANDIT_EVAL_ENABLED) and run
asynchronously to avoid blocking the response path.
"""

import logging
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Dict

logger = logging.getLogger(__name__)


# ============================================================================
# Evaluator Results
# ============================================================================

@dataclass
class EvalResult:
    """Result from a single evaluator."""
    metric_name: str
    score: float            # [0, 1]
    confidence: float       # [0, 1] — how confident we are in this score
    raw_response: str = ""  # LLM's raw output for debugging


# ============================================================================
# Self-Consistency Evaluator
# ============================================================================

class SelfConsistencyEvaluator:
    """Asks the LLM if it would give the same answer to the same question.

    Score: 1.0 = YES (consistent), 0.0 = NO (inconsistent).
    Cost: 1 LLM call.
    """

    PROMPT_TEMPLATE = """You previously answered the following question:

Question: {question}

Your answer was:
{answer}

Would you give essentially the same answer if asked again? Consider:
- Are the key facts correct?
- Is the reasoning sound?
- Would you change any significant part?

Reply with exactly one word: YES or NO"""

    def evaluate(
        self,
        question: str,
        answer: str,
        llm_call: Callable[[str], str],
    ) -> EvalResult:
        """Run self-consistency check.

        Args:
            question: Original user query.
            answer: Generated response.
            llm_call: Function that sends a prompt to the LLM and returns text.

        Returns:
            EvalResult with score 1.0 (consistent) or 0.0 (inconsistent).
        """
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                question=question[:500],
                answer=answer[:1000],
            )
            raw = llm_call(prompt).strip().upper()

            if "YES" in raw:
                score = 1.0
            elif "NO" in raw:
                score = 0.0
            else:
                score = 0.5  # Ambiguous
                logger.debug(f"[SelfConsistency] Ambiguous response: {raw[:100]}")

            return EvalResult(
                metric_name="self_consistency",
                score=score,
                confidence=0.8 if score != 0.5 else 0.3,
                raw_response=raw[:200],
            )
        except Exception as e:
            logger.warning(f"[SelfConsistency] Error: {e}")
            return EvalResult(
                metric_name="self_consistency",
                score=0.5,
                confidence=0.0,
                raw_response=str(e),
            )


# ============================================================================
# Judge Evaluator
# ============================================================================

class JudgeEvaluator:
    """Binary PASS/FAIL rubric evaluation by LLM.

    Score: 1.0 = PASS, 0.0 = FAIL.
    Cost: 1 LLM call.
    """

    PROMPT_TEMPLATE = """Evaluate this response to the given question.

Question: {question}

Response: {answer}

Evaluation criteria:
1. Does the response address the question directly?
2. Is the information accurate (to the best of your knowledge)?
3. Is the response well-structured and clear?
4. Is the response complete (not cut off or missing key points)?

Based on these criteria, does this response PASS or FAIL?
Reply with exactly one word: PASS or FAIL"""

    def evaluate(
        self,
        question: str,
        answer: str,
        llm_call: Callable[[str], str],
    ) -> EvalResult:
        """Run judge evaluation.

        Args:
            question: Original user query.
            answer: Generated response.
            llm_call: Function that sends a prompt to the LLM and returns text.

        Returns:
            EvalResult with score 1.0 (PASS) or 0.0 (FAIL).
        """
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                question=question[:500],
                answer=answer[:1500],
            )
            raw = llm_call(prompt).strip().upper()

            if "PASS" in raw:
                score = 1.0
            elif "FAIL" in raw:
                score = 0.0
            else:
                score = 0.5
                logger.debug(f"[Judge] Ambiguous response: {raw[:100]}")

            return EvalResult(
                metric_name="judge_score",
                score=score,
                confidence=0.85 if score != 0.5 else 0.3,
                raw_response=raw[:200],
            )
        except Exception as e:
            logger.warning(f"[Judge] Error: {e}")
            return EvalResult(
                metric_name="judge_score",
                score=0.5,
                confidence=0.0,
                raw_response=str(e),
            )


# ============================================================================
# Stepwise Coherence Evaluator
# ============================================================================

class StepwiseCoherenceEvaluator:
    """Evaluates logical coherence of the response.

    Score: 1.0 = HIGH, 0.5 = MEDIUM, 0.0 = LOW.
    Cost: 1 LLM call.
    """

    PROMPT_TEMPLATE = """Evaluate the logical coherence of this response.

Question: {question}

Response: {answer}

Consider:
- Do the ideas flow logically from one to the next?
- Are there contradictions within the response?
- Does each step/point follow from the previous one?
- Is the overall argument or explanation coherent?

Rate the coherence as exactly one word: HIGH, MEDIUM, or LOW"""

    def evaluate(
        self,
        question: str,
        answer: str,
        llm_call: Callable[[str], str],
    ) -> EvalResult:
        """Run coherence evaluation.

        Args:
            question: Original user query.
            answer: Generated response.
            llm_call: Function that sends a prompt to the LLM and returns text.

        Returns:
            EvalResult with score 1.0/0.5/0.0 for HIGH/MEDIUM/LOW.
        """
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                question=question[:500],
                answer=answer[:1500],
            )
            raw = llm_call(prompt).strip().upper()

            if "HIGH" in raw:
                score = 1.0
            elif "MEDIUM" in raw:
                score = 0.5
            elif "LOW" in raw:
                score = 0.0
            else:
                score = 0.5
                logger.debug(f"[Coherence] Ambiguous response: {raw[:100]}")

            return EvalResult(
                metric_name="coherence_score",
                score=score,
                confidence=0.75 if raw in ("HIGH", "MEDIUM", "LOW") else 0.3,
                raw_response=raw[:200],
            )
        except Exception as e:
            logger.warning(f"[Coherence] Error: {e}")
            return EvalResult(
                metric_name="coherence_score",
                score=0.5,
                confidence=0.0,
                raw_response=str(e),
            )


# ============================================================================
# Reward Signal Collector
# ============================================================================

class RewardSignalCollector:
    """Orchestrates all evaluators and collects reward signals.

    Can run evaluators in parallel using a thread pool. Results are
    collected as a dict of metric_name → score suitable for
    CompositeRewardComputer.compute().
    """

    def __init__(self):
        self._self_consistency = SelfConsistencyEvaluator()
        self._judge = JudgeEvaluator()
        self._coherence = StepwiseCoherenceEvaluator()
        from aura.pools import llm_pool
        self._executor = llm_pool()
        # atexit cleanup now handled by aura.pools

    def collect_sync(
        self,
        question: str,
        answer: str,
        llm_call: Callable[[str], str],
    ) -> Dict[str, float]:
        """Run all evaluators synchronously and return metrics dict.

        Args:
            question: Original user query.
            answer: Generated response.
            llm_call: Function that sends a prompt to the LLM.

        Returns:
            Dict of metric_name → score, ready for CompositeRewardComputer.
        """
        metrics = {}

        for evaluator in [self._self_consistency, self._judge, self._coherence]:
            try:
                result = evaluator.evaluate(question, answer, llm_call)
                metrics[result.metric_name] = result.score
            except Exception as e:
                logger.warning(f"[RewardCollector] Evaluator error: {e}")

        return metrics

    def collect_async(
        self,
        question: str,
        answer: str,
        llm_call: Callable[[str], str],
    ) -> Future:
        """Run all evaluators asynchronously in thread pool.

        Args:
            question: Original user query.
            answer: Generated response.
            llm_call: Function that sends a prompt to the LLM.

        Returns:
            Future[Dict[str, float]] that resolves to the metrics dict.
        """
        return self._executor.submit(
            self.collect_sync, question, answer, llm_call
        )

    def shutdown(self):
        """Mark collector as inactive. Does NOT shut down the shared pool."""
        # self._executor comes from llm_pool() — a shared resource.
        # Calling shutdown() on it would break all other users.
        pass
