"""
Introspection Circuit for AURA
==============================

Enables AURA to "know when it doesn't know" through multi-signal
uncertainty detection and automatic verification triggering.

Based on research from:
- Anthropic's Introspection Research (2025)
- Semantic Entropy (Nature, 2024)
- Uncertainty Quantification Survey (KDD 2025)
- SelfCheckGPT (EMNLP 2023)

Core capabilities:
1. Query classification (factual/opinion/creative)
2. Verbalized confidence extraction
3. Multi-sample consistency checking
4. Automatic verification triggering
5. Epistemic marker insertion

References:
- https://transformer-circuits.pub/2025/introspection/index.html
- https://www.nature.com/articles/s41586-024-07421-0
- https://arxiv.org/abs/2503.15850
"""

import re
import math
import logging
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types for confidence calibration"""
    FACTUAL = "factual"          # Verifiable facts (dates, names, numbers)
    PROCEDURAL = "procedural"    # How-to, step-by-step
    ANALYTICAL = "analytical"    # Reasoning, analysis
    OPINION = "opinion"          # Subjective, preferences
    CREATIVE = "creative"        # Generation, brainstorming
    CONVERSATIONAL = "conversational"  # Chitchat, greetings
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Discrete confidence levels for decision making"""
    HIGH = "high"           # >= 0.8 - Respond directly
    MEDIUM = "medium"       # 0.5-0.8 - Respond with hedging
    LOW = "low"             # 0.3-0.5 - Verify before responding
    VERY_LOW = "very_low"   # < 0.3 - Abstain or heavy verification


class IntrospectionAction(Enum):
    """Actions based on confidence assessment"""
    RESPOND = "respond"              # Answer directly
    RESPOND_HEDGED = "respond_hedged"  # Answer with uncertainty markers
    VERIFY_THEN_RESPOND = "verify"   # Search/verify first
    ABSTAIN = "abstain"              # Decline to answer
    ASK_CLARIFICATION = "clarify"    # Request more info


@dataclass
class ConfidenceSignal:
    """A single confidence signal from one detection method"""
    source: str              # e.g., "verbalized", "consistency", "query_type"
    value: float             # 0.0 to 1.0
    weight: float = 1.0      # Importance weight
    reasoning: str = ""      # Why this score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntrospectionResult:
    """Complete result of introspection analysis"""
    query: str
    query_type: QueryType
    confidence: float                      # Combined confidence 0-1
    confidence_level: ConfidenceLevel
    action: IntrospectionAction
    signals: List[ConfidenceSignal]

    # Response modifications
    should_verify: bool = False
    verification_query: Optional[str] = None
    epistemic_markers: List[str] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "query_type": self.query_type.value,
            "confidence": round(self.confidence, 3),
            "confidence_level": self.confidence_level.value,
            "action": self.action.value,
            "should_verify": self.should_verify,
            "verification_query": self.verification_query,
            "epistemic_markers": self.epistemic_markers,
            "signals": [
                {
                    "source": s.source,
                    "value": round(s.value, 3),
                    "weight": s.weight,
                    "reasoning": s.reasoning
                }
                for s in self.signals
            ],
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class IntrospectionConfig:
    """Configuration for the Introspection Circuit"""

    def __init__(
        self,
        # Confidence thresholds
        high_confidence_threshold: float = 0.80,
        medium_confidence_threshold: float = 0.50,
        low_confidence_threshold: float = 0.30,

        # Verification triggers
        verify_factual_below: float = 0.70,
        verify_procedural_below: float = 0.60,
        abstain_below: float = 0.25,

        # Multi-sample settings
        enable_consistency_check: bool = True,
        consistency_samples: int = 3,
        consistency_temperature: float = 0.7,

        # Feature flags
        enable_epistemic_markers: bool = True,
        enable_auto_verification: bool = True,
        enable_query_classification: bool = True,

        # Integration
        use_fluxmind: bool = True,
        use_guardian: bool = True,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.verify_factual_below = verify_factual_below
        self.verify_procedural_below = verify_procedural_below
        self.abstain_below = abstain_below
        self.enable_consistency_check = enable_consistency_check
        self.consistency_samples = consistency_samples
        self.consistency_temperature = consistency_temperature
        self.enable_epistemic_markers = enable_epistemic_markers
        self.enable_auto_verification = enable_auto_verification
        self.enable_query_classification = enable_query_classification
        self.use_fluxmind = use_fluxmind
        self.use_guardian = use_guardian


class IntrospectionCircuit:
    """
    Main Introspection Circuit for AURA

    Detects uncertainty before responding and triggers appropriate actions:
    - High confidence: Respond directly
    - Medium confidence: Respond with epistemic markers
    - Low confidence: Verify via search/tools first
    - Very low confidence: Abstain or ask for clarification
    """

    # Query type detection patterns
    FACTUAL_PATTERNS = [
        r'\b(what|who|when|where|which)\s+(is|are|was|were)\b',
        r'\b(how many|how much|how old|how long|how far)\b',
        r'\b(capital of|president of|CEO of|founder of)\b',
        r'\b(born|died|founded|invented|discovered)\b',
        r'\b(in what year|on what date)\b',
        r'\b(true or false|is it true)\b',
        r'\b(definition of|meaning of|what does .* mean)\b',
    ]

    PROCEDURAL_PATTERNS = [
        r'\b(how (do|can|should|to)|steps to|guide to)\b',
        r'\b(tutorial|instructions|procedure)\b',
        r'\b(install|setup|configure|implement)\b',
        r'\b(fix|solve|resolve|debug)\b',
        r'\b(create|build|make|write)\b',
    ]

    ANALYTICAL_PATTERNS = [
        r'\b(why|explain|analyze|compare|contrast)\b',
        r'\b(pros and cons|advantages|disadvantages)\b',
        r'\b(difference between|relationship between)\b',
        r'\b(cause|effect|impact|consequence)\b',
        r'\b(evaluate|assess|critique)\b',
    ]

    OPINION_PATTERNS = [
        r'\b(best|worst|favorite|recommend)\b',
        r'\b(should i|would you|do you think)\b',
        r'\b(opinion|preference|suggestion)\b',
        r'\b(like|dislike|prefer)\b',
    ]

    CREATIVE_PATTERNS = [
        r'\b(write|compose|create|generate|imagine)\b',
        r'\b(story|poem|song|essay|script)\b',
        r'\b(brainstorm|ideas for|suggest)\b',
        r'\b(design|invent|come up with)\b',
    ]

    CONVERSATIONAL_PATTERNS = [
        r'^(hi|hello|hey|greetings|good morning|good evening)',
        r'\b(how are you|what\'s up|thanks|thank you)\b',
        r'^(bye|goodbye|see you|take care)',
        r'\b(nice to meet|pleased to)\b',
    ]

    # Epistemic markers for hedging
    HEDGE_PHRASES = {
        ConfidenceLevel.MEDIUM: [
            "I believe",
            "Based on my understanding",
            "If I recall correctly",
            "I think",
        ],
        ConfidenceLevel.LOW: [
            "I'm not entirely certain, but",
            "This might not be fully accurate, but",
            "I have limited confidence in this, but",
            "You may want to verify this, but",
        ],
        ConfidenceLevel.VERY_LOW: [
            "I'm quite uncertain about this",
            "I don't have reliable information on this",
            "I'm not confident I can answer this accurately",
        ],
    }

    # Verbalized confidence extraction patterns
    CONFIDENCE_PATTERNS = [
        # Percentage patterns
        (r'(\d{1,3})%\s*confident', lambda m: int(m.group(1)) / 100),
        (r'confidence[:\s]+(\d{1,3})%', lambda m: int(m.group(1)) / 100),
        (r'(\d{1,3})%\s*(?:sure|certain)', lambda m: int(m.group(1)) / 100),

        # Numeric patterns (0-1 or 0-10)
        (r'confidence[:\s]+(0?\.\d+)', lambda m: float(m.group(1))),
        (r'confidence[:\s]+(\d+)/10', lambda m: int(m.group(1)) / 10),

        # Verbal patterns
        (r'\b(very confident|highly confident|absolutely certain)\b', lambda m: 0.95),
        (r'\b(confident|fairly certain|pretty sure)\b', lambda m: 0.80),
        (r'\b(somewhat confident|moderately sure)\b', lambda m: 0.60),
        (r'\b(not very confident|somewhat unsure|uncertain)\b', lambda m: 0.40),
        (r'\b(not confident|very unsure|highly uncertain)\b', lambda m: 0.25),
        (r'\b(don\'t know|no idea|cannot say)\b', lambda m: 0.10),
    ]

    def __init__(
        self,
        llm_func: Callable[[str, Optional[str]], str],
        config: Optional[IntrospectionConfig] = None,
        search_func: Optional[Callable[[str], str]] = None,
        fluxmind: Optional[Any] = None,
        guardian: Optional[Any] = None,
    ):
        """
        Initialize the Introspection Circuit.

        Args:
            llm_func: Function (prompt, system) -> response
            config: Configuration options
            search_func: Optional search function for verification
            fluxmind: Optional FluxMind instance for calibration
            guardian: Optional Guardian instance for failure prediction
        """
        self.llm = llm_func
        self.config = config or IntrospectionConfig()
        self.search_func = search_func
        self.fluxmind = fluxmind
        self.guardian = guardian

        # Statistics
        self.stats = {
            "total_queries": 0,
            "verifications_triggered": 0,
            "abstentions": 0,
            "avg_confidence": 0.0,
            "query_type_counts": Counter(),
        }

        # Cache for repeated queries
        self._cache: Dict[str, IntrospectionResult] = {}

        logger.info("Introspection Circuit initialized")

    def analyze(
        self,
        query: str,
        response: Optional[str] = None,
        context: str = "",
        force_consistency_check: bool = False,
    ) -> IntrospectionResult:
        """
        Analyze a query/response for uncertainty.

        Args:
            query: The user's query
            response: Optional pre-generated response to analyze
            context: Additional context
            force_consistency_check: Force multi-sample check

        Returns:
            IntrospectionResult with confidence and recommended action
        """
        start_time = time.time()
        signals: List[ConfidenceSignal] = []

        # 1. Classify query type
        query_type = self._classify_query(query) if self.config.enable_query_classification else QueryType.UNKNOWN

        # Add query type signal
        query_type_confidence = self._get_query_type_base_confidence(query_type)
        signals.append(ConfidenceSignal(
            source="query_type",
            value=query_type_confidence,
            weight=0.3,
            reasoning=f"Query type '{query_type.value}' has base confidence {query_type_confidence:.2f}",
        ))

        # 2. Get verbalized confidence from LLM
        if response:
            verbalized = self._extract_verbalized_confidence(response)
        else:
            verbalized, response = self._get_response_with_confidence(query, context)

        if verbalized is not None:
            signals.append(ConfidenceSignal(
                source="verbalized",
                value=verbalized,
                weight=0.4,
                reasoning=f"LLM self-reported confidence: {verbalized:.2f}",
            ))

        # 3. Check consistency (optional, for factual queries)
        if (self.config.enable_consistency_check and
            (force_consistency_check or query_type == QueryType.FACTUAL)):
            consistency = self._check_consistency(query, response, context)
            if consistency is not None:
                signals.append(ConfidenceSignal(
                    source="consistency",
                    value=consistency,
                    weight=0.5,
                    reasoning=f"Multi-sample consistency: {consistency:.2f}",
                ))

        # 4. Integrate with FluxMind (if available)
        if self.config.use_fluxmind and self.fluxmind:
            fluxmind_score = self._get_fluxmind_confidence(query, response)
            if fluxmind_score is not None:
                signals.append(ConfidenceSignal(
                    source="fluxmind",
                    value=fluxmind_score,
                    weight=0.4,
                    reasoning=f"FluxMind calibrated confidence: {fluxmind_score:.2f}",
                ))

        # 5. Integrate with Guardian (if available)
        if self.config.use_guardian and self.guardian:
            guardian_score = self._get_guardian_confidence(query, response)
            if guardian_score is not None:
                signals.append(ConfidenceSignal(
                    source="guardian",
                    value=guardian_score,
                    weight=0.3,
                    reasoning=f"Guardian failure prediction: {guardian_score:.2f} (inverted)",
                ))

        # 6. Combine signals
        confidence = self._combine_signals(signals)
        confidence_level = self._get_confidence_level(confidence)

        # 7. Determine action
        action = self._determine_action(query_type, confidence, confidence_level)

        # 8. Generate verification query if needed
        should_verify = action in [IntrospectionAction.VERIFY_THEN_RESPOND]
        verification_query = self._generate_verification_query(query) if should_verify else None

        # 9. Get epistemic markers
        epistemic_markers = self._get_epistemic_markers(confidence_level) if self.config.enable_epistemic_markers else []

        processing_time = (time.time() - start_time) * 1000

        result = IntrospectionResult(
            query=query,
            query_type=query_type,
            confidence=confidence,
            confidence_level=confidence_level,
            action=action,
            signals=signals,
            should_verify=should_verify,
            verification_query=verification_query,
            epistemic_markers=epistemic_markers,
            processing_time_ms=processing_time,
        )

        # Update stats
        self._update_stats(result)

        logger.info(f"Introspection: {query_type.value} query, confidence={confidence:.2f}, action={action.value}")

        return result

    def pre_response_check(
        self,
        query: str,
        context: str = "",
    ) -> Tuple[IntrospectionResult, Optional[str]]:
        """
        Quick pre-check before generating a response.
        Returns introspection result and optional verification info.

        Use this to decide whether to verify before responding.
        """
        result = self.analyze(query, context=context)

        verification_info = None
        if result.should_verify and self.search_func and self.config.enable_auto_verification:
            try:
                verification_info = self.search_func(result.verification_query or query)
                logger.info(f"Auto-verification triggered for: {query[:50]}...")
            except Exception as e:
                logger.error(f"Verification failed: {e}")

        return result, verification_info

    def wrap_response(
        self,
        response: str,
        introspection: IntrospectionResult,
    ) -> str:
        """
        Wrap a response with appropriate epistemic markers based on confidence.
        """
        if not self.config.enable_epistemic_markers:
            return response

        if introspection.confidence_level == ConfidenceLevel.HIGH:
            return response

        markers = introspection.epistemic_markers
        if not markers:
            return response

        # Add a hedge phrase at the beginning
        import random
        hedge = random.choice(markers)

        # Don't double-hedge if response already starts with uncertainty
        uncertainty_starters = ["i'm not", "i don't", "i think", "i believe", "it seems", "perhaps", "maybe"]
        if any(response.lower().startswith(s) for s in uncertainty_starters):
            return response

        return f"{hedge}, {response[0].lower()}{response[1:]}"

    def _classify_query(self, query: str) -> QueryType:
        """Classify the query type based on patterns"""
        query_lower = query.lower().strip()

        # Check each pattern type
        pattern_checks = [
            (self.CONVERSATIONAL_PATTERNS, QueryType.CONVERSATIONAL),
            (self.FACTUAL_PATTERNS, QueryType.FACTUAL),
            (self.PROCEDURAL_PATTERNS, QueryType.PROCEDURAL),
            (self.ANALYTICAL_PATTERNS, QueryType.ANALYTICAL),
            (self.CREATIVE_PATTERNS, QueryType.CREATIVE),
            (self.OPINION_PATTERNS, QueryType.OPINION),
        ]

        for patterns, qtype in pattern_checks:
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return qtype

        return QueryType.UNKNOWN

    def _get_query_type_base_confidence(self, query_type: QueryType) -> float:
        """Get base confidence for a query type"""
        # Different query types have different inherent uncertainty
        base_confidence = {
            QueryType.CONVERSATIONAL: 0.95,  # Chitchat - high confidence
            QueryType.CREATIVE: 0.85,        # Creative - no "wrong" answer
            QueryType.OPINION: 0.80,         # Opinion - subjective
            QueryType.ANALYTICAL: 0.65,      # Analysis - moderate
            QueryType.PROCEDURAL: 0.60,      # How-to - can be wrong
            QueryType.FACTUAL: 0.50,         # Facts - need verification
            QueryType.UNKNOWN: 0.60,         # Unknown - moderate default
        }
        return base_confidence.get(query_type, 0.60)

    def _get_response_with_confidence(
        self,
        query: str,
        context: str = "",
    ) -> Tuple[Optional[float], str]:
        """Generate response with verbalized confidence"""

        prompt = f"""Answer the following question. After your answer, on a new line, state your confidence level as a percentage (0-100%).

If you are unsure or lack knowledge about the topic, it's better to say "I don't know" or express low confidence rather than guess.

{f"Context: {context}" if context else ""}

Question: {query}

Answer:"""

        system = """You are a helpful assistant that accurately assesses its own confidence.
When answering:
- If you're certain, say so (e.g., "Confidence: 95%")
- If you're unsure, express it honestly (e.g., "Confidence: 40%")
- If you don't know, admit it (e.g., "I don't have reliable information on this. Confidence: 10%")
Never pretend to know something you don't."""

        try:
            response = self.llm(prompt, system)
            confidence = self._extract_verbalized_confidence(response)
            return confidence, response
        except Exception as e:
            logger.error(f"Error getting response with confidence: {e}")
            return None, ""

    def _extract_verbalized_confidence(self, response: str) -> Optional[float]:
        """Extract confidence score from response text"""
        if not response:
            return None

        response_lower = response.lower()

        for pattern, extractor in self.CONFIDENCE_PATTERNS:
            match = re.search(pattern, response_lower)
            if match:
                try:
                    confidence = extractor(match)
                    return max(0.0, min(1.0, confidence))
                except (ValueError, TypeError, AttributeError):
                    continue

        return None

    def _check_consistency(
        self,
        query: str,
        original_response: str,
        context: str = "",
    ) -> Optional[float]:
        """Check consistency across multiple samples"""
        if not original_response:
            return None

        try:
            # Generate additional samples
            samples = [original_response]

            prompt = f"{context}\n\nQuestion: {query}" if context else f"Question: {query}"

            for _ in range(self.config.consistency_samples - 1):
                response = self.llm(prompt, None)
                samples.append(response)

            # Calculate consistency (simplified - semantic similarity would be better)
            consistency = self._calculate_sample_consistency(samples)
            return consistency

        except Exception as e:
            logger.error(f"Error in consistency check: {e}")
            return None

    def _calculate_sample_consistency(self, samples: List[str]) -> float:
        """Calculate consistency score from multiple samples"""
        if len(samples) < 2:
            return 1.0

        # Simple approach: compare key terms overlap
        # In production, use semantic similarity or NLI

        def extract_key_terms(text: str) -> set:
            # Extract words, numbers, and key phrases
            words = re.findall(r'\b\w+\b', text.lower())
            # Filter out common words
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                        'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                        'she', 'we', 'they', 'and', 'or', 'but', 'if', 'then'}
            return set(w for w in words if w not in stopwords and len(w) > 2)

        term_sets = [extract_key_terms(s) for s in samples]

        if not any(term_sets):
            return 0.5

        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(term_sets)):
            for j in range(i + 1, len(term_sets)):
                if term_sets[i] or term_sets[j]:
                    intersection = len(term_sets[i] & term_sets[j])
                    union = len(term_sets[i] | term_sets[j])
                    if union > 0:
                        similarities.append(intersection / union)

        if not similarities:
            return 0.5

        return sum(similarities) / len(similarities)

    def _get_fluxmind_confidence(self, query: str, response: str) -> Optional[float]:
        """Get confidence from FluxMind if available"""
        if not self.fluxmind:
            return None

        try:
            # FluxMind should have a calibrate method
            if hasattr(self.fluxmind, 'calibrate'):
                result = self.fluxmind.calibrate(query, response)
                if isinstance(result, dict):
                    return result.get('confidence', result.get('score'))
                return float(result)
        except Exception as e:
            logger.debug(f"FluxMind integration error: {e}")

        return None

    def _get_guardian_confidence(self, query: str, response: str) -> Optional[float]:
        """Get confidence from Guardian (inverted failure probability)"""
        if not self.guardian:
            return None

        try:
            # Guardian predicts failure, so we invert
            if hasattr(self.guardian, 'predict_failure'):
                failure_prob = self.guardian.predict_failure(query, response)
                return 1.0 - failure_prob
            elif hasattr(self.guardian, 'assess'):
                result = self.guardian.assess(query)
                if isinstance(result, dict):
                    risk = result.get('risk_score', result.get('failure_probability', 0.5))
                    return 1.0 - risk
        except Exception as e:
            logger.debug(f"Guardian integration error: {e}")

        return None

    def _combine_signals(self, signals: List[ConfidenceSignal]) -> float:
        """Combine multiple confidence signals into a single score"""
        if not signals:
            return 0.5

        # Weighted average
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return 0.5

        weighted_sum = sum(s.value * s.weight for s in signals)
        combined = weighted_sum / total_weight

        return max(0.0, min(1.0, combined))

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to discrete level"""
        if confidence >= self.config.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.config.medium_confidence_threshold:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.config.low_confidence_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _determine_action(
        self,
        query_type: QueryType,
        confidence: float,
        confidence_level: ConfidenceLevel,
    ) -> IntrospectionAction:
        """Determine the appropriate action based on query type and confidence"""

        # Very low confidence -> abstain or clarify
        if confidence < self.config.abstain_below:
            return IntrospectionAction.ABSTAIN

        # Factual queries have stricter thresholds
        if query_type == QueryType.FACTUAL:
            if confidence < self.config.verify_factual_below:
                return IntrospectionAction.VERIFY_THEN_RESPOND

        # Procedural queries also need accuracy
        if query_type == QueryType.PROCEDURAL:
            if confidence < self.config.verify_procedural_below:
                return IntrospectionAction.VERIFY_THEN_RESPOND

        # General confidence-based actions
        if confidence_level == ConfidenceLevel.HIGH:
            return IntrospectionAction.RESPOND
        elif confidence_level == ConfidenceLevel.MEDIUM:
            return IntrospectionAction.RESPOND_HEDGED
        elif confidence_level == ConfidenceLevel.LOW:
            if query_type in [QueryType.FACTUAL, QueryType.PROCEDURAL]:
                return IntrospectionAction.VERIFY_THEN_RESPOND
            return IntrospectionAction.RESPOND_HEDGED
        else:
            return IntrospectionAction.ABSTAIN

    def _generate_verification_query(self, query: str) -> str:
        """Generate a search query for verification"""
        # Simple approach: clean up the query for search
        # Remove question words at start
        search_query = re.sub(r'^(what|who|when|where|why|how|is|are|was|were|do|does|did|can|could|would|should)\s+', '', query.lower())
        # Remove question marks
        search_query = search_query.rstrip('?').strip()
        return search_query

    def _get_epistemic_markers(self, confidence_level: ConfidenceLevel) -> List[str]:
        """Get appropriate epistemic markers for the confidence level"""
        return self.HEDGE_PHRASES.get(confidence_level, [])

    def _update_stats(self, result: IntrospectionResult):
        """Update internal statistics"""
        self.stats["total_queries"] += 1
        self.stats["query_type_counts"][result.query_type.value] += 1

        if result.should_verify:
            self.stats["verifications_triggered"] += 1

        if result.action == IntrospectionAction.ABSTAIN:
            self.stats["abstentions"] += 1

        # Rolling average confidence
        n = self.stats["total_queries"]
        old_avg = self.stats["avg_confidence"]
        self.stats["avg_confidence"] = old_avg + (result.confidence - old_avg) / n

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit statistics"""
        return {
            **self.stats,
            "query_type_counts": dict(self.stats["query_type_counts"]),
            "verification_rate": (
                self.stats["verifications_triggered"] / max(1, self.stats["total_queries"])
            ),
            "abstention_rate": (
                self.stats["abstentions"] / max(1, self.stats["total_queries"])
            ),
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_queries": 0,
            "verifications_triggered": 0,
            "abstentions": 0,
            "avg_confidence": 0.0,
            "query_type_counts": Counter(),
        }


# Convenience functions
def create_introspection_circuit(
    llm_func: Callable[[str, Optional[str]], str],
    **config_kwargs
) -> IntrospectionCircuit:
    """Create an Introspection Circuit with custom config"""
    config = IntrospectionConfig(**config_kwargs)
    return IntrospectionCircuit(llm_func=llm_func, config=config)


def quick_confidence_check(
    query: str,
    llm_func: Callable[[str, Optional[str]], str],
) -> float:
    """Quick confidence check without full analysis"""
    circuit = IntrospectionCircuit(llm_func=llm_func)
    result = circuit.analyze(query)
    return result.confidence
