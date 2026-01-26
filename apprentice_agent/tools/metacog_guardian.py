"""
Metacognitive Guardian: Failure prediction and proactive intervention.

Research basis:
- Agentic Metacognition paper: 75.78% → 83.56% success rate with meta-layer
- Key insight: Agents that predict their own failures perform better

Integration points:
- Inner Monologue: Monitor "uncertain" thoughts
- EvoEmo: Detect emotional mismatches
- Knowledge Graph: Store failure patterns
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import json
import os


class FailureType(Enum):
    """Types of failures the guardian monitors"""
    KNOWLEDGE_GAP = "knowledge_gap"           # Don't know enough about topic
    TOOL_MISMATCH = "tool_mismatch"           # Wrong tool selected for task
    AMBIGUOUS_REQUEST = "ambiguous_request"   # Unclear what user wants
    CONFIDENCE_DROP = "confidence_drop"       # Sudden uncertainty spike
    EMOTIONAL_MISMATCH = "emotional_mismatch" # Misreading user's emotional state
    SKILL_BOUNDARY = "skill_boundary"         # Task beyond current capabilities
    HALLUCINATION_RISK = "hallucination_risk" # High risk of making things up
    CONTEXT_OVERFLOW = "context_overflow"     # Too much context, losing track
    LOOP_DETECTED = "loop_detected"           # Stuck in repetitive pattern


class InterventionType(Enum):
    """Types of interventions the guardian can trigger"""
    CONFIDENCE_WARNING = "confidence_warning"      # Warn user about uncertainty
    REQUEST_CLARIFICATION = "request_clarification" # Ask user for more info
    SUGGEST_ALTERNATIVE = "suggest_alternative"     # Offer different approach
    HUMAN_HANDOFF = "human_handoff"                # Explicitly ask for help
    TOOL_SWITCH = "tool_switch"                    # Recommend different tool
    PAUSE_AND_EXPLAIN = "pause_and_explain"        # Stop and explain the issue
    EMOTIONAL_ADJUSTMENT = "emotional_adjustment"  # Adjust response tone
    ABORT_TASK = "abort_task"                      # Stop task entirely


@dataclass
class FailurePrediction:
    """A predicted failure with diagnosis and recommendation"""
    id: str
    timestamp: datetime
    failure_type: FailureType
    probability: float              # 0.0 to 1.0
    severity: float                 # 0.0 to 1.0 (how bad if it happens)

    # Diagnosis
    indicators: Dict[str, float]    # What signals triggered this
    reasoning: str                  # Human-readable explanation

    # Context snapshot
    current_task: str
    current_tool: Optional[str]

    # Recommendation
    recommended_intervention: InterventionType
    intervention_message: str       # Message to show user
    alternative_approaches: List[str]


@dataclass
class GuardianConfig:
    """Configuration for the guardian"""
    warning_threshold: float = 0.3      # Probability to show warning
    intervention_threshold: float = 0.6  # Probability to intervene
    abort_threshold: float = 0.9         # Probability to abort task
    monitoring_level: str = "medium"     # low, medium, high, critical
    learn_from_outcomes: bool = True


class MetacognitiveGuardian:
    """
    Meta-layer that monitors Aura's reasoning and predicts failures.

    Usage in agent.py:
        guardian = MetacognitiveGuardian(inner_monologue, evoemo, knowledge_graph)

        # Before processing
        prediction = guardian.assess_risk(user_message, selected_tool, context)
        if prediction and prediction.probability > guardian.config.intervention_threshold:
            return guardian.execute_intervention(prediction)

        # After response
        guardian.record_outcome(prediction_id, was_successful, user_feedback)
    """

    def __init__(self,
                 inner_monologue=None,
                 evoemo=None,
                 knowledge_graph=None,
                 config: GuardianConfig = None,
                 data_dir: str = None):

        self.monologue = inner_monologue
        self.evoemo = evoemo
        self.kg = knowledge_graph
        self.config = config or GuardianConfig()

        # Default data directory
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, "data", "metacog_guardian")
        self.data_dir = data_dir

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "patterns"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "outcomes"), exist_ok=True)

        # Load failure patterns from past experiences
        self.failure_patterns = self._load_failure_patterns()

        # Session tracking
        self.session_predictions: List[FailurePrediction] = []
        self.intervention_count = 0

        # Track recent tasks to detect loops
        self._recent_tasks: List[str] = []
        self._max_recent_tasks = 10

    # ==================== MAIN API ====================

    def assess_risk(self,
                   task: str,
                   tool: Optional[str] = None,
                   context: Dict[str, Any] = None) -> Optional[FailurePrediction]:
        """
        Assess risk of failure for current task/tool combination.

        Args:
            task: The user's request or current task description
            tool: The tool being considered (if any)
            context: Additional context (conversation history, etc.)

        Returns:
            FailurePrediction if risk exceeds warning threshold, else None
        """
        context = context or {}

        # Track for loop detection
        self._recent_tasks.append(task.lower().strip())
        if len(self._recent_tasks) > self._max_recent_tasks:
            self._recent_tasks.pop(0)

        # Gather all indicators
        indicators = self._gather_indicators(task, tool, context)

        # Run prediction logic
        prediction = self._predict_failure(task, tool, indicators, context)

        if prediction and prediction.probability >= self.config.warning_threshold:
            self.session_predictions.append(prediction)

            # Log to Inner Monologue if available
            if self.monologue and hasattr(self.monologue, 'think'):
                try:
                    self.monologue.think(
                        thought_type="uncertain",
                        content=f"Guardian Alert: {prediction.failure_type.value} "
                               f"({prediction.probability:.0%} probability). "
                               f"{prediction.reasoning}"
                    )
                except Exception:
                    pass

            return prediction

        return None

    def execute_intervention(self, prediction: FailurePrediction) -> Dict[str, Any]:
        """
        Execute the recommended intervention.

        Returns:
            Dict with:
            - message: str to show user
            - should_abort: bool
            - adjustments: list of internal adjustments to make
        """
        self.intervention_count += 1

        # Log decision
        if self.monologue and hasattr(self.monologue, 'think'):
            try:
                self.monologue.think(
                    thought_type="decide",
                    content=f"Executing {prediction.recommended_intervention.value} intervention"
                )
            except Exception:
                pass

        result = {
            "message": prediction.intervention_message,
            "should_abort": prediction.recommended_intervention == InterventionType.ABORT_TASK,
            "adjustments": [],
            "prediction_id": prediction.id
        }

        # Add specific adjustments based on intervention type
        if prediction.recommended_intervention == InterventionType.TOOL_SWITCH:
            if prediction.alternative_approaches:
                result["adjustments"].append({
                    "type": "switch_tool",
                    "value": prediction.alternative_approaches[0]
                })

        elif prediction.recommended_intervention == InterventionType.EMOTIONAL_ADJUSTMENT:
            result["adjustments"].append({
                "type": "adjust_tone",
                "value": "more_supportive"
            })

        return result

    def record_outcome(self,
                      prediction_id: str,
                      was_successful: bool,
                      user_feedback: Optional[str] = None):
        """
        Record the actual outcome to improve future predictions.

        Call this after getting user feedback (thumbs up/down, explicit feedback, etc.)
        """
        # Find the prediction
        prediction = next(
            (p for p in self.session_predictions if p.id == prediction_id),
            None
        )

        if not prediction:
            return

        outcome = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "failure_type": prediction.failure_type.value,
            "predicted_probability": prediction.probability,
            "intervention_type": prediction.recommended_intervention.value,
            "actual_success": was_successful,
            "prediction_correct": not was_successful,  # We predicted failure
            "user_feedback": user_feedback,
            "task": prediction.current_task,
            "tool": prediction.current_tool,
            "indicators": prediction.indicators
        }

        # Save outcome
        self._save_outcome(outcome)

        # If we were wrong or right, learn from it
        if self.config.learn_from_outcomes:
            self._learn_from_outcome(prediction, outcome)

        # Log reflection
        if self.monologue and hasattr(self.monologue, 'think'):
            try:
                self.monologue.think(
                    thought_type="reflect",
                    content=f"Outcome recorded: prediction was "
                           f"{'correct' if outcome['prediction_correct'] else 'incorrect'}. "
                           f"Task {'failed' if not was_successful else 'succeeded'}."
                )
            except Exception:
                pass

    # ==================== INDICATOR GATHERING ====================

    def _gather_indicators(self,
                          task: str,
                          tool: Optional[str],
                          context: Dict[str, Any]) -> Dict[str, float]:
        """Gather all signals that might indicate failure risk."""
        indicators = {}

        # 1. Inner Monologue signals
        if self.monologue:
            try:
                recent_thoughts = []
                if hasattr(self.monologue, 'get_recent'):
                    recent_thoughts = self.monologue.get_recent(n=10)
                elif hasattr(self.monologue, 'thoughts'):
                    recent_thoughts = list(self.monologue.thoughts)[-10:]

                if recent_thoughts:
                    uncertain_count = sum(1 for t in recent_thoughts
                                         if (t.get("type") == "uncertain" if isinstance(t, dict)
                                             else getattr(t, 'thought_type', '') == "uncertain"))
                    indicators["uncertainty_ratio"] = uncertain_count / len(recent_thoughts)
                else:
                    indicators["uncertainty_ratio"] = 0.0
            except Exception:
                indicators["uncertainty_ratio"] = 0.0
        else:
            indicators["uncertainty_ratio"] = 0.0

        # 2. EvoEmo emotional signals
        if self.evoemo:
            try:
                state = {}
                if hasattr(self.evoemo, 'get_current_state'):
                    state = self.evoemo.get_current_state()
                elif hasattr(self.evoemo, 'current_state'):
                    state = self.evoemo.current_state
                elif hasattr(self.evoemo, 'emotions'):
                    state = self.evoemo.emotions

                indicators["user_frustration"] = state.get("frustrated", 0.0)
                indicators["user_stress"] = state.get("stressed", 0.0)
                indicators["user_confusion"] = state.get("curious", 0.0) * 0.5
            except Exception:
                indicators["user_frustration"] = 0.0
                indicators["user_stress"] = 0.0
                indicators["user_confusion"] = 0.0
        else:
            indicators["user_frustration"] = 0.0
            indicators["user_stress"] = 0.0
            indicators["user_confusion"] = 0.0

        # 3. Task analysis
        indicators["task_ambiguity"] = self._assess_ambiguity(task)
        indicators["task_complexity"] = self._assess_complexity(task)

        # 4. Tool analysis
        if tool:
            indicators["tool_match_score"] = self._assess_tool_match(task, tool)
            indicators["tool_historical_success"] = self._get_tool_success_rate(tool)
        else:
            indicators["tool_match_score"] = 1.0
            indicators["tool_historical_success"] = 1.0

        # 5. Knowledge coverage
        indicators["knowledge_coverage"] = self._assess_knowledge_coverage(task)

        # 6. Context from caller
        indicators["confidence"] = context.get("confidence", 0.7)
        max_context = context.get("max_context", 4096)
        context_length = context.get("context_length", 0)
        indicators["context_length"] = context_length / max_context if max_context > 0 else 0

        # 7. Pattern matching against past failures
        similar_failures = self._find_similar_failures(task, tool)
        indicators["similar_failure_count"] = min(1.0, len(similar_failures) / 5)

        # 8. Loop detection
        indicators["loop_risk"] = self._detect_loop_risk(task)

        return indicators

    def _assess_ambiguity(self, task: str) -> float:
        """Assess how ambiguous the task is (0 = clear, 1 = very ambiguous)."""
        task_lower = task.lower()

        ambiguity_signals = [
            # Vague pronouns without context
            task_lower.count(" it ") + task_lower.count(" this ") + task_lower.count(" that "),
            # Questions without specifics
            1 if task.strip().endswith("?") and len(task.split()) < 5 else 0,
            # Very short requests
            1 if len(task.split()) < 3 else 0,
            # Words indicating uncertainty
            sum(1 for word in ["something", "maybe", "probably", "might", "whatever", "stuff", "things"]
                if word in task_lower),
        ]

        score = min(1.0, sum(ambiguity_signals) / 5)
        return score

    def _assess_complexity(self, task: str) -> float:
        """Assess task complexity (0 = simple, 1 = very complex)."""
        task_lower = task.lower()

        complexity_signals = [
            # Multiple steps indicated
            sum(1 for word in ["then", "after", "next", "also", "and then", "finally"]
                if word in task_lower) / 3,
            # Technical terms
            sum(1 for word in ["api", "database", "algorithm", "function", "class", "async",
                              "deploy", "configure", "integrate", "optimize"]
                if word in task_lower) / 5,
            # Length as proxy
            min(1.0, len(task.split()) / 50),
            # Multiple questions
            task.count("?") / 3,
        ]

        return min(1.0, sum(complexity_signals) / len(complexity_signals))

    def _assess_tool_match(self, task: str, tool: str) -> float:
        """Assess how well the tool matches the task (0 = poor, 1 = perfect)."""
        # Simple keyword matching
        tool_keywords = {
            "web_search": ["search", "find", "look up", "google", "news", "latest", "current"],
            "code_executor": ["code", "python", "run", "execute", "script", "program", "function"],
            "file_manager": ["file", "folder", "directory", "save", "create", "delete", "move"],
            "browser": ["website", "url", "browse", "open", "page", "click"],
            "vision_analysis": ["image", "picture", "photo", "see", "look at", "analyze image"],
            "calculator": ["calculate", "math", "compute", "add", "subtract", "multiply"],
        }

        keywords = tool_keywords.get(tool, [])
        if not keywords:
            return 0.7  # Unknown tool, assume moderate match

        task_lower = task.lower()
        matches = sum(1 for kw in keywords if kw in task_lower)

        return min(1.0, matches / 2) if matches > 0 else 0.3

    def _get_tool_success_rate(self, tool: str) -> float:
        """Get historical success rate for a tool."""
        outcomes_file = os.path.join(self.data_dir, "outcomes", "all_outcomes.jsonl")
        if not os.path.exists(outcomes_file):
            return 0.7  # Default assumption

        successes = 0
        total = 0

        try:
            with open(outcomes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        outcome = json.loads(line)
                        if outcome.get("tool") == tool:
                            total += 1
                            if outcome.get("actual_success"):
                                successes += 1
        except Exception:
            return 0.7

        if total == 0:
            return 0.7

        return successes / total

    def _assess_knowledge_coverage(self, task: str) -> float:
        """Assess how well Knowledge Graph covers the task topics."""
        if not self.kg:
            return 0.5  # No KG, assume moderate coverage

        # Extract key terms from task
        words = task.lower().split()
        important_words = [w for w in words if len(w) > 4 and w.isalpha()]

        if not important_words:
            return 0.7

        # Check how many are in Knowledge Graph
        found = 0
        check_words = important_words[:5]  # Check up to 5 words

        for word in check_words:
            try:
                nodes = []
                if hasattr(self.kg, 'find_nodes'):
                    nodes = self.kg.find_nodes(label=word)
                elif hasattr(self.kg, 'get_node_by_label'):
                    node = self.kg.get_node_by_label(word)
                    nodes = [node] if node else []
                elif hasattr(self.kg, 'search'):
                    nodes = self.kg.search(word)

                if nodes:
                    found += 1
            except Exception:
                pass

        return found / len(check_words) if check_words else 0.7

    def _find_similar_failures(self, task: str, tool: Optional[str]) -> List[Dict]:
        """Find similar past failures."""
        similar = []
        task_words = set(task.lower().split())

        for pattern in self.failure_patterns:
            # Simple similarity: shared words
            pattern_words = set(pattern.get("task_keywords", []))

            overlap = len(pattern_words & task_words)
            if overlap >= 2:
                similar.append(pattern)
            elif tool and pattern.get("tool") == tool:
                similar.append(pattern)

        return similar[:10]  # Return up to 10 similar failures

    def _detect_loop_risk(self, task: str) -> float:
        """Detect if we're stuck in a loop."""
        if len(self._recent_tasks) < 3:
            return 0.0

        task_lower = task.lower().strip()

        # Count similar recent tasks
        similar_count = sum(1 for t in self._recent_tasks[-5:]
                          if t == task_lower or
                          (len(t) > 10 and t[:10] == task_lower[:10]))

        return min(1.0, similar_count / 3)

    # ==================== FAILURE PREDICTION ====================

    def _predict_failure(self,
                        task: str,
                        tool: Optional[str],
                        indicators: Dict[str, float],
                        context: Dict[str, Any]) -> Optional[FailurePrediction]:
        """
        Predict if and what type of failure is likely.

        Uses rule-based prediction (can be upgraded to ML later).
        """
        predictions = []

        # Rule 1: High uncertainty in recent thinking
        if indicators.get("uncertainty_ratio", 0) > 0.4:
            predictions.append(self._create_prediction(
                failure_type=FailureType.KNOWLEDGE_GAP,
                probability=indicators["uncertainty_ratio"],
                reasoning=f"Recent thoughts show {indicators['uncertainty_ratio']:.0%} uncertainty",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 2: User frustration detected
        if indicators.get("user_frustration", 0) > 0.6:
            predictions.append(self._create_prediction(
                failure_type=FailureType.EMOTIONAL_MISMATCH,
                probability=indicators["user_frustration"],
                reasoning="User appears frustrated - current approach may not be meeting their needs",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 3: High ambiguity
        if indicators.get("task_ambiguity", 0) > 0.5:
            predictions.append(self._create_prediction(
                failure_type=FailureType.AMBIGUOUS_REQUEST,
                probability=indicators["task_ambiguity"],
                reasoning="Request is ambiguous - multiple interpretations possible",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 4: Poor tool match
        if indicators.get("tool_match_score", 1.0) < 0.4:
            predictions.append(self._create_prediction(
                failure_type=FailureType.TOOL_MISMATCH,
                probability=1.0 - indicators["tool_match_score"],
                reasoning=f"Tool '{tool}' may not be the best choice for this task",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 5: Low knowledge coverage
        if indicators.get("knowledge_coverage", 1.0) < 0.3:
            predictions.append(self._create_prediction(
                failure_type=FailureType.KNOWLEDGE_GAP,
                probability=1.0 - indicators["knowledge_coverage"],
                reasoning="Limited knowledge about the topics in this request",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 6: Similar past failures
        if indicators.get("similar_failure_count", 0) > 0.4:
            predictions.append(self._create_prediction(
                failure_type=FailureType.SKILL_BOUNDARY,
                probability=indicators["similar_failure_count"],
                reasoning="Similar tasks have failed before",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 7: Low confidence from context
        if indicators.get("confidence", 1.0) < 0.3:
            predictions.append(self._create_prediction(
                failure_type=FailureType.CONFIDENCE_DROP,
                probability=1.0 - indicators["confidence"],
                reasoning=f"Confidence level is only {indicators['confidence']:.0%}",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 8: Context overflow risk
        if indicators.get("context_length", 0) > 0.85:
            predictions.append(self._create_prediction(
                failure_type=FailureType.CONTEXT_OVERFLOW,
                probability=indicators["context_length"],
                reasoning="Context is nearly full - may lose important information",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 9: High complexity + low confidence
        complexity = indicators.get("task_complexity", 0)
        confidence = indicators.get("confidence", 0.7)
        if complexity > 0.6 and confidence < 0.5:
            predictions.append(self._create_prediction(
                failure_type=FailureType.SKILL_BOUNDARY,
                probability=(complexity + (1 - confidence)) / 2,
                reasoning="Complex task combined with low confidence",
                task=task, tool=tool, indicators=indicators
            ))

        # Rule 10: Loop detected
        if indicators.get("loop_risk", 0) > 0.5:
            predictions.append(self._create_prediction(
                failure_type=FailureType.LOOP_DETECTED,
                probability=indicators["loop_risk"],
                reasoning="Detected repetitive pattern - may be stuck in a loop",
                task=task, tool=tool, indicators=indicators
            ))

        # Return highest probability prediction
        if predictions:
            return max(predictions, key=lambda p: p.probability)

        return None

    def _create_prediction(self,
                          failure_type: FailureType,
                          probability: float,
                          reasoning: str,
                          task: str,
                          tool: Optional[str],
                          indicators: Dict[str, float]) -> FailurePrediction:
        """Create a FailurePrediction with appropriate intervention."""

        # Select intervention based on failure type and probability
        intervention = self._select_intervention(failure_type, probability)

        # Generate intervention message
        message = self._generate_intervention_message(failure_type, probability, reasoning, intervention)

        # Generate alternatives
        alternatives = self._generate_alternatives(failure_type, task, tool)

        return FailurePrediction(
            id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{failure_type.value[:4]}",
            timestamp=datetime.now(),
            failure_type=failure_type,
            probability=probability,
            severity=self._assess_severity(failure_type),
            indicators=indicators,
            reasoning=reasoning,
            current_task=task,
            current_tool=tool,
            recommended_intervention=intervention,
            intervention_message=message,
            alternative_approaches=alternatives
        )

    def _select_intervention(self,
                            failure_type: FailureType,
                            probability: float) -> InterventionType:
        """Select appropriate intervention based on failure type and probability."""

        # Map failure types to default interventions
        intervention_map = {
            FailureType.KNOWLEDGE_GAP: InterventionType.REQUEST_CLARIFICATION,
            FailureType.TOOL_MISMATCH: InterventionType.TOOL_SWITCH,
            FailureType.AMBIGUOUS_REQUEST: InterventionType.REQUEST_CLARIFICATION,
            FailureType.CONFIDENCE_DROP: InterventionType.CONFIDENCE_WARNING,
            FailureType.EMOTIONAL_MISMATCH: InterventionType.EMOTIONAL_ADJUSTMENT,
            FailureType.SKILL_BOUNDARY: InterventionType.HUMAN_HANDOFF,
            FailureType.HALLUCINATION_RISK: InterventionType.PAUSE_AND_EXPLAIN,
            FailureType.CONTEXT_OVERFLOW: InterventionType.PAUSE_AND_EXPLAIN,
            FailureType.LOOP_DETECTED: InterventionType.SUGGEST_ALTERNATIVE,
        }

        intervention = intervention_map.get(failure_type, InterventionType.CONFIDENCE_WARNING)

        # Escalate if probability is very high
        if probability >= self.config.abort_threshold:
            intervention = InterventionType.ABORT_TASK
        elif probability >= self.config.intervention_threshold:
            if intervention == InterventionType.CONFIDENCE_WARNING:
                intervention = InterventionType.HUMAN_HANDOFF

        return intervention

    def _generate_intervention_message(self,
                                       failure_type: FailureType,
                                       probability: float,
                                       reasoning: str,
                                       intervention: InterventionType) -> str:
        """Generate user-facing intervention message."""

        if intervention == InterventionType.CONFIDENCE_WARNING:
            return (f"**Heads up:** I'm only {(1-probability):.0%} confident about this.\n\n"
                   f"{reasoning}\n\n"
                   f"I'll do my best, but please double-check my answer.")

        elif intervention == InterventionType.REQUEST_CLARIFICATION:
            return (f"I want to make sure I help you properly.\n\n"
                   f"{reasoning}\n\n"
                   f"Could you provide a bit more detail about what you're looking for?")

        elif intervention == InterventionType.HUMAN_HANDOFF:
            return (f"I need to be honest with you - I'm not confident I can handle this well.\n\n"
                   f"**What I'm struggling with:** {reasoning}\n\n"
                   f"**My confidence:** {(1-probability):.0%}\n\n"
                   f"Would you like to:\n"
                   f"1. Give me more context or clarification?\n"
                   f"2. Try a different approach?\n"
                   f"3. Handle this part yourself?\n\n"
                   f"I'd rather ask for help than give you a bad answer.")

        elif intervention == InterventionType.TOOL_SWITCH:
            return (f"I think there might be a better approach.\n\n"
                   f"{reasoning}\n\n"
                   f"Let me try a different method.")

        elif intervention == InterventionType.SUGGEST_ALTERNATIVE:
            return (f"I have a suggestion.\n\n"
                   f"{reasoning}\n\n"
                   f"Would you like me to try an alternative approach?")

        elif intervention == InterventionType.EMOTIONAL_ADJUSTMENT:
            return (f"I sense this might be frustrating. Let me take a step back and approach this differently.\n\n"
                   f"What would be most helpful for you right now?")

        elif intervention == InterventionType.PAUSE_AND_EXPLAIN:
            return (f"Let me pause and explain something.\n\n"
                   f"{reasoning}\n\n"
                   f"I want to be transparent about my limitations here.")

        elif intervention == InterventionType.ABORT_TASK:
            return (f"I need to stop here.\n\n"
                   f"{reasoning}\n\n"
                   f"I don't think I can complete this task reliably. "
                   f"It would be better for you to handle this directly or try a different approach.")

        return f"Note: {reasoning}"

    def _generate_alternatives(self,
                              failure_type: FailureType,
                              task: str,
                              tool: Optional[str]) -> List[str]:
        """Generate alternative approaches."""
        alternatives = []

        if failure_type == FailureType.TOOL_MISMATCH:
            tool_suggestions = {
                "web_search": ["Try asking me directly - I might know this", "Use browser for specific sites"],
                "code_executor": ["Let me explain the concept instead", "Use a simpler approach"],
                "browser": ["Use web_search for general queries", "I can describe how to do it manually"],
            }
            alternatives = tool_suggestions.get(tool, ["Try rephrasing the request"])

        elif failure_type == FailureType.AMBIGUOUS_REQUEST:
            alternatives = [
                "Be more specific about what you want",
                "Provide an example of the expected output",
                "Break this into smaller, clearer steps"
            ]

        elif failure_type == FailureType.KNOWLEDGE_GAP:
            alternatives = [
                "Let me search for more information",
                "Can you provide some context or background?",
                "Try asking about a specific aspect"
            ]

        elif failure_type == FailureType.SKILL_BOUNDARY:
            alternatives = [
                "Break this into simpler sub-tasks",
                "Handle the complex part manually",
                "Use a specialized tool or service"
            ]

        elif failure_type == FailureType.LOOP_DETECTED:
            alternatives = [
                "Try rephrasing your request",
                "Let's start fresh with a different approach",
                "Break the task into smaller steps"
            ]

        return alternatives[:3]

    def _assess_severity(self, failure_type: FailureType) -> float:
        """Assess how severe a failure would be."""
        severity_map = {
            FailureType.KNOWLEDGE_GAP: 0.4,
            FailureType.TOOL_MISMATCH: 0.3,
            FailureType.AMBIGUOUS_REQUEST: 0.3,
            FailureType.CONFIDENCE_DROP: 0.5,
            FailureType.EMOTIONAL_MISMATCH: 0.6,
            FailureType.SKILL_BOUNDARY: 0.7,
            FailureType.HALLUCINATION_RISK: 0.8,
            FailureType.CONTEXT_OVERFLOW: 0.5,
            FailureType.LOOP_DETECTED: 0.4,
        }
        return severity_map.get(failure_type, 0.5)

    # ==================== LEARNING ====================

    def _learn_from_outcome(self, prediction: FailurePrediction, outcome: Dict):
        """Learn from prediction outcome to improve future predictions."""

        if not outcome["actual_success"]:
            # Task failed - add to failure patterns
            pattern = {
                "task_keywords": prediction.current_task.lower().split()[:10],
                "tool": prediction.current_tool,
                "failure_type": prediction.failure_type.value,
                "indicators": prediction.indicators,
                "timestamp": datetime.now().isoformat()
            }
            self.failure_patterns.append(pattern)
            self._save_failure_patterns()

            # Add to Knowledge Graph if available
            if self.kg:
                try:
                    if hasattr(self.kg, 'add_node'):
                        self.kg.add_node(
                            node_type="event",
                            label=f"Failure: {prediction.failure_type.value}",
                            properties={
                                "task_preview": prediction.current_task[:100],
                                "tool": prediction.current_tool,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                except Exception:
                    pass

    # ==================== PERSISTENCE ====================

    def _load_failure_patterns(self) -> List[Dict]:
        """Load failure patterns from disk."""
        patterns_file = os.path.join(self.data_dir, "patterns", "failure_patterns.jsonl")
        patterns = []

        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            patterns.append(json.loads(line))
            except Exception:
                pass

        return patterns

    def _save_failure_patterns(self):
        """Save failure patterns to disk."""
        patterns_file = os.path.join(self.data_dir, "patterns", "failure_patterns.jsonl")

        try:
            with open(patterns_file, 'w', encoding='utf-8') as f:
                for pattern in self.failure_patterns[-100:]:  # Keep last 100
                    f.write(json.dumps(pattern) + '\n')
        except Exception:
            pass

    def _save_outcome(self, outcome: Dict):
        """Save outcome to disk."""
        outcomes_file = os.path.join(self.data_dir, "outcomes", "all_outcomes.jsonl")

        try:
            with open(outcomes_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(outcome) + '\n')
        except Exception:
            pass

    # ==================== STATS & MONITORING ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get guardian statistics."""
        return {
            "session_predictions": len(self.session_predictions),
            "interventions_triggered": self.intervention_count,
            "failure_patterns_learned": len(self.failure_patterns),
            "monitoring_level": self.config.monitoring_level,
            "thresholds": {
                "warning": self.config.warning_threshold,
                "intervention": self.config.intervention_threshold,
                "abort": self.config.abort_threshold
            }
        }

    def set_monitoring_level(self, level: str):
        """
        Set monitoring intensity.

        Levels:
        - "low": Only critical risks (thresholds: 0.5, 0.8, 0.95)
        - "medium": Balanced monitoring (thresholds: 0.3, 0.6, 0.9)
        - "high": Sensitive monitoring (thresholds: 0.2, 0.5, 0.85)
        - "critical": Maximum sensitivity (thresholds: 0.1, 0.4, 0.8)
        """
        thresholds = {
            "low": (0.5, 0.8, 0.95),
            "medium": (0.3, 0.6, 0.9),
            "high": (0.2, 0.5, 0.85),
            "critical": (0.1, 0.4, 0.8)
        }

        if level in thresholds:
            self.config.monitoring_level = level
            self.config.warning_threshold = thresholds[level][0]
            self.config.intervention_threshold = thresholds[level][1]
            self.config.abort_threshold = thresholds[level][2]

    def reset_session(self):
        """Reset session-specific tracking."""
        self.session_predictions = []
        self.intervention_count = 0
        self._recent_tasks = []


# Convenience function
def get_guardian(inner_monologue=None, evoemo=None, knowledge_graph=None) -> MetacognitiveGuardian:
    """Get or create a MetacognitiveGuardian instance."""
    return MetacognitiveGuardian(
        inner_monologue=inner_monologue,
        evoemo=evoemo,
        knowledge_graph=knowledge_graph
    )
