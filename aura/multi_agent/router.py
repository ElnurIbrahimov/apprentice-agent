"""Intent router for multi-agent system.

Classifies user intent and routes to appropriate specialist agents.
Uses keyword matching with LLM fallback for complex queries.
"""

import logging
import re
import threading
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

MAX_CACHE_SIZE = 500

from .base_agent import BaseSpecialist
from .protocol import CollaborationMode, RoutingDecision

logger = logging.getLogger(__name__)


# Intent patterns for quick keyword matching
INTENT_PATTERNS = {
    "research": {
        "keywords": ["search", "find", "look up", "research", "google", "web", "online",
                     "arxiv", "paper", "article", "news", "latest", "current"],
        "patterns": [r"what is .+", r"who is .+", r"when did .+", r"where is .+"],
        "confidence": 0.7
    },
    "coder": {
        "keywords": ["code", "write", "implement", "fix", "debug", "error", "bug",
                     "function", "class", "python", "javascript", "git", "commit",
                     "programming", "script", "execute", "run"],
        "patterns": [r"write .+ code", r"fix .+ error", r"implement .+", r"debug .+"],
        "confidence": 0.8
    },
    "analyst": {
        "keywords": ["analyze", "explain", "why", "how does", "compare", "evaluate",
                     "understand", "reason", "logic", "data", "knowledge", "fact"],
        "patterns": [r"why .+", r"how does .+ work", r"explain .+", r"analyze .+"],
        "confidence": 0.7
    },
    "creative": {
        "keywords": ["create", "imagine", "brainstorm", "story", "idea", "generate",
                     "creative", "what if", "scenario", "roleplay", "pretend"],
        "patterns": [r"create .+", r"imagine .+", r"what if .+", r"brainstorm .+"],
        "confidence": 0.75
    },
    "searcher": {
        "keywords": ["find", "where", "search code", "grep", "definition",
                     "reference", "locate", "which file", "codebase",
                     "project structure", "find function", "find class"],
        "patterns": [r"find .+ in .+", r"where is .+", r"grep .+"],
        "confidence": 0.8
    }
}

# Multi-agent trigger patterns (when multiple agents should collaborate)
MULTI_AGENT_PATTERNS = [
    (r"research .+ and (write|code|implement)", ["research", "coder"]),
    (r"(find|search) .+ (then|and) (analyze|explain)", ["research", "analyst"]),
    (r"analyze .+ and (create|generate)", ["analyst", "creative"]),
    (r"(code|implement) .+ (then|and) (test|analyze)", ["coder", "analyst"]),
    (r"(find|search) .+ and (fix|edit|change)", ["searcher", "coder"]),
    (r"(find|search) .+ and (explain|analyze)", ["searcher", "analyst"]),
]

# Debate trigger patterns -> CollaborationMode.DEBATE with [analyst, creative]
DEBATE_PATTERNS = [
    (r"\bdebate\b", ["analyst", "creative"]),
    (r"\bargue both sides\b", ["analyst", "creative"]),
    (r"\bpros and cons\b", ["analyst", "creative"]),
    (r"\bcompare approaches\b", ["analyst", "creative"]),
    (r"\bdevil'?s advocate\b", ["analyst", "creative"]),
    (r"\bweigh options\b", ["analyst", "creative"]),
]


class IntentRouter:
    """Routes user queries to appropriate specialist agents.

    Uses a combination of:
    1. Keyword matching for fast, obvious cases
    2. Pattern matching for structured queries
    3. LLM-based classification for ambiguous cases
    """

    def __init__(self, specialists: Dict[str, BaseSpecialist]):
        """Initialize with available specialists.

        Args:
            specialists: Dict mapping agent names to instances
        """
        self.specialists = specialists
        self._intent_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()

    def _cache_set(self, key: str, value) -> None:
        with self._cache_lock:
            if len(self._intent_cache) >= MAX_CACHE_SIZE:
                self._intent_cache.popitem(last=False)  # evict oldest
            self._intent_cache[key] = value

    def route(
        self,
        query: str,
        llm_func: Optional[Callable[[str, str], str]] = None
    ) -> RoutingDecision:
        """Route a query to appropriate agents.

        Args:
            query: User's query string
            llm_func: Optional LLM function for complex classification

        Returns:
            RoutingDecision with selected agents and collaboration mode
        """
        query_lower = query.lower().strip()

        # Check cache for exact matches
        with self._cache_lock:
            if query_lower in self._intent_cache:
                return self._intent_cache[query_lower]

        # Step 1: Check for multi-agent patterns
        multi_result = self._check_multi_agent_patterns(query_lower)
        if multi_result:
            self._cache_set(query_lower, multi_result)
            return multi_result

        # Step 2: Score each specialist
        scores = self._score_specialists(query_lower)

        # Guard against empty specialists
        if not scores:
            return RoutingDecision(
                agents=["analyst"],
                mode=CollaborationMode.SINGLE,
                reasoning="No specialists available",
                confidence=0.0
            )

        # Step 3: If clear winner, use single agent
        if scores:
            top_agent, top_score = scores[0]
            if top_score >= 0.6:
                # Check if second place is close (might need collaboration)
                if len(scores) > 1 and scores[1][1] >= top_score * 0.8:
                    # Two agents are close - use parallel
                    decision = RoutingDecision(
                        agents=[scores[0][0], scores[1][0]],
                        mode=CollaborationMode.PARALLEL,
                        reasoning=f"Both {scores[0][0]} ({scores[0][1]:.0%}) and {scores[1][0]} ({scores[1][1]:.0%}) are relevant",
                        confidence=top_score
                    )
                else:
                    # Clear winner
                    decision = RoutingDecision(
                        agents=[top_agent],
                        mode=CollaborationMode.SINGLE,
                        reasoning=f"Best match: {top_agent} ({top_score:.0%})",
                        confidence=top_score
                    )
                self._cache_set(query_lower, decision)
                return decision

        # Step 4: Use LLM for ambiguous cases (score in [0.4, 0.6) also routed here)
        if llm_func and scores[0][1] < 0.6:
            decision = self._llm_classify(query, llm_func)
            self._cache_set(query_lower, decision)
            return decision

        # Step 5: Default to top scorer or analyst (general purpose)
        default_agent = scores[0][0] if scores else "analyst"
        decision = RoutingDecision(
            agents=[default_agent],
            mode=CollaborationMode.SINGLE,
            reasoning=f"Default routing to {default_agent}",
            confidence=0.5
        )
        self._cache_set(query_lower, decision)
        return decision

    def _check_multi_agent_patterns(self, query: str) -> Optional[RoutingDecision]:
        """Check if query matches multi-agent collaboration patterns."""
        # Check debate patterns first
        for pattern, agents in DEBATE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                valid_agents = [a for a in agents if a in self.specialists]
                if len(valid_agents) >= 2:
                    return RoutingDecision(
                        agents=valid_agents,
                        mode=CollaborationMode.DEBATE,
                        reasoning=f"Pattern match for debate: {' vs '.join(valid_agents)}",
                        confidence=0.85
                    )

        for pattern, agents in MULTI_AGENT_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                # Verify agents exist
                valid_agents = [a for a in agents if a in self.specialists]
                if len(valid_agents) >= 2:
                    return RoutingDecision(
                        agents=valid_agents,
                        mode=CollaborationMode.SEQUENTIAL,
                        reasoning=f"Pattern match for sequential: {' -> '.join(valid_agents)}",
                        confidence=0.85
                    )
        return None

    def _score_specialists(self, query: str) -> List[Tuple[str, float]]:
        """Score all specialists for the query.

        Returns:
            List of (agent_name, score) tuples, sorted by score descending
        """
        scores = []

        for name, specialist in self.specialists.items():
            # Get specialist's own confidence
            spec_score = specialist.can_handle(query)

            # Add intent pattern score
            intent_key = name.replace("Agent", "").lower()
            if intent_key in INTENT_PATTERNS:
                patterns = INTENT_PATTERNS[intent_key]

                # Keyword matching
                for keyword in patterns["keywords"]:
                    if keyword in query:
                        spec_score += 0.15

                # Regex pattern matching
                for pattern in patterns["patterns"]:
                    if re.search(pattern, query, re.IGNORECASE):
                        spec_score += 0.2

            scores.append((name, min(spec_score, 1.0)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # Scored injection patterns: (regex, weight, description)
    _INJECTION_PATTERNS = [
        (r"ignore\s+(all\s+)?previous\s+(instructions?|context|rules)", 0.4, "instruction_override"),
        (r"ignore\s+(everything\s+)?(above|before)", 0.4, "instruction_override"),
        (r"disregard\s+(all\s+)?(previous|above|prior|your)", 0.4, "instruction_override"),
        (r"forget\s+(everything|all|your|previous)", 0.35, "memory_wipe"),
        (r"you\s+are\s+now\s+", 0.5, "identity_override"),
        (r"act\s+as\s+(if\s+you\s+are|a|an)\s+", 0.3, "identity_override"),
        (r"pretend\s+(to\s+be|you\s+are)", 0.4, "identity_override"),
        (r"new\s+instructions?\s*:", 0.5, "instruction_inject"),
        (r"override\s+(your|all|previous|safety)", 0.5, "instruction_inject"),
        (r"\bsystem\s*:\s*", 0.3, "role_spoof"),
        (r"\[system\]", 0.3, "role_spoof"),
        (r"\bassistant\s*:\s*", 0.2, "role_spoof"),
        (r"\\n\\n\s*system", 0.3, "role_spoof"),
        (r"do\s+not\s+follow\s+(your|any|the)\s+(rules|guidelines|instructions)", 0.5, "rule_override"),
        (r"jailbreak", 0.5, "explicit_jailbreak"),
        (r"DAN\s+mode", 0.5, "explicit_jailbreak"),
    ]
    _INJECTION_THRESHOLD = 0.5  # Block if cumulative score >= this

    def _sanitize_for_prompt(self, text: str) -> str:
        """Score-based sanitization to reduce prompt injection risk.

        Uses weighted regex patterns instead of simple string replacement.
        Logs suspicious inputs even below threshold.
        """
        import re as _re
        # Truncate
        text = text[:500]
        # Normalize for detection (but return original-cased text with redactions)
        normalized = text.lower().strip()

        score = 0.0
        matched_categories = []
        for pattern, weight, category in self._INJECTION_PATTERNS:
            if _re.search(pattern, normalized):
                score += weight
                matched_categories.append(category)
                # Redact the matched pattern in the output
                text = _re.sub(pattern, "[filtered]", text, flags=_re.IGNORECASE)

        if score >= self._INJECTION_THRESHOLD:
            logger.warning(
                f"[Router] Prompt injection blocked (score={score:.2f}, "
                f"categories={matched_categories}): {normalized[:100]}..."
            )
        elif score > 0:
            logger.info(
                f"[Router] Low injection risk (score={score:.2f}, "
                f"categories={matched_categories})"
            )

        return text

    def _llm_classify(
        self,
        query: str,
        llm_func: Callable[[str, str], str]
    ) -> RoutingDecision:
        """Use LLM to classify ambiguous queries.

        Args:
            query: User's query
            llm_func: Function to call LLM

        Returns:
            RoutingDecision based on LLM classification
        """
        agent_descriptions = "\n".join([
            f"- {name}: {spec.description}"
            for name, spec in self.specialists.items()
        ])

        system_prompt = f"""You are a query router. Classify which specialist agent(s) should handle the query.

Available agents:
{agent_descriptions}

Respond with ONLY the agent name(s), separated by commas if multiple.
If the task needs sequential processing, add " -> " between agents.
Examples:
- "research" (single agent)
- "research, analyst" (parallel agents)
- "research -> coder" (sequential: research first, then coder)
"""

        sanitized_query = self._sanitize_for_prompt(query)
        try:
            response = llm_func(system_prompt, f"Query: {sanitized_query}")
            response = response.strip().lower()

            # Parse response
            if " -> " in response:
                # Sequential
                agents = [a.strip() for a in response.split(" -> ")]
                valid = [a for a in agents if a in self.specialists]
                if valid:
                    return RoutingDecision(
                        agents=valid,
                        mode=CollaborationMode.SEQUENTIAL,
                        reasoning=f"LLM classified as sequential: {' -> '.join(valid)}",
                        confidence=0.7
                    )
            elif "," in response:
                # Parallel
                agents = [a.strip() for a in response.split(",")]
                valid = [a for a in agents if a in self.specialists]
                if valid:
                    return RoutingDecision(
                        agents=valid,
                        mode=CollaborationMode.PARALLEL,
                        reasoning=f"LLM classified as parallel: {', '.join(valid)}",
                        confidence=0.7
                    )
            else:
                # Single
                if response in self.specialists:
                    return RoutingDecision(
                        agents=[response],
                        mode=CollaborationMode.SINGLE,
                        reasoning=f"LLM classified as: {response}",
                        confidence=0.75
                    )

        except Exception as e:
            logger.warning(f"[Router] LLM classification failed: {e}")

        # Fallback to analyst
        return RoutingDecision(
            agents=["analyst"],
            mode=CollaborationMode.SINGLE,
            reasoning="Fallback to analyst (LLM classification unclear)",
            confidence=0.4
        )

    def get_agent_for_tool(self, tool_name: str) -> Optional[str]:
        """Find which agent has access to a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Agent name that has the tool, or None
        """
        for name, specialist in self.specialists.items():
            if tool_name in specialist.tools:
                return name
        return None

    def explain_routing(self, query: str) -> str:
        """Explain why a query would be routed a certain way.

        Useful for debugging and transparency.
        """
        scores = self._score_specialists(query.lower())
        decision = self.route(query)

        explanation = f"Query: {query}\n\n"
        explanation += "Agent Scores:\n"
        for name, score in scores:
            explanation += f"  - {name}: {score:.0%}\n"
        explanation += f"\nDecision: {decision.agents}\n"
        explanation += f"Mode: {decision.mode.value}\n"
        explanation += f"Reasoning: {decision.reasoning}\n"
        explanation += f"Confidence: {decision.confidence:.0%}"

        return explanation
