"""Inter-agent communication protocol.

Defines the message structures for agent communication and collaboration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CollaborationMode(Enum):
    """How agents collaborate on a task."""
    SINGLE = "single"          # One agent handles entire task
    SEQUENTIAL = "sequential"  # Agents work in sequence, passing results
    PARALLEL = "parallel"      # Agents work simultaneously, results merged
    DEBATE = "debate"          # Agents critique and refine each other's work


@dataclass
class AgentMessage:
    """Message passed between agents or from user.

    Attributes:
        content: The message text
        sender: Who sent this ("user", "orchestrator", or agent name)
        recipient: Target agent (None for broadcast)
        context: Shared context (previous results, user preferences, etc.)
        metadata: Additional info (timestamps, routing decisions, etc.)
    """
    content: str
    sender: str = "user"
    recipient: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def with_context(self, key: str, value: Any) -> "AgentMessage":
        """Return a new message with additional context."""
        new_context = {**self.context, key: value}
        return AgentMessage(
            content=self.content,
            sender=self.sender,
            recipient=self.recipient,
            context=new_context,
            metadata=self.metadata,
            timestamp=self.timestamp
        )

    def forward_to(self, recipient: str, new_sender: str = "orchestrator") -> "AgentMessage":
        """Forward this message to another agent."""
        return AgentMessage(
            content=self.content,
            sender=new_sender,
            recipient=recipient,
            context=self.context,
            metadata={**self.metadata, "forwarded_from": self.sender},
            timestamp=datetime.now()
        )


@dataclass
class AgentResult:
    """Result returned by an agent after processing.

    Attributes:
        success: Whether the task completed successfully
        response: The agent's response text
        agent: Name of the agent that produced this
        tools_used: List of tool names that were invoked
        confidence: Agent's confidence in the response (0-1)
        needs_followup: Whether another agent should continue
        followup_agent: Suggested agent for followup (if any)
        artifacts: Any produced artifacts (code, files, data)
        thinking: Agent's reasoning process (for transparency)
    """
    success: bool
    response: str
    agent: str
    tools_used: List[str] = field(default_factory=list)
    confidence: float = 1.0
    needs_followup: bool = False
    followup_agent: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    thinking: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_context(self) -> Dict[str, Any]:
        """Convert to context dict for passing to next agent."""
        return {
            "previous_agent": self.agent,
            "previous_response": self.response,
            "previous_tools": self.tools_used,
            "previous_confidence": self.confidence,
            "artifacts": self.artifacts
        }

    def merge_with(self, other: "AgentResult") -> "AgentResult":
        """Merge with another result (for parallel execution)."""
        return AgentResult(
            success=self.success and other.success,
            response=f"**{self.agent}:**\n{self.response}\n\n**{other.agent}:**\n{other.response}",
            agent="merged",
            tools_used=list(set(self.tools_used + other.tools_used)),
            confidence=(self.confidence + other.confidence) / 2,
            needs_followup=self.needs_followup or other.needs_followup,
            artifacts={**self.artifacts, **other.artifacts},
            execution_time=max(self.execution_time, other.execution_time)
        )


@dataclass
class RoutingDecision:
    """Decision made by the router about which agents to use.

    Attributes:
        agents: List of agent names to invoke
        mode: How agents should collaborate
        reasoning: Why this routing was chosen
        confidence: Confidence in the routing decision
    """
    agents: List[str]
    mode: CollaborationMode = CollaborationMode.SINGLE
    reasoning: str = ""
    confidence: float = 1.0

    @property
    def is_multi_agent(self) -> bool:
        """Whether multiple agents are involved."""
        return len(self.agents) > 1


@dataclass
class ConversationTurn:
    """A single turn in the multi-agent conversation."""
    user_message: AgentMessage
    routing: RoutingDecision
    results: List[AgentResult]
    final_response: str
    timestamp: datetime = field(default_factory=datetime.now)
