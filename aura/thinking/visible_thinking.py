"""
VisibleThinking - Show AURA's Internal Reasoning

Makes AURA's thought process visible to users:
- Shows what AURA is considering
- Reveals decision points
- Makes reasoning transparent
- Creates a sense of authentic thinking

This makes AURA feel more human and trustworthy.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    """Types of thoughts AURA can have."""
    ANALYZING = "analyzing"         # Processing input
    CONSIDERING = "considering"     # Weighing options
    PLANNING = "planning"           # Formulating approach
    RECALLING = "recalling"         # Accessing memory
    QUESTIONING = "questioning"     # Uncertain, need info
    CONNECTING = "connecting"       # Making connections
    DECIDING = "deciding"           # Making a choice
    REFLECTING = "reflecting"       # Thinking about thinking


@dataclass
class Thought:
    """A single thought in the thinking process."""
    content: str
    thought_type: ThoughtType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.7
    source: str = ""  # What triggered this thought

    def format_display(self) -> str:
        """Format thought for display."""
        icons = {
            ThoughtType.ANALYZING: "🔍",
            ThoughtType.CONSIDERING: "🤔",
            ThoughtType.PLANNING: "📋",
            ThoughtType.RECALLING: "💭",
            ThoughtType.QUESTIONING: "❓",
            ThoughtType.CONNECTING: "🔗",
            ThoughtType.DECIDING: "✅",
            ThoughtType.REFLECTING: "💫"
        }
        icon = icons.get(self.thought_type, "💭")
        return f"{icon} {self.content}"


@dataclass
class ThoughtProcess:
    """A complete thought process for a task."""
    task: str
    thoughts: List[Thought] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    conclusion: str = ""

    def add_thought(self, content: str, thought_type: ThoughtType, **kwargs) -> Thought:
        """Add a thought to the process."""
        thought = Thought(content=content, thought_type=thought_type, **kwargs)
        self.thoughts.append(thought)
        return thought

    def complete(self, conclusion: str) -> None:
        """Mark the thought process as complete."""
        self.conclusion = conclusion
        self.completed_at = datetime.now().isoformat()

    def format_stream(self) -> str:
        """Format as a thinking stream for display."""
        lines = [f"*Thinking about: {self.task}*\n"]
        for thought in self.thoughts:
            lines.append(thought.format_display())
        if self.conclusion:
            lines.append(f"\n*Conclusion: {self.conclusion}*")
        return "\n".join(lines)

    def format_compact(self) -> str:
        """Format as a compact summary."""
        if not self.thoughts:
            return ""
        key_thoughts = [t for t in self.thoughts if t.confidence > 0.6][-3:]
        if not key_thoughts:
            key_thoughts = self.thoughts[-2:]
        return " → ".join(t.content[:50] for t in key_thoughts)


class VisibleThinking:
    """
    System for making AURA's thinking visible.

    Features:
    - Stream thoughts in real-time
    - Format for different display modes
    - Connect to memory and emotion
    """

    # Thought templates for common situations
    TEMPLATES = {
        "greeting": [
            ("Let me see who this is...", ThoughtType.RECALLING),
            ("Checking our conversation history...", ThoughtType.RECALLING),
        ],
        "question": [
            ("Interesting question...", ThoughtType.ANALYZING),
            ("Let me think about this...", ThoughtType.CONSIDERING),
            ("I should consider multiple angles here", ThoughtType.PLANNING),
        ],
        "task": [
            ("Breaking this down into steps...", ThoughtType.PLANNING),
            ("What's the best approach?", ThoughtType.CONSIDERING),
            ("Let me work through this systematically", ThoughtType.PLANNING),
        ],
        "error": [
            ("Hmm, something's not right...", ThoughtType.ANALYZING),
            ("Let me check what went wrong", ThoughtType.ANALYZING),
            ("Maybe I should try a different approach", ThoughtType.CONSIDERING),
        ],
        "memory": [
            ("This reminds me of something...", ThoughtType.RECALLING),
            ("We've discussed something similar before", ThoughtType.CONNECTING),
        ],
        "uncertain": [
            ("I'm not entirely sure about this...", ThoughtType.QUESTIONING),
            ("Let me think more carefully", ThoughtType.REFLECTING),
            ("I should be honest about my uncertainty", ThoughtType.DECIDING),
        ],
    }

    def __init__(
        self,
        show_thoughts: bool = True,
        detail_level: str = "normal"  # "minimal", "normal", "verbose"
    ):
        """
        Initialize visible thinking system.

        Args:
            show_thoughts: Whether to show thoughts to user
            detail_level: How much detail to show
        """
        self.show_thoughts = show_thoughts
        self.detail_level = detail_level
        self.current_process: Optional[ThoughtProcess] = None
        self.history: List[ThoughtProcess] = []

        # Callbacks for streaming thoughts
        self._thought_callbacks: List[Callable[[Thought], None]] = []

    def start_thinking(self, task: str) -> ThoughtProcess:
        """
        Start a new thought process.

        Args:
            task: What AURA is thinking about

        Returns:
            The new ThoughtProcess
        """
        if self.current_process:
            self.history.append(self.current_process)

        self.current_process = ThoughtProcess(task=task)
        return self.current_process

    def think(
        self,
        content: str,
        thought_type: ThoughtType = ThoughtType.CONSIDERING,
        confidence: float = 0.7,
        source: str = ""
    ) -> Optional[Thought]:
        """
        Add a thought to the current process.

        Args:
            content: The thought content
            thought_type: Type of thought
            confidence: How confident (affects display)
            source: What triggered this thought

        Returns:
            The thought (or None if no active process)
        """
        if not self.current_process:
            self.current_process = ThoughtProcess(task="thinking")

        thought = self.current_process.add_thought(
            content=content,
            thought_type=thought_type,
            confidence=confidence,
            source=source
        )

        # Notify callbacks
        for callback in self._thought_callbacks:
            try:
                callback(thought)
            except Exception as e:
                logger.error(f"Thought callback error: {e}")

        return thought

    def use_template(self, template_name: str) -> List[Thought]:
        """
        Add thoughts from a template.

        Args:
            template_name: Name of the template to use

        Returns:
            List of thoughts added
        """
        thoughts = []
        template = self.TEMPLATES.get(template_name, [])

        for content, thought_type in template:
            thought = self.think(content, thought_type)
            if thought:
                thoughts.append(thought)

        return thoughts

    def conclude(self, conclusion: str) -> ThoughtProcess:
        """
        Complete the current thought process.

        Args:
            conclusion: Final conclusion

        Returns:
            The completed ThoughtProcess
        """
        if self.current_process:
            self.current_process.complete(conclusion)
            completed = self.current_process
            self.history.append(completed)
            self.current_process = None
            return completed

        # No active process, create minimal one
        process = ThoughtProcess(task="conclusion")
        process.complete(conclusion)
        return process

    def register_callback(self, callback: Callable[[Thought], None]) -> None:
        """Register a callback for new thoughts (for streaming)."""
        self._thought_callbacks.append(callback)

    def get_formatted_thinking(self) -> str:
        """Get current thinking formatted for display."""
        if not self.current_process:
            return ""

        if self.detail_level == "minimal":
            return self.current_process.format_compact()
        elif self.detail_level == "verbose":
            return self.current_process.format_stream()
        else:  # normal
            # Show last few thoughts
            recent = self.current_process.thoughts[-3:]
            return "\n".join(t.format_display() for t in recent)

    def generate_thinking_prefix(self, query: str) -> str:
        """
        Generate a brief thinking prefix for a response.

        Args:
            query: The user's query

        Returns:
            A brief thought to show before responding
        """
        query_lower = query.lower()

        # Detect query type and generate appropriate thought
        if any(w in query_lower for w in ["what is", "explain", "how does"]):
            return "🤔 *Let me explain this clearly...*"

        elif any(w in query_lower for w in ["help", "fix", "error", "problem"]):
            return "🔍 *Let me look into this...*"

        elif any(w in query_lower for w in ["why", "reason", "because"]):
            return "💭 *Good question, let me think...*"

        elif any(w in query_lower for w in ["create", "write", "make", "build"]):
            return "📋 *Planning the approach...*"

        elif any(w in query_lower for w in ["remember", "last time", "before"]):
            return "💭 *Let me check my memory...*"

        elif "?" in query:
            return "🤔 *Considering this...*"

        return ""  # No prefix needed

    def introspect(self) -> Dict:
        """Get introspection data about thinking patterns."""
        all_thoughts = []
        for process in self.history[-20:]:  # Recent history
            all_thoughts.extend(process.thoughts)

        if not all_thoughts:
            return {"status": "No thinking history yet"}

        type_counts = {}
        for thought in all_thoughts:
            ttype = thought.thought_type.value
            type_counts[ttype] = type_counts.get(ttype, 0) + 1

        avg_confidence = sum(t.confidence for t in all_thoughts) / len(all_thoughts)

        return {
            "total_thoughts": len(all_thoughts),
            "thought_types": type_counts,
            "avg_confidence": round(avg_confidence, 2),
            "processes_completed": len(self.history),
            "most_common": max(type_counts, key=type_counts.get) if type_counts else None
        }


if __name__ == "__main__":
    print("=" * 60)
    print("VisibleThinking - Test")
    print("=" * 60)

    thinking = VisibleThinking(detail_level="verbose")

    # Simulate thinking about a question
    print("\n--- Thinking about a coding question ---")

    thinking.start_thinking("How to write a Python decorator?")
    thinking.use_template("question")
    thinking.think("Decorators wrap functions...", ThoughtType.RECALLING, source="memory")
    thinking.think("Should I show a simple example first?", ThoughtType.CONSIDERING)
    thinking.think("Yes, start simple, then build up", ThoughtType.DECIDING, confidence=0.9)

    print(thinking.get_formatted_thinking())

    result = thinking.conclude("Explain with a basic example, then show advanced usage")
    print(f"\nConclusion: {result.conclusion}")

    # Test prefix generation
    print("\n--- Thinking prefixes ---")
    queries = [
        "What is a closure in Python?",
        "Help me fix this bug",
        "Why does this happen?",
        "Create a function to sort a list",
        "Do you remember what we discussed?",
    ]

    for q in queries:
        prefix = thinking.generate_thinking_prefix(q)
        print(f"  '{q[:30]}...' -> {prefix}")

    # Introspection
    print("\n--- Introspection ---")
    intro = thinking.introspect()
    for k, v in intro.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Test complete!")
