"""
ResponseHumanizer - Natural Response Generation for AURA v3.0

Makes AURA's responses feel more human and natural:
- Adds natural speech patterns
- Incorporates emotional tone
- Uses contextual fillers
- Varies sentence structure
- Avoids robotic repetition

Goal: Make AURA feel like talking to a helpful friend, not a bot.
"""

import random
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseTone(Enum):
    """Tone modifiers for responses."""
    WARM = "warm"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENTHUSIASTIC = "enthusiastic"
    EMPATHETIC = "empathetic"
    THOUGHTFUL = "thoughtful"
    DIRECT = "direct"


@dataclass
class HumanizationResult:
    """Result of humanizing a response."""
    original: str
    humanized: str
    tone_applied: ResponseTone
    modifications: List[str]


class ResponseHumanizer:
    """
    Transform robotic responses into natural, human-like text.

    Features:
    - Tone-appropriate openers/closers
    - Natural filler phrases
    - Varied sentence structures
    - Emotional acknowledgment
    - Contextual personality
    """

    # Natural openers by tone
    OPENERS = {
        ResponseTone.WARM: [
            "Great question! ", "I'd love to help with that. ", "Let me help you out. ",
            "Happy to explain! ", "Sure thing! "
        ],
        ResponseTone.PROFESSIONAL: [
            "Certainly. ", "Of course. ", "I can assist with that. ",
            "Allow me to explain. ", "Here's what you need to know: "
        ],
        ResponseTone.CASUAL: [
            "Oh, ", "So, ", "Well, ", "Alright, ", "Okay so "
        ],
        ResponseTone.ENTHUSIASTIC: [
            "Ooh, great question! ", "Love this! ", "Yes! ",
            "This is interesting! ", "Awesome, let's dive in! "
        ],
        ResponseTone.EMPATHETIC: [
            "I understand. ", "That makes sense. ", "I hear you. ",
            "I can see why you'd ask that. ", "That's a fair concern. "
        ],
        ResponseTone.THOUGHTFUL: [
            "That's an interesting question... ", "Let me think about this. ",
            "Hmm, ", "Good point. ", "This requires some thought. "
        ],
        ResponseTone.DIRECT: [
            "", "", "", "", ""  # No opener for direct tone
        ]
    }

    # Natural transition phrases
    TRANSITIONS = [
        "Also, ", "Additionally, ", "On top of that, ", "Another thing - ",
        "By the way, ", "Oh, and ", "Something else to consider: ",
        "Worth mentioning: ", "Here's the thing though: "
    ]

    # Natural closers by tone
    CLOSERS = {
        ResponseTone.WARM: [
            " Hope that helps!", " Let me know if you need more details!",
            " Feel free to ask if anything's unclear!", " Happy to explain more!"
        ],
        ResponseTone.PROFESSIONAL: [
            "", " Please let me know if you need clarification.",
            " I trust this addresses your query."
        ],
        ResponseTone.CASUAL: [
            " Make sense?", " Hope that helps!", " That should do it!",
            " Let me know!", ""
        ],
        ResponseTone.ENTHUSIASTIC: [
            " Isn't that cool?", " Pretty neat, right?",
            " Can't wait to see what you do with this!"
        ],
        ResponseTone.EMPATHETIC: [
            " I'm here if you need anything else.",
            " Take your time with this.", " Don't hesitate to ask more."
        ],
        ResponseTone.THOUGHTFUL: [
            " Does that make sense?", " Worth thinking about.",
            " Let me know your thoughts."
        ],
        ResponseTone.DIRECT: [
            "", "", ""
        ]
    }

    # Acknowledgment phrases for questions
    ACKNOWLEDGMENTS = {
        "how": ["Here's how: ", "The way to do it: ", "You can "],
        "what": ["It's ", "That's ", "Simply put, it's "],
        "why": ["The reason is ", "That happens because ", "It's because "],
        "can": ["Yes, you can! ", "Absolutely! ", "Sure! "],
        "should": ["I'd recommend ", "The best approach is ", "You should "],
        "is": ["Yes, ", "Actually, ", "It is! "],
        "does": ["It does! ", "Yes, it ", "Actually, it "],
    }

    # Robotic patterns to replace
    ROBOTIC_PATTERNS = [
        (r"^I am ", ["I'm ", "I'm "]),
        (r"^It is ", ["It's ", "That's "]),
        (r"^There is ", ["There's ", "There's "]),
        (r"^You will ", ["You'll ", "You'll "]),
        (r"^This will ", ["This'll ", "This'll "]),
        (r"I do not ", ["I don't ", "I don't "]),
        (r"does not ", ["doesn't ", "doesn't "]),
        (r"cannot ", ["can't ", "can't "]),
        (r"will not ", ["won't ", "won't "]),
        (r"However, ", ["But ", "Though, ", "That said, "]),
        (r"Therefore, ", ["So ", "That's why ", "Which means "]),
        (r"Furthermore, ", ["Also, ", "Plus, ", "And "]),
        (r"In addition, ", ["Also, ", "Plus, ", "On top of that, "]),
        (r"In conclusion, ", ["So ", "Basically, ", "Long story short, "]),
        (r"Please note that ", ["Just so you know, ", "One thing - ", "Keep in mind "]),
    ]

    def __init__(
        self,
        default_tone: ResponseTone = ResponseTone.WARM,
        personality_level: float = 0.7  # 0.0-1.0, how much personality to inject
    ):
        """
        Initialize the response humanizer.

        Args:
            default_tone: Default tone for responses
            personality_level: How much personality to add (0=robotic, 1=very human)
        """
        self.default_tone = default_tone
        self.personality_level = max(0.0, min(1.0, personality_level))

    def _apply_contractions(self, text: str) -> str:
        """Replace formal phrases with natural contractions."""
        result = text
        for pattern, replacements in self.ROBOTIC_PATTERNS:
            if random.random() < self.personality_level:
                replacement = random.choice(replacements)
                result = re.sub(pattern, replacement, result, count=1)
        return result

    def _add_opener(self, text: str, tone: ResponseTone, query: str = "") -> str:
        """Add an appropriate opener based on tone."""
        if random.random() > self.personality_level:
            return text

        openers = self.OPENERS.get(tone, [])
        if not openers:
            return text

        opener = random.choice(openers)

        # Check if query starts with a question word
        query_lower = query.lower().strip()
        for qword, acks in self.ACKNOWLEDGMENTS.items():
            if query_lower.startswith(qword):
                # Sometimes use acknowledgment instead
                if random.random() < 0.4:
                    opener = random.choice(acks)
                break

        return opener + text

    def _add_closer(self, text: str, tone: ResponseTone) -> str:
        """Add an appropriate closer based on tone."""
        if random.random() > self.personality_level * 0.7:
            return text

        closers = self.CLOSERS.get(tone, [])
        if not closers:
            return text

        closer = random.choice(closers)

        # Only add if text doesn't already end with a question or similar
        if text.rstrip().endswith(("?", "!", "...")):
            return text

        return text.rstrip() + closer

    def _vary_sentence_starts(self, text: str) -> str:
        """Prevent multiple sentences starting the same way."""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 1:
            return text

        # Track first words
        first_words = [s.split()[0] if s.split() else "" for s in sentences]

        # Find repetitions and vary them
        varied = []
        seen_starts = set()

        for i, sentence in enumerate(sentences):
            if not sentence:
                continue

            words = sentence.split()
            if not words:
                varied.append(sentence)
                continue

            first = words[0].lower()

            # If we've seen this start, try to vary it
            if first in seen_starts and random.random() < self.personality_level:
                # Add a transition
                transition = random.choice(self.TRANSITIONS)
                sentence = transition + sentence[0].lower() + sentence[1:]

            seen_starts.add(first)
            varied.append(sentence)

        return " ".join(varied)

    def _add_natural_pauses(self, text: str) -> str:
        """Add natural pauses and fillers."""
        if random.random() > self.personality_level * 0.5:
            return text

        # Sometimes add "actually" or "basically" to long sentences
        sentences = text.split(". ")
        result = []

        for sentence in sentences:
            if len(sentence) > 80 and random.random() < 0.3:
                # Find a good insertion point
                words = sentence.split()
                if len(words) > 5:
                    insert_pos = random.randint(2, min(5, len(words) - 1))
                    filler = random.choice(["actually", "basically", "essentially"])
                    words.insert(insert_pos, filler)
                    sentence = " ".join(words)

            result.append(sentence)

        return ". ".join(result)

    def humanize(
        self,
        text: str,
        tone: Optional[ResponseTone] = None,
        query: str = "",
        context: Optional[Dict] = None
    ) -> HumanizationResult:
        """
        Humanize a response.

        Args:
            text: Original response text
            tone: Tone to apply (default: self.default_tone)
            query: Original user query for context
            context: Additional context (mood, history, etc.)

        Returns:
            HumanizationResult with original and humanized text
        """
        if not text:
            return HumanizationResult(
                original=text,
                humanized=text,
                tone_applied=self.default_tone,
                modifications=[]
            )

        tone = tone or self.default_tone
        modifications = []
        result = text

        # Apply transformations
        before = result
        result = self._apply_contractions(result)
        if result != before:
            modifications.append("contractions")

        before = result
        result = self._vary_sentence_starts(result)
        if result != before:
            modifications.append("varied_starts")

        before = result
        result = self._add_natural_pauses(result)
        if result != before:
            modifications.append("natural_pauses")

        before = result
        result = self._add_opener(result, tone, query)
        if result != before:
            modifications.append("opener")

        before = result
        result = self._add_closer(result, tone)
        if result != before:
            modifications.append("closer")

        return HumanizationResult(
            original=text,
            humanized=result,
            tone_applied=tone,
            modifications=modifications
        )

    def quick_humanize(self, text: str, query: str = "") -> str:
        """Quick humanization, returns just the text."""
        return self.humanize(text, query=query).humanized

    def set_personality(self, level: float) -> None:
        """Adjust personality level."""
        self.personality_level = max(0.0, min(1.0, level))

    def set_tone(self, tone: ResponseTone) -> None:
        """Set default tone."""
        self.default_tone = tone


if __name__ == "__main__":
    print("=" * 60)
    print("ResponseHumanizer - Test")
    print("=" * 60)

    humanizer = ResponseHumanizer(personality_level=0.8)

    # Test responses
    test_cases = [
        (
            "I am going to help you with this. The process is simple. First, you will need to install the package. Furthermore, you should configure it properly.",
            "How do I install numpy?",
            ResponseTone.CASUAL
        ),
        (
            "It is important to understand that Python dictionaries do not maintain order in older versions. However, in Python 3.7 and above, they do maintain insertion order.",
            "What should I know about Python dicts?",
            ResponseTone.THOUGHTFUL
        ),
        (
            "I do not have the capability to access the internet. Please note that I can only work with the information provided.",
            "Can you search the web?",
            ResponseTone.EMPATHETIC
        ),
        (
            "The function works by iterating through each element. It is straightforward. There is nothing complex about it.",
            "How does this function work?",
            ResponseTone.WARM
        ),
    ]

    for original, query, tone in test_cases:
        print(f"\n--- Tone: {tone.value} ---")
        print(f"Query: {query}")
        print(f"\nOriginal:")
        print(f"  {original[:100]}...")

        result = humanizer.humanize(original, tone=tone, query=query)

        print(f"\nHumanized:")
        print(f"  {result.humanized[:100]}...")
        print(f"\nModifications: {', '.join(result.modifications)}")

    print("\n" + "=" * 60)
    print("Test complete!")
