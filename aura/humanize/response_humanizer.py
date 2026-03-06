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

    # Natural openers by tone - GENUINE, not corporate
    OPENERS = {
        ResponseTone.WARM: [
            "Oh! ", "So, ", "Okay, ", "Right, ", "Hmm, "
        ],
        ResponseTone.PROFESSIONAL: [
            "Right, ", "So, ", "Here's the thing: ", "", ""
        ],
        ResponseTone.CASUAL: [
            "Oh, ", "So, ", "Well, ", "Okay so ", "Hmm, "
        ],
        ResponseTone.ENTHUSIASTIC: [
            "Ooh! ", "Oh! ", "Yes! ", "Okay okay, ",
            "Oh man, "
        ],
        ResponseTone.EMPATHETIC: [
            "Hey, ", "Look, ", "I get it. ", "", ""
        ],
        ResponseTone.THOUGHTFUL: [
            "Hmm, ", "So... ", "Let me think... ", "Okay, ", ""
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

    # Natural closers by tone - GENUINE, conversational
    CLOSERS = {
        ResponseTone.WARM: [
            "", " Make sense?", " Let me know!", ""
        ],
        ResponseTone.PROFESSIONAL: [
            "", "", ""
        ],
        ResponseTone.CASUAL: [
            " Make sense?", "", " Yeah?",
            " Let me know!", ""
        ],
        ResponseTone.ENTHUSIASTIC: [
            " Right?!", " Pretty cool, huh?", ""
        ],
        ResponseTone.EMPATHETIC: [
            "", " I'm here.", ""
        ],
        ResponseTone.THOUGHTFUL: [
            " ...if that makes sense?", "", ""
        ],
        ResponseTone.DIRECT: [
            "", "", ""
        ]
    }

    # Acknowledgment phrases for questions - natural, not robotic
    ACKNOWLEDGMENTS = {
        "how": ["So, ", "Okay so ", ""],
        "what": ["It's ", "That's ", "Oh, it's "],
        "why": ["So basically, ", "It's because ", "Well, "],
        "can": ["Yeah! ", "Yep! ", "Sure! "],
        "should": ["I'd say ", "Probably ", ""],
        "is": ["Yeah, ", "Yep, ", "It is! "],
        "does": ["It does! ", "Yeah, it ", "Yep! "],
    }

    # Spontaneous expressions for genuine reactions
    SPONTANEOUS = {
        "surprise": ["Oh!", "Whoa!", "Huh!", "Wait,"],
        "thinking": ["Hmm...", "Let me see...", "So...", "Okay..."],
        "agreement": ["Yeah!", "Right!", "Exactly!", "Yes!"],
        "empathy": ["Oof.", "Ugh.", "Ah.", "Oh no."],
        "excitement": ["Ooh!", "Nice!", "YES!", "Oh man!"],
    }

    # Robotic patterns to replace with genuine language
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
        (r"However, ", ["But ", "Though ", "That said, "]),
        (r"Therefore, ", ["So ", "That's why ", ""]),
        (r"Furthermore, ", ["Also, ", "Plus, ", "And "]),
        (r"In addition, ", ["Also, ", "Plus, ", ""]),
        (r"In conclusion, ", ["So ", "Basically, ", ""]),
        (r"Please note that ", ["Just so you know, ", "Oh, ", ""]),
        # Remove corporate speak
        (r"I would be happy to ", ["I can ", "Sure, I'll ", ""]),
        (r"I am happy to ", ["Sure! ", "Yeah, ", ""]),
        (r"Certainly! ", ["Sure! ", "Yeah! ", ""]),
        (r"Absolutely! ", ["Yeah! ", "Yep! ", "For sure! "]),
        (r"That's great news! ", ["Oh nice! ", "Awesome! ", ""]),
        (r"I'm sorry to hear that", ["That sucks", "Oof", "That's rough"]),
        (r"Congratulations! ", ["Nice! ", "Congrats! ", "Awesome! "]),
        (r"I understand your frustration", ["Yeah, that's frustrating", "Ugh, I get it", ""]),
        (r"I appreciate your patience", ["Thanks for waiting", "Sorry about the wait", ""]),
        (r"Please feel free to", ["You can ", "Go ahead and ", ""]),
        (r"Don't hesitate to", ["Just ", "Feel free to ", ""]),
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

        # Don't add openers to very short responses (looks weird)
        if len(text) < 30:
            return text

        # Don't add openers to generic acknowledgment phrases
        generic_responses = ["got it", "i hear you", "okay", "ok", "sure", "yes",
                           "no", "thanks", "thank you", "alright", "noted"]
        text_lower = text.lower().strip().rstrip("!?.")
        if text_lower in generic_responses:
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

        # Don't add opener if text already starts with a similar phrase
        text_start = text[:20].lower()
        opener_word = opener.strip().lower().rstrip("!,.")
        if opener_word and opener_word in text_start:
            return text

        return opener + text

    def _add_closer(self, text: str, tone: ResponseTone) -> str:
        """Add an appropriate closer based on tone."""
        if random.random() > self.personality_level * 0.7:
            return text

        # Don't add closers to very short responses (looks weird)
        if len(text) < 30:
            return text

        # Don't add closers to generic responses
        generic_responses = ["got it", "i hear you", "okay", "ok", "sure", "yes",
                           "no", "thanks", "thank you", "alright", "noted"]
        text_lower = text.lower().strip().rstrip("!?.")
        if text_lower in generic_responses:
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

    def _add_genuine_reaction(self, text: str, query: str) -> str:
        """Add genuine emotional reactions based on query context."""
        if not query or random.random() > self.personality_level:
            return text

        query_lower = query.lower()

        # Success/excitement detection
        success_words = ["got the job", "passed", "won", "made it", "finally", "worked"]
        if any(w in query_lower for w in success_words):
            prefix = random.choice(["Wait, REALLY?! ", "NO WAY! ", "Oh my god! ", "YES!! "])
            return prefix + text

        # Struggle detection
        struggle_words = ["struggling", "frustrated", "stuck", "stressed", "anxious"]
        if any(w in query_lower for w in struggle_words):
            prefix = random.choice(["Hey... ", "Oof. ", "I hear you. ", ""])
            return prefix + text

        # Bad news detection
        bad_words = ["failed", "rejected", "didn't get", "lost"]
        if any(w in query_lower for w in bad_words):
            prefix = random.choice(["Oh no... ", "Ugh, that sucks. ", "I'm sorry. ", ""])
            return prefix + text

        # Question surprise
        if "?" in query and random.random() < 0.2:
            prefix = random.choice(self.SPONTANEOUS.get("thinking", [""]))
            if prefix:
                return prefix + " " + text

        return text

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

        # Add genuine emotional reactions based on context
        before = result
        result = self._add_genuine_reaction(result, query)
        if result != before:
            modifications.append("genuine_reaction")

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
