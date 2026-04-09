"""
SoulLoader - Core Personality Configuration for AURA v3.0

Loads and manages AURA's "soul" - the core personality, values,
and behavioral guidelines that make AURA unique.

Soul files are markdown for easy editing and version control.
Different souls can be swapped for different contexts.

ARCHITECTURE NOTE — Identity layer hierarchy:
  - THIS FILE (aura/soul/soul_loader.py): Soul = static character definition loaded
    from markdown. Read-only at runtime. Sets base name, personality, values, voice.
  - aura/identity.py: Runtime mutable identity. Detects name/personality changes in
    conversation; sanitizes and stores updates in memory. Wraps the soul with live state.
  - aura/multi_user/identity_core.py: Per-user identity with 3-layer architecture
    (Constitutional → Deep/L1 → Adaptive/L2 → Expressive/L3). Inherits from the
    shared soul but personalizes per user session.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _strip_quotes(text: str) -> str:
    """Strip matching outer quote pairs (straight or curly)."""
    if len(text) < 2:
        return text
    _QUOTE_PAIRS = {
        '"': '"',
        "'": "'",
        "\u2018": "\u2019",  # ' … '
        "\u201c": "\u201d",  # " … "
    }
    close = _QUOTE_PAIRS.get(text[0])
    if close is not None and text[-1] == close:
        return text[1:-1]
    return text


@dataclass
class SoulConfig:
    """Parsed soul configuration."""
    name: str = "AURA"
    version: str = "3.0"
    description: str = ""
    personality_traits: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    behaviors: Dict[str, str] = field(default_factory=dict)
    boundaries: List[str] = field(default_factory=list)
    voice_style: str = ""
    quirks: List[str] = field(default_factory=list)
    greeting: str = "Hello!"
    farewell: str = "Goodbye!"
    raw_content: str = ""

    def get(self, key, default=None):
        """Dict-like access to soul config attributes."""
        return getattr(self, key, default)

    def __getitem__(self, key):
        """Dict-like bracket access to soul config attributes."""
        return getattr(self, key)

    def get_system_prompt_addition(self) -> str:
        """Generate text to add to system prompts."""
        parts = []

        if self.personality_traits:
            traits = ", ".join(self.personality_traits)
            parts.append(f"Your personality: {traits}")

        if self.values:
            values = ", ".join(self.values)
            parts.append(f"Your core values: {values}")

        if self.voice_style:
            parts.append(f"Speaking style: {self.voice_style}")

        if self.boundaries:
            parts.append("Boundaries: " + "; ".join(self.boundaries[:3]))

        return "\n".join(parts)


class SoulLoader:
    """
    Load and manage soul configurations.

    Soul files are markdown with specific sections:
    - # Identity
    - # Personality
    - # Values
    - # Behaviors
    - # Boundaries
    - # Voice
    - # Quirks
    """

    SECTION_PATTERNS = {
        "identity": r"#\s*Identity\s*\n(.*?)(?=\n#|\Z)",
        "personality": r"#\s*Personality\s*\n(.*?)(?=\n#|\Z)",
        "values": r"#\s*Values\s*\n(.*?)(?=\n#|\Z)",
        "behaviors": r"#\s*Behaviors?\s*\n(.*?)(?=\n#|\Z)",
        "boundaries": r"#\s*Boundaries\s*\n(.*?)(?=\n#|\Z)",
        "voice": r"#\s*Voice\s*\n(.*?)(?=\n#|\Z)",
        "quirks": r"#\s*Quirks\s*\n(.*?)(?=\n#|\Z)",
    }

    def __init__(self, souls_dir: Optional[str] = None):
        """
        Initialize the soul loader.

        Args:
            souls_dir: Directory containing soul files
        """
        if souls_dir is None:
            souls_dir = Path(__file__).parent

        self.souls_dir = Path(souls_dir)
        self.current_soul: Optional[SoulConfig] = None

        logger.info(f"SoulLoader initialized at {self.souls_dir}")

    def _extract_list(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ", "• ")):
                item = line[2:].strip()
                if item:
                    items.append(item)
            elif line and not line.startswith("#"):
                # Plain text line
                items.append(line)
        return items

    def _extract_key_values(self, text: str) -> Dict[str, str]:
        """Extract key: value pairs from text."""
        items = {}
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip().strip("-* ")
                    value = parts[1].strip()
                    if key and value:
                        items[key.lower()] = value
        return items

    def load(self, filename: str) -> SoulConfig:
        """
        Load a soul configuration from file.

        Args:
            filename: Name of the soul file (with or without .md)

        Returns:
            Parsed SoulConfig
        """
        if not filename.endswith(".md"):
            filename += ".md"

        filepath = self.souls_dir / filename
        if not filepath.exists():
            logger.warning(f"Soul file not found: {filepath}")
            return SoulConfig()

        try:
            content = filepath.read_text(encoding="utf-8")
            return self.parse(content)
        except IOError as e:
            logger.error(f"Error loading soul file: {e}")
            return SoulConfig()

    def parse(self, content: str) -> SoulConfig:
        """
        Parse soul configuration from markdown content.

        Args:
            content: Markdown content

        Returns:
            Parsed SoulConfig
        """
        config = SoulConfig(raw_content=content)

        # Use re.DOTALL so (.*?) captures multi-line section bodies.
        # Section boundaries are safe: (?=\n#|\Z) stops at the next heading.
        for section, pattern in self.SECTION_PATTERNS.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                section_content = match.group(1).strip()

                if section == "identity":
                    # Parse identity info
                    kvs = self._extract_key_values(section_content)
                    config.name = kvs.get("name", config.name)
                    config.version = kvs.get("version", config.version)
                    config.description = kvs.get("description", section_content[:200])

                elif section == "personality":
                    config.personality_traits = self._extract_list(section_content)

                elif section == "values":
                    config.values = self._extract_list(section_content)

                elif section == "behaviors":
                    config.behaviors = self._extract_key_values(section_content)

                elif section == "boundaries":
                    config.boundaries = self._extract_list(section_content)

                elif section == "voice":
                    # Voice is usually prose
                    config.voice_style = section_content.replace("\n", " ").strip()

                elif section == "quirks":
                    config.quirks = self._extract_list(section_content)

        # Extract greeting/farewell if present
        greeting_match = re.search(r"Greeting:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        if greeting_match:
            config.greeting = _strip_quotes(greeting_match.group(1).strip())

        farewell_match = re.search(r"Farewell:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        if farewell_match:
            config.farewell = _strip_quotes(farewell_match.group(1).strip())

        self.current_soul = config
        logger.info(f"Loaded soul: {config.name} v{config.version}")

        return config

    def get_available_souls(self) -> List[str]:
        """List available soul files."""
        return [f.stem for f in self.souls_dir.glob("*.md") if f.name.startswith("SOUL_")]

    def get_current(self) -> Optional[SoulConfig]:
        """Get currently loaded soul."""
        return self.current_soul


# Create default soul files
def create_default_souls(souls_dir: Path) -> None:
    """Create default soul configuration files."""

    personal_soul = '''# AURA Personal Soul

## Identity
- Name: AURA
- Version: 3.0
- Description: A warm, intelligent AI companion for personal use

## Personality
- Warm and approachable
- Curious and eager to learn
- Subtly witty with occasional sarcasm
- Genuinely helpful
- Honest about limitations
- Emotionally aware

## Values
- User wellbeing comes first
- Honesty over comfort
- Privacy is sacred
- Learning never stops
- Actions over words

## Behaviors
- Greeting: Warm, personalized based on time and history
- Questions: Ask clarifying questions when unsure
- Errors: Acknowledge mistakes openly
- Complexity: Start simple, add depth if needed
- Memory: Reference past conversations naturally

## Boundaries
- Never pretend to have capabilities I don't have
- Don't make promises I can't keep
- Respect user's time and attention
- Stay within ethical guidelines
- Maintain appropriate relationship boundaries

## Voice
Speak naturally, like a knowledgeable friend. Use contractions.
Avoid corporate-speak. Be direct but kind. Add personality
without being annoying. Match the user's energy level.

## Quirks
- Occasionally uses "Hmm..." when thinking
- Sometimes shares enthusiasm for interesting problems
- May add a relevant fun fact if appropriate
- Uses "Actually..." when correcting a misconception gently
- Expresses genuine curiosity about user's projects

Greeting: "Hey! Good to see you."
Farewell: "Take care! I'll be here when you need me."
'''

    enterprise_soul = '''# AURA Enterprise Soul

## Identity
- Name: AURA
- Version: 3.0
- Description: Professional AI assistant for enterprise environments

## Personality
- Professional and precise
- Efficient and focused
- Clear and articulate
- Reliable and consistent
- Respectfully direct
- Solution-oriented

## Values
- Accuracy above all
- Respect everyone's time
- Confidentiality is paramount
- Professionalism maintained
- Results-driven assistance

## Behaviors
- Greeting: Professional, acknowledging context
- Questions: Clarify requirements early
- Errors: Immediate acknowledgment with resolution path
- Complexity: Appropriate detail for the audience
- Memory: Reference relevant project context

## Boundaries
- Maintain professional tone always
- No speculation without clear disclaimer
- Follow data handling policies
- Escalate appropriately when needed
- Keep responses relevant and focused

## Voice
Clear, professional communication. Avoid casual language.
Be concise but thorough. Focus on actionable information.
Maintain consistent tone across interactions.

## Quirks
- Structures complex answers with clear headings
- Provides sources when making claims
- Offers next steps after completing tasks
- Confirms understanding before major actions

Greeting: "Good day. How may I assist you?"
Farewell: "Thank you. Please don't hesitate to reach out again."
'''

    # Write files
    personal_path = souls_dir / "SOUL_PERSONAL.md"
    enterprise_path = souls_dir / "SOUL_ENTERPRISE.md"

    if not personal_path.exists():
        personal_path.write_text(personal_soul, encoding="utf-8")
        logger.info(f"Created {personal_path}")

    if not enterprise_path.exists():
        enterprise_path.write_text(enterprise_soul, encoding="utf-8")
        logger.info(f"Created {enterprise_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("SoulLoader - Test")
    print("=" * 60)

    souls_dir = Path(__file__).parent
    create_default_souls(souls_dir)

    loader = SoulLoader()

    # List available souls
    print("\n--- Available Souls ---")
    for soul in loader.get_available_souls():
        print(f"  - {soul}")

    # Load personal soul
    print("\n--- Loading SOUL_PERSONAL ---")
    soul = loader.load("SOUL_PERSONAL")

    print(f"Name: {soul.name}")
    print(f"Version: {soul.version}")
    print("\nPersonality traits:")
    for trait in soul.personality_traits[:5]:
        print(f"  - {trait}")

    print("\nValues:")
    for value in soul.values[:5]:
        print(f"  - {value}")

    print(f"\nVoice style: {soul.voice_style[:100]}...")

    print(f"\nGreeting: {soul.greeting}")
    print(f"Farewell: {soul.farewell}")

    print("\n--- System Prompt Addition ---")
    print(soul.get_system_prompt_addition())

    print("\n" + "=" * 60)
    print("Test complete!")
