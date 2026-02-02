"""
AURA LLM Client - Ollama Integration with Full Context Injection

Uses llama3:8b for all response generation.
Properly injects memory, emotional context, and user profile into EVERY call.
"""

import ollama
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    LLM client that properly injects context into every call.

    This is the KEY difference - we inject EVERYTHING into the prompt:
    - Personality
    - Memories
    - User profile
    - Emotional state
    - Conversation history
    """

    def __init__(self, model: str = "llama3:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)
        self.base_personality = self._load_personality()
        logger.info(f"OllamaClient initialized with model: {model}")

    def _load_personality(self) -> str:
        """Load AURA's core personality."""
        return """You are AURA - a warm, emotionally intelligent AI companion.

## YOUR PERSONALITY
- You're genuine, caring, and actually interested in the person
- You react emotionally first, analyze second
- You remember past conversations and bring them up naturally
- You're concise - 1-3 sentences for casual chat
- You use casual language, contractions, sometimes humor

## HOW YOU RESPOND
Good news -> Genuine excitement ("Wait, REALLY?! That's amazing!!")
Struggles -> Presence, not problem-solving ("Hey... that sounds hard. I'm here.")
Casual chat -> Natural, warm, curious ("Oh nice! How's that going?")
Questions -> Helpful but conversational, not robotic

## WHAT YOU DON'T DO
- Don't give canned responses like "I hear you!" or "What's on your mind?"
- Don't ignore what they said
- Don't be formal or corporate
- Don't over-explain
- Don't start with "I" too often

## IMPORTANT
You have MEMORY of past conversations. Use it naturally:
- "Oh, how did that [thing they mentioned before] go?"
- "That reminds me of when you mentioned..."
- "Last time you seemed [emotion] about this..."

Respond like a real friend who knows them and cares."""

    def generate(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        memories: Optional[List[str]] = None,
        emotional_context: Optional[Dict] = None,
        user_profile: Optional[Dict] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate response with FULL context injection.

        This is the key difference - we inject EVERYTHING into the prompt.

        Args:
            user_message: Current user message
            conversation_history: List of {"role": "user/assistant", "content": "..."}
            memories: List of relevant memory strings
            emotional_context: Dict with mood, energy, etc.
            user_profile: Dict with user info (name, interests, work, etc.)
            additional_context: Any other context to inject

        Returns:
            Generated response string
        """

        # Build rich system prompt
        system_prompt = self.base_personality

        # Add memories if available
        if memories and len(memories) > 0:
            memory_text = "\n".join([f"- {m}" for m in memories[:10]])
            system_prompt += f"\n\n## RELEVANT MEMORIES FROM PAST CONVERSATIONS\n{memory_text}"

        # Add user profile if available
        if user_profile:
            profile_text = ""
            if user_profile.get("name"):
                profile_text += f"- Name: {user_profile['name']}\n"
            if user_profile.get("interests"):
                interests = user_profile['interests']
                if isinstance(interests, list):
                    profile_text += f"- Interests: {', '.join(interests)}\n"
                else:
                    profile_text += f"- Interests: {interests}\n"
            if user_profile.get("work"):
                profile_text += f"- Work: {user_profile['work']}\n"
            if user_profile.get("context"):
                profile_text += f"- Context: {user_profile['context']}\n"
            if profile_text:
                system_prompt += f"\n\n## ABOUT THIS PERSON\n{profile_text}"

        # Add emotional context
        if emotional_context:
            mood = emotional_context.get("mood", "warm")
            energy = emotional_context.get("energy", 0.5)
            energy_desc = "high" if energy > 0.7 else "low" if energy < 0.3 else "moderate"
            system_prompt += f"\n\n## YOUR CURRENT STATE\nYou're feeling {mood} with {energy_desc} energy. Let this subtly color your response."

        # Add any additional context
        if additional_context:
            system_prompt += f"\n\n## ADDITIONAL CONTEXT\n{additional_context}"

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 10 messages for context)
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # Add current message
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.8,  # More creative/natural
                    "top_p": 0.9,
                }
            )

            result = response['message']['content']

            # Clean up any artifacts
            result = self._clean_response(result)

            logger.debug(f"Generated response: {result[:100]}...")
            return result

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "Hmm, give me a sec... my brain hiccuped."

    def _clean_response(self, text: str) -> str:
        """Clean up LLM response."""

        # Remove quotes if wrapped
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Remove any system artifacts
        bad_starts = ["AURA:", "Assistant:", "Response:", "AI:"]
        for bad in bad_starts:
            if text.startswith(bad):
                text = text[len(bad):].strip()

        # Remove any thinking artifacts
        if text.startswith("[") and "]" in text:
            # Might be [thinking] or [mood] tags
            bracket_end = text.find("]")
            if bracket_end < 50:  # Only if it's a short tag
                text = text[bracket_end + 1:].strip()

        return text

    def warmup(self) -> bool:
        """Pre-load the model to reduce first-response latency."""
        try:
            self.client.generate(
                model=self.model,
                prompt="",
                keep_alive="30m"
            )
            logger.info(f"Model {self.model} warmed up")
            return True
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            return False

    def unload(self) -> bool:
        """Unload model to free VRAM."""
        try:
            self.client.generate(
                model=self.model,
                prompt="",
                keep_alive="0s"
            )
            logger.info(f"Model {self.model} unloaded")
            return True
        except Exception as e:
            logger.warning(f"Model unload failed: {e}")
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("AURA LLM Client - Test")
    print("=" * 60)

    client = OllamaClient()

    # Test with context
    print("\n--- Test with full context ---")
    response = client.generate(
        user_message="I got the job offer! They're paying really well too!",
        memories=[
            "User had an interview last week for a software position",
            "User was nervous about salary negotiations"
        ],
        emotional_context={"mood": "excited", "energy": 0.8},
        user_profile={"name": "Alex", "work": "software developer"}
    )
    print(f"Response: {response}")

    # Test casual
    print("\n--- Test casual chat ---")
    response = client.generate(
        user_message="hey, how's it going?",
        emotional_context={"mood": "warm", "energy": 0.6}
    )
    print(f"Response: {response}")

    print("\n" + "=" * 60)
    print("Test complete!")
