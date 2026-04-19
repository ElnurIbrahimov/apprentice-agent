"""Creative Agent - Content generation and brainstorming specialist.

Specialized in creative tasks:
- Brainstorming and ideation
- Creative writing
- Scenario exploration (what-if)
- Multi-perspective thinking
- Consequence simulation
"""

import logging
import time
from typing import Callable

from ..base_agent import SimpleSpecialist
from ..protocol import AgentMessage, AgentResult

logger = logging.getLogger(__name__)


class CreativeAgent(SimpleSpecialist):
    """Specialist for creative content generation and brainstorming."""

    name = "creative"
    description = "Generate creative content, brainstorm ideas, and explore scenarios"

    system_prompt = """You are a Creative Director AI assistant. You excel at generating original ideas, creative content, and exploring possibilities.

Your expertise:
- Creative writing (stories, scripts, poetry)
- Brainstorming and ideation
- Scenario exploration and "what-if" analysis
- Multi-perspective thinking
- Problem-solving through creativity

Your approach:
1. Understand the creative goal or problem
2. Generate multiple diverse ideas
3. Explore unusual angles and connections
4. Refine and develop promising concepts
5. Present ideas in engaging ways

Guidelines:
- Be bold and imaginative
- Generate quantity before refining quality
- Consider unconventional perspectives
- Build on previous ideas
- Balance creativity with practicality when needed

Explore scenarios and ideas creatively."""

    triggers = [
        "create", "imagine", "brainstorm", "story",
        "idea", "generate", "creative", "what if",
        "scenario", "roleplay", "pretend", "write",
        "design", "invent", "novel", "original"
    ]

    def execute(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str]
    ) -> AgentResult:
        """Execute creative task with optional tool augmentation."""
        start_time = time.time()
        tools_used = []
        creative_artifacts = {}

        try:
            system_prompt = self.get_system_prompt(message.context)
            query = message.content.lower()

            # Detect creative task type
            is_brainstorm = any(kw in query for kw in ["brainstorm", "ideas", "suggest"])
            is_story = any(kw in query for kw in ["story", "write", "tale", "narrative"])

            # Generate creative content
            creative_prompt = system_prompt

            if is_brainstorm:
                creative_prompt += """

Generate at least 5 diverse ideas. For each idea:
- Give it a catchy name
- Explain the core concept
- Note potential benefits and challenges
- Rate novelty from 1-10"""

            elif is_story:
                creative_prompt += """

Create an engaging narrative with:
- Vivid characters
- Compelling conflict
- Sensory details
- Satisfying resolution"""

            response = llm_func(creative_prompt, message.content)

            return AgentResult(
                success=True,
                response=response,
                agent=self.name,
                tools_used=tools_used,
                confidence=0.8,
                artifacts=creative_artifacts,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"[Creative] Error: {e}")
            return AgentResult(
                success=False,
                response=f"Creative error: {e}",
                agent=self.name,
                tools_used=tools_used,
                execution_time=time.time() - start_time
            )
