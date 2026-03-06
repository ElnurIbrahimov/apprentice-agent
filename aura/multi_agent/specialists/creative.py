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
from typing import Any, Callable, Dict

from ..base_agent import ToolUsingSpecialist
from ..protocol import AgentMessage, AgentResult

logger = logging.getLogger(__name__)


class CreativeAgent(ToolUsingSpecialist):
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

When exploring scenarios, use: [TOOL: worldsim] scenario description
For self-critique, use: [TOOL: mirrormind] content to evaluate
For multi-perspective debate, use: [TOOL: cognitive_theater] topic"""

    tools = [
        "worldsim",
        "mirrormind",
        "cognitive_theater",
    ]

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
            is_whatif = "what if" in query or "scenario" in query
            is_brainstorm = any(kw in query for kw in ["brainstorm", "ideas", "suggest"])
            is_story = any(kw in query for kw in ["story", "write", "tale", "narrative"])
            is_debate = any(kw in query for kw in ["perspective", "debate", "argue", "both sides"])

            # Use WorldSim for "what-if" scenarios
            worldsim_result = None
            if is_whatif and "worldsim" in self._available_tools:
                logger.info("[Creative] Running WorldSim scenario...")
                worldsim_result = self._execute_tool("worldsim", message.content)
                tools_used.append("worldsim")
                if worldsim_result.get("success"):
                    creative_artifacts["worldsim"] = worldsim_result

            # Use CognitiveTheater for multi-perspective debate
            theater_result = None
            if is_debate and "cognitive_theater" in self._available_tools:
                logger.info("[Creative] Running CognitiveTheater debate...")
                theater_result = self._execute_tool("cognitive_theater", message.content)
                tools_used.append("cognitive_theater")
                if theater_result.get("success"):
                    creative_artifacts["debate"] = theater_result

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

            elif is_whatif and worldsim_result:
                creative_prompt += f"""

WorldSim Analysis:
{worldsim_result}

Build on this simulation to explore the scenario further."""

            elif is_debate and theater_result:
                creative_prompt += f"""

Multi-Perspective Debate:
{theater_result}

Synthesize these perspectives into a balanced conclusion."""

            response = llm_func(creative_prompt, message.content)

            # Optional: Use MirrorMind for self-critique
            if "mirrormind" in self._available_tools and len(response) > 200:
                logger.info("[Creative] Running MirrorMind self-critique...")
                critique = self._execute_tool("mirrormind", response[:1000])
                tools_used.append("mirrormind")
                if critique.get("success") and critique.get("suggestions"):
                    creative_artifacts["critique"] = critique
                    response += f"\n\n*Self-Critique: {critique.get('suggestions', '')}*"

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
