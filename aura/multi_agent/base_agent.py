"""Base class for specialist agents.

All specialist agents inherit from BaseSpecialist and implement
domain-specific logic while sharing common infrastructure.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .protocol import AgentMessage, AgentResult

logger = logging.getLogger(__name__)


class BaseSpecialist(ABC):
    """Base class for all specialist agents.

    Each specialist has:
    - A unique name and description
    - A system prompt that defines its personality/role
    - A list of tools it can access
    - Trigger keywords that activate it
    """

    # Subclasses should override these
    name: str = "base"
    description: str = "Base specialist agent"
    system_prompt: str = "You are a helpful assistant."
    tools: List[str] = []  # Tool names this agent can use
    triggers: List[str] = []  # Keywords that suggest this agent

    def __init__(self, tool_registry: Dict[str, Any]):
        """Initialize with access to shared tool registry.

        Args:
            tool_registry: Dict mapping tool names to tool instances
        """
        self.tool_registry = tool_registry
        self._available_tools = {
            name: tool for name, tool in tool_registry.items()
            if name in self.tools
        }
        logger.info(f"[{self.name}] Initialized with {len(self._available_tools)} tools")

    def can_handle(self, message: str) -> float:
        """Determine if this agent can handle the message.

        Args:
            message: The user's message

        Returns:
            Confidence score 0-1 (0 = can't handle, 1 = perfect match)
        """
        message_lower = message.lower()
        score = 0.0

        # Check for trigger keywords
        for trigger in self.triggers:
            if trigger.lower() in message_lower:
                score += 0.3

        # Check for tool-related keywords
        tool_keywords = self._get_tool_keywords()
        for keyword in tool_keywords:
            if keyword.lower() in message_lower:
                score += 0.2

        # Cap at 1.0
        return min(score, 1.0)

    def _get_tool_keywords(self) -> List[str]:
        """Get keywords from available tools."""
        keywords = []
        for tool in self._available_tools.values():
            if hasattr(tool, 'name'):
                keywords.append(tool.name)
            if hasattr(tool, 'description'):
                # Extract key words from description
                desc = getattr(tool, 'description', '')
                words = re.findall(r'\b\w{4,}\b', desc.lower())
                keywords.extend(words[:5])  # First 5 significant words
        return list(set(keywords))

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name if available to this agent."""
        return self._available_tools.get(name)

    def get_system_prompt(self, context: Dict[str, Any] | None = None) -> str:
        """Get the full system prompt with optional context injection.

        Args:
            context: Additional context to include

        Returns:
            Complete system prompt string
        """
        prompt = self.system_prompt

        # Add available tools info
        if self._available_tools:
            tool_info = "\n\nYou have access to these tools:\n"
            for name, tool in self._available_tools.items():
                desc = getattr(tool, 'description', 'No description')[:100]
                tool_info += f"- {name}: {desc}\n"
            prompt += tool_info

        # Add context if provided
        if context:
            if context.get("previous_response"):
                prompt += f"\n\nPrevious agent ({context.get('previous_agent', 'unknown')}) said:\n"
                prompt += context["previous_response"][:500]

            if context.get("artifacts"):
                prompt += f"\n\nAvailable artifacts: {list(context['artifacts'].keys())}"

        return prompt

    @abstractmethod
    def execute(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str]
    ) -> AgentResult:
        """Execute the agent's task.

        Args:
            message: The incoming message to process
            llm_func: Function to call LLM (system_prompt, user_message) -> response

        Returns:
            AgentResult with the response and metadata
        """
        pass

    def _execute_tool(self, tool_name: str, action: str) -> Dict[str, Any]:
        """Execute a tool action.

        Args:
            tool_name: Name of the tool to use
            action: Action string to pass to the tool

        Returns:
            Result dict from tool execution
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not available"}

        try:
            if hasattr(tool, 'execute'):
                return tool.execute(action)
            elif callable(tool):
                return tool(action)
            else:
                return {"success": False, "error": f"Tool '{tool_name}' not executable"}
        except Exception as e:
            logger.error(f"[{self.name}] Tool {tool_name} error: {e}")
            return {"success": False, "error": str(e)}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, str]]:
        """Parse tool calls from LLM response.

        Looks for patterns like:
        - [TOOL: tool_name] action
        - <tool>tool_name</tool> <action>action</action>
        - {"tool": "name", "action": "action"}

        Args:
            response: LLM response text

        Returns:
            List of {tool, action} dicts
        """
        tool_calls = []

        # Pattern 1: [TOOL: name] action
        pattern1 = r'\[TOOL:\s*(\w+)\]\s*(.+?)(?=\[TOOL:|$)'
        matches = re.findall(pattern1, response, re.DOTALL | re.IGNORECASE)
        for tool, action in matches:
            tool_calls.append({"tool": tool.strip(), "action": action.strip()})

        # Pattern 2: JSON-style
        import json
        json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
        for match in re.findall(json_pattern, response):
            try:
                parsed = json.loads(match)
                if "tool" in parsed:
                    tool_calls.append({
                        "tool": parsed["tool"],
                        "action": parsed.get("action", "")
                    })
            except json.JSONDecodeError:
                pass

        return tool_calls

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' tools={len(self._available_tools)}>"


class SimpleSpecialist(BaseSpecialist):
    """A simple specialist that just uses LLM without tools.

    Good for tasks that don't require external tools, like:
    - Explaining concepts
    - Summarizing text
    - Generating creative content
    """

    def execute(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str]
    ) -> AgentResult:
        """Execute by simply calling LLM with system prompt."""
        start_time = time.time()

        try:
            system_prompt = self.get_system_prompt(message.context)
            response = llm_func(system_prompt, message.content)

            return AgentResult(
                success=True,
                response=response,
                agent=self.name,
                tools_used=[],
                confidence=0.9,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"[{self.name}] Execution error: {e}")
            return AgentResult(
                success=False,
                response=f"Error: {e}",
                agent=self.name,
                execution_time=time.time() - start_time
            )


class ToolUsingSpecialist(BaseSpecialist):
    """A specialist that can use tools based on LLM decisions.

    The LLM decides which tools to call and the specialist
    executes them, then synthesizes a final response.
    """

    max_tool_calls: int = 3  # Maximum tool calls per execution

    def execute(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str]
    ) -> AgentResult:
        """Execute by letting LLM decide tool usage."""
        start_time = time.time()
        tools_used = []
        tool_results = []

        try:
            # First LLM call to decide on tools
            system_prompt = self.get_system_prompt(message.context)
            system_prompt += "\n\nTo use a tool, write: [TOOL: tool_name] action_description"
            system_prompt += "\nYou can use multiple tools. After using tools, provide your final answer."

            response = llm_func(system_prompt, message.content)

            # Parse and execute tool calls
            tool_calls = self._parse_tool_calls(response)
            for _i, call in enumerate(tool_calls[:self.max_tool_calls]):
                tool_name = call["tool"]
                action = call["action"]

                logger.info(f"[{self.name}] Executing tool: {tool_name}")
                result = self._execute_tool(tool_name, action)
                tools_used.append(tool_name)
                tool_results.append({
                    "tool": tool_name,
                    "action": action,
                    "result": result
                })

            # If tools were used, do a follow-up call to synthesize
            if tool_results:
                synthesis_prompt = system_prompt + "\n\nTool execution results:\n"
                for tr in tool_results:
                    synthesis_prompt += f"\n[{tr['tool']}]: {tr['result']}\n"

                final_response = llm_func(
                    synthesis_prompt,
                    f"Based on the tool results above, answer the user's question: {message.content}"
                )
            else:
                final_response = response

            return AgentResult(
                success=True,
                response=final_response,
                agent=self.name,
                tools_used=tools_used,
                confidence=0.85 if tools_used else 0.7,
                artifacts={"tool_results": tool_results} if tool_results else {},
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"[{self.name}] Execution error: {e}")
            return AgentResult(
                success=False,
                response=f"Error: {e}",
                agent=self.name,
                tools_used=tools_used,
                execution_time=time.time() - start_time
            )
