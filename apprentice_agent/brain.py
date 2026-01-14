"""Ollama API integration as the agent's reasoning engine."""

import re
from typing import Optional
import ollama

from .config import Config


class OllamaBrain:
    """Handles all interactions with Ollama API for reasoning and decision-making."""

    def __init__(self):
        self.client = ollama.Client(host=Config.OLLAMA_HOST)
        self.model = Config.MODEL_NAME
        self.conversation_history: list[dict] = []

    def think(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = True
    ) -> str:
        """Generate a response using Ollama for reasoning tasks."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if use_history:
            messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages
        )

        assistant_message = response["message"]["content"]

        if use_history:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def observe(self, context: dict) -> str:
        """Process observations about the current state."""
        prompt = f"""Context:
{self._format_context(context)}

List 3-5 key observations. Be brief."""

        return self.think(prompt, system_prompt=self._observer_prompt())

    def plan(self, goal: str, observations: str, available_tools: list[str]) -> str:
        """Create a plan to achieve the goal based on observations."""
        tool_descriptions = self._get_tool_descriptions(available_tools)

        prompt = f"""Goal: {goal}

Observations: {observations[:500]}

Available tools:
{tool_descriptions}

Create a short 3-5 step plan. Be specific about which tool to use for each step."""

        return self.think(prompt, system_prompt=self._planner_prompt())

    def decide_action(self, plan: str, available_tools: list[str]) -> dict:
        """Decide the next action to take based on the plan."""
        tool_descriptions = self._get_tool_descriptions(available_tools)

        prompt = f"""Plan: {plan[:500]}

Available tools:
{tool_descriptions}

Pick ONE action from the plan. Reply ONLY in this format:

TOOL: <tool_name>
ACTION: <the query or path>
REASONING: <one sentence why>

IMPORTANT:
- To list/read LOCAL files -> use filesystem
- To search the INTERNET -> use web_search
- To summarize what you found -> use summarize

Examples:
- List local files: TOOL: filesystem / ACTION: list C:/Users/project / REASONING: need to see directory contents
- Read a file: TOOL: filesystem / ACTION: read C:/Users/project/readme.md / REASONING: need file contents
- Search online: TOOL: web_search / ACTION: latest AI news 2024 / REASONING: need current information from internet"""

        response = self.think(prompt, system_prompt=self._actor_prompt())
        return self._parse_action_response(response)

    def evaluate(self, action: str, result: str, goal: str) -> dict:
        """Evaluate the result of an action."""
        # Truncate result to avoid overwhelming the model
        result_truncated = result[:1000] if len(result) > 1000 else result

        prompt = f"""Goal: {goal}
Action: {action}
Result: {result_truncated}

Reply ONLY in this format:

SUCCESS: yes OR no
PROGRESS: one sentence about progress
NEXT: continue OR complete OR retry

If the goal is achieved, say NEXT: complete"""

        response = self.think(prompt, system_prompt=self._evaluator_prompt())
        return self._parse_evaluation_response(response)

    def summarize_for_memory(self, episode: dict) -> str:
        """Create a memory-worthy summary of an episode."""
        prompt = f"""Summarize in 2-3 sentences:
Goal: {episode.get('goal', 'N/A')}
Actions: {episode.get('actions', [])}
Outcome: {episode.get('outcome', 'N/A')}"""

        return self.think(prompt, system_prompt=self._memory_prompt(), use_history=False)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def _format_context(self, context: dict) -> str:
        """Format context dictionary for prompts."""
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def summarize(self, content: str, goal: str) -> str:
        """Summarize content in relation to a goal."""
        prompt = f"""Goal: {goal}

Content to summarize:
{content[:2000]}

Write a clear, concise summary (3-5 sentences) of the key points relevant to the goal."""

        return self.think(prompt, system_prompt="You summarize information clearly and concisely.", use_history=False)

    def _get_tool_descriptions(self, available_tools: list[str]) -> str:
        """Get clear descriptions for available tools."""
        descriptions = {
            "filesystem": "filesystem - list or read LOCAL files on this computer. Use for any file/directory operations. ACTION: 'list <path>' or 'read <path>'",
            "web_search": "web_search - search the INTERNET for information. Only use for online searches. ACTION: the search query",
            "summarize": "summarize - summarize gathered information. ACTION: 'results'"
        }
        return "\n".join(descriptions.get(t, t) for t in available_tools)

    def _parse_action_response(self, response: str) -> dict:
        """Parse the action decision response with better extraction for local models."""
        result = {"tool": None, "action": None, "reasoning": None, "raw": response}

        # Try to find TOOL, ACTION, REASONING in the response
        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("TOOL:"):
                tool = line[5:].strip().lower()
                # Clean up common variations
                tool = tool.replace("**", "").replace("`", "").strip()
                if "summar" in tool:
                    tool = "summarize"
                elif "web" in tool or "search" in tool:
                    tool = "web_search"
                elif "file" in tool or "fs" in tool:
                    tool = "filesystem"
                result["tool"] = tool
            elif line.upper().startswith("ACTION:"):
                action = line[7:].strip()
                # Clean up the action - remove common prefixes local models add
                action = self._clean_action(action)
                result["action"] = action
            elif line.upper().startswith("REASONING:"):
                result["reasoning"] = line[10:].strip()

        # Fallback: try to extract from less structured responses
        if not result["tool"]:
            response_lower = response.lower()
            # Check for filesystem indicators first (more specific)
            if "filesystem" in response_lower or "list " in response_lower or "read " in response_lower or "directory" in response_lower:
                result["tool"] = "filesystem"
            elif "summarize" in response_lower or "summary" in response_lower:
                result["tool"] = "summarize"
            elif "web_search" in response_lower or "internet" in response_lower or "online" in response_lower:
                result["tool"] = "web_search"

        if not result["action"] and result["tool"] == "web_search":
            # Try to extract a search query from the response
            result["action"] = self._extract_search_query(response)

        return result

    def _clean_action(self, action: str) -> str:
        """Clean up action string from verbose local model outputs."""
        # Remove common prefixes that local models add
        prefixes_to_remove = [
            "use web_search tool to search for",
            "use web_search to search for",
            "search for",
            "search the web for",
            "search:",
            "query:",
            "use filesystem to",
            "use the",
        ]

        action_lower = action.lower()
        for prefix in prefixes_to_remove:
            if action_lower.startswith(prefix):
                action = action[len(prefix):].strip()
                action_lower = action.lower()

        # Remove quotes if present
        action = action.strip('"\'')

        # Remove markdown formatting
        action = action.replace("**", "").replace("`", "")

        return action.strip()

    def _extract_search_query(self, response: str) -> str:
        """Extract a search query from a verbose response."""
        # Look for quoted text
        quoted = re.findall(r'["\']([^"\']+)["\']', response)
        if quoted:
            return quoted[0]

        # Look for text after common patterns
        patterns = [
            r'search (?:for |query[: ]+)?["\']?([^"\'\n]+)',
            r'query[: ]+([^\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"\'')

        # Fallback: return a default query based on context
        return "latest news"

    def _parse_evaluation_response(self, response: str) -> dict:
        """Parse the evaluation response."""
        result = {"success": False, "progress": None, "next": None, "raw": response}

        response_lower = response.lower()

        for line in response.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if line_lower.startswith("success:"):
                result["success"] = "yes" in line_lower or "true" in line_lower
            elif line_lower.startswith("progress:"):
                result["progress"] = line_stripped[9:].strip()
            elif line_lower.startswith("next:"):
                next_val = line_stripped[5:].strip().lower()
                result["next"] = next_val
                # Check if goal is complete
                if "complete" in next_val or "done" in next_val or "achieved" in next_val:
                    result["success"] = True

        # Fallback detection
        if result["progress"] is None:
            if "success" in response_lower or "found" in response_lower:
                result["progress"] = "Made progress"

        return result

    def _default_system_prompt(self) -> str:
        return "You are a helpful AI assistant. Be concise and direct."

    def _observer_prompt(self) -> str:
        return "You analyze situations. List only key observations. Be very brief."

    def _planner_prompt(self) -> str:
        return "You create simple action plans. Use numbered steps. Be brief and practical."

    def _actor_prompt(self) -> str:
        return """You select actions. Follow the output format exactly.
Do not explain or add extra text. Just output TOOL, ACTION, REASONING lines."""

    def _evaluator_prompt(self) -> str:
        return """You evaluate results. Follow the format exactly.
Say 'NEXT: complete' when the goal is achieved."""

    def _memory_prompt(self) -> str:
        return "Summarize in 2-3 short sentences. Focus on what happened and what was learned."
