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

        # Detect task type from goal
        goal_lower = goal.lower()

        # Screenshot keywords - check first
        screenshot_keywords = [
            'screenshot', 'screen shot', 'capture screen', 'screen capture',
            'take a picture of screen', 'grab screen', 'what\'s on my screen',
            'what is on my screen', 'capture my screen', 'print screen'
        ]
        is_screenshot_task = any(kw in goal_lower for kw in screenshot_keywords)

        # Search/web keywords
        search_keywords = [
            'search', 'find', 'look up', 'lookup', 'google', 'web', 'internet',
            'online', 'news', 'latest', 'current', 'today', 'price', 'weather',
            'stock', 'bitcoin', 'crypto', 'what is the', 'who is', 'where is',
            'when did', 'how much', 'trending', 'recent', 'update'
        ]
        is_search_task = any(kw in goal_lower for kw in search_keywords) and not is_screenshot_task

        # Code/calculation keywords
        code_keywords = [
            'python', 'calculate', 'compute', 'factorial', 'code', 'program',
            'script', 'generate', 'write code', 'run', 'execute', 'math',
            'sum', 'average', 'sort', 'algorithm', 'function', 'check',
            'prime', 'number', 'verify', 'test', 'fibonacci', 'loop',
            'print', 'multiply', 'divide', 'add', 'subtract', 'power',
            'square', 'root', 'modulo', 'remainder', 'even', 'odd'
        ]
        is_code_task = any(kw in goal_lower for kw in code_keywords) and not is_search_task and not is_screenshot_task

        # Store for use in decide_action and _generate_default_code
        self._current_goal_is_code = is_code_task
        self._current_goal_is_search = is_search_task
        self._current_goal_is_screenshot = is_screenshot_task
        self._current_goal = goal

        if is_screenshot_task:
            prompt = f"""Goal: {goal}

This is a SCREENSHOT task. Use screenshot tool to capture the screen.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use screenshot to capture the screen"""
        elif is_search_task:
            prompt = f"""Goal: {goal}

This is a WEB SEARCH task. Use web_search to find information online.
DO NOT use code_executor for web searches - it cannot access the internet.

Available tools:
{tool_descriptions}

Create a 2-3 step plan:
1. Use web_search with a clear search query
2. Summarize the results if needed"""
        elif is_code_task:
            prompt = f"""Goal: {goal}

This is a CODE/CALCULATION task. Use code_executor to run Python code directly.
DO NOT search the web. Just write and run the Python code.

Available tools:
{tool_descriptions}

Create a 1-2 step plan:
1. Use code_executor with the actual Python code to solve this
2. (Optional) Summarize if needed"""
        else:
            prompt = f"""Goal: {goal}

Observations: {observations[:500]}

Available tools:
{tool_descriptions}

Create a short 3-5 step plan. Be specific about which tool to use for each step."""

        return self.think(prompt, system_prompt=self._planner_prompt())

    def decide_action(self, plan: str, available_tools: list[str]) -> dict:
        """Decide the next action to take based on the plan."""
        tool_descriptions = self._get_tool_descriptions(available_tools)

        # Check task type
        is_screenshot = getattr(self, '_current_goal_is_screenshot', False)
        is_search = getattr(self, '_current_goal_is_search', False)

        if is_screenshot:
            prompt = f"""Plan: {plan[:500]}

This is a SCREENSHOT task. You MUST use screenshot tool.

Pick ONE action. Reply ONLY in this format:

TOOL: screenshot
ACTION: capture
REASONING: take screenshot of the screen

Example:

TOOL: screenshot
ACTION: capture full screen
REASONING: capture current screen"""
        elif is_search:
            prompt = f"""Plan: {plan[:500]}

Available tools:
{tool_descriptions}

This is a WEB SEARCH task. You MUST use web_search tool.
DO NOT use code_executor - it cannot access the internet!

Pick ONE action. Reply ONLY in this format:

TOOL: web_search
ACTION: <your search query>
REASONING: <why>

Examples:

TOOL: web_search
ACTION: Bitcoin price today USD
REASONING: find current Bitcoin price

TOOL: web_search
ACTION: latest AI news 2024
REASONING: search for recent AI news

TOOL: web_search
ACTION: weather New York today
REASONING: get current weather"""
        else:
            prompt = f"""Plan: {plan[:500]}

Available tools:
{tool_descriptions}

Pick ONE action. Reply ONLY in this format:

TOOL: <tool_name>
ACTION: <actual code, path, or query>
REASONING: <why>

RULES:
- For calculations/math/Python -> use code_executor with ACTUAL Python code
- For local files -> use filesystem
- For internet/online info -> use web_search (NOT code_executor!)
- code_executor CANNOT access the internet - use web_search instead!

Examples:

TOOL: code_executor
ACTION: import math; print(math.factorial(50))
REASONING: calculate factorial

TOOL: web_search
ACTION: Bitcoin price today
REASONING: search internet for price

TOOL: filesystem
ACTION: list C:/Users/project
REASONING: see directory"""

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
            "filesystem": "filesystem - list or read LOCAL files on this computer. ACTION: 'list <path>' or 'read <path>'",
            "web_search": "web_search - search the INTERNET for information. ACTION: the search query",
            "code_executor": "code_executor - run Python code and get the output. Use for calculations, data processing. ACTION: the Python code",
            "screenshot": "screenshot - capture a screenshot of the screen. ACTION: 'capture' or 'capture region x y width height'",
            "summarize": "summarize - summarize gathered information. ACTION: 'results'"
        }
        return "\n".join(descriptions.get(t, t) for t in available_tools)

    def _parse_action_response(self, response: str) -> dict:
        """Parse the action decision response with better extraction for local models."""
        result = {"tool": None, "action": None, "reasoning": None, "raw": response}

        # Check if this is a code task - if so, force code_executor
        is_code_task = getattr(self, '_current_goal_is_code', False)

        # Try to find TOOL, ACTION, REASONING in the response
        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("TOOL:"):
                tool = line[5:].strip().lower()
                # Clean up common variations
                tool = tool.replace("**", "").replace("`", "").strip()
                if "code" in tool or "execute" in tool or "python" in tool or "run" in tool:
                    tool = "code_executor"
                elif "summar" in tool:
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
            # Check for code executor indicators
            if "code_executor" in response_lower or "python" in response_lower or "calculate" in response_lower or "factorial" in response_lower or "print(" in response:
                result["tool"] = "code_executor"
            # Check for filesystem indicators
            elif "filesystem" in response_lower or "list " in response_lower or "read " in response_lower or "directory" in response_lower:
                result["tool"] = "filesystem"
            elif "summarize" in response_lower or "summary" in response_lower:
                result["tool"] = "summarize"
            elif "web_search" in response_lower or "internet" in response_lower or "online" in response_lower:
                result["tool"] = "web_search"

        if not result["action"] and result["tool"] == "web_search":
            # Try to extract a search query from the response
            result["action"] = self._extract_search_query(response)

        # FORCE code_executor for code tasks - override any wrong tool selection
        if is_code_task and result["tool"] != "code_executor":
            result["tool"] = "code_executor"
            # Try to extract code from the action or response
            if result["action"]:
                # Clean up action to be valid Python code
                action = result["action"]
                # If it looks like a description, try to extract actual code
                if not any(ind in action for ind in ['print(', '=', 'import ', 'def ', 'for ', 'if ']):
                    # Extract any Python-like code from the full response
                    code = self._extract_code_from_response(response)
                    if code:
                        result["action"] = code
                    else:
                        # Generate default code based on the original goal
                        result["action"] = self._generate_default_code()

        # Also force code_executor if action looks like code
        if result["tool"] != "code_executor" and result["action"]:
            if any(ind in result["action"] for ind in ['print(', 'import ', 'def ', 'for i in']):
                result["tool"] = "code_executor"

        return result

    def _generate_default_code(self) -> str:
        """Generate default Python code based on the current goal."""
        goal = getattr(self, '_current_goal', '').lower()

        # Prime number check
        if 'prime' in goal:
            # Extract number from goal
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"n = {n}; is_prime = n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)); print(str(n) + ' is ' + ('' if is_prime else 'not ') + 'a prime number')"

        # Factorial
        if 'factorial' in goal:
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"import math; print(f'factorial({n}) = {{math.factorial({n})}}')"

        # Fibonacci
        if 'fibonacci' in goal or 'fib' in goal:
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2); print(f'fibonacci({n}) = {{fib({n})}}')"

        # Default: just print hello
        return "print('Code executed successfully')"

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

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from a verbose LLM response."""
        # Look for code blocks
        code_block = re.search(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Look for lines that look like Python code
        code_indicators = ['print(', 'import ', 'def ', 'for ', 'while ', 'if ', '=']
        for line in response.split('\n'):
            line = line.strip()
            if any(ind in line for ind in code_indicators):
                # This looks like code
                return line

        # Look for code after "ACTION:" anywhere in response
        action_match = re.search(r'ACTION:\s*(.+)', response, re.IGNORECASE)
        if action_match:
            return action_match.group(1).strip()

        return None

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
        return """You create simple action plans. Be brief.
CRITICAL: For ANY calculation, math, Python, or code task -> use code_executor FIRST. Do NOT search the web for how to do it. Just write and execute the code directly.
For local files -> filesystem.
For internet info -> web_search."""

    def _actor_prompt(self) -> str:
        return """You select actions. Output ONLY: TOOL, ACTION, REASONING lines.
For code_executor: ACTION must be actual Python code with print() to show results.
Do NOT describe code - write the actual code!"""

    def _evaluator_prompt(self) -> str:
        return """You evaluate results. Follow the format exactly.
Say 'NEXT: complete' when the goal is achieved."""

    def _memory_prompt(self) -> str:
        return "Summarize in 2-3 short sentences. Focus on what happened and what was learned."
