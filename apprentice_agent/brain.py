"""Ollama API integration as the agent's reasoning engine."""

import re
from enum import Enum
from typing import Optional
import ollama

from .config import Config
from .identity import get_identity_prompt


class TaskType(Enum):
    """Types of tasks for model routing."""
    SIMPLE = "simple"       # Greetings, short answers, basic queries
    REASONING = "reasoning" # Planning, evaluation, complex decisions
    CODE = "code"           # Code generation, calculations
    VISION = "vision"       # Image analysis


class OllamaBrain:
    """Handles all interactions with Ollama API for reasoning and decision-making."""

    def __init__(self):
        self.client = ollama.Client(host=Config.OLLAMA_HOST)
        self.model = Config.MODEL_NAME
        self.conversation_history: list[dict] = []
        self._last_model_used: str = self.model  # Track for metacognition

    def think(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        task_type: Optional[TaskType] = None
    ) -> str:
        """Generate a response using Ollama for reasoning tasks.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            use_history: Whether to include conversation history
            task_type: Type of task for model routing (auto-detected if None)
        """
        # Select model based on task type
        model = self._select_model(prompt, task_type)
        self._last_model_used = model

        # Prepend identity to system prompt
        identity_prompt = get_identity_prompt()
        if system_prompt:
            full_system_prompt = f"{identity_prompt}\n\n{system_prompt}"
        else:
            full_system_prompt = identity_prompt

        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        if use_history:
            messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=model,
            messages=messages
        )

        assistant_message = response["message"]["content"]

        if use_history:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def _select_model(self, prompt: str, task_type: Optional[TaskType] = None) -> str:
        """Select the appropriate model based on task type.

        Model routing:
        - SIMPLE (qwen2:1.5b): Greetings, short answers, basic queries
        - REASONING (llama3:8b): Planning, evaluation, complex decisions
        - CODE (deepseek-coder:6.7b): Code generation, debugging, scripts
        - VISION (llava): Image analysis

        Args:
            prompt: The prompt to analyze
            task_type: Explicit task type, or None for auto-detection

        Returns:
            Model name to use
        """
        # If task type is explicitly provided, use it
        if task_type:
            if task_type == TaskType.SIMPLE:
                return Config.MODEL_FAST
            elif task_type == TaskType.VISION:
                return Config.MODEL_VISION
            elif task_type == TaskType.CODE:
                return Config.MODEL_CODE
            else:  # REASONING
                return Config.MODEL_REASON

        # Auto-detect task type from prompt
        prompt_lower = prompt.lower()

        # Vision tasks
        if any(kw in prompt_lower for kw in ['image', 'picture', 'screenshot', 'photo', 'analyze image']):
            return Config.MODEL_VISION

        # Identity questions - route to reasoning model (follows system prompts better)
        identity_patterns = [
            'what is your name', 'who are you', 'your name', 'are you called',
            'what should i call you', 'introduce yourself', 'tell me about yourself',
            'what are you', 'are you an ai', 'are you a bot', 'what model are you'
        ]
        if any(pattern in prompt_lower for pattern in identity_patterns):
            return Config.MODEL_REASON

        # Simple tasks - greetings, basic questions (excluding identity questions)
        simple_patterns = [
            'hello', 'hi ', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'thanks', 'thank you',
            'bye', 'goodbye', 'yes', 'no', 'ok', 'okay'
        ]
        if any(pattern in prompt_lower for pattern in simple_patterns):
            # Check if it's ONLY a simple greeting (short prompt)
            if len(prompt.split()) < 10:
                return Config.MODEL_FAST

        # Browser tasks - route to reasoning model
        browser_patterns = [
            'browse', 'open website', 'go to', 'visit url', 'visit site',
            'navigate to', 'click', 'google search', 'open page', 'web page',
            'browser', 'url'
        ]
        if any(pattern in prompt_lower for pattern in browser_patterns):
            return Config.MODEL_REASON

        # Code/calculation tasks - route to specialized code model
        code_patterns = [
            'calculate', 'compute', 'factorial', 'fibonacci', 'prime',
            'print(', 'import ', 'def ', 'for ', 'while ', 'python',
            'code', 'script', 'function', 'algorithm',
            'debug', 'fix this', 'fix the', 'write a script', 'implement',
            'refactor', 'class ', 'method', 'variable', 'loop',
            'error', 'exception', 'traceback', 'bug', 'syntax'
        ]
        if any(pattern in prompt_lower for pattern in code_patterns):
            return Config.MODEL_CODE

        # Default to reasoning model for complex tasks
        return Config.MODEL_REASON

    def get_last_model_used(self) -> str:
        """Get the model used in the last think() call."""
        return self._last_model_used

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

        # Vision/image analysis keywords
        vision_keywords = [
            'analyze', 'describe', 'what do you see', 'look at this image',
            'what\'s in this image', 'what is in this image', 'examine image',
            'read this image', 'interpret', 'identify', 'recognize'
        ]
        is_vision_task = any(kw in goal_lower for kw in vision_keywords)

        # Screenshot keywords
        screenshot_keywords = [
            'screenshot', 'screen shot', 'capture screen', 'screen capture',
            'take a picture of screen', 'grab screen', 'what\'s on my screen',
            'what is on my screen', 'capture my screen', 'print screen'
        ]
        is_screenshot_task = any(kw in goal_lower for kw in screenshot_keywords)

        # Combined screenshot + vision task (e.g., "take a screenshot and describe it")
        is_screenshot_and_vision = is_screenshot_task and (
            is_vision_task or
            'and describe' in goal_lower or
            'and analyze' in goal_lower or
            'and tell me' in goal_lower or
            'then describe' in goal_lower or
            'then analyze' in goal_lower
        )

        # Search/web keywords
        search_keywords = [
            'search', 'find', 'look up', 'lookup', 'google', 'web', 'internet',
            'online', 'news', 'latest', 'current', 'today', 'price', 'weather',
            'stock', 'bitcoin', 'crypto', 'what is the', 'who is', 'where is',
            'when did', 'how much', 'trending', 'recent', 'update'
        ]
        is_search_task = any(kw in goal_lower for kw in search_keywords) and not is_screenshot_task and not is_vision_task

        # Code/calculation keywords
        code_keywords = [
            'python', 'calculate', 'compute', 'factorial', 'code', 'program',
            'script', 'generate', 'write code', 'run', 'execute', 'math',
            'sum', 'average', 'sort', 'algorithm', 'function', 'check',
            'prime', 'number', 'verify', 'test', 'fibonacci', 'loop',
            'print', 'multiply', 'divide', 'add', 'subtract', 'power',
            'square', 'root', 'modulo', 'remainder', 'even', 'odd'
        ]
        is_code_task = any(kw in goal_lower for kw in code_keywords) and not is_search_task and not is_screenshot_task and not is_vision_task

        # PDF keywords
        pdf_keywords = [
            'pdf', '.pdf', 'document', 'read pdf', 'extract pdf', 'summarize pdf',
            'pdf file', 'open pdf', 'pdf content', 'pdf text', 'pdf pages',
            'search pdf', 'find in pdf', 'pdf info', 'pdf metadata'
        ]
        is_pdf_task = any(kw in goal_lower for kw in pdf_keywords)

        # Clipboard keywords - check for explicit clipboard mentions
        clipboard_keywords = [
            'clipboard', 'paste', 'copied', 'what\'s in my clipboard',
            'what is in my clipboard', 'read clipboard', 'write clipboard',
            'copy to clipboard', 'analyze clipboard', 'clipboard content'
        ]
        # Only mark as PURE clipboard task if clipboard is mentioned WITHOUT other tool keywords
        # This allows multi-tool tasks like "read clipboard then search web"
        has_clipboard = 'clipboard' in goal_lower
        has_other_tools = is_search_task or is_code_task or is_screenshot_task or is_vision_task or is_pdf_task
        is_clipboard_task = has_clipboard and not has_other_tools

        # System control keywords
        system_control_keywords = [
            'system info', 'system information', 'cpu usage', 'cpu', 'ram usage',
            'ram', 'memory usage', 'gpu', 'gpu usage', 'disk usage', 'disk space',
            'get volume', 'set volume', 'volume level', 'get brightness',
            'set brightness', 'brightness level', 'open app', 'launch app',
            'open notepad', 'open calculator', 'open browser', 'open chrome',
            'open firefox', 'open vscode', 'open terminal', 'lock screen',
            'show me system', 'what is my cpu', 'what is my ram', 'how much ram',
            'how much memory', 'computer info', 'pc info', 'machine info'
        ]
        is_system_control_task = any(kw in goal_lower for kw in system_control_keywords) and not is_code_task

        # Notification keywords
        notification_keywords = [
            'remind', 'reminder', 'notify', 'notification', 'alert me',
            'schedule', 'every day', 'every morning', 'every evening',
            'daily at', 'weekly', 'weekdays', 'in 5 minutes', 'in 10 minutes',
            'in 30 minutes', 'in an hour', 'in 2 hours', 'set reminder',
            'set alarm', 'remind me', 'alert when', 'notify when',
            'list reminders', 'show reminders', 'cancel reminder', 'clear reminders'
        ]
        is_notification_task = any(kw in goal_lower for kw in notification_keywords)

        # Store for use in decide_action and _generate_default_code
        self._current_goal_is_code = is_code_task
        self._current_goal_is_search = is_search_task
        self._current_goal_is_screenshot = is_screenshot_task
        self._current_goal_is_vision = is_vision_task
        self._current_goal_is_screenshot_and_vision = is_screenshot_and_vision
        self._current_goal_is_pdf = is_pdf_task
        self._current_goal_is_clipboard = is_clipboard_task
        self._current_goal_is_system_control = is_system_control_task
        self._current_goal_is_notification = is_notification_task
        self._current_goal = goal

        if is_clipboard_task:
            prompt = f"""Goal: {goal}

This is a CLIPBOARD task. Use clipboard tool to read, write, or analyze clipboard content.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use clipboard to read/write/analyze the clipboard"""
        elif is_screenshot_and_vision:
            prompt = f"""Goal: {goal}

This is a SCREENSHOT + VISION task. First capture the screen, then analyze it.

Available tools:
{tool_descriptions}

Create a 2-step plan:
1. Use screenshot to capture the screen
2. Use vision to analyze/describe the captured screenshot"""
        elif is_vision_task and not is_screenshot_task:
            prompt = f"""Goal: {goal}

This is a VISION/IMAGE ANALYSIS task. Use vision tool to analyze an image.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use vision to analyze the image"""
        elif is_screenshot_task:
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
        elif is_pdf_task:
            prompt = f"""Goal: {goal}

This is a PDF task. Use pdf_reader tool to read, extract, or search PDF content.

Available tools:
{tool_descriptions}

Create a 1-2 step plan:
1. Use pdf_reader to read/extract/search the PDF
2. (Optional) Summarize the content if needed"""
        elif is_system_control_task:
            prompt = f"""Goal: {goal}

This is a SYSTEM CONTROL task. Use system_control tool to get system info, control volume/brightness, or launch apps.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use system_control to execute the system command"""
        elif is_notification_task:
            prompt = f"""Goal: {goal}

This is a NOTIFICATION task. Use notifications tool to set reminders, schedule notifications, or create conditional alerts.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use notifications to add/list/remove reminder or scheduled task"""
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
        is_vision = getattr(self, '_current_goal_is_vision', False)
        is_screenshot_and_vision = getattr(self, '_current_goal_is_screenshot_and_vision', False)
        is_clipboard = getattr(self, '_current_goal_is_clipboard', False)
        screenshot_path = getattr(self, '_last_screenshot_path', None)

        # Check clipboard FIRST (before vision which also has "analyze")
        if is_clipboard:
            prompt = f"""Plan: {plan[:500]}

This is a CLIPBOARD task. Use clipboard tool to read, write, or analyze clipboard content.

Pick ONE action. Reply ONLY in this format:

TOOL: clipboard
ACTION: <read/write/analyze> [text to copy]
REASONING: <why>

Examples:

TOOL: clipboard
ACTION: read
REASONING: get current clipboard content

TOOL: clipboard
ACTION: analyze
REASONING: detect clipboard content type

TOOL: clipboard
ACTION: write "hello world"
REASONING: copy text to clipboard"""
        # Check combined screenshot+vision (before screenshot alone)
        elif is_screenshot_and_vision:
            if screenshot_path:
                # Screenshot already taken, now use vision
                prompt = f"""Plan: {plan[:500]}

This is a SCREENSHOT + VISION task. Screenshot was already taken at: {screenshot_path}

NOW use vision tool to analyze the captured screenshot.

Reply ONLY in this format:

TOOL: vision
ACTION: analyze {screenshot_path}
REASONING: analyze the captured screenshot"""
            else:
                # Need to take screenshot first
                prompt = f"""Plan: {plan[:500]}

This is a SCREENSHOT + VISION task. Take screenshot first, then analyze it.

Screenshot has NOT been taken yet. Use screenshot tool first.

Reply ONLY in this format:

TOOL: screenshot
ACTION: capture
REASONING: need to capture screen first"""
        elif is_screenshot:
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
        elif is_vision:
            # Vision-only task (not combined with screenshot)
            prompt = f"""Plan: {plan[:500]}

This is a VISION/IMAGE ANALYSIS task. Use vision tool to analyze an image.

Pick ONE action. Reply ONLY in this format:

TOOL: vision
ACTION: <analyze/describe/read> <image_path>
REASONING: <why>

Examples:

TOOL: vision
ACTION: analyze screenshots/screenshot_20260115_152610.png
REASONING: analyze what is in the image

TOOL: vision
ACTION: describe screen screenshots/latest.png
REASONING: describe what is on screen

TOOL: vision
ACTION: read text document.png
REASONING: extract text from image"""
        elif getattr(self, '_current_goal_is_pdf', False):
            # PDF task
            prompt = f"""Plan: {plan[:500]}

This is a PDF task. Use pdf_reader tool to read, extract, or search PDF content.

Pick ONE action. Reply ONLY in this format:

TOOL: pdf_reader
ACTION: <read/extract/search/info> <pdf_path> [pages/query]
REASONING: <why>

Examples:

TOOL: pdf_reader
ACTION: read C:/Documents/report.pdf
REASONING: read entire PDF content

TOOL: pdf_reader
ACTION: read C:/Documents/report.pdf pages 1-5
REASONING: read first 5 pages

TOOL: pdf_reader
ACTION: search C:/Documents/report.pdf "revenue"
REASONING: find pages mentioning revenue

TOOL: pdf_reader
ACTION: info C:/Documents/report.pdf
REASONING: get PDF metadata and page count"""
        elif getattr(self, '_current_goal_is_system_control', False):
            # System control task
            prompt = f"""Plan: {plan[:500]}

This is a SYSTEM CONTROL task. Use system_control tool.

Pick ONE action. Reply ONLY in this format:

TOOL: system_control
ACTION: <get_system_info/get_volume/set_volume/get_brightness/set_brightness/open_app/lock_screen> [args]
REASONING: <why>

Examples:

TOOL: system_control
ACTION: get_system_info
REASONING: get CPU, RAM, GPU, and disk usage

TOOL: system_control
ACTION: set_volume 50
REASONING: set volume to 50%

TOOL: system_control
ACTION: open_app notepad
REASONING: launch notepad application

TOOL: system_control
ACTION: get_brightness
REASONING: get current screen brightness"""
        elif getattr(self, '_current_goal_is_notification', False):
            # Notification task
            prompt = f"""Plan: {plan[:500]}

This is a NOTIFICATION task. Use notifications tool.

Pick ONE action. Reply ONLY in this format:

TOOL: notifications
ACTION: <add_reminder/add_scheduled/add_condition/list/remove/clear> [args]
REASONING: <why>

Examples:

TOOL: notifications
ACTION: add_reminder "take a break" in 30 minutes
REASONING: set a reminder for 30 minutes

TOOL: notifications
ACTION: add_scheduled "standup meeting" 9:00 AM daily
REASONING: schedule daily notification at 9 AM

TOOL: notifications
ACTION: add_condition "high CPU alert" cpu 80
REASONING: alert when CPU exceeds 80%

TOOL: notifications
ACTION: list
REASONING: show all scheduled tasks"""
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

        # Check for multi-step tasks (screenshot + vision)
        goal_lower = goal.lower()
        is_screenshot_and_vision = getattr(self, '_current_goal_is_screenshot_and_vision', False)

        # If this is a combined task and we just did screenshot, continue to vision
        if is_screenshot_and_vision and 'screenshot' in action.lower() and 'success' in result_truncated.lower():
            extra_instruction = """
IMPORTANT: This is a 2-step task (screenshot + describe/analyze).
If only screenshot was done, say NEXT: continue (still need to analyze the image).
Only say NEXT: complete if BOTH screenshot AND vision/description are done."""
        else:
            extra_instruction = ""

        prompt = f"""Goal: {goal}
Action: {action}
Result: {result_truncated}
{extra_instruction}
Reply ONLY in this format:

SUCCESS: yes OR no
CONFIDENCE: 0-100 (how confident are you the goal is fully achieved)
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
            "vision": "vision - analyze images using AI vision model. ACTION: 'analyze <image_path>' or 'describe screen <path>' or 'read text <path>'",
            "pdf_reader": "pdf_reader - read, extract text, or search PDF files. ACTION: 'read <path>' or 'extract <path> pages 1-5' or 'search <path> query' or 'info <path>'",
            "clipboard": "clipboard - read, write, or analyze clipboard content. ACTION: 'read' or 'write <text>' or 'analyze'",
            "system_control": "system_control - get system info (CPU, RAM, GPU, disk), control volume/brightness, open apps, lock screen. ACTION: 'get_system_info' or 'get_volume' or 'set_volume <level>' or 'open_app <name>'",
            "notifications": "notifications - set reminders, schedule notifications, create conditional alerts. ACTION: 'add_reminder <msg> in <time>' or 'add_scheduled <msg> <time> <repeat>' or 'list' or 'remove <id>'",
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
                elif "screenshot" in tool or "screen" in tool or "capture" in tool:
                    tool = "screenshot"
                elif "vision" in tool or "llava" in tool or "image" in tool or "analyze" in tool:
                    tool = "vision"
                elif "pdf" in tool or "document" in tool:
                    tool = "pdf_reader"
                elif "clipboard" in tool or "copy" in tool or "paste" in tool:
                    tool = "clipboard"
                elif "system" in tool or "control" in tool:
                    tool = "system_control"
                elif "notif" in tool or "remind" in tool or "schedule" in tool or "alert" in tool:
                    tool = "notifications"
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
            elif "screenshot" in response_lower or "capture screen" in response_lower:
                result["tool"] = "screenshot"
            elif "vision" in response_lower or "analyze image" in response_lower or "describe image" in response_lower:
                result["tool"] = "vision"
            elif "pdf_reader" in response_lower or "read pdf" in response_lower or "extract pdf" in response_lower:
                result["tool"] = "pdf_reader"
            elif "clipboard" in response_lower or "paste" in response_lower or "copy to" in response_lower:
                result["tool"] = "clipboard"
            elif "system_control" in response_lower or "system info" in response_lower or "cpu" in response_lower or "ram" in response_lower or "volume" in response_lower or "brightness" in response_lower:
                result["tool"] = "system_control"
            elif "notification" in response_lower or "reminder" in response_lower or "remind" in response_lower or "schedule" in response_lower or "alert" in response_lower:
                result["tool"] = "notifications"

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

        # FORCE clipboard for clipboard tasks - override tool AND action
        is_clipboard_task = getattr(self, '_current_goal_is_clipboard', False)
        if is_clipboard_task:
            result["tool"] = "clipboard"
            goal = getattr(self, '_current_goal', '').lower()
            current_action = (result.get("action") or "").lower()

            # Determine correct action based on goal keywords
            if 'analyze' in goal or 'type' in goal or 'detect' in goal:
                result["action"] = "analyze"
            elif 'copy' in goal or 'write' in goal:
                # Extract text to copy if present in action
                if '"' in current_action:
                    # Keep action with quoted text
                    pass
                elif 'write' in current_action and len(current_action) > 10:
                    # Keep action if it has text after "write"
                    pass
                else:
                    result["action"] = "write"
            else:
                # Default to read for "what's in clipboard", "paste", etc.
                result["action"] = "read"

        # FORCE system_control for system control tasks
        is_system_control_task = getattr(self, '_current_goal_is_system_control', False)
        if is_system_control_task:
            result["tool"] = "system_control"
            goal = getattr(self, '_current_goal', '').lower()
            current_action = (result.get("action") or "").lower()

            # Determine correct action based on goal keywords
            if 'volume' in goal:
                if 'set' in goal or any(c.isdigit() for c in goal):
                    result["action"] = current_action if 'volume' in current_action else "set_volume"
                else:
                    result["action"] = "get_volume"
            elif 'brightness' in goal:
                if 'set' in goal or any(c.isdigit() for c in goal):
                    result["action"] = current_action if 'brightness' in current_action else "set_brightness"
                else:
                    result["action"] = "get_brightness"
            elif 'open' in goal or 'launch' in goal:
                result["action"] = current_action if 'open' in current_action else "open_app"
            elif 'lock' in goal:
                result["action"] = "lock_screen"
            else:
                # Default to get_system_info for "system info", "cpu", "ram", etc.
                result["action"] = "get_system_info"

        # FORCE notifications for notification tasks
        is_notification_task = getattr(self, '_current_goal_is_notification', False)
        if is_notification_task:
            result["tool"] = "notifications"
            goal = getattr(self, '_current_goal', '').lower()
            current_action = (result.get("action") or "").lower()

            # Determine correct action based on goal keywords
            if 'list' in goal or 'show' in goal or 'all' in goal:
                result["action"] = "list"
            elif 'remove' in goal or 'delete' in goal or 'cancel' in goal:
                result["action"] = current_action if 'remove' in current_action else "remove"
            elif 'clear' in goal:
                result["action"] = "clear"
            elif 'condition' in goal or 'alert when' in goal or 'notify when' in goal or ('cpu' in goal and ('above' in goal or 'exceed' in goal or '%' in goal)):
                result["action"] = current_action if 'condition' in current_action else "add_condition"
            elif 'schedule' in goal or 'every day' in goal or 'daily' in goal or 'weekday' in goal or 'weekly' in goal or 'every morning' in goal or 'every evening' in goal:
                result["action"] = current_action if 'scheduled' in current_action else "add_scheduled"
            else:
                # Default to add_reminder for "remind me", "in 30 minutes", etc.
                result["action"] = current_action if 'reminder' in current_action else "add_reminder"

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
        result = {"success": False, "confidence": 0, "progress": None, "next": None, "raw": response}

        response_lower = response.lower()

        for line in response.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if line_lower.startswith("success:"):
                result["success"] = "yes" in line_lower or "true" in line_lower
            elif line_lower.startswith("confidence:"):
                # Extract numeric confidence value
                conf_str = line_stripped[11:].strip()
                # Extract first number found
                conf_match = re.search(r'\d+', conf_str)
                if conf_match:
                    result["confidence"] = min(100, max(0, int(conf_match.group())))
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
