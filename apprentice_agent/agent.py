"""Main agent implementation with observe/plan/act/evaluate/remember loop."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .brain import OllamaBrain
from .memory import MemorySystem
from .metacognition import MetacognitionLogger
from .tools import FileSystemTool, WebSearchTool, CodeExecutorTool, ScreenshotTool, VisionTool, PDFReaderTool, ClipboardTool


class AgentPhase(Enum):
    """Phases of the agent loop."""
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    EVALUATE = "evaluate"
    REMEMBER = "remember"


@dataclass
class AgentState:
    """Current state of the agent."""
    goal: str = ""
    phase: AgentPhase = AgentPhase.OBSERVE
    observations: str = ""
    current_plan: str = ""
    last_action: Optional[dict] = None
    last_result: Optional[dict] = None
    evaluation: Optional[dict] = None
    iteration: int = 0
    completed: bool = False
    history: list = field(default_factory=list)
    gathered_content: str = ""  # Store content gathered from searches for summarization


class ApprenticeAgent:
    """An AI agent that learns and acts using observe/plan/act/evaluate/remember."""

    def __init__(self):
        self.brain = OllamaBrain()
        self.memory = MemorySystem()
        self.tools = {
            "filesystem": FileSystemTool(),
            "web_search": WebSearchTool(),
            "code_executor": CodeExecutorTool(),
            "screenshot": ScreenshotTool(),
            "vision": VisionTool(),
            "pdf_reader": PDFReaderTool(),
            "clipboard": ClipboardTool()
        }
        self.state = AgentState()
        self.max_iterations = 10
        self.metacognition = MetacognitionLogger()

    def run(self, goal: str, context: Optional[dict] = None) -> dict:
        """Run the agent loop to achieve a goal."""
        self.state = AgentState(goal=goal)
        context = context or {}
        # Clear screenshot path for new task
        self.brain._last_screenshot_path = None
        # Start metacognition tracking for this goal
        self.metacognition.start_goal(goal)

        print(f"\n{'='*60}")
        print(f"Agent starting with goal: {goal}")
        print(f"{'='*60}\n")

        while not self.state.completed and self.state.iteration < self.max_iterations:
            self.state.iteration += 1
            self.metacognition.increment_iteration()
            print(f"\n--- Iteration {self.state.iteration} ---")

            # Phase 1: OBSERVE
            self._observe(context)

            # Phase 2: PLAN
            self._plan()

            # Phase 3: ACT
            self._act()

            # Phase 4: EVALUATE
            self._evaluate()

            # Phase 5: REMEMBER
            self._remember()

        return self._get_final_result()

    def _observe(self, context: dict) -> None:
        """Phase 1: Observe the current state and context."""
        self.state.phase = AgentPhase.OBSERVE
        print(f"[OBSERVE] Analyzing current context...")

        # Gather context including relevant memories
        relevant_memories = self.memory.recall(self.state.goal, n_results=3)

        observation_context = {
            "goal": self.state.goal,
            "iteration": self.state.iteration,
            "relevant_memories": [m["content"] for m in relevant_memories],
            "last_action": self.state.last_action,
            "last_result": self.state.last_result,
            **context
        }

        self.state.observations = self.brain.observe(observation_context)
        print(f"Observations: {self.state.observations[:200]}...")

    def _plan(self) -> None:
        """Phase 2: Create or update the plan."""
        self.state.phase = AgentPhase.PLAN
        print(f"[PLAN] Creating action plan...")

        # Include summarize as an available tool
        available_tools = list(self.tools.keys()) + ["summarize"]
        self.state.current_plan = self.brain.plan(
            self.state.goal,
            self.state.observations,
            available_tools
        )
        print(f"Plan: {self.state.current_plan[:200]}...")

    def _act(self) -> None:
        """Phase 3: Execute an action."""
        self.state.phase = AgentPhase.ACT
        print(f"[ACT] Deciding and executing action...")

        # Include summarize as an available tool
        available_tools = list(self.tools.keys()) + ["summarize"]
        action_decision = self.brain.decide_action(
            self.state.current_plan,
            available_tools
        )

        self.state.last_action = action_decision
        print(f"Action: {action_decision.get('tool')} - {action_decision.get('action')}")

        # Execute the action
        self.state.last_result = self._execute_action(action_decision)
        print(f"Result: {str(self.state.last_result)[:200]}...")

    def _evaluate(self) -> None:
        """Phase 4: Evaluate the result."""
        self.state.phase = AgentPhase.EVALUATE
        print(f"[EVALUATE] Assessing result...")

        action_str = f"{self.state.last_action.get('tool')}: {self.state.last_action.get('action')}"
        result_str = str(self.state.last_result)

        self.state.evaluation = self.brain.evaluate(
            action_str,
            result_str,
            self.state.goal
        )

        print(f"Success: {self.state.evaluation.get('success')}")
        print(f"Confidence: {self.state.evaluation.get('confidence', 0)}%")
        print(f"Progress: {self.state.evaluation.get('progress')}")

        # Log to metacognition system
        self.metacognition.log_evaluation(
            tool=self.state.last_action.get('tool'),
            action=self.state.last_action.get('action'),
            confidence=self.state.evaluation.get('confidence', 0),
            success=self.state.evaluation.get('success', False),
            progress=self.state.evaluation.get('progress'),
            next_step=self.state.evaluation.get('next'),
            result_summary=str(self.state.last_result)[:500]
        )

        # Check if goal is achieved
        if self.state.evaluation.get("success") and "complete" in self.state.evaluation.get("next", "").lower():
            # Override for combined screenshot+vision tasks
            is_screenshot_and_vision = getattr(self.brain, '_current_goal_is_screenshot_and_vision', False)
            last_tool = self.state.last_action.get('tool', '').lower() if self.state.last_action else ''

            if is_screenshot_and_vision and last_tool == 'screenshot':
                # Don't mark as complete - still need to do vision analysis
                print("Screenshot done, continuing to vision analysis...")
                self.state.evaluation['next'] = 'continue'
            else:
                self.state.completed = True
                print("Goal achieved!")

    def _remember(self) -> None:
        """Phase 5: Store important information in memory."""
        self.state.phase = AgentPhase.REMEMBER
        print(f"[REMEMBER] Storing experience...")

        # Record this iteration in history
        episode = {
            "iteration": self.state.iteration,
            "goal": self.state.goal,
            "action": self.state.last_action,
            "result": self.state.last_result,
            "evaluation": self.state.evaluation,
            "timestamp": datetime.now().isoformat()
        }
        self.state.history.append(episode)

        # Store significant learnings in long-term memory
        if self.state.evaluation:
            memory_content = self.brain.summarize_for_memory({
                "goal": self.state.goal,
                "actions": [self.state.last_action],
                "outcome": self.state.evaluation.get("progress", ""),
                "lessons": self.state.evaluation.get("next", "")
            })

            self.memory.remember(
                memory_content,
                memory_type="episodic",
                metadata={
                    "goal": self.state.goal,
                    "success": self.state.evaluation.get("success", False),
                    "iteration": self.state.iteration
                }
            )
            print(f"Stored memory: {memory_content[:100]}...")

    def _execute_action(self, action_decision: dict) -> dict:
        """Execute an action using the appropriate tool."""
        tool_name = action_decision.get("tool", "").lower()
        action = action_decision.get("action", "")

        # Handle summarize tool specially (it's not in self.tools)
        if tool_name == "summarize":
            return self._execute_summarize()

        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()) + ["summarize"]
            }

        tool = self.tools[tool_name]

        # Parse the action into tool method and arguments
        try:
            result = self._parse_and_execute_tool_action(tool, tool_name, action)
            # Store successful search results for later summarization
            if tool_name == "web_search" and result.get("success"):
                self._store_search_results(result)
            # Store screenshot path for combined screenshot+vision tasks
            if tool_name == "screenshot" and result.get("success"):
                self.brain._last_screenshot_path = result.get("path")
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_summarize(self) -> dict:
        """Execute the summarize action on gathered content."""
        if not self.state.gathered_content:
            return {
                "success": False,
                "error": "No content to summarize. Run a web search first."
            }

        summary = self.brain.summarize(self.state.gathered_content, self.state.goal)
        return {
            "success": True,
            "summary": summary,
            "message": "Content summarized successfully"
        }

    def _store_search_results(self, result: dict) -> None:
        """Store search results for later summarization."""
        results = result.get("results", [])
        if results:
            content_parts = []
            for r in results[:5]:  # Limit to top 5 results
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                if title or snippet:
                    content_parts.append(f"- {title}: {snippet}")

            new_content = "\n".join(content_parts)
            # Append to existing content
            if self.state.gathered_content:
                self.state.gathered_content += "\n\n" + new_content
            else:
                self.state.gathered_content = new_content

    def _parse_and_execute_tool_action(self, tool: Any, tool_name: str, action: str) -> dict:
        """Parse action string and execute on tool."""
        action_lower = action.lower()

        if tool_name == "filesystem":
            if "read" in action_lower:
                # Extract file path from action
                path = self._extract_path(action)
                return tool.read_file(path) if path else {"success": False, "error": "No path specified"}
            elif "write" in action_lower:
                return {"success": False, "error": "Write action needs path and content"}
            elif "list" in action_lower:
                path = self._extract_path(action) or "."
                return tool.list_directory(path)
            elif "search" in action_lower:
                pattern = self._extract_pattern(action)
                return tool.search_files(pattern) if pattern else {"success": False, "error": "No pattern"}
            else:
                return {"success": False, "error": f"Unknown filesystem action: {action}"}

        elif tool_name == "web_search":
            if "news" in action_lower:
                query = self._extract_query(action)
                return tool.news(query) if query else {"success": False, "error": "No query"}
            elif "search" in action_lower or "find" in action_lower:
                query = self._extract_query(action)
                return tool.search(query) if query else {"success": False, "error": "No query"}
            elif "answer" in action_lower:
                query = self._extract_query(action)
                return tool.instant_answer(query) if query else {"success": False, "error": "No query"}
            else:
                # Default to search
                return tool.search(action)

        elif tool_name == "code_executor":
            # Extract Python code from the action
            code = self._extract_code(action)
            if code:
                return tool.execute(code)
            else:
                return {"success": False, "error": "No code provided"}

        elif tool_name == "screenshot":
            # Handle screenshot actions
            if "region" in action_lower:
                # Extract region coordinates if specified
                import re
                numbers = re.findall(r'\d+', action)
                if len(numbers) >= 4:
                    x, y, w, h = int(numbers[0]), int(numbers[1]), int(numbers[2]), int(numbers[3])
                    return tool.take_screenshot_region(x, y, w, h)
                return {"success": False, "error": "Region needs x, y, width, height"}
            elif "monitor" in action_lower or "list" in action_lower:
                return tool.list_monitors()
            else:
                # Default: capture full screen (primary monitor)
                return tool.take_screenshot(monitor=1)

        elif tool_name == "vision":
            # Handle vision/image analysis actions
            image_path = self._extract_path(action)
            if not image_path:
                # Check if there's a recent screenshot
                screenshots_dir = Path("screenshots")
                if screenshots_dir.exists():
                    screenshots = sorted(screenshots_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if screenshots:
                        image_path = str(screenshots[0])

            if not image_path:
                return {"success": False, "error": "No image path found. Take a screenshot first or specify an image path."}

            # Determine question from action
            if "read" in action_lower or "text" in action_lower:
                return tool.read_text(image_path)
            elif "screen" in action_lower:
                return tool.describe_screen(image_path)
            else:
                return tool.analyze_image(image_path)

        elif tool_name == "pdf_reader":
            # Handle PDF reading actions
            pdf_path = self._extract_path(action)
            if not pdf_path:
                # Try extracting from original goal
                pdf_path = self._extract_path(self.state.goal)
            if not pdf_path:
                return {"success": False, "error": "No PDF path specified"}

            if "info" in action_lower or "metadata" in action_lower:
                return tool.info(pdf_path)
            elif "search" in action_lower or "find" in action_lower:
                # Extract search query
                query = self._extract_query(action)
                if not query:
                    return {"success": False, "error": "No search query provided"}
                return tool.search(pdf_path, query)
            elif "extract" in action_lower:
                # Extract pages specification from action first, then from goal
                pages = self._extract_pages(action)
                if not pages:
                    pages = self._extract_pages(self.state.goal)
                pages = pages or "all"
                return tool.extract_text(pdf_path, pages)
            else:
                # Default: read PDF
                # Extract pages specification from action first, then from goal
                pages = self._extract_pages(action)
                if not pages:
                    pages = self._extract_pages(self.state.goal)
                pages = pages or "all"
                summarize = "summar" in action_lower
                return tool.read(pdf_path, pages, summarize)

        elif tool_name == "clipboard":
            # Handle clipboard actions
            if "analyze" in action_lower or "detect" in action_lower or "type" in action_lower:
                return tool.analyze()
            elif "write" in action_lower or "copy" in action_lower:
                # Extract text to copy
                text = self._extract_clipboard_text(action)
                if not text:
                    # Try from original goal
                    text = self._extract_clipboard_text(self.state.goal)
                if text:
                    return tool.write(text)
                return {"success": False, "error": "No text specified to copy"}
            else:
                # Default: read clipboard (paste, read, what's in clipboard)
                return tool.read()

        return {"success": False, "error": f"Cannot parse action for {tool_name}"}

    def _extract_path(self, action: str) -> Optional[str]:
        """Extract file path from action string."""
        import re
        # Look for quoted strings first
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        # Look for Windows paths (C:/... or C:\...)
        win_paths = re.findall(r'[A-Za-z]:[/\\][\w./\\-]+', action)
        if win_paths:
            return win_paths[0]
        # Look for Unix paths or relative paths
        unix_paths = re.findall(r'(?:^|[ ])([./][\w./\\-]+)', action)
        if unix_paths:
            return unix_paths[0].strip()
        # Look for path patterns with extensions
        paths = re.findall(r'[\w./\\-]+\.\w+', action)
        return paths[0] if paths else None

    def _extract_pattern(self, action: str) -> Optional[str]:
        """Extract search pattern from action string."""
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        patterns = re.findall(r'\*[\w.*]+|\w+\.\w+', action)
        return patterns[0] if patterns else "*.py"

    def _extract_query(self, action: str) -> Optional[str]:
        """Extract search query from action string."""
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        # Remove common command words
        query = re.sub(r'\b(search|find|look|for|up|query|news|about)\b', '', action, flags=re.I)
        return query.strip() or None

    def _extract_code(self, action: str) -> Optional[str]:
        """Extract Python code from action string."""
        import re

        # Look for code in triple backticks
        triple_backtick = re.search(r'```(?:python)?\s*(.*?)```', action, re.DOTALL)
        if triple_backtick:
            return triple_backtick.group(1).strip()

        # Look for code in single backticks
        single_backtick = re.search(r'`([^`]+)`', action)
        if single_backtick:
            return single_backtick.group(1).strip()

        # Look for code after common prefixes
        prefixes = ['run:', 'execute:', 'code:', 'python:']
        action_lower = action.lower()
        for prefix in prefixes:
            if prefix in action_lower:
                idx = action_lower.index(prefix) + len(prefix)
                return action[idx:].strip()

        # If it looks like code (has print, =, import, def, etc.), use as-is
        code_indicators = ['print(', 'import ', 'def ', 'class ', '=', 'for ', 'while ', 'if ']
        if any(ind in action for ind in code_indicators):
            return action.strip()

        # Otherwise return the whole action as potential code
        return action.strip() if action.strip() else None

    def _extract_pages(self, action: str) -> Optional[str]:
        """Extract page specification from action string."""
        import re
        action_lower = action.lower()

        # Check for "first page" or "last page" first
        if 'first page' in action_lower or 'first' in action_lower and 'page' in action_lower:
            return "first"
        if 'last page' in action_lower or 'last' in action_lower and 'page' in action_lower:
            return "last"
        if 'all page' in action_lower:
            return "all"

        # Look for page patterns like "pages 1-3", "page 5", "pages 1,3,5"
        # Pattern: "pages 1-3" or "page 1-3" or "pages 1, 3, 5"
        page_patterns = [
            r'pages?\s+([0-9]+(?:\s*[-,]\s*[0-9]+)*)',  # "pages 1-3" or "pages 1,3,5"
            r'page\s+([0-9]+(?:\s*-\s*[0-9]+)?)',       # "page 5" or "page 1-3"
        ]
        for pattern in page_patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                # Clean up any extra spaces in the result
                result = re.sub(r'\s+', '', match.group(1))
                return result
        return None

    def _extract_clipboard_text(self, action: str) -> Optional[str]:
        """Extract text to copy to clipboard from action string."""
        import re
        # Look for quoted text first
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]

        # Look for text after common patterns
        patterns = [
            r'copy\s+(?:this\s+)?(?:to\s+clipboard)?[:\s]+(.+)',
            r'write\s+(?:to\s+clipboard)?[:\s]+(.+)',
            r'clipboard[:\s]+(.+)',
            r'copy[:\s]+(.+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Remove common suffixes
                text = re.sub(r'\s+to\s+clipboard.*$', '', text, flags=re.IGNORECASE)
                return text.strip('"\'')
        return None

    def _get_final_result(self) -> dict:
        """Compile the final result of the agent run."""
        return {
            "goal": self.state.goal,
            "completed": self.state.completed,
            "iterations": self.state.iteration,
            "final_evaluation": self.state.evaluation,
            "history": self.state.history
        }

    def chat(self, message: str) -> str:
        """Simple chat interface for one-off interactions."""
        return self.brain.think(message)

    def recall_memories(self, query: str, n: int = 5) -> list:
        """Recall relevant memories."""
        return self.memory.recall(query, n_results=n)
