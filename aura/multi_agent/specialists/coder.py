"""Coder Agent - Code writing and execution specialist.

Specialized in software development tasks:
- Writing code in various languages
- Debugging and fixing errors
- Code execution and testing
- Git operations
- File system operations
"""

import logging
import time
import re
from typing import Any, Callable, Dict

from ..base_agent import ToolUsingSpecialist
from ..protocol import AgentMessage, AgentResult

logger = logging.getLogger(__name__)


class CoderAgent(ToolUsingSpecialist):
    """Specialist for code writing, debugging, and execution."""

    name = "coder"
    description = "Write, debug, execute code and manage git operations"

    system_prompt = """You are a Senior Software Engineer AI assistant. You write clean, efficient, well-documented code.

Your expertise:
- Python, JavaScript, TypeScript, and other popular languages
- Code debugging and error fixing
- Best practices and design patterns
- Git version control
- Testing and code quality

Your approach:
1. Understand the requirements clearly
2. Plan the implementation
3. Write clean, readable code with comments
4. Test the code when possible
5. Explain your implementation

Guidelines:
- Write production-quality code
- Include error handling
- Add docstrings and comments
- Follow language-specific conventions
- Suggest improvements and alternatives

When executing code, use: [TOOL: code_executor] your_code_here
When working with files, use: [TOOL: filesystem] action path
For git operations, use: [TOOL: git] git_command"""

    tools = [
        "code_executor",
        "filesystem",
        "git",
    ]

    triggers = [
        "code", "write", "implement", "fix", "debug",
        "error", "bug", "function", "class", "python",
        "javascript", "git", "commit", "programming",
        "script", "execute", "run", "compile"
    ]

    max_tool_calls = 5  # Allow more tool calls for code tasks

    def execute(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str]
    ) -> AgentResult:
        """Execute coding task with code generation and optional execution."""
        start_time = time.time()
        tools_used = []
        code_artifacts = {}

        try:
            query = message.content.lower()
            system_prompt = self.get_system_prompt(message.context)

            # Detect task type
            is_debug = any(kw in query for kw in ["fix", "debug", "error", "bug", "issue"])
            is_execute = any(kw in query for kw in ["run", "execute", "test"])
            is_git = any(kw in query for kw in ["git", "commit", "push", "pull", "branch"])

            # Handle git operations
            if is_git:
                return self._handle_git(message, llm_func, system_prompt, start_time)

            # Generate code with LLM
            code_prompt = system_prompt
            if is_debug:
                code_prompt += "\n\nFocus on identifying and fixing the bug. Explain what was wrong and how you fixed it."
            elif is_execute:
                code_prompt += "\n\nProvide executable code. Use [TOOL: code_executor] to run the code."

            response = llm_func(code_prompt, message.content)

            # Extract code blocks
            code_blocks = self._extract_code_blocks(response)
            if code_blocks:
                code_artifacts["generated_code"] = code_blocks

            # Execute code if requested
            execution_results = []
            if is_execute and code_blocks and "code_executor" in self._available_tools:
                for i, code in enumerate(code_blocks[:2]):  # Execute max 2 blocks
                    logger.info(f"[Coder] Executing code block {i+1}...")
                    result = self._execute_tool("code_executor", code["code"])
                    tools_used.append("code_executor")
                    execution_results.append({
                        "language": code.get("language", "unknown"),
                        "result": result
                    })

                # Add execution results to response
                if execution_results:
                    response += "\n\n**Execution Results:**\n"
                    for i, er in enumerate(execution_results):
                        response += f"\n*Block {i+1} ({er['language']}):*\n"
                        if er["result"].get("success"):
                            output = er["result"].get("output", "No output")
                            response += f"```\n{output}\n```\n"
                        else:
                            error = er["result"].get("error", "Unknown error")
                            response += f"Error: {error}\n"

                    code_artifacts["execution_results"] = execution_results

            return AgentResult(
                success=True,
                response=response,
                agent=self.name,
                tools_used=tools_used,
                confidence=0.9 if not is_execute else (0.85 if execution_results and all(r["result"].get("success") for r in execution_results) else 0.7),
                artifacts=code_artifacts,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"[Coder] Error: {e}")
            return AgentResult(
                success=False,
                response=f"Coding error: {e}",
                agent=self.name,
                tools_used=tools_used,
                execution_time=time.time() - start_time
            )

    def _handle_git(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str],
        system_prompt: str,
        start_time: float
    ) -> AgentResult:
        """Handle git-specific operations."""
        tools_used = []

        if "git" not in self._available_tools:
            return AgentResult(
                success=False,
                response="Git tool not available.",
                agent=self.name,
                execution_time=time.time() - start_time
            )

        # Let LLM decide git command
        git_prompt = system_prompt + """

For git operations, use: [TOOL: git] git_command
Examples:
- [TOOL: git] status
- [TOOL: git] add -A
- [TOOL: git] commit -m "message"
- [TOOL: git] push"""

        response = llm_func(git_prompt, message.content)

        # Execute git commands
        tool_calls = self._parse_tool_calls(response)
        git_results = []

        for call in tool_calls:
            if call["tool"] == "git":
                result = self._execute_tool("git", call["action"])
                tools_used.append("git")
                git_results.append({
                    "command": call["action"],
                    "result": result
                })

        # Format response with git results
        if git_results:
            response += "\n\n**Git Operations:**\n"
            for gr in git_results:
                response += f"\n`git {gr['command']}`:\n"
                if gr["result"].get("success"):
                    output = gr["result"].get("output", "Success")
                    response += f"```\n{output}\n```\n"
                else:
                    error = gr["result"].get("error", "Failed")
                    response += f"Error: {error}\n"

        return AgentResult(
            success=True,
            response=response,
            agent=self.name,
            tools_used=tools_used,
            artifacts={"git_results": git_results},
            execution_time=time.time() - start_time
        )

    def _extract_code_blocks(self, text: str) -> list:
        """Extract code blocks from markdown-formatted text."""
        blocks = []

        # Pattern for fenced code blocks
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        for lang, code in matches:
            blocks.append({
                "language": lang or "text",
                "code": code.strip()
            })

        return blocks
