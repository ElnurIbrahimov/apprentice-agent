"""Searcher Agent - Code search and codebase exploration specialist.

Specialized in navigating and searching codebases:
- Grep for patterns across files
- Find definitions (functions, classes, variables)
- Find references to symbols
- Explore project structure
"""

import logging
import re
import time
from typing import Callable

from ..base_agent import ToolUsingSpecialist
from ..protocol import AgentMessage, AgentResult

logger = logging.getLogger(__name__)


class SearcherAgent(ToolUsingSpecialist):
    """Specialist for searching codebases: grep, definitions, references, structure."""

    name = "searcher"
    description = "Search codebases: grep, find definitions, references, project structure"

    system_prompt = """You are a Code Search Specialist. You excel at navigating codebases quickly.

Your expertise:
- Finding function/class definitions across large codebases
- Grepping for patterns, error messages, and usage examples
- Mapping project structure and key files
- Tracing references and call chains

Your approach:
1. Identify what the user is looking for (definition, usage, pattern, structure)
2. Use the most efficient search method
3. Present results clearly with file paths and line numbers

When searching code, use: [TOOL: code_search] action
When reading files for context, use: [TOOL: code_edit] read_file path"""

    tools = ["code_search", "code_edit"]

    triggers = [
        "find", "where", "search code", "grep", "definition",
        "reference", "locate", "which file", "codebase",
        "project structure", "find function", "find class",
    ]

    max_tool_calls = 5

    def execute(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str]
    ) -> AgentResult:
        """Execute search by detecting intent and calling CodeSearchTool directly."""
        start_time = time.time()
        tools_used = []
        artifacts = {}

        try:
            query = message.content
            query_lower = query.lower()
            search_tool = self.get_tool("code_search")

            if not search_tool:
                # Fallback to LLM-based response
                return super().execute(message, llm_func)

            result = None

            # Detect search intent and call appropriate method
            if re.search(r'\b(definition|def |class |where is .+ defined)\b', query_lower):
                # Extract the name to find
                name = self._extract_name(query)
                if name:
                    result = search_tool.find_definition(name=name)
                    tools_used.append("code_search.find_definition")

            elif re.search(r'\b(reference|usage|used|called|imported)\b', query_lower):
                name = self._extract_name(query)
                if name:
                    result = search_tool.find_references(name=name)
                    tools_used.append("code_search.find_references")

            elif re.search(r'\b(structure|tree|layout|overview)\b', query_lower):
                path = self._extract_path(query) or "."
                result = search_tool.project_structure(path=path)
                tools_used.append("code_search.project_structure")

            elif re.search(r'\b(grep|search for|find .+ in|pattern)\b', query_lower):
                pattern = self._extract_pattern(query)
                if pattern:
                    path = self._extract_path(query) or "."
                    result = search_tool.grep(pattern=pattern, path=path)
                    tools_used.append("code_search.grep")

            # If no specific intent matched, try grep with the query as pattern
            if result is None:
                name = self._extract_name(query)
                if name:
                    result = search_tool.find_definition(name=name)
                    tools_used.append("code_search.find_definition")
                    # If no definitions, try grep
                    if not result.get("success") or not result.get("definitions"):
                        result = search_tool.grep(pattern=name, path=".")
                        tools_used[-1] = "code_search.grep"

            # Format results
            if result and result.get("success"):
                response = self._format_result(result, query)
                artifacts["search_results"] = result
            else:
                # Fallback: let LLM answer
                system_prompt = self.get_system_prompt(message.context)
                response = llm_func(system_prompt, query)

            return AgentResult(
                success=True,
                response=response,
                agent=self.name,
                tools_used=tools_used,
                confidence=0.9 if tools_used else 0.6,
                artifacts=artifacts,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"[Searcher] Error: {e}")
            return AgentResult(
                success=False,
                response=f"Search error: {e}",
                agent=self.name,
                tools_used=tools_used,
                execution_time=time.time() - start_time,
            )

    def _extract_name(self, query: str) -> str:
        """Extract a symbol/function/class name from the query."""
        # Try quoted strings first
        m = re.search(r"['\"](\w+)['\"]", query)
        if m:
            return m.group(1)
        # Try "find X", "definition of X", "where is X"
        m = re.search(r"(?:find|definition of|where is|locate|references? (?:to|of))\s+(\w+)", query, re.I)
        if m:
            return m.group(1)
        # Last word that looks like an identifier
        words = re.findall(r'\b[A-Za-z_]\w*\b', query)
        # Filter common words
        skip = {"find", "where", "is", "the", "in", "all", "search", "for", "code",
                "definition", "reference", "class", "function", "method", "grep", "project"}
        candidates = [w for w in words if w.lower() not in skip]
        return candidates[-1] if candidates else ""

    def _extract_path(self, query: str) -> str:
        """Extract a file path from the query."""
        m = re.search(r'(?:in|from|at)\s+([./\w-]+(?:/[./\w-]+)*)', query)
        return m.group(1) if m else ""

    def _extract_pattern(self, query: str) -> str:
        """Extract a grep pattern from the query."""
        m = re.search(r"(?:grep|search for|find)\s+['\"](.+?)['\"]", query, re.I)
        if m:
            return m.group(1)
        m = re.search(r"(?:grep|search for|find)\s+(\S+)", query, re.I)
        return m.group(1) if m else ""

    def _format_result(self, result: dict, query: str) -> str:
        """Format search results into a readable response."""
        parts = []

        if "definitions" in result:
            defs = result["definitions"]
            parts.append(f"Found {len(defs)} definition(s):\n")
            for d in defs[:15]:
                parts.append(f"  {d['file']}:{d['line']} ({d['kind']}) — {d['text']}")

        elif "references" in result:
            refs = result["references"]
            parts.append(f"Found {len(refs)} reference(s):\n")
            for r in refs[:20]:
                parts.append(f"  {r['file']}:{r['line']} — {r['text']}")

        elif "matches" in result:
            matches = result["matches"]
            total = result.get("total_matches", len(matches))
            parts.append(f"Found {total} match(es):\n")
            for m in matches[:20]:
                parts.append(f"  {m['file']}:{m['line']} — {m['text']}")
            if total > 20:
                parts.append(f"\n  ... and {total - 20} more matches")

        elif "tree" in result:
            parts.append(result["tree"])
            stats = result.get("stats", {})
            parts.append(f"\n{stats.get('files', 0)} files, {stats.get('dirs', 0)} dirs")

        else:
            parts.append(str(result))

        return "\n".join(parts)
