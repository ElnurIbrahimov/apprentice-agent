"""Research Agent - Information gathering specialist.

Specialized in finding information from various sources:
- Web search
- Academic papers (arXiv)
- PDF documents
- Browser automation
"""

import logging
import time
from typing import Callable

from ..base_agent import ToolUsingSpecialist
from ..protocol import AgentMessage, AgentResult

logger = logging.getLogger(__name__)


class ResearchAgent(ToolUsingSpecialist):
    """Specialist for information gathering and research tasks."""

    name = "research"
    description = "Find and gather information from web, papers, and documents"

    system_prompt = """You are a Research Specialist AI assistant. Your expertise is finding accurate, relevant information from various sources.

Your approach:
1. Understand what information the user needs
2. Determine the best source (web search, academic papers, local documents)
3. Search and gather relevant information
4. Synthesize findings into a clear, well-sourced response

Guidelines:
- Always cite your sources
- Distinguish between facts and opinions
- Acknowledge when information might be outdated
- If you can't find something, say so honestly
- Prioritize authoritative sources

When using tools, format as: [TOOL: tool_name] your action
Example: [TOOL: web_search] latest developments in AI safety 2024"""

    tools = [
        "web_search",
        "arxiv_search",
        "pdf_reader",
        "browser",
        "local_rag",
    ]

    triggers = [
        "search", "find", "look up", "research", "google",
        "web", "online", "arxiv", "paper", "article",
        "news", "latest", "current", "discover"
    ]

    max_tool_calls = 3

    def _execute_tool(self, tool_name, action):
        """Override to use fallback chain for web search."""
        if tool_name in ("web_search", "search_web", "search"):
            try:
                from aura.tools.search_fallback import web_search_with_fallback
                return web_search_with_fallback(query=action, max_results=8)
            except ImportError:
                pass
        return super()._execute_tool(tool_name, action)

    def execute(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str]
    ) -> AgentResult:
        """Execute research task with multi-source search."""
        start_time = time.time()
        tools_used = []
        sources = []

        try:
            # Determine search strategy
            query = message.content.lower()

            # Academic query detection
            is_academic = any(kw in query for kw in [
                "paper", "research", "study", "arxiv", "academic",
                "scientific", "journal", "publication"
            ])

            # Get system prompt with context
            system_prompt = self.get_system_prompt(message.context)

            # First, let LLM plan the search
            plan_prompt = system_prompt + """

Based on the user's query, plan your research approach.
What sources should you check? What search terms should you use?
Format tool calls as: [TOOL: tool_name] search query"""

            plan_response = llm_func(plan_prompt, message.content)

            # Parse and execute tool calls from plan
            tool_calls = self._parse_tool_calls(plan_response)

            # If no explicit tool calls, add default search
            if not tool_calls:
                if is_academic:
                    tool_calls.append({"tool": "arxiv_search", "action": message.content})
                else:
                    tool_calls.append({"tool": "web_search", "action": message.content})

            # Execute searches
            search_results = []
            citations = []
            for call in tool_calls[:self.max_tool_calls]:
                tool_name = call["tool"]
                action = call["action"]

                if tool_name not in self._available_tools:
                    # Try fallback tools
                    if tool_name == "arxiv_search" and "web_search" in self._available_tools:
                        tool_name = "web_search"
                        action = f"arxiv {action}"
                    else:
                        continue

                logger.info(f"[Research] Searching with {tool_name}: {action[:50]}...")
                result = self._execute_tool(tool_name, action)
                tools_used.append(tool_name)

                if result.get("success"):
                    search_results.append({
                        "source": tool_name,
                        "query": action,
                        "result": result
                    })
                    sources.append(tool_name)

                    # Collect citations from tool results
                    if "citations" in result:
                        # Tavily-style: already extracted citations
                        for c in result["citations"]:
                            c["id"] = len(citations) + 1
                            citations.append(c)
                    elif "results" in result and isinstance(result["results"], list):
                        # Web search style: extract from results list
                        for item in result["results"]:
                            if item.get("url"):
                                citations.append({
                                    "id": len(citations) + 1,
                                    "title": item.get("title", item.get("url", "")),
                                    "url": item["url"],
                                    "snippet": item.get("snippet", item.get("content", ""))[:200],
                                    "score": item.get("score", 0),
                                })

            # Build sources section for synthesis prompt
            sources_text = ""
            if citations:
                sources_text = "\n\nSources:\n"
                for c in citations[:10]:
                    sources_text += f"[{c['id']}] {c['title']} — {c['url']}\n"

            # Synthesize findings
            if search_results:
                synthesis_prompt = system_prompt + "\n\nSearch Results:\n"
                for sr in search_results:
                    synthesis_prompt += f"\n[{sr['source']}] Query: {sr['query']}\n"
                    result_text = str(sr['result'])[:1000]  # Truncate long results
                    synthesis_prompt += f"Results: {result_text}\n"

                synthesis_prompt += sources_text
                synthesis_prompt += "\n\nBased on these search results, provide a comprehensive answer. Use [1], [2] etc inline to cite sources by number when referencing information."

                final_response = llm_func(synthesis_prompt, message.content)
            else:
                # No search results, use LLM knowledge
                final_response = llm_func(
                    system_prompt,
                    f"I couldn't find external sources. Please answer based on your knowledge: {message.content}"
                )
                final_response = f"*Note: No external sources found. Based on general knowledge:*\n\n{final_response}"

            return AgentResult(
                success=True,
                response=final_response,
                agent=self.name,
                tools_used=tools_used,
                confidence=0.85 if search_results else 0.6,
                artifacts={"search_results": search_results, "sources": sources, "citations": citations},
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"[Research] Error: {e}")
            return AgentResult(
                success=False,
                response=f"Research error: {e}",
                agent=self.name,
                tools_used=tools_used,
                execution_time=time.time() - start_time
            )
