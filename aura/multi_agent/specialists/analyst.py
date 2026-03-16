"""Analyst Agent - Data analysis and reasoning specialist.

Specialized in analytical tasks:
- Data analysis and interpretation
- Logical reasoning
- Fact-checking and verification
- Knowledge graph queries
- Document analysis with RAG
"""

import logging
import time
from typing import Any, Callable, Dict

from ..base_agent import ToolUsingSpecialist
from ..protocol import AgentMessage, AgentResult

logger = logging.getLogger(__name__)


class AnalystAgent(ToolUsingSpecialist):
    """Specialist for data analysis, reasoning, and fact verification."""

    name = "analyst"
    description = "Analyze data, reason about facts, and provide explanations"

    system_prompt = """You are a Data Analyst and Critical Thinker AI assistant. You excel at analyzing information, finding patterns, and providing well-reasoned explanations.

Your expertise:
- Data analysis and interpretation
- Logical reasoning and argumentation
- Fact-checking and verification
- Pattern recognition
- Clear explanations of complex topics

Your approach:
1. Gather relevant data from available sources
2. Analyze the information systematically
3. Identify patterns, correlations, and insights
4. Consider multiple perspectives
5. Present findings with supporting evidence

Guidelines:
- Base conclusions on evidence
- Acknowledge uncertainty when present
- Consider alternative explanations
- Use clear, structured reasoning
- Cite sources and data points

When querying knowledge, use: [TOOL: knowledge_graph] query
When searching documents, use: [TOOL: local_rag] search query"""

    tools = [
        "knowledge_graph",
        "local_rag",
    ]

    triggers = [
        "analyze", "explain", "why", "how does",
        "compare", "evaluate", "understand", "reason",
        "logic", "data", "knowledge", "fact", "pattern",
        "evidence", "conclude", "interpret"
    ]

    def execute(
        self,
        message: AgentMessage,
        llm_func: Callable[[str, str], str]
    ) -> AgentResult:
        """Execute analytical task with knowledge retrieval."""
        start_time = time.time()
        tools_used = []
        analysis_data = {}

        try:
            system_prompt = self.get_system_prompt(message.context)
            query = message.content

            # Step 1: Gather context from knowledge sources
            context_parts = []

            # Check knowledge graph
            if "knowledge_graph" in self._available_tools:
                logger.info("[Analyst] Querying knowledge graph...")
                kg_result = self._execute_tool("knowledge_graph", f"query {query}")
                tools_used.append("knowledge_graph")
                if kg_result.get("success") and kg_result.get("results"):
                    context_parts.append(f"Knowledge Graph:\n{kg_result['results']}")
                    analysis_data["kg_results"] = kg_result

            # Check local documents (RAG)
            if "local_rag" in self._available_tools:
                logger.info("[Analyst] Searching local documents...")
                rag_result = self._execute_tool("local_rag", f"search {query}")
                tools_used.append("local_rag")
                if rag_result.get("success") and rag_result.get("results"):
                    context_parts.append(f"Document Search:\n{rag_result['results']}")
                    analysis_data["rag_results"] = rag_result

            # Step 2: Synthesize analysis
            fluxmind_reasoning = None
            analysis_prompt = system_prompt

            if context_parts:
                analysis_prompt += "\n\n**Available Context:**\n"
                analysis_prompt += "\n---\n".join(context_parts)

            if fluxmind_reasoning:
                analysis_prompt += f"\n\n**Calibrated Reasoning (FluxMind):**\n{fluxmind_reasoning}"

            analysis_prompt += """

Now provide a thorough analysis. Structure your response:
1. **Summary**: Brief overview of the topic
2. **Analysis**: Detailed examination with evidence
3. **Insights**: Key findings and patterns
4. **Confidence**: How certain are you about the conclusions"""

            response = llm_func(analysis_prompt, message.content)

            # Determine confidence based on sources used
            confidence = 0.6  # Base confidence
            if "knowledge_graph" in tools_used:
                confidence += 0.1
            if "local_rag" in tools_used:
                confidence += 0.15
            confidence = min(confidence, 0.95)

            return AgentResult(
                success=True,
                response=response,
                agent=self.name,
                tools_used=tools_used,
                confidence=confidence,
                artifacts=analysis_data,
                thinking=f"Analyzed using {len(tools_used)} tools. Context sources: {len(context_parts)}",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"[Analyst] Error: {e}")
            return AgentResult(
                success=False,
                response=f"Analysis error: {e}",
                agent=self.name,
                tools_used=tools_used,
                execution_time=time.time() - start_time
            )
