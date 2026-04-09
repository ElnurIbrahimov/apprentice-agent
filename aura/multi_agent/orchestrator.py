"""Multi-Agent Orchestrator.

Coordinates specialist agents to handle complex tasks through
routing, collaboration, and response synthesis.
"""

import concurrent.futures
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .base_agent import BaseSpecialist
from .protocol import (
    AgentMessage,
    AgentResult,
    CollaborationMode,
    ConversationTurn,
    RoutingDecision,
)
from .router import IntentRouter
from .specialists import (
    AnalystAgent,
    CoderAgent,
    CreativeAgent,
    ResearchAgent,
    SearcherAgent,
)

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Orchestrates multiple specialist agents for complex tasks.

    The orchestrator:
    1. Routes incoming queries to appropriate specialists
    2. Coordinates multi-agent collaboration
    3. Synthesizes responses from multiple agents
    4. Maintains conversation context
    """

    def __init__(
        self,
        tool_registry: Dict[str, Any],
        llm_func: Callable[[str, str], str],
        specialists: Optional[Dict[str, BaseSpecialist]] = None
    ):
        """Initialize the orchestrator.

        Args:
            tool_registry: Shared tool instances
            llm_func: Function to call LLM (system_prompt, user_message) -> response
            specialists: Optional custom specialists (uses defaults if None)
        """
        self.tool_registry = tool_registry
        self.llm_func = llm_func

        # Initialize specialists
        if specialists:
            self.specialists = specialists
        else:
            self.specialists = self._create_default_specialists()

        # Initialize router
        self.router = IntentRouter(self.specialists)

        # Shared pool — centralized in aura.pools
        from aura.pools import bg_pool
        self._executor = bg_pool()

        # Conversation history
        self.history: List[ConversationTurn] = []

        logger.info(f"[Orchestrator] Initialized with {len(self.specialists)} specialists")

    def _create_default_specialists(self) -> Dict[str, BaseSpecialist]:
        """Create default specialist agents."""
        return {
            "research": ResearchAgent(self.tool_registry),
            "coder": CoderAgent(self.tool_registry),
            "analyst": AnalystAgent(self.tool_registry),
            "creative": CreativeAgent(self.tool_registry),
            "searcher": SearcherAgent(self.tool_registry),
        }

    def chat(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a user query and return the response.

        This is the main entry point for the multi-agent system.

        Args:
            query: User's message
            context: Optional additional context

        Returns:
            Final response string
        """
        start_time = time.time()

        # Create message
        message = AgentMessage(
            content=query,
            sender="user",
            context=context or {}
        )

        # Add conversation context
        if self.history:
            last_turn = self.history[-1]
            message.context["previous_query"] = last_turn.user_message.content
            message.context["previous_response"] = last_turn.final_response[:500]

        # Route to specialists
        routing = self.router.route(query, self.llm_func)
        logger.info(f"[Orchestrator] Routing: {routing.agents} ({routing.mode.value})")

        # Execute based on collaboration mode
        if routing.mode == CollaborationMode.SINGLE:
            results = [self._execute_single(routing.agents[0], message)]

        elif routing.mode == CollaborationMode.PARALLEL:
            results = self._execute_parallel(routing.agents, message)

        elif routing.mode == CollaborationMode.SEQUENTIAL:
            results = self._execute_sequential(routing.agents, message)

        elif routing.mode == CollaborationMode.DEBATE:
            results = self._execute_debate(routing.agents, message)

        else:
            # Fallback
            results = [self._execute_single(routing.agents[0], message)]

        # Synthesize final response
        final_response = self._synthesize_response(results, routing)

        # Record turn
        turn = ConversationTurn(
            user_message=message,
            routing=routing,
            results=results,
            final_response=final_response
        )
        self.history.append(turn)

        # Keep history manageable
        if len(self.history) > 20:
            self.history = self.history[-15:]

        exec_time = time.time() - start_time
        logger.info(f"[Orchestrator] Completed in {exec_time:.2f}s")

        return final_response

    def _execute_single(self, agent_name: str, message: AgentMessage) -> AgentResult:
        """Execute a single agent."""
        if agent_name not in self.specialists:
            return AgentResult(
                success=False,
                response=f"Agent '{agent_name}' not found",
                agent=agent_name
            )

        agent = self.specialists[agent_name]
        return agent.execute(message, self.llm_func)

    def _execute_parallel(
        self,
        agent_names: List[str],
        message: AgentMessage
    ) -> List[AgentResult]:
        """Execute multiple agents in parallel."""
        results = []

        futures = {
            self._executor.submit(self._execute_single, name, message): name
            for name in agent_names if name in self.specialists
        }

        if futures:
            try:
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        agent_name = futures[future]
                        logger.error(f"[Orchestrator] Parallel execution error for {agent_name}: {e}")
                        results.append(AgentResult(
                            success=False,
                            response=f"Error: {e}",
                            agent=agent_name
                        ))
            except concurrent.futures.TimeoutError:
                logger.error("[Orchestrator] Parallel execution timed out after 60s")
                for future, name in futures.items():
                    if not future.done():
                        future.cancel()
                        results.append(AgentResult(
                            success=False,
                            response="Timed out after 60s",
                            agent=name
                        ))

        return results

    def _execute_sequential(
        self,
        agent_names: List[str],
        message: AgentMessage
    ) -> List[AgentResult]:
        """Execute agents in sequence, passing context between them."""
        results = []
        current_message = message

        for agent_name in agent_names:
            if agent_name not in self.specialists:
                continue

            # Execute agent
            result = self._execute_single(agent_name, current_message)
            results.append(result)

            if not result.success:
                break  # Stop chain on failure

            # Pass result to next agent via context
            current_message = current_message.with_context(
                "previous_result", result.to_context()
            )
            current_message.context.update(result.to_context())

        return results

    def _execute_debate(
        self,
        agent_names: List[str],
        message: AgentMessage
    ) -> List[AgentResult]:
        """Execute agents in a debate format.

        Agent 1 proposes -> Agent 2 critiques -> Agent 1 revises
        """
        if len(agent_names) < 2:
            return self._execute_sequential(agent_names, message)

        results = []

        # Round 1: First agent proposes
        proposer = agent_names[0]
        proposal = self._execute_single(proposer, message)
        results.append(proposal)

        if not proposal.success:
            return results

        # Round 2: Second agent critiques
        critic = agent_names[1]
        critique_message = message.with_context("proposal", {
            "from": proposer,
            "content": proposal.response
        })
        critique_message.content = f"Critique this proposal and suggest improvements:\n\n{proposal.response[:1000]}"

        critique = self._execute_single(critic, critique_message)
        results.append(critique)

        # Round 3: First agent revises based on critique
        if critique.success:
            revision_message = message.with_context("critique", {
                "from": critic,
                "content": critique.response
            })
            revision_message.content = f"""Original request: {message.content}

Your proposal: {proposal.response[:500]}

Critique received: {critique.response[:500]}

Please revise your response based on the critique."""

            revision = self._execute_single(proposer, revision_message)
            results.append(revision)

        return results

    def _synthesize_response(
        self,
        results: List[AgentResult],
        routing: RoutingDecision
    ) -> str:
        """Synthesize final response from agent results."""
        if not results:
            return "No agents were able to handle your request."

        # Single agent - just return its response
        if len(results) == 1:
            result = results[0]
            if result.success:
                return result.response
            else:
                return f"Error from {result.agent}: {result.response}"

        # Multiple agents - synthesize
        successful_results = [r for r in results if r.success]

        if not successful_results:
            errors = "\n".join([f"- {r.agent}: {r.response}" for r in results])
            return f"All agents encountered errors:\n{errors}"

        # For parallel: merge responses
        if routing.mode == CollaborationMode.PARALLEL:
            synthesis_prompt = """You are synthesizing responses from multiple AI specialists.
Combine their insights into a unified, coherent response.
Remove redundancy but preserve unique contributions from each."""

            combined = "\n\n---\n\n".join([
                f"**{r.agent.title()} Agent:**\n{r.response}"
                for r in successful_results
            ])

            return self.llm_func(
                synthesis_prompt,
                f"Synthesize these specialist responses:\n\n{combined}"
            )

        # For sequential: use last successful result
        elif routing.mode == CollaborationMode.SEQUENTIAL:
            last_result = successful_results[-1]
            # Add attribution
            agents_involved = " -> ".join([r.agent for r in successful_results])
            return f"*[Agents: {agents_involved}]*\n\n{last_result.response}"

        # For debate: use the final revision
        elif routing.mode == CollaborationMode.DEBATE:
            if len(successful_results) >= 3:
                return successful_results[-1].response  # Final revision
            elif len(successful_results) == 2:
                # Proposal and critique, no revision
                return f"""**Proposal ({successful_results[0].agent}):**
{successful_results[0].response}

**Critique ({successful_results[1].agent}):**
{successful_results[1].response}"""
            else:
                return successful_results[0].response

        # Default: join responses
        return "\n\n---\n\n".join([r.response for r in successful_results])

    def route_preview(self, query: str) -> Dict[str, Any]:
        """Preview routing decision without executing.

        Useful for debugging and transparency.
        """
        # Preview uses pattern-only routing without LLM fallback
        routing = self.router.route(query, llm_func=None)
        scores = self.router._score_specialists(query.lower())

        return {
            "query": query,
            "selected_agents": routing.agents,
            "mode": routing.mode.value,
            "reasoning": routing.reasoning,
            "confidence": routing.confidence,
            "all_scores": {name: score for name, score in scores}
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "specialists": list(self.specialists.keys()),
            "specialist_details": {
                name: {
                    "description": spec.description,
                    "tools": spec.tools,
                    "triggers": spec.triggers[:5]
                }
                for name, spec in self.specialists.items()
            },
            "conversation_turns": len(self.history),
            "available_tools": list(self.tool_registry.keys())
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        logger.info("[Orchestrator] History cleared")
