"""
Reasoning Tree Tool for AURA
============================

Tool interface for MCTS-based deep reasoning.
Enables AURA to explore multiple reasoning paths for complex problems.

Usage:
    "Think deeply about X"
    "Explore different approaches to Y"
    "Reason through Z step by step"
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .mcts_reasoning import (
    MCTSReasoning,
    MCTSConfig,
    MCTSResult,
    MCTSNode,
    ThoughtType,
)

logger = logging.getLogger(__name__)


class ReasoningTreeTool:
    """
    Tool for deep reasoning using Monte Carlo Tree Search.

    This tool enables AURA to:
    - Explore multiple reasoning paths simultaneously
    - Backtrack from dead ends automatically
    - Learn from failed approaches via reflection
    - Find optimal solutions through deliberate search
    """

    name = "reasoning_tree"
    description = """Deep reasoning tool that explores multiple solution paths simultaneously.
    Use this for complex problems that benefit from exploring different approaches.
    The tool builds a tree of thoughts and finds the best reasoning path."""

    # Keywords that trigger this tool
    keywords = [
        "think deeply",
        "deep reasoning",
        "explore approaches",
        "reasoning tree",
        "multiple paths",
        "deliberate",
        "step by step reasoning",
        "complex problem",
        "analyze thoroughly",
        "mcts",
        "tree of thoughts",
    ]

    def __init__(
        self,
        llm_func: Callable[[str, Optional[str]], str],
        config: Optional[MCTSConfig] = None,
    ):
        """
        Initialize the Reasoning Tree Tool.

        Args:
            llm_func: Function that takes (prompt, system_prompt) and returns LLM response
            config: Optional MCTS configuration
        """
        self.llm_func = llm_func
        self.config = config or MCTSConfig(
            max_iterations=30,
            max_depth=10,
            branching_factor=5,
            exploration_weight=1.414,
            timeout_seconds=120.0,
            enable_reflection=True,
            enable_pruning=True,
        )

        self.mcts: Optional[MCTSReasoning] = None
        self.last_result: Optional[MCTSResult] = None
        self._lock = threading.Lock()

        # Callbacks for UI updates
        self._progress_callback: Optional[Callable[[Dict], None]] = None

    def set_progress_callback(self, callback: Callable[[Dict], None]):
        """Set callback for progress updates"""
        self._progress_callback = callback

    def think_deeply(
        self,
        problem: str,
        context: str = "",
        max_iterations: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform deep reasoning on a problem using MCTS.

        Args:
            problem: The problem or question to reason about
            context: Additional context (e.g., conversation history)
            max_iterations: Override max iterations
            max_depth: Override max depth

        Returns:
            Dictionary with reasoning result
        """
        logger.info(f"Starting deep reasoning: {problem[:100]}...")

        # Create config with overrides
        config = MCTSConfig(
            max_iterations=max_iterations or self.config.max_iterations,
            max_depth=max_depth or self.config.max_depth,
            branching_factor=self.config.branching_factor,
            exploration_weight=self.config.exploration_weight,
            timeout_seconds=self.config.timeout_seconds,
            enable_reflection=self.config.enable_reflection,
            enable_pruning=self.config.enable_pruning,
        )

        # Initialize MCTS
        mcts = MCTSReasoning(
            llm_func=self.llm_func,
            config=config,
        )

        # Set up progress reporting
        if self._progress_callback:
            mcts.on_iteration_complete = self._on_iteration

        # Run search
        result = mcts.search(problem, context)
        with self._lock:
            self.mcts = mcts
            self.last_result = result

        # Format output
        return self._format_result(result)

    def explore_options(
        self,
        question: str,
        num_options: int = 5,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Explore multiple options/approaches for a decision.

        Args:
            question: The decision question
            num_options: Number of options to explore
            context: Additional context

        Returns:
            Dictionary with explored options and analysis
        """
        logger.info(f"Exploring options for: {question[:100]}...")

        # Use a configuration optimized for option exploration
        config = MCTSConfig(
            max_iterations=num_options * 5,
            max_depth=5,
            branching_factor=num_options,
            exploration_weight=2.0,  # Higher exploration
            timeout_seconds=60.0,
            enable_reflection=False,  # Faster exploration
            enable_pruning=False,
        )

        mcts = MCTSReasoning(
            llm_func=self.llm_func,
            config=config,
        )

        result = mcts.search(question, context)
        with self._lock:
            self.mcts = mcts
            self.last_result = result

        # Extract options from first-level children
        options = []
        if result.tree and result.tree.children:
            for i, child in enumerate(result.tree.children):
                options.append({
                    "option": i + 1,
                    "description": child.thought.content,
                    "score": round(child.avg_value, 2),
                    "visits": child.visits,
                    "rationale": child.thought.metadata.get("rationale", ""),
                })

        if not options:
            return {"success": False, "error": "MCTS search produced no options. Check LLM availability."}
        sorted_options = sorted(options, key=lambda x: x["score"], reverse=True)
        best = sorted_options[0]
        return {
            "success": True,
            "question": question,
            "options": sorted_options,
            "recommendation": best["description"],
            "iterations": result.iterations,
            "time_taken": round(result.time_taken, 2),
        }

    def get_reasoning_path(self) -> Dict[str, Any]:
        """
        Get the best reasoning path from the last search.

        Returns:
            Dictionary with the reasoning steps
        """
        with self._lock:
            last_result = self.last_result
        if not last_result:
            return {"success": False, "error": "No reasoning performed yet"}

        steps = []
        for node in last_result.best_path:
            if node.thought.type != ThoughtType.ROOT:
                steps.append({
                    "step": len(steps) + 1,
                    "type": node.thought.type.value,
                    "content": node.thought.content,
                    "confidence": round(node.thought.confidence, 2),
                    "value": round(node.avg_value, 2),
                })

        return {
            "success": True,
            "steps": steps,
            "conclusion": last_result.best_answer,
            "confidence": round(last_result.confidence, 2),
        }

    def get_tree_visualization(self) -> Dict[str, Any]:
        """
        Get the full reasoning tree for visualization.

        Returns:
            Dictionary with tree structure
        """
        with self._lock:
            mcts = self.mcts
        if not mcts:
            return {"success": False, "error": "No reasoning tree available"}

        tree_data = mcts.get_tree_visualization()

        return {
            "success": True,
            "tree": tree_data.get("root", {}),
            "stats": tree_data.get("stats", {}),
        }

    def get_reflections(self) -> Dict[str, Any]:
        """
        Get reflections/lessons learned from the reasoning process.

        Returns:
            Dictionary with reflections
        """
        with self._lock:
            last_result = self.last_result
        if not last_result:
            return {"success": False, "error": "No reasoning performed yet"}

        reflections = []
        for r in last_result.reflections:
            reflections.append({
                "critique": r.critique,
                "lessons": r.lessons,
                "alternatives": r.suggested_alternatives,
            })

        return {
            "success": True,
            "reflections": reflections,
            "total_lessons": sum(len(r.lessons) for r in last_result.reflections),
        }

    def status(self) -> Dict[str, Any]:
        """
        Get the current status of the reasoning tool.

        Returns:
            Dictionary with status information
        """
        return {
            "success": True,
            "tool": "reasoning_tree",
            "description": "MCTS-based deep reasoning with tree search",
            "config": {
                "max_iterations": self.config.max_iterations,
                "max_depth": self.config.max_depth,
                "branching_factor": self.config.branching_factor,
                "exploration_weight": self.config.exploration_weight,
                "reflection_enabled": self.config.enable_reflection,
            },
            "last_result": {
                "available": self.last_result is not None,
                "iterations": self.last_result.iterations if self.last_result else 0,
                "confidence": round(self.last_result.confidence, 2) if self.last_result else 0,
            } if self.last_result else None,
        }

    def _on_iteration(self, iteration: int, root: MCTSNode):
        """Callback for iteration progress"""
        if self._progress_callback:
            self._progress_callback({
                "type": "iteration",
                "iteration": iteration,
                "nodes": self.mcts.nodes_created if self.mcts else 0,
                "reflections": len(self.mcts.reflections) if self.mcts else 0,
            })

    def _format_result(self, result: MCTSResult) -> Dict[str, Any]:
        """Format MCTS result for output"""
        # Build reasoning steps
        steps = []
        for node in result.best_path:
            if node.thought.type != ThoughtType.ROOT:
                steps.append({
                    "type": node.thought.type.value,
                    "content": node.thought.content,
                    "confidence": round(node.thought.confidence, 2),
                    "value": round(node.avg_value, 2),
                })

        # Build summary
        summary = f"""## Deep Reasoning Result

**Confidence:** {result.confidence:.0%}
**Iterations:** {result.iterations}
**Nodes Explored:** {result.nodes_explored}
**Time:** {result.time_taken:.1f}s

### Reasoning Path
"""
        for i, step in enumerate(steps, 1):
            summary += f"\n{i}. [{step['type']}] {step['content'][:200]}"

        summary += f"\n\n### Conclusion\n{result.best_answer}"

        if result.reflections:
            summary += "\n\n### Lessons Learned"
            for r in result.reflections[-2:]:  # Last 2 reflections
                for lesson in r.lessons[:2]:
                    summary += f"\n- {lesson}"

        return {
            "success": result.success,
            "answer": result.best_answer,
            "confidence": round(result.confidence, 2),
            "reasoning_steps": steps,
            "summary": summary,
            "metadata": {
                "iterations": result.iterations,
                "nodes_explored": result.nodes_explored,
                "time_taken": round(result.time_taken, 2),
                "reflections_count": len(result.reflections),
                **result.metadata,
            },
        }

    # Main execution method for tool interface
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a reasoning action.

        Args:
            action: The action to perform (think_deeply, explore_options, status, etc.)
            **kwargs: Additional arguments for the action

        Returns:
            Result dictionary
        """
        action = action.lower().strip()

        if action in ["think", "think_deeply", "reason", "analyze"]:
            problem = kwargs.get("problem") or kwargs.get("query") or kwargs.get("input", "")
            return self.think_deeply(
                problem=problem,
                context=kwargs.get("context", ""),
                max_iterations=kwargs.get("max_iterations"),
                max_depth=kwargs.get("max_depth"),
            )

        elif action in ["explore", "explore_options", "options"]:
            question = kwargs.get("question") or kwargs.get("query") or kwargs.get("input", "")
            return self.explore_options(
                question=question,
                num_options=kwargs.get("num_options", 5),
                context=kwargs.get("context", ""),
            )

        elif action in ["path", "reasoning_path", "steps"]:
            return self.get_reasoning_path()

        elif action in ["tree", "visualization", "visualize"]:
            return self.get_tree_visualization()

        elif action in ["reflections", "lessons"]:
            return self.get_reflections()

        elif action in ["status", "info"]:
            return self.status()

        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "available_actions": [
                    "think_deeply", "explore_options", "reasoning_path",
                    "tree", "reflections", "status"
                ],
            }


# Quick function for integration
def deep_reason(
    problem: str,
    llm_func: Callable[[str, Optional[str]], str],
    context: str = "",
) -> Dict[str, Any]:
    """
    Quick function to perform deep reasoning on a problem.

    Args:
        problem: The problem to reason about
        llm_func: LLM function
        context: Additional context

    Returns:
        Reasoning result
    """
    tool = ReasoningTreeTool(llm_func=llm_func)
    return tool.think_deeply(problem, context)
