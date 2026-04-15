"""
MCTS Reasoning Tree for AURA
============================

Monte Carlo Tree Search implementation for LLM reasoning, based on:
- LATS (Language Agent Tree Search) - ICML 2024
- Tree of Thoughts (ToT) - NeurIPS 2023
- ReST-MCTS* - NeurIPS 2024

This enables AURA to explore multiple reasoning paths simultaneously,
backtrack when needed, and find optimal solutions through deliberate search.

Core Algorithm:
1. Selection - Navigate tree using UCT (Upper Confidence Bound for Trees)
2. Expansion - Generate candidate thoughts/actions using LLM
3. Evaluation - Score each state using LLM as value function
4. Backpropagation - Update values up the tree
5. Reflection - Generate self-critique on failed paths

References:
- https://arxiv.org/abs/2310.04406 (LATS)
- https://arxiv.org/abs/2305.10601 (Tree of Thoughts)
- https://github.com/THUDM/ReST-MCTS
"""

import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """State of a reasoning node"""
    PENDING = "pending"      # Not yet evaluated
    EXPLORING = "exploring"  # Currently being explored
    EVALUATED = "evaluated"  # Has been scored
    TERMINAL = "terminal"    # Reached conclusion (success or failure)
    PRUNED = "pruned"        # Pruned from search


class ThoughtType(Enum):
    """Type of thought/action at a node"""
    ROOT = "root"            # Initial problem statement
    REASONING = "reasoning"  # Chain of thought step
    ACTION = "action"        # Tool invocation
    OBSERVATION = "observation"  # Result from action
    CONCLUSION = "conclusion"    # Final answer
    REFLECTION = "reflection"    # Self-critique


@dataclass
class Thought:
    """Represents a single thought or action in the reasoning tree"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ThoughtType = ThoughtType.REASONING
    content: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Reflection:
    """Self-reflection on a reasoning path"""
    critique: str = ""
    lessons: List[str] = field(default_factory=list)
    suggested_alternatives: List[str] = field(default_factory=list)
    score: float = 0.0


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search reasoning tree.

    Each node represents a state in the reasoning process, containing:
    - The thought/action that led to this state
    - Value estimates from evaluations
    - Visit counts for UCT calculation
    - Links to parent and children
    """

    def __init__(
        self,
        thought: Thought,
        parent: Optional['MCTSNode'] = None,
        context: str = "",
    ):
        self.id = str(uuid.uuid4())[:8]
        self.thought = thought
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.context = context  # Accumulated context up to this point

        # Lock for thread-safe backpropagation
        self._lock = threading.Lock()

        # MCTS statistics
        self.visits = 0
        self.value = 0.0  # Cumulative value
        self.avg_value = 0.0  # Average value (value / visits)

        # Evaluation results
        self.state = NodeState.PENDING
        self.reflection: Optional[Reflection] = None
        self.is_terminal = False
        self.is_successful = False

        # Tree metadata
        self.depth = parent.depth + 1 if parent else 0
        self.created_at = datetime.now().isoformat()
        self.expanded_at: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0

    @property
    def is_fully_expanded(self) -> bool:
        """Check if all children have been visited at least once"""
        return all(child.visits > 0 for child in self.children)

    def get_ucb1(self, exploration_weight: float = 1.414) -> float:
        """
        Calculate Upper Confidence Bound for Trees (UCT/UCB1)

        UCB1 = V(s)/N(s) + c * sqrt(ln(N(parent)) / N(s))

        Where:
        - V(s) = cumulative value of this node
        - N(s) = visit count of this node
        - N(parent) = visit count of parent
        - c = exploration weight (default sqrt(2) ≈ 1.414)
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have infinite UCB

        if self.parent is None:
            return self.avg_value

        exploitation = self.avg_value
        exploration = exploration_weight * math.sqrt(
            math.log(max(1, self.parent.visits)) / self.visits
        )

        return exploitation + exploration

    def select_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """Select the child with highest UCB1 score"""
        if not self.children:
            return self

        return max(self.children, key=lambda c: c.get_ucb1(exploration_weight))

    def add_child(self, thought: Thought, context: str = "") -> 'MCTSNode':
        """Add a new child node"""
        child = MCTSNode(
            thought=thought,
            parent=self,
            context=context or self.context
        )
        self.children.append(child)
        return child

    def backpropagate(self, value: float) -> None:
        """Iterative backpropagation up the tree."""
        node = self
        while node is not None:
            with node._lock:
                node.visits += 1
                node.value += value
                node.avg_value = node.value / node.visits
            node = node.parent

    def get_path(self) -> List['MCTSNode']:
        """Get the path from root to this node"""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_path_thoughts(self) -> List[Thought]:
        """Get all thoughts along the path from root to this node"""
        return [node.thought for node in self.get_path()]

    def to_dict(self, depth_limit: int = 5) -> Dict[str, Any]:
        """Convert node to dictionary for serialization/visualization"""
        return {
            "id": self.id,
            "thought": {
                "id": self.thought.id,
                "type": self.thought.type.value,
                "content": self.thought.content[:200] + "..." if len(self.thought.content) > 200 else self.thought.content,
                "confidence": self.thought.confidence,
            },
            "depth": self.depth,
            "visits": self.visits,
            "value": round(self.avg_value, 3),
            "ucb1": round(self.get_ucb1(), 3) if self.parent else None,
            "state": self.state.value,
            "is_terminal": self.is_terminal,
            "is_successful": self.is_successful,
            "children_count": len(self.children),
            "children": [child.to_dict(depth_limit - 1) for child in self.children] if depth_limit > 0 else [],
        }


class MCTSConfig:
    """Configuration for MCTS reasoning"""

    def __init__(
        self,
        max_iterations: int = 30,
        max_depth: int = 10,
        branching_factor: int = 5,
        exploration_weight: float = 1.414,
        value_weight: float = 0.5,  # λ in combined value function
        min_confidence_threshold: float = 0.3,
        timeout_seconds: float = 120.0,
        enable_reflection: bool = True,
        enable_pruning: bool = True,
        pruning_threshold: float = 0.2,
        # SOTA additions
        max_token_budget: int = 50000,
        stagnation_window: int = 5,
        stagnation_exploration_boost: float = 0.5,
        beam_width: int = 3,
        max_json_retries: int = 2,
        # Learned value function
        use_learned_value: bool = True,
        learned_value_weight: float = 0.3,
    ):
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.exploration_weight = exploration_weight
        self.value_weight = value_weight
        self.min_confidence_threshold = min_confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.enable_reflection = enable_reflection
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold
        self.max_token_budget = max_token_budget
        self.stagnation_window = stagnation_window
        self.stagnation_exploration_boost = stagnation_exploration_boost
        self.beam_width = beam_width
        self.max_json_retries = max_json_retries
        self.use_learned_value = use_learned_value
        self.learned_value_weight = learned_value_weight


@dataclass
class MCTSResult:
    """Result of MCTS reasoning"""
    success: bool
    best_path: List[MCTSNode]
    best_answer: str
    confidence: float
    iterations: int
    nodes_explored: int
    time_taken: float
    tree: MCTSNode  # Root of the tree
    reflections: List[Reflection]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TreeCache:
    """Saves/loads MCTS tree state for warm-starting similar problems."""

    CACHE_DIR = Path(os.getenv("AURA_DATA_DIR", "data")) / "mcts_cache"

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _problem_hash(self, problem: str) -> str:
        """MD5 hash of problem text."""
        return hashlib.md5(problem.encode()).hexdigest()

    def save_tree(self, problem: str, root: MCTSNode):
        """Serialize full tree to JSON file."""
        data = {
            "problem": problem,
            "timestamp": datetime.now().isoformat(),
            "tree": self._serialize_node(root),
        }
        path = self.CACHE_DIR / f"{self._problem_hash(problem)}.json"
        path.write_text(json.dumps(data, indent=2))

    def load_tree(self, problem: str) -> Optional[Dict]:
        """Load cached tree if exists."""
        path = self.CACHE_DIR / f"{self._problem_hash(problem)}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def fuzzy_match_cached_problem(self, problem: str, threshold: float = 0.6) -> Optional[Dict]:
        """Find a cached tree for a similar problem using word-overlap (Jaccard)."""
        try:
            problem_words = set(problem.lower().split())
            if len(problem_words) < 2:
                return None

            best_match = None
            best_score = 0.0

            for cache_file in self.CACHE_DIR.glob("*.json"):
                try:
                    cached = json.loads(cache_file.read_text())
                    cached_words = set(cached.get("problem", "").lower().split())
                    if not cached_words:
                        continue
                    # Jaccard similarity
                    intersection = len(problem_words & cached_words)
                    union = len(problem_words | cached_words)
                    score = intersection / union if union > 0 else 0.0
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = cached
                except Exception:
                    continue

            if best_match:
                logger.debug(f"[TreeCache] Fuzzy match found (score={best_score:.2f})")
            return best_match
        except Exception as e:
            logger.debug(f"[TreeCache] Fuzzy match failed: {e}")
            return None

    def _serialize_node(self, node: MCTSNode) -> Dict:
        """Full serialization (not truncated like to_dict)."""
        return {
            "id": node.id,
            "thought": {
                "id": node.thought.id,
                "type": node.thought.type.value,
                "content": node.thought.content,
                "confidence": node.thought.confidence,
                "metadata": node.thought.metadata,
                "timestamp": node.thought.timestamp,
            },
            "depth": node.depth,
            "visits": node.visits,
            "value": node.value,
            "avg_value": node.avg_value,
            "state": node.state.value,
            "is_terminal": node.is_terminal,
            "is_successful": node.is_successful,
            "children": [self._serialize_node(c) for c in node.children],
        }


class MCTSReasoning:
    """
    Monte Carlo Tree Search for LLM Reasoning

    This class implements LATS-style reasoning where the LLM serves three roles:
    1. Action/Thought Generator - Generates candidate next steps
    2. Value Function - Evaluates the quality of reasoning states
    3. Reflection Generator - Critiques failed paths to improve future search

    Usage:
        mcts = MCTSReasoning(llm_func=my_llm, config=MCTSConfig())
        result = mcts.search("What is the best approach to solve X?")
    """

    def __init__(
        self,
        llm_func: Callable[[str, Optional[str]], str],
        config: Optional[MCTSConfig] = None,
        tool_executor: Optional[Callable[[str, Dict], Any]] = None,
    ):
        """
        Initialize MCTS Reasoning.

        Args:
            llm_func: Function that takes (prompt, system_prompt) and returns LLM response
            config: MCTS configuration
            tool_executor: Optional function to execute tool calls
        """
        self.llm = llm_func
        self.config = config or MCTSConfig()
        self.tool_executor = tool_executor

        # Search state
        self.root: Optional[MCTSNode] = None
        self.reflections: List[Reflection] = []
        self.iteration_count = 0
        self.nodes_created = 0
        self.start_time: Optional[float] = None
        self.tokens_used: int = 0
        self.current_best: Optional[MCTSResult] = None
        self._value_history: List[float] = []
        self._stagnation_boost: float = 0.0

        # Thread pool for parallel node evaluation
        self._eval_pool = ThreadPoolExecutor(max_workers=4)

        # Learned value predictor (lazy — loads pickle on first predict call)
        self._value_predictor = None
        if self.config.use_learned_value:
            try:
                from aura.tools.mcts_value_fn import ValuePredictor
                self._value_predictor = ValuePredictor()
            except Exception as e:
                logger.debug(f"[MCTS] learned value predictor unavailable: {e}")

        # Callbacks for UI updates
        self.on_node_created: Optional[Callable[[MCTSNode], None]] = None
        self.on_node_evaluated: Optional[Callable[[MCTSNode, float], None]] = None
        self.on_iteration_complete: Optional[Callable[[int, MCTSNode], None]] = None

    def search(self, problem: str, context: str = "") -> MCTSResult:
        """
        Run MCTS search to find the best reasoning path.

        Args:
            problem: The problem/question to solve
            context: Additional context (e.g., conversation history)

        Returns:
            MCTSResult with the best reasoning path and answer
        """
        self.start_time = time.time()
        self.iteration_count = 0
        self.nodes_created = 0
        self.reflections = []
        self.tokens_used = 0
        self.current_best = None
        self._value_history = []
        self._stagnation_boost = 0.0

        # Create root node
        root_thought = Thought(
            type=ThoughtType.ROOT,
            content=problem,
            confidence=1.0,
        )
        self.root = MCTSNode(thought=root_thought, context=context)
        self.root.state = NodeState.EVALUATED
        self.nodes_created += 1

        logger.info(f"Starting MCTS search for: {problem[:100]}...")

        # Warm-start from cached tree if similar problem exists
        try:
            cache = TreeCache()
            cached = cache.fuzzy_match_cached_problem(problem, threshold=0.6)
            if cached and "tree" in cached:
                grafted = self._deserialize_cached_tree(cached["tree"])
                if grafted and grafted.children:
                    self.root = grafted
                    self.root.thought = root_thought  # Use current problem text
                    self.root.state = NodeState.EVALUATED
                    self.nodes_created += sum(1 for _ in self._get_all_nodes())
                    logger.info(f"[WarmStart] Grafted cached tree ({self.nodes_created} nodes)")
        except Exception as e:
            logger.debug(f"[WarmStart] Skipped: {e}")

        # Main MCTS loop (beam search)
        while not self._should_stop():
            self.iteration_count += 1

            beam = self._select_beam(self.root)

            for node in beam:
                if not node.is_terminal and node.depth < self.config.max_depth:
                    expanded = self._expand(node)
                    # Parallel evaluation of expanded children
                    if len(expanded) > 1 and self._eval_pool:
                        futures = {
                            self._eval_pool.submit(self._evaluate_and_backprop, child): child
                            for child in expanded
                        }
                        for future in as_completed(futures):
                            try:
                                reward, child = future.result()
                                child.backpropagate(reward)
                                if self.on_node_evaluated:
                                    self.on_node_evaluated(child, reward)
                            except Exception as e:
                                logger.debug(f"[MCTS] Parallel eval error: {e}")
                    else:
                        for child in expanded:
                            reward = self._evaluate(child)
                            child.backpropagate(reward)
                            if self.on_node_evaluated:
                                self.on_node_evaluated(child, reward)
                else:
                    reward = self._evaluate(node)
                    node.backpropagate(reward)

            if self.config.enable_reflection:
                for node in beam:
                    self._maybe_reflect(node)

            if self.config.enable_pruning:
                self._prune_low_value_branches()

            # Update anytime best result
            self._update_current_best()

            # Check stagnation and adapt exploration
            self._check_stagnation()

            if self.on_iteration_complete:
                try:
                    self.on_iteration_complete(self.iteration_count, self.root, self.current_best)
                except TypeError:
                    self.on_iteration_complete(self.iteration_count, self.root)

            logger.debug(f"Iteration {self.iteration_count}: beam size {len(beam)}, tokens {self.tokens_used}")

        # Find best path
        best_path = self._get_best_path()
        best_answer = self._extract_answer(best_path)
        confidence = best_path[-1].avg_value if best_path else 0.0

        time_taken = time.time() - self.start_time

        result = MCTSResult(
            success=confidence >= self.config.min_confidence_threshold,
            best_path=best_path,
            best_answer=best_answer,
            confidence=confidence,
            iterations=self.iteration_count,
            nodes_explored=self.nodes_created,
            time_taken=time_taken,
            tree=self.root,
            reflections=self.reflections,
            metadata={
                "max_depth_reached": max(n.depth for n in self._get_all_nodes()),
                "terminal_nodes": sum(1 for n in self._get_all_nodes() if n.is_terminal),
                "successful_terminals": sum(1 for n in self._get_all_nodes() if n.is_successful),
                "total_tokens_used": self.tokens_used,
            }
        )

        logger.info(f"MCTS complete: {self.iteration_count} iterations, {self.nodes_created} nodes, confidence={confidence:.2f}")

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    # ==================================================================
    # UPGRADE: Lightweight Monte Carlo Rollout
    # ==================================================================

    def _rollout(self, node: MCTSNode, depth_limit: int = 3) -> float:
        """Fast rollout: estimate terminal value with a short LLM call."""
        try:
            path_context = self._build_path_context(node)
            prompt = (
                f"PROBLEM:\n{self.root.thought.content}\n\n"
                f"CURRENT PATH:\n{path_context}\n\n"
                f"Score the likelihood this path leads to a correct solution. "
                f"Return ONLY JSON: {{\"rollout_score\": 0.X}}"
            )
            response = self.llm(prompt, "Score 0-1. Be brief.")
            self.tokens_used += self._estimate_tokens(prompt) + self._estimate_tokens(response)
            data = self._parse_json(response) if hasattr(self, '_parse_json') else json.loads(response)
            return max(0.0, min(1.0, float(data.get("rollout_score", 0.5))))
        except Exception:
            return 0.5

    # ==================================================================
    # UPGRADE: Evaluate + Backprop helper for parallel execution
    # ==================================================================

    def _evaluate_and_backprop(self, child: MCTSNode) -> Tuple[float, MCTSNode]:
        """Evaluate a node (used for parallel futures). Returns (reward, node)."""
        reward = self._evaluate(child)
        return reward, child

    # ==================================================================
    # UPGRADE: Adaptive branching factor
    # ==================================================================

    def _get_adaptive_branching(self, node: MCTSNode) -> int:
        """Dynamic branching: fewer branches on confident nodes, more on uncertain."""
        base = self.config.branching_factor
        avg_val = node.avg_value if node.visits > 0 else 0.5
        if avg_val > 0.8:
            return max(2, base // 2)
        elif avg_val < 0.3:
            return base
        else:
            return max(2, round(2 + (base - 2) * (1 - avg_val)))

    # ==================================================================
    # UPGRADE: Deserialize cached tree for warm-starting
    # ==================================================================

    def _deserialize_cached_tree(self, tree_dict: Dict) -> Optional[MCTSNode]:
        """Reconstruct MCTSNode from cached JSON, scaling visits for re-exploration."""
        try:
            def _reconstruct(data: Dict, parent=None) -> MCTSNode:
                thought = Thought(
                    id=data["thought"].get("id", str(uuid.uuid4())),
                    type=ThoughtType(data["thought"]["type"]),
                    content=data["thought"]["content"],
                    confidence=data["thought"].get("confidence", 0.5),
                    metadata=data["thought"].get("metadata", {}),
                )
                node = MCTSNode(thought=thought, parent=parent)
                node.visits = max(1, data.get("visits", 0) // 2)  # Scale down
                node.value = data.get("value", 0.0) / 2
                node.avg_value = data.get("avg_value", 0.0)
                node.state = NodeState(data.get("state", "pending"))
                node.is_terminal = data.get("is_terminal", False)
                node.is_successful = data.get("is_successful", False)
                for child_data in data.get("children", []):
                    child = _reconstruct(child_data, parent=node)
                    node.children.append(child)
                return node
            return _reconstruct(tree_dict)
        except Exception as e:
            logger.debug(f"[MCTS] Tree deserialization failed: {e}")
            return None

    def get_current_best(self) -> Optional[MCTSResult]:
        """Return the current best result (anytime access)."""
        return self.current_best

    def _should_stop(self) -> bool:
        """Check if search should stop"""
        # Token budget
        if self.tokens_used >= self.config.max_token_budget:
            logger.info("MCTS stopped: token budget exhausted")
            return True

        # Timeout
        if time.time() - self.start_time > self.config.timeout_seconds:
            logger.info("MCTS stopped: timeout")
            return True

        # Max iterations
        if self.iteration_count >= self.config.max_iterations:
            logger.info("MCTS stopped: max iterations")
            return True

        # Found high-confidence solution
        best_terminal = self._get_best_terminal()
        if best_terminal and best_terminal.avg_value >= 0.95:
            logger.info("MCTS stopped: found high-confidence solution")
            return True

        return False

    def _get_adaptive_exploration(self) -> float:
        """Adaptive exploration weight that decays over iterations, boosted on stagnation."""
        c_init = self.config.exploration_weight
        t = self.iteration_count
        max_iter = self.config.max_iterations
        base = c_init * math.sqrt(max(0.0, 1.0 - t / max_iter))
        return max(0.01, base + self._stagnation_boost)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Navigate tree using UCT until reaching a leaf.
        """
        current = node
        exploration = self._get_adaptive_exploration()

        while current.children and not current.is_terminal:
            valid_children = [c for c in current.children if c.state != NodeState.PRUNED]
            if not valid_children:
                return current

            # If not fully expanded, return current for expansion
            if not all(c.visits > 0 for c in valid_children):
                # Return an unvisited child
                unvisited = [c for c in valid_children if c.visits == 0]
                if unvisited:
                    return unvisited[0]

            # Select best child using adaptive UCB1
            current = max(valid_children, key=lambda c: c.get_ucb1(exploration))

        return current

    def _select_beam(self, root: MCTSNode) -> List[MCTSNode]:
        """Select top-K leaves by UCB1 for beam search (K = config.beam_width)."""
        exploration = self._get_adaptive_exploration()
        all_nodes = self._get_all_nodes()
        leaves = [
            n for n in all_nodes
            if n.is_leaf and n.state != NodeState.PRUNED and n is not root
        ]
        # Include non-fully-expanded internal nodes as candidates too
        expandable = [
            n for n in all_nodes
            if not n.is_leaf and not n.is_fully_expanded and n.state != NodeState.PRUNED
        ]
        candidates = leaves + expandable
        if not candidates:
            # Fall back to single select
            return [self._select(root)]
        # Sort by UCB1 descending, pick top beam_width
        candidates.sort(key=lambda c: c.get_ucb1(exploration), reverse=True)
        return candidates[:self.config.beam_width]

    def _check_stagnation(self):
        """Detect stagnation and boost exploration when best value plateaus."""
        best_path = self._get_best_path()
        current_val = best_path[-1].avg_value if best_path else 0.0
        self._value_history.append(current_val)

        window = self.config.stagnation_window
        if len(self._value_history) >= window:
            recent = self._value_history[-window:]
            if max(recent) <= recent[0]:
                # No improvement in window — boost exploration
                self._stagnation_boost = self.config.stagnation_exploration_boost
                logger.debug(f"Stagnation detected, boosting exploration by {self._stagnation_boost}")
            else:
                # Improving — reset boost
                self._stagnation_boost = 0.0

    def _update_current_best(self):
        """Update the anytime best result."""
        best_path = self._get_best_path()
        best_answer = self._extract_answer(best_path)
        confidence = best_path[-1].avg_value if best_path else 0.0
        self.current_best = MCTSResult(
            success=confidence >= self.config.min_confidence_threshold,
            best_path=best_path,
            best_answer=best_answer,
            confidence=confidence,
            iterations=self.iteration_count,
            nodes_explored=self.nodes_created,
            time_taken=time.time() - self.start_time,
            tree=self.root,
            reflections=self.reflections,
            metadata={"total_tokens_used": self.tokens_used},
        )

    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Expansion phase: Generate candidate next thoughts using LLM.

        LATS extension: When tool_executor is available, candidates can include
        tool actions. Action nodes are executed immediately and their results
        become observation child nodes, grounding the search in real data.
        """
        if node.is_terminal:
            return []

        node.state = NodeState.EXPLORING
        node.expanded_at = datetime.now().isoformat()

        # Build prompt for generating candidate thoughts
        path_context = self._build_path_context(node)
        reflection_context = self._build_reflection_context()

        # Include tool-use instructions when tool_executor is available
        tool_action_guidance = ""
        if self.tool_executor:
            tool_action_guidance = """
You can also suggest TOOL ACTIONS that gather real information. For tool actions, set type to "action" and include a "tool" field with the tool name and "tool_args" with the arguments.

Available tool actions:
- {{"type": "action", "tool": "search_web", "tool_args": {{"query": "..."}}, "thought": "Search for X to verify...", "rationale": "...", "confidence": 0.X}}
- {{"type": "action", "tool": "code_executor", "tool_args": {{"code": "..."}}, "thought": "Calculate X to check...", "rationale": "...", "confidence": 0.X}}
- {{"type": "action", "tool": "read_file", "tool_args": {{"path": "..."}}, "thought": "Read file to understand...", "rationale": "...", "confidence": 0.X}}

Tool actions get executed and their results are used as evidence. Mix reasoning steps with tool actions for best results."""

        # Adaptive branching: fewer on confident nodes, more on uncertain
        adaptive_bf = self._get_adaptive_branching(node)

        prompt = f"""You are reasoning step by step to solve a problem. Generate {adaptive_bf} different possible next thoughts or actions.

PROBLEM:
{self.root.thought.content}

REASONING PATH SO FAR:
{path_context}

{reflection_context}

Generate {adaptive_bf} DIFFERENT candidate next steps. Each should be a distinct approach or thought.
For each candidate, provide:
1. The thought/action (what to think or do next)
2. A brief rationale (why this might be good)
3. Estimated confidence (0.0 to 1.0)
{tool_action_guidance}

Format your response as JSON:
{{
    "candidates": [
        {{"thought": "...", "rationale": "...", "confidence": 0.X, "type": "reasoning|action|conclusion"}},
        ...
    ]
}}

Think creatively and consider multiple angles. Include at least one unconventional approach."""

        try:
            response = self.llm(prompt, "You are a careful reasoning assistant exploring multiple solution paths.")
            self.tokens_used += self._estimate_tokens(prompt) + self._estimate_tokens(response)
            candidates = self._parse_candidates(response)
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            candidates = []

        # Create child nodes for each candidate
        new_nodes = []
        for candidate in candidates[:self.config.branching_factor]:
            thought_type = self._parse_thought_type(candidate.get("type", "reasoning"))
            tool_name = candidate.get("tool")
            tool_args = candidate.get("tool_args", {})

            thought = Thought(
                type=thought_type,
                content=candidate.get("thought", ""),
                confidence=float(candidate.get("confidence", 0.5)),
                metadata={
                    "rationale": candidate.get("rationale", ""),
                    "tool": tool_name,
                    "tool_args": tool_args,
                }
            )

            # Build new context
            new_context = f"{node.context}\n\nStep {node.depth + 1}: {thought.content}"

            child = node.add_child(thought, new_context)
            child.state = NodeState.PENDING

            # Check if this is a terminal node (conclusion)
            if thought.type == ThoughtType.CONCLUSION:
                child.is_terminal = True

            # LATS: Execute tool actions and create observation child nodes
            if thought_type == ThoughtType.ACTION and tool_name and self.tool_executor:
                observation_node = self._execute_tool_action(child, tool_name, tool_args)
                if observation_node:
                    new_nodes.append(observation_node)
                    self.nodes_created += 2  # action + observation
                else:
                    new_nodes.append(child)
                    self.nodes_created += 1
            else:
                new_nodes.append(child)
                self.nodes_created += 1

            if self.on_node_created:
                self.on_node_created(child)

        node.state = NodeState.EVALUATED

        return new_nodes

    def _execute_tool_action(
        self, action_node: MCTSNode, tool_name: str, tool_args: Dict
    ) -> Optional[MCTSNode]:
        """
        LATS tool integration: Execute a tool call and create an observation node.

        This grounds MCTS reasoning in real external data by actually running
        tools (search, code execution, file reads, etc.) and feeding results
        back into the tree as observation nodes.

        Args:
            action_node: The action node that requested the tool call
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Observation child node with tool result, or None on failure
        """
        try:
            logger.debug(f"[MCTS-LATS] Executing tool: {tool_name}({tool_args})")
            result = self.tool_executor(tool_name, tool_args)

            # Convert result to string for the observation
            if isinstance(result, dict):
                result_text = json.dumps(result, default=str)
            elif isinstance(result, str):
                result_text = result
            else:
                result_text = str(result)

            # Truncate very long results to avoid blowing up context
            max_result_len = 2000
            if len(result_text) > max_result_len:
                result_text = result_text[:max_result_len] + f"\n... [truncated, {len(result_text)} chars total]"

            # Create observation node as child of the action node
            obs_thought = Thought(
                type=ThoughtType.OBSERVATION,
                content=f"[Tool: {tool_name}] {result_text}",
                confidence=0.7,  # External data gets moderate-high base confidence
                metadata={
                    "tool": tool_name,
                    "tool_args": tool_args,
                    "result_length": len(result_text),
                    "source": "tool_execution",
                }
            )

            obs_context = (
                f"{action_node.context}\n\n"
                f"Observation (from {tool_name}): {result_text[:500]}"
            )

            obs_node = action_node.add_child(obs_thought, obs_context)
            obs_node.state = NodeState.EVALUATED

            logger.debug(f"[MCTS-LATS] Tool {tool_name} returned {len(result_text)} chars")

            if self.on_node_created:
                self.on_node_created(obs_node)

            return obs_node

        except Exception as e:
            logger.warning(f"[MCTS-LATS] Tool execution failed for {tool_name}: {e}")
            # Create a failed observation so the tree knows this path didn't work
            fail_thought = Thought(
                type=ThoughtType.OBSERVATION,
                content=f"[Tool: {tool_name}] FAILED: {str(e)[:200]}",
                confidence=0.1,
                metadata={"tool": tool_name, "error": str(e)},
            )
            fail_node = action_node.add_child(fail_thought, action_node.context)
            fail_node.state = NodeState.EVALUATED
            return fail_node

    def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluation phase: Score reasoning state using rollout bootstrap + full LLM eval.

        Returns a value between 0.0 and 1.0:
        - 1.0 = Correct/optimal solution
        - 0.5 = Reasonable but incomplete
        - 0.0 = Wrong/dead end
        """
        # Fast rollout estimate as prior
        rollout_value = self._rollout(node, depth_limit=2)

        path_context = self._build_path_context(node)

        prompt = f"""Evaluate the quality of this reasoning path for solving the problem.

PROBLEM:
{self.root.thought.content}

REASONING PATH:
{path_context}

CURRENT STEP:
{node.thought.content}

Evaluate this reasoning state on these criteria:
1. Correctness: Is the reasoning logically sound?
2. Progress: Does this move toward solving the problem?
3. Completeness: If this is a conclusion, is it complete and correct?
4. Efficiency: Is this a productive path or a dead end?

Provide your evaluation as JSON:
{{
    "score": 0.X,  // Overall score from 0.0 to 1.0
    "is_correct": true/false,  // Is the reasoning correct so far?
    "is_complete": true/false,  // Is this a complete solution?
    "reasoning": "Brief explanation of the score"
}}"""

        try:
            response = self.llm(prompt, "You are an expert evaluator assessing reasoning quality. Be critical but fair.")
            self.tokens_used += self._estimate_tokens(prompt) + self._estimate_tokens(response)
            evaluation = self._parse_evaluation(response)

            score = float(evaluation.get("score", 0.5))
            is_complete = evaluation.get("is_complete", False)
            is_correct = evaluation.get("is_correct", True)

            # Mark terminal if complete
            if is_complete:
                node.is_terminal = True
                node.is_successful = is_correct and score >= 0.7

            node.state = NodeState.EVALUATED

            # Combine LLM score with self-consistency
            # V(s) = λ * LM_score + (1-λ) * self_consistency
            eval_value = self.config.value_weight * score + (1 - self.config.value_weight) * node.thought.confidence

            # Blend with rollout: alpha fades rollout influence as visits increase
            alpha = min(0.7, 0.5 + node.visits * 0.02)
            combined_value = (1 - alpha) * rollout_value + alpha * eval_value

            # Learned value function: a third term weighted by its own confidence.
            # Predictor returns (0.5, 0.0) when untrained — a no-op mix.
            if self._value_predictor is not None and self.config.use_learned_value:
                try:
                    learned_pred, learned_conf = self._value_predictor.predict(
                        node.thought.content,
                        path_context,
                    )
                    if learned_conf > 0.0:
                        w = learned_conf * self.config.learned_value_weight
                        combined_value = (1.0 - w) * combined_value + w * learned_pred
                except Exception as e:
                    logger.debug(f"[MCTS] learned value predict failed: {e}")

            return combined_value

        except Exception as e:
            logger.error(f"Error evaluating node: {e}")
            return rollout_value  # Fall back to rollout estimate

    def _maybe_reflect(self, node: MCTSNode):
        """
        Generate reflection on low-value paths to improve future search.
        """
        # Only reflect on evaluated nodes with low scores
        if node.avg_value >= 0.5 or node.visits < 2:
            return

        # Don't reflect too often
        if len(self.reflections) >= self.iteration_count // 5:
            return

        path_context = self._build_path_context(node)

        prompt = f"""Analyze why this reasoning path is not working well and suggest improvements.

PROBLEM:
{self.root.thought.content}

REASONING PATH (low score: {node.avg_value:.2f}):
{path_context}

Reflect on:
1. What went wrong in this reasoning?
2. What assumptions were incorrect?
3. What alternative approaches should be tried?

Provide your reflection as JSON:
{{
    "critique": "What went wrong...",
    "lessons": ["Lesson 1", "Lesson 2"],
    "alternatives": ["Alternative approach 1", "Alternative approach 2"]
}}"""

        try:
            response = self.llm(prompt, "You are a thoughtful critic analyzing reasoning failures.")
            reflection_data = self._parse_json(response)

            reflection = Reflection(
                critique=reflection_data.get("critique", ""),
                lessons=reflection_data.get("lessons", []),
                suggested_alternatives=reflection_data.get("alternatives", []),
                score=node.avg_value,
            )

            self.reflections.append(reflection)
            node.reflection = reflection

            logger.debug(f"Generated reflection: {reflection.critique[:100]}...")

        except Exception as e:
            logger.error(f"Error generating reflection: {e}")

    def _prune_low_value_branches(self):
        """Prune branches with consistently low values"""
        def prune_recursive(node: MCTSNode):
            if node.is_terminal:
                return

            for child in node.children[:]:  # Copy list for modification
                if child.visits >= 3 and child.avg_value < self.config.pruning_threshold:
                    child.state = NodeState.PRUNED
                    logger.debug(f"Pruned node at depth {child.depth} with value {child.avg_value:.2f}")
                else:
                    prune_recursive(child)

        prune_recursive(self.root)

    def _get_best_path(self) -> List[MCTSNode]:
        """Get the highest-value path through the tree"""
        best_terminal = self._get_best_terminal()

        if best_terminal:
            return best_terminal.get_path()

        # No terminal found - return highest value leaf path
        all_nodes = self._get_all_nodes()
        leaves = [n for n in all_nodes if n.is_leaf and n.state != NodeState.PRUNED]

        if leaves:
            best_leaf = max(leaves, key=lambda n: n.avg_value)
            return best_leaf.get_path()

        return [self.root]

    def _get_best_terminal(self) -> Optional[MCTSNode]:
        """Get the best terminal (conclusion) node"""
        terminals = [n for n in self._get_all_nodes() if n.is_terminal and n.is_successful]

        if terminals:
            return max(terminals, key=lambda n: n.avg_value)

        # Fall back to any terminal
        all_terminals = [n for n in self._get_all_nodes() if n.is_terminal]
        if all_terminals:
            return max(all_terminals, key=lambda n: n.avg_value)

        return None

    def _get_all_nodes(self) -> List[MCTSNode]:
        """Iterative BFS to collect all nodes."""
        result = []
        if not self.root:
            return result
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            result.append(node)
            queue.extend(node.children)
        return result

    def _extract_answer(self, path: List[MCTSNode]) -> str:
        """Extract the final answer from a reasoning path"""
        if not path:
            return "No solution found."

        # Find conclusion nodes in path
        conclusions = [n for n in path if n.thought.type == ThoughtType.CONCLUSION]
        if conclusions:
            return conclusions[-1].thought.content

        # Otherwise, summarize the path
        steps = [n.thought.content for n in path[1:]]  # Skip root

        if steps:
            return "Based on reasoning:\n" + "\n".join(f"- {s}" for s in steps[-3:])

        return "Reasoning incomplete."

    def _build_path_context(self, node: MCTSNode) -> str:
        """Build a string representation of the path to this node"""
        path = node.get_path()

        lines = []
        for i, n in enumerate(path):
            if n.thought.type == ThoughtType.ROOT:
                lines.append(f"[ROOT] {n.thought.content}")
            else:
                prefix = "  " * (n.depth - 1)
                lines.append(f"{prefix}[Step {i}] ({n.thought.type.value}) {n.thought.content}")

        return "\n".join(lines)

    def _build_reflection_context(self) -> str:
        """Build context from past reflections"""
        if not self.reflections:
            return ""

        recent = self.reflections[-3:]  # Last 3 reflections

        lines = ["LESSONS FROM PREVIOUS ATTEMPTS:"]
        for r in recent:
            for lesson in r.lessons[:2]:
                lines.append(f"- {lesson}")

        return "\n".join(lines)

    def _parse_candidates(self, response: str) -> List[Dict]:
        """Parse candidate thoughts from LLM response"""
        data = self._parse_json(response)
        return data.get("candidates", [])

    def _parse_evaluation(self, response: str) -> Dict:
        """Parse evaluation from LLM response"""
        return self._parse_json(response)

    def _parse_json(self, response: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks. Retries via LLM on failure."""
        raw = response

        # Try to extract JSON from code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        # Try to find JSON object
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        # Structured output retry via LLM
        for attempt in range(self.config.max_json_retries):
            try:
                fix_prompt = f"The following text should be valid JSON but has errors. Fix it and return ONLY valid JSON:\n{raw}"
                fixed = self.llm(fix_prompt, "Return only valid JSON, no explanation.")
                self.tokens_used += self._estimate_tokens(fix_prompt) + self._estimate_tokens(fixed)
                start = fixed.find("{")
                end = fixed.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(fixed[start:end])
            except Exception:
                continue

        logger.warning(f"Failed to parse JSON from response: {response[:100]}...")
        return {}

    def _parse_thought_type(self, type_str: str) -> ThoughtType:
        """Parse thought type from string"""
        type_map = {
            "reasoning": ThoughtType.REASONING,
            "action": ThoughtType.ACTION,
            "conclusion": ThoughtType.CONCLUSION,
            "observation": ThoughtType.OBSERVATION,
            "reflection": ThoughtType.REFLECTION,
        }
        return type_map.get(type_str.lower(), ThoughtType.REASONING)

    def get_tree_visualization(self) -> Dict[str, Any]:
        """Get the tree structure for visualization"""
        if not self.root:
            return {}

        return {
            "root": self.root.to_dict(),
            "stats": {
                "iterations": self.iteration_count,
                "nodes": self.nodes_created,
                "reflections": len(self.reflections),
                "max_depth": max(n.depth for n in self._get_all_nodes()) if self._get_all_nodes() else 0,
            }
        }


# Convenience function for simple usage
def mcts_reason(
    problem: str,
    llm_func: Callable[[str, Optional[str]], str],
    max_iterations: int = 30,
    branching_factor: int = 5,
    max_depth: int = 10,
) -> MCTSResult:
    """
    Simple function to run MCTS reasoning on a problem.

    Args:
        problem: The problem to solve
        llm_func: LLM function (prompt, system) -> response
        max_iterations: Maximum MCTS iterations
        branching_factor: Number of candidates per expansion
        max_depth: Maximum tree depth

    Returns:
        MCTSResult with the solution
    """
    config = MCTSConfig(
        max_iterations=max_iterations,
        branching_factor=branching_factor,
        max_depth=max_depth,
    )

    mcts = MCTSReasoning(llm_func=llm_func, config=config)
    return mcts.search(problem)
