# MCTS Reasoning - Monte Carlo Tree Search for Thought

## Overview
Applies Monte Carlo Tree Search (traditionally used in game AI like AlphaGo) to reasoning tasks. Explores multiple thought paths and selects the best one.

## How It Works
1. **Selection**: Choose most promising thought branch (UCB1 formula)
2. **Expansion**: Generate new thoughts from selected node
3. **Simulation**: Evaluate thought quality via LLM
4. **Backpropagation**: Update scores up the tree

## Configuration
```python
MCTSConfig(
    max_iterations: int = 50,
    exploration_weight: float = 1.414,  # UCB1 constant
    max_depth: int = 5,
    min_quality_threshold: float = 0.3,
)
```

## Thought Types
- HYPOTHESIS, ANALYSIS, SYNTHESIS, EVALUATION, CONCLUSION

## Node States
- UNEXPLORED, EXPLORING, EVALUATED, PRUNED, SELECTED

## Key Function
```python
result = mcts_reason("What is the best approach to implement caching?")
# Returns: MCTSResult with best_path, all explored nodes, confidence score
```

## Use Cases
- Complex multi-step reasoning
- Exploring multiple solution approaches
- Decision-making under uncertainty

## Files
- `aura/tools/mcts_reasoning.py` - Core MCTS engine
- `aura/tools/reasoning_tree_tool.py` - Tool wrapper with `deep_reason()`
