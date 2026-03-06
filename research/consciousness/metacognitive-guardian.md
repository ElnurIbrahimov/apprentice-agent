# Metacognitive Guardian

## Overview
Monitors AURA's own cognitive processes and predicts failures before they happen. Acts as a "supervisor" over the agent's reasoning.

## Failure Types Detected
- **HALLUCINATION** - Generating false information
- **LOOP** - Repeating the same action without progress
- **OVERCONFIDENCE** - Being too certain about uncertain answers
- **SCOPE_CREEP** - Drifting from the original question
- **RESOURCE_EXHAUSTION** - Using too many iterations
- **CONTRADICTION** - Conflicting with previous statements

## Intervention Types
- **WARN** - Log a warning, continue
- **REDIRECT** - Suggest alternative approach
- **PAUSE** - Stop and ask user for clarification
- **ABORT** - Stop the current action entirely

## Configuration
```python
GuardianConfig(
    max_iterations=10,
    confidence_threshold=0.7,
    loop_detection_window=5,
    hallucination_sensitivity=0.8,
)
```

## Integration
- Connected to `InnerMonologue` for thought monitoring
- Connected to `EvoEmo` for emotional state awareness
- Runs checks at each phase transition in the cognitive loop

## File Location
- `aura/tools/metacog_guardian.py`
