# Introspection Circuit - Confidence Calibration

## Overview
Evaluates AURA's confidence in its own responses. Detects when the agent should say "I don't know" instead of guessing.

## Confidence Levels
- **HIGH** (0.8-1.0): Very confident, proceed
- **MEDIUM** (0.5-0.8): Somewhat confident, add caveats
- **LOW** (0.2-0.5): Uncertain, suggest verification
- **UNKNOWN** (0-0.2): No idea, say so explicitly

## Confidence Signals
- Knowledge domain match
- Query specificity
- Response consistency (multiple attempts agree)
- Source availability
- Temporal relevance (is info current?)

## Query Types
- FACTUAL, OPINION, PROCEDURAL, CREATIVE, ANALYTICAL

## Actions Based on Confidence
- **PROCEED**: High confidence, answer directly
- **CAVEAT**: Medium confidence, answer with qualifiers
- **SEARCH**: Low confidence, search for information first
- **DEFER**: Unknown, tell user to verify elsewhere

## Key Functions
```python
result = quick_confidence_check("What is the capital of France?")
# Returns: IntrospectionResult with level=HIGH, action=PROCEED
```

## Files
- `aura/tools/introspection_circuit.py` - Core engine
- `aura/tools/introspection_tool.py` - Tool wrapper
