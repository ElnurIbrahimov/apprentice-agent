# Cognitive Theater - Multi-Perspective Decision Making

## Overview
When AURA faces a decision question, it spawns multiple internal "perspectives" that debate the options before reaching a conclusion. Inspired by Minsky's Society of Mind.

## How It Works
1. `is_decision_question(text)` detects if input requires a decision
2. Spawns a `Deliberation` with multiple viewpoints
3. Each perspective argues for/against options
4. Perspectives are synthesized into a balanced recommendation

## Perspectives (Examples)
- **Pragmatist**: What's the most practical option?
- **Optimist**: What's the best-case scenario?
- **Skeptic**: What could go wrong?
- **Creative**: Is there an unconventional approach?
- **Analyst**: What do the data say?

## Key Classes
- `CognitiveTheater` - Main orchestrator
- `Deliberation` - A single decision-making session
- `is_decision_question()` - Detection function

## File Location
- `aura/tools/cognitive_theater.py`

## Use Cases
- "Should I use Redis or PostgreSQL for caching?"
- "Which framework is better for this project?"
- Complex tradeoff analysis
