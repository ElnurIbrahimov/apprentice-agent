# Reflexion - Learning from Mistakes

## Overview
When AURA produces a wrong answer or a tool fails, Reflexion captures what went wrong and generates improvement strategies for future attempts.

## Reflexion Loop
1. **Attempt**: Try to answer/execute
2. **Evaluate**: Check result against expected outcome
3. **Reflect**: If wrong, analyze what went wrong
4. **Store**: Save reflection for future reference
5. **Retry**: Apply learned lessons on next attempt

## Built-in Evaluators
- `code_syntax_evaluator` - Check if code is syntactically valid
- `function_evaluator` - Check if function produces expected output
- `json_evaluator` - Check if JSON is valid
- `answer_completeness_evaluator` - Check if answer fully addresses question

## Data Model
```python
@dataclass
class Reflection:
    attempt: str          # What was tried
    result: str           # What happened
    error: str            # What went wrong
    lesson: str           # What to do differently
    strategy: str         # Improved approach
```

## Key Classes
- `ReflexionEngine` - Core engine
- `ReflexionResult` - Outcome of a reflexion cycle

## File Location
- `aura/tools/reflexion.py`
