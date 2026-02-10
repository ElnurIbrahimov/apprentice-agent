# Claude Code — Extended Thinking

**Tags:** claude-code, thinking, reasoning, deep-analysis
**Created:** 2025
**Category:** claude-code

---

## What Is Extended Thinking
Extended thinking gives Claude additional "thinking time" before responding. It uses extra tokens for internal reasoning that isn't shown directly but improves output quality.

## Enabling Extended Thinking
- **Toggle**: `Option+T` (Mac) / `Alt+T` (Windows/Linux)
- **Model picker**: Available in model selection
- **Setting**: `"alwaysThinkingEnabled": true` in settings
- **Environment**: `MAX_THINKING_TOKENS` to set budget

## When to Use Extended Thinking

### Good Use Cases
| Scenario | Why |
|----------|-----|
| Architecture decisions | Need to weigh tradeoffs |
| Complex debugging | Multiple interacting systems |
| Algorithm design | Correctness matters |
| Security review | Need thorough analysis |
| Large refactors | Many files, many implications |
| Unfamiliar codebase | Need to understand before acting |

### Bad Use Cases (Waste of Tokens)
| Scenario | Why |
|----------|-----|
| Simple file edits | Straightforward, no reasoning needed |
| Running commands | Just execute, don't think |
| Formatting changes | Mechanical, not analytical |
| Reading files | No reasoning required |
| Simple Q&A | Fast answer is fine |

## How It Works Internally
1. Claude receives your prompt
2. Extended thinking generates internal reasoning (chain-of-thought)
3. Internal reasoning is NOT shown to you
4. Final response is informed by the extended reasoning
5. Results in more accurate, well-considered answers

## Budget Control
```bash
# Set max thinking tokens
export MAX_THINKING_TOKENS=10000

# Or per-session
claude --max-budget-usd 10
```

## Tips
- Extended thinking is most valuable for the FIRST response in a complex task
- Once Claude understands the problem, subsequent responses may not need it
- Toggle off for simple follow-up edits
- Combine with Plan mode for thorough analysis before any code changes
- Works best with Opus model (most capable reasoning)

## Interleaved Thinking
```bash
claude --betas interleaved-thinking
```
Allows thinking to happen between tool calls, not just at the start. Useful for complex multi-step tasks.
