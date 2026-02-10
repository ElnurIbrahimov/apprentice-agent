# Claude Code — Context Window Management

**Tags:** claude-code, context, tokens, optimization
**Created:** 2025
**Category:** claude-code

---

## How Context Works
- Each conversation has a context window limit (varies by model)
- As conversation grows, older messages get compressed
- Automatic compaction summarizes when approaching limits

## Key Commands
| Command | Purpose |
|---------|---------|
| `/context` | Visualize usage as colored grid |
| `/cost` | Show token usage statistics |
| `/compact` | Manual compaction with optional focus |

## Strategies for Context Efficiency

### Use Subagents
- Each subagent has independent context
- Exploration output stays isolated
- Only summaries return to main conversation
- Best for: verbose search results, large file reads

### Manual Compaction
```
/compact focus on the authentication changes
```
Compacts with instructions to preserve specific information.

### Best Practices
1. Use `Task` tool for research-heavy work (isolates output)
2. Compact regularly during long sessions
3. Use `Explore` subagent for codebase searches
4. Avoid reading very large files in main context
5. Let auto-compaction handle routine management
6. Use `/cost` to monitor token usage

## Extended Thinking
- Toggle with `Option+T` / `Alt+T`
- Uses additional tokens for complex reasoning
- Budget controlled via `MAX_THINKING_TOKENS`
- Best for: architectural decisions, complex debugging

## Token Limits by Model
- **Opus 4.6** — Largest context, extended thinking
- **Sonnet 4.5** — Standard context
- **Haiku 4.5** — Smaller context, fastest
