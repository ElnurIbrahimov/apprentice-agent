# Claude Code — Cost Optimization

**Tags:** claude-code, cost, tokens, optimization
**Created:** 2025
**Category:** claude-code

---

## Understanding Costs

### Token Pricing (Approximate)
| Model | Input | Output |
|-------|-------|--------|
| Opus 4.6 | Most expensive | Highest quality |
| Sonnet 4.5 | Mid-range | Best value |
| Haiku 4.5 | Cheapest | Fastest |

### Check Current Usage
```
/cost          # Token usage this session
/usage         # Plan limits and rate limit status
/stats         # Daily usage, sessions, streaks
```

---

## Cost Reduction Strategies

### 1. Choose the Right Model
| Task | Best Model | Why |
|------|-----------|-----|
| Simple edits, renames | Haiku | Fast, cheap |
| Feature development | Sonnet | Good balance |
| Architecture, complex reasoning | Opus | Best quality |
| Quick searches, exploration | Haiku (via Explore agent) | Minimal cost |

Switch models mid-session: `Alt+P` or `/model`

### 2. Use Subagents for Research
Subagents use independent context — results are summarized before returning. This prevents main context bloat.

```
# Instead of reading 20 files yourself:
Task(subagent_type="Explore", prompt="Find all API endpoints")
# Returns concise summary, not 20 file contents
```

### 3. Compact Regularly
```
/compact focus on the auth changes
```
Reduces context size while preserving important info.

### 4. Be Specific in Prompts
More specific = fewer iterations = fewer tokens:
- Bad: "Fix the bug" (Claude reads many files searching)
- Good: "Fix the null check in src/auth.py line 42" (direct fix)

### 5. Use CLAUDE.md
Put project rules in CLAUDE.md instead of repeating them every prompt. Loaded once, used throughout.

### 6. Budget Controls
```bash
claude --max-budget-usd 5          # Hard cap per session
```

### 7. Use `--print` for One-Shots
```bash
claude -p "what does this error mean: [error]" --model haiku
```
Single query, no interactive overhead.

### 8. Batch Related Changes
Instead of 5 separate requests:
```
# One request:
"In src/api/routes.py:
1. Add rate limiting to /login endpoint
2. Add input validation to /register
3. Add error handling to /reset-password
4. Update the docstrings for all three"
```

### 9. Avoid Re-Reading Files
If you already discussed a file, reference it by name instead of asking Claude to read it again.

### 10. Use Plan Mode First
```
/plan
```
Read-only exploration costs less than trial-and-error coding.

---

## Cost Monitoring Best Practices
1. Check `/cost` periodically during long sessions
2. Set `--max-budget-usd` for automated/CI usage
3. Use Haiku for Explore subagents (default)
4. Compact before context fills up (auto-compaction costs tokens too)
5. End sessions when done — don't leave context growing idle
