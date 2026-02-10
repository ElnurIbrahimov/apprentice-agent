# Claude Code — Model Comparison

**Tags:** claude-code, models, opus, sonnet, haiku
**Created:** 2025
**Category:** claude-code

---

## Available Models

### Claude Opus 4.6
- **ID**: `claude-opus-4-6`
- **Best for**: Complex reasoning, architecture, hard bugs, code review
- **Speed**: Slowest
- **Cost**: Highest
- **Context**: Largest
- **Extended thinking**: Best support
- **When to use**:
  - Designing system architecture
  - Debugging complex multi-file issues
  - Security audits
  - Large refactoring tasks
  - Anything requiring deep reasoning

### Claude Sonnet 4.5
- **ID**: `claude-sonnet-4-5-20250929`
- **Best for**: Daily development, feature work, general coding
- **Speed**: Medium
- **Cost**: Mid-range
- **Context**: Standard
- **When to use**:
  - Feature implementation
  - Code writing
  - Test creation
  - Documentation
  - Most everyday tasks

### Claude Haiku 4.5
- **ID**: `claude-haiku-4-5-20251001`
- **Best for**: Quick tasks, searches, simple edits
- **Speed**: Fastest
- **Cost**: Cheapest
- **Context**: Smaller
- **When to use**:
  - Codebase exploration (Explore subagent)
  - Simple file edits
  - Quick questions
  - Running commands
  - Batch operations

---

## Model Selection Guide

| Task | Recommended | Why |
|------|------------|-----|
| "Fix this typo" | Haiku | Trivial, fast |
| "Add a login form" | Sonnet | Standard feature |
| "Design auth system" | Opus | Architecture |
| "Find all API endpoints" | Haiku (Explore) | Search task |
| "Review PR for security" | Opus | Deep analysis |
| "Write unit tests" | Sonnet | Standard coding |
| "Explain this code" | Sonnet | Understanding |
| "Debug race condition" | Opus | Complex reasoning |
| "Rename variables" | Haiku | Mechanical |
| "Create database schema" | Opus | Design decisions |

## Switching Models

### Mid-Session
- `Alt+P` / `Option+P` — Model picker
- `/model` — Select model and effort level

### Per-Session
```bash
claude --model sonnet
claude --model opus
claude --model haiku
```

### For Subagents
```python
Task(subagent_type="Explore", model="haiku")  # Cheap searches
Task(subagent_type="Plan", model="opus")       # Deep planning
```

## Cost vs Quality Tradeoff
```
Haiku  ████░░░░░░ Quality  ██████████ Speed  █░░░░░░░░░ Cost
Sonnet ███████░░░ Quality  ██████░░░░ Speed  ████░░░░░░ Cost
Opus   ██████████ Quality  ███░░░░░░░ Speed  ████████░░ Cost
```

## Pro Tips
- Start with Sonnet, escalate to Opus only when needed
- Use Haiku for all Explore/search subagents
- Opus for the first response on complex tasks, then switch to Sonnet for follow-ups
- Set default model in settings to avoid re-selecting each session
