# Claude Code — Permission System

**Tags:** claude-code, permissions, security
**Created:** 2025
**Category:** claude-code

---

## Permission Modes
| Mode | Behavior |
|------|----------|
| `default` | Prompts for permission on first use |
| `acceptEdits` | Auto-accept file edits |
| `plan` | Read-only exploration with plan approval |
| `dontAsk` | Auto-deny unless pre-approved |
| `delegate` | Coordination-only for agent team leads |
| `bypassPermissions` | Skip all prompts |

## Permission Rules

### Bash — Wildcard Patterns
```json
{
  "allow": [
    "Bash(npm run *)",
    "Bash(git commit *)",
    "Bash(* --version)"
  ]
}
```

### Read/Edit — Gitignore-Style Patterns
```json
{
  "deny": [
    "Read(.env*)",
    "Edit(/src/**/*.ts)",
    "Read(~/Documents/*)"
  ]
}
```

### WebFetch — Domain-Based
```json
{
  "deny": ["WebFetch(domain:example.com)"]
}
```

### MCP — Server and Tool Based
```json
{
  "allow": [
    "mcp__github",
    "mcp__github__search_repositories"
  ]
}
```

### Task (Subagents)
```json
{
  "deny": ["Task(Explore)", "Task(my-custom-agent)"]
}
```

## Rule Precedence
1. **Deny** rules (highest priority)
2. **Ask** rules
3. **Allow** rules (lowest priority)

## Switching Modes at Runtime
- `Shift+Tab` or `Alt+M` — Cycle through permission modes
