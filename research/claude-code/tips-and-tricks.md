# Claude Code — Tips & Tricks for Power Users

**Tags:** claude-code, tips, productivity, best-practices
**Created:** 2025
**Category:** claude-code

---

## Productivity Tips

### 1. Use CLAUDE.md Effectively
Put project conventions, architecture decisions, and common patterns in CLAUDE.md so Claude remembers across sessions:
```markdown
# Project Rules
- Use TypeScript strict mode
- Tests go in __tests__/ directories
- Use pnpm, not npm
- API responses follow {success, data, error} format
```

### 2. Named Sessions for Context
```bash
claude -r "auth-refactor"      # Resume by name
/rename auth-refactor          # Name current session
```

### 3. Bash Mode for Quick Commands
Type `!git status` instead of asking Claude to run git status.

### 4. Background Long Tasks
`Ctrl+B` to background a long-running operation, continue working, check with `/tasks`.

### 5. Use Plan Mode for Safety
```bash
claude --permission-mode plan
```
Explore and plan without accidentally modifying files.

### 6. Multi-Directory Projects
```bash
claude --add-dir ../frontend ../shared-lib
```

### 7. Pre-approve Common Commands
```json
{
  "permissions": {
    "allow": [
      "Bash(npm run *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(python -m pytest *)"
    ]
  }
}
```

### 8. Auto-Format on Edit
Use hooks to run prettier/black after every file edit:
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{"type": "command", "command": "npx prettier --write $CLAUDE_FILE_PATH"}]
    }]
  }
}
```

### 9. Cost Control
```bash
claude --max-budget-usd 5     # Cap spending
/cost                          # Check usage mid-session
```

### 10. Image Input
`Ctrl+V` to paste a screenshot directly into the prompt. Great for UI bugs.

### 11. Structured Output
```bash
claude -p "list all API endpoints" --json-schema '{"type":"array","items":{"type":"object","properties":{"method":{"type":"string"},"path":{"type":"string"}}}}'
```

### 12. Debug Issues
```bash
/doctor                        # Health check
/debug                         # Session debug log
claude --debug "api,mcp"       # Filtered debug output
```

### 13. Fork Don't Rewind
When exploring multiple approaches, use `/rewind` to fork rather than lose conversation history.

### 14. Modular Rules
Instead of one giant CLAUDE.md, use `.claude/rules/`:
```
.claude/rules/
├── code-style.md
├── testing.md
├── git-workflow.md
└── api-patterns.md
```

### 15. Remote Sessions
Start on your phone via claude.ai, then teleport to terminal:
```bash
claude --teleport
```
