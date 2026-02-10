# Claude Code — Team & Enterprise Setup

**Tags:** claude-code, team, enterprise, managed-settings
**Created:** 2025
**Category:** claude-code

---

## Shared Project Configuration

### Project Settings (`.claude/settings.json`)
Committed to version control, shared across team:
```json
{
  "permissions": {
    "allow": ["Bash(npm run *)", "Bash(npx jest *)"],
    "deny": ["Read(.env*)", "Bash(rm -rf *)"]
  },
  "env": {
    "NODE_ENV": "development"
  },
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{"type": "command", "command": "npx prettier --write"}]
    }]
  }
}
```

### Project MCP Servers (`.mcp.json`)
Shared MCP configuration:
```json
{
  "mcpServers": {
    "github": {
      "transport": "http",
      "url": "https://api.githubcopilot.com/mcp/"
    }
  }
}
```

### Project CLAUDE.md
Team-wide instructions and conventions:
```markdown
# Project Rules
- Use TypeScript strict mode
- All API responses: { success: boolean, data?: T, error?: string }
- Tests required for all new endpoints
- Use pnpm, not npm or yarn
```

### Modular Rules (`.claude/rules/`)
```
.claude/rules/
├── code-style.md        # Formatting, naming conventions
├── testing.md           # Test requirements, patterns
├── git-workflow.md      # Branch naming, PR requirements
├── security.md          # Security policies
└── api-patterns.md      # API design rules
```

---

## Managed Settings (Enterprise)

### Installation Locations
| OS | Path |
|----|------|
| macOS | `/Library/Application Support/ClaudeCode/` |
| Linux | `/etc/claude-code/` |
| Windows | `C:\Program Files\ClaudeCode\` |

### Managed-Only Settings
| Setting | Purpose |
|---------|---------|
| `disableBypassPermissionsMode` | Prevent permission bypass |
| `allowManagedPermissionRulesOnly` | Only managed permission rules |
| `allowManagedHooksOnly` | Only managed hooks |
| `strictKnownMarketplaces` | Restrict plugin sources |
| `allowedMcpServers` | Whitelist MCP servers |
| `deniedMcpServers` | Blacklist MCP servers |

### Example Managed Policy
```json
{
  "disableBypassPermissionsMode": true,
  "allowManagedPermissionRulesOnly": true,
  "permissions": {
    "deny": [
      "Bash(curl * | bash)",
      "Bash(wget * | bash)",
      "Read(.env*)",
      "Read(*credentials*)",
      "Read(*secret*)"
    ]
  },
  "allowedMcpServers": [
    {"serverName": "github"},
    {"serverUrl": "https://mcp.company.com/*"}
  ],
  "deniedMcpServers": [
    {"serverName": "untrusted-server"}
  ]
}
```

---

## Team Workflows

### Onboarding New Developers
1. Clone repo (CLAUDE.md + .claude/ included)
2. Install Claude Code
3. Run `claude --init` — picks up project settings automatically
4. Team conventions are enforced via CLAUDE.md and hooks

### Code Review with Claude
```bash
# Create review agent
# .claude/agents/reviewer.md
---
name: code-reviewer
tools: Read, Glob, Grep
disallowedTools: Write, Edit, Bash
model: sonnet
---

Review code changes for bugs, security issues, and style violations.
Follow the rules in .claude/rules/code-style.md.
```

### Consistent Formatting
Use hooks to auto-format on every edit:
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [
        {"type": "command", "command": "npx prettier --write"},
        {"type": "command", "command": "npx eslint --fix"}
      ]
    }]
  }
}
```

---

## Security for Teams
1. Never commit API keys — use `deny` rules for `.env*`
2. Use managed settings to enforce policies system-wide
3. Restrict MCP servers to approved list
4. Audit hooks regularly
5. Use `allowManagedPermissionRulesOnly` in enterprise
6. Enable audit logging for compliance
7. Set `disableBypassPermissionsMode` to prevent bypasses
