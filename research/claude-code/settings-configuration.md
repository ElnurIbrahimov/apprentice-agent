# Claude Code — Settings & Configuration

**Tags:** claude-code, settings, configuration, claude-md
**Created:** 2025
**Category:** claude-code

---

## Configuration Scopes (Precedence Order)
1. Managed settings (system-wide, highest priority)
2. Command-line arguments
3. Local project settings (`.claude/*.local.*`)
4. Shared project settings (`.claude/`)
5. User settings (`~/.claude/`)

## Settings File Example
```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": ["Bash(npm run *)", "Read(src/**)"],
    "ask": ["Bash(git push *)"],
    "deny": ["WebFetch", "Read(.env*)"]
  },
  "env": {
    "CUSTOM_VAR": "value"
  },
  "model": "claude-sonnet-4-5-20250929",
  "outputStyle": "Explanatory",
  "defaultMode": "default",
  "hooks": {},
  "additionalDirectories": ["/path/to/dir"],
  "alwaysThinkingEnabled": true
}
```

## Environment Variables
| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | API key |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | Override max tokens |
| `CLAUDE_CODE_SHELL` | Override shell detection |
| `MAX_MCP_OUTPUT_TOKENS` | MCP tool output limit |
| `ENABLE_TOOL_SEARCH` | MCP tool search (`auto`, `true`, `false`) |
| `MAX_THINKING_TOKENS` | Extended thinking budget |

## CLAUDE.md Memory System

### Hierarchy (highest to lowest)
1. Managed policy CLAUDE.md (system-wide)
2. Project CLAUDE.md (repo root)
3. Nested project rules (`.claude/rules/`)
4. User CLAUDE.md (`~/.claude/CLAUDE.md`)
5. Project local CLAUDE.md (`./CLAUDE.local.md`)
6. Auto memory (`~/.claude/projects/<project>/memory/`)

### Features
- `@path/to/import` syntax for importing other files
- Recursive imports up to 5 hops
- `.claude/rules/*.md` for modular instructions
- Path-specific rules with YAML frontmatter (`paths` field)
- Glob pattern support in rules

### Auto Memory
- Stored at `~/.claude/projects/<project>/memory/MEMORY.md`
- First 200 lines loaded at session start
- Claude writes notes about patterns, debugging, architecture
- Topic files on demand

## Directory Structure
```
project/
├── .claude/
│   ├── settings.json          # Project settings
│   ├── settings.local.json    # Local (gitignored)
│   ├── CLAUDE.md              # Project memory
│   ├── hooks/                 # Hook scripts
│   ├── agents/                # Custom subagents
│   ├── rules/                 # Modular rules
│   ├── skills/                # Custom skills
│   └── commands/              # Custom commands
├── CLAUDE.md                  # Alt project memory
├── .mcp.json                  # Project MCP config
└── .claude.json               # User MCP config
```
