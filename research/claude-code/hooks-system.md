# Claude Code — Hooks System

**Tags:** claude-code, hooks, automation, events
**Created:** 2025
**Category:** claude-code

---

## Hook Events
| Event | When It Fires | Matcher |
|-------|---------------|---------|
| `SessionStart` | Session begins/resumes | `startup`, `resume`, `clear`, `compact` |
| `UserPromptSubmit` | Prompt submitted before processing | No matcher |
| `PreToolUse` | Before tool executes (can block) | Tool name |
| `PermissionRequest` | Permission dialog appears | Tool name |
| `PostToolUse` | After tool succeeds | Tool name |
| `PostToolUseFailure` | After tool fails | Tool name |
| `Notification` | Notification sent | Type string |
| `SubagentStart` | Subagent spawned | Agent type |
| `SubagentStop` | Subagent completes | Agent type |
| `Stop` | Claude finishes responding | No matcher |
| `PreCompact` | Before context compaction | `manual`, `auto` |
| `SessionEnd` | Session terminates | `clear`, `logout`, etc. |

## Hook Types
1. **Command hooks** — Run shell commands
2. **Prompt-based hooks** — Use Claude model for decisions
3. **Agent-based hooks** — Use subagent for verification

## Configuration Example
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "npx prettier --write"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "~/scripts/validate-command.sh"
          }
        ]
      }
    ]
  }
}
```

## Hook Input/Output
- **Input**: JSON via stdin — `session_id`, `cwd`, `hook_event_name`, tool-specific data
- **Output**: Exit codes:
  - **0** — Action proceeds
  - **2** — Action blocked (reason in stderr)
  - **Other** — Action proceeds, stderr logged

## Hook Locations
1. `~/.claude/settings.json` — User-level (all projects)
2. `.claude/settings.json` — Project-level (shared)
3. `.claude/settings.local.json` — Local (gitignored)
4. Skill/agent frontmatter — Component-scoped

## Common Use Cases
- Auto-format code after edits (prettier, black)
- Validate shell commands before execution
- Run linting after file changes
- Log all tool usage for audit
- Block dangerous operations
