# Claude Code — Subagents & Multi-Agent Systems

**Tags:** claude-code, subagents, multi-agent, task-tool
**Created:** 2025
**Category:** claude-code

---

## Built-in Subagent Types
| Agent | Model | Tools | Use Case |
|-------|-------|-------|----------|
| Explore | Haiku | Read-only | Fast codebase search |
| Plan | Inherit | Read-only | Research before planning |
| General-purpose | Inherit | All | Complex multi-step tasks |
| Bash | Inherit | Bash only | Shell commands |

## Creating Custom Subagents

### Via `/agents` Command
- Create user-level or project-level
- Generate with Claude or manually
- Select tools, model, color

### File-Based (`.claude/agents/code-reviewer.md`)
```markdown
---
name: code-reviewer
description: Expert code reviewer. Use proactively after code changes.
tools: Read, Glob, Grep, Bash
disallowedTools: Write, Edit
model: sonnet
permissionMode: default
maxTurns: 20
memory: user
skills:
  - code-style
  - security-patterns
mcpServers:
  - github
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./validate-readonly.sh"
---

You are a senior code reviewer. Focus on quality, security, and best practices.
```

### Scope Locations
- CLI flag: `--agents` (session only)
- Project: `.claude/agents/`
- User: `~/.claude/agents/`
- Plugin: Plugin's `agents/` directory

## Subagent Features
- **Persistent memory**: `memory: user|project|local`
- **Tool restrictions**: `tools` allowlist + `disallowedTools`
- **Skill preloading**: `skills` field injects skill content
- **Hooks**: `PreToolUse`, `PostToolUse`, `Stop` events
- **Max turns**: `maxTurns` limits execution
- **Context isolation**: Each subagent has independent context

## Agent Teams
- Enable with `/agents-team` command
- Multiple agents work in parallel
- Each has independent context
- Lead can delegate or approve work
- `delegate` permission mode for coordinators

## Using Task Tool (from Claude's perspective)
```
Task tool -> specify subagent_type -> prompt -> get results back
```
- `subagent_type`: "Explore", "Plan", "general-purpose", "Bash", or custom
- Results returned as single message
- Can run in background with `run_in_background: true`
- Can resume with `resume: agent_id`
