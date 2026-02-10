# Claude Code — Skills & Plugins

**Tags:** claude-code, skills, plugins, extensibility
**Created:** 2025
**Category:** claude-code

---

## Skills

### What Are Skills
Markdown files with YAML frontmatter that define specialized capabilities. Invoked with `/skillname`.

### Creating Skills (`.claude/skills/`)
```markdown
---
name: test-runner
context: fork
description: Run project tests and report results
---

Run the test suite with npm test and provide a summary of passing/failing tests.
Include code coverage if available.
```

### Skill Features
- Markdown files with YAML frontmatter
- Invoke with `/skillname`
- Can include supporting files
- Restrict tool access per skill
- Run in subagent (fork context)
- Can be user-level or project-level

### Frontmatter Options
- `name` — Display name
- `context` — `fork` (subagent) or `inline`
- `description` — What the skill does
- `tools` — Allowed tools
- `disallowedTools` — Blocked tools
- `model` — Override model

---

## Plugins

### Three-Layer System
1. **Skills** — Specialized prompts and commands
2. **Agents** — Custom subagents
3. **MCP servers** — External integrations

### Installing Plugins
- `/plugin` command in Claude Code
- Official Anthropic marketplace
- Custom GitHub repositories
- Local paths
- Remote URLs

### Plugin Structure
```
my-plugin/
├── skills/
│   ├── test-runner.md
│   └── code-review.md
├── agents/
│   └── reviewer.md
├── mcp/
│   └── config.json
└── plugin.json
```

---

## Built-in Skills Available
Skills vary by installation but commonly include:
- `/commit` — Create git commits with good messages
- `/review-pr` — Review pull requests
- `/init` — Initialize project
- Plus any custom skills in `.claude/skills/`
