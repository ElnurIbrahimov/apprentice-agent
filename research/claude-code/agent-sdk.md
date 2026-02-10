# Claude Code — Agent SDK

**Tags:** claude-code, sdk, python, typescript, automation
**Created:** 2025
**Category:** claude-code

---

## Overview
Build custom AI agents with the same tools, agent loop, and context management as Claude Code. Available for Python and TypeScript.

## Installation
```bash
# Python
pip install claude-agent-sdk

# TypeScript
npm install @anthropic-ai/claude-agent-sdk
```

## Basic Usage

### Python
```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    async for message in query(
        prompt="Find and fix the bug in auth.py",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Bash"]
        )
    ):
        print(message)

asyncio.run(main())
```

### TypeScript
```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Find and fix the bug in auth.py",
  options: { allowedTools: ["Read", "Edit", "Bash"] }
})) {
  console.log(message);
}
```

## Built-in Tools
- **Read** — Read files
- **Write** — Create files
- **Edit** — Modify existing files
- **Bash** — Run shell commands
- **Glob** — Find files by pattern
- **Grep** — Search file contents
- **WebSearch** — Search the web
- **WebFetch** — Fetch web pages
- **AskUserQuestion** — Clarifying questions

## Key Features
- Hooks: PreToolUse, PostToolUse, SessionStart, SessionEnd
- Subagents: Spawn specialized agents
- MCP: Connect external tools/databases
- Permissions: Control tool access
- Sessions: Maintain context across exchanges
- Skills: Specialized capabilities
- Structured output with `--json-schema`

## Headless Mode
```bash
claude -p "fix the bug" --output-format json --max-turns 10
```
- No interactive prompts
- Structured JSON output
- For CI/CD pipelines and automation

## Use Cases
- Automated code review in CI/CD
- Batch file processing
- Custom development tools
- Integration into existing workflows
- Testing automation
