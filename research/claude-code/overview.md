# Claude Code — Complete Overview

**Tags:** claude-code, cli, anthropic, ai-tools
**Created:** 2025
**Category:** claude-code

---

## What Is Claude Code
Claude Code is Anthropic's official CLI tool for AI-assisted software engineering. It runs in the terminal, connects to Claude models (Opus 4.6, Sonnet 4.5, Haiku 4.5), and provides an agentic coding assistant with file editing, shell execution, web search, and 500+ MCP integrations.

## Key Capabilities
- Interactive REPL for conversational coding
- File reading, writing, and editing with diffs
- Shell command execution with sandboxing
- Web search and page fetching
- MCP (Model Context Protocol) for external tool integration
- Multi-agent systems with subagents
- VS Code and JetBrains IDE integration
- Agent SDK for programmatic use (Python + TypeScript)
- Hooks system for automation
- Custom skills and plugins
- Chrome browser automation
- Plan mode for safe read-only analysis
- Extended thinking for complex reasoning

## Models Available
| Model | ID | Use Case |
|-------|-----|----------|
| Opus 4.6 | claude-opus-4-6 | Most capable, complex tasks |
| Sonnet 4.5 | claude-sonnet-4-5-20250929 | Balanced speed/quality |
| Haiku 4.5 | claude-haiku-4-5-20251001 | Fast, low cost |

## Core Architecture
```
User Input -> Permission Check -> Tool Selection -> Execution -> Response
```
- Permission system controls what tools can do
- Hooks fire at each stage for automation
- Context window auto-compacts when approaching limits
- Subagents isolate heavy tasks from main context
