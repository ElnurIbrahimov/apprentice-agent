# Claude Code — CLI Flags & Options

**Tags:** claude-code, cli, flags, configuration
**Created:** 2025
**Category:** claude-code

---

## Core Flags
| Flag | Description | Example |
|------|-------------|---------|
| `--add-dir` | Add additional working directories | `claude --add-dir ../apps ../lib` |
| `--agent` | Specify subagent for session | `claude --agent my-custom-agent` |
| `--agents` | Define custom subagents via JSON | `claude --agents '{"reviewer":{...}}'` |
| `-c, --continue` | Load most recent conversation | `claude -c` |
| `-p, --print` | Query via SDK, then exit | `claude -p "explain this"` |
| `-r, --resume` | Resume specific session by ID/name | `claude -r "auth-refactor"` |
| `--remote` | Create web session on claude.ai | `claude --remote "Fix login bug"` |
| `--teleport` | Resume web session in terminal | `claude --teleport` |

## Permission & Execution Flags
| Flag | Description |
|------|-------------|
| `--permission-mode` | `default`, `plan`, `acceptEdits`, `dontAsk`, `bypassPermissions` |
| `--allowedTools` | Tools that execute without prompting |
| `--disallowedTools` | Tools that cannot be used |
| `--dangerously-skip-permissions` | Skip all permission prompts |

## Advanced Flags
| Flag | Description |
|------|-------------|
| `--debug` | Enable debug mode with optional category filtering |
| `--chrome` | Enable Chrome browser integration |
| `--ide` | Auto-connect to IDE if available |
| `--model` | Set model (`sonnet`, `opus`, `haiku`) |
| `--max-turns` | Limit agentic turns (print mode only) |
| `--max-budget-usd` | Max dollar spend before stopping |
| `--verbose` | Enable verbose logging |

## Output Format Flags
| Flag | Description |
|------|-------------|
| `--output-format` | `text`, `json`, or `stream-json` |
| `--json-schema` | Get validated JSON matching JSON Schema |

## System Prompt Flags
| Flag | Description |
|------|-------------|
| `--system-prompt` | Replace entire default prompt |
| `--append-system-prompt` | Append custom text to default prompt |
| `--system-prompt-file` | Load system prompt from file |

## MCP & Plugin Flags
| Flag | Description |
|------|-------------|
| `--mcp-config` | Load MCP servers from JSON files or strings |
| `--strict-mcp-config` | Only use MCP servers from `--mcp-config` |
| `--plugin-dir` | Load plugins from directories |

## Authentication
| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | API key for Claude |
| `CLAUDE_CODE_USE_BEDROCK=1` | Use Amazon Bedrock |
| `CLAUDE_CODE_USE_VERTEX=1` | Use Google Vertex AI |
| `CLAUDE_CODE_USE_FOUNDRY=1` | Use Microsoft Azure |
