# Claude Code — MCP (Model Context Protocol) Integration

**Tags:** claude-code, mcp, integrations, tools
**Created:** 2025
**Category:** claude-code

---

## What Is MCP
Model Context Protocol lets Claude Code connect to external tools, databases, and services. 500+ servers available.

## Popular MCP Servers
- **Business**: Slack, Asana, Jira, Notion, Airtable
- **Development**: GitHub, GitLab, Stripe, SendGrid
- **Data**: PostgreSQL, MySQL, MongoDB, BigQuery
- **Monitoring**: Sentry, Datadog, New Relic
- **Browsers**: Playwright, Puppeteer

## Installation Methods

### HTTP Servers (Recommended)
```bash
claude mcp add --transport http notion https://mcp.notion.com/mcp
claude mcp add --transport http github https://api.githubcopilot.com/mcp/
```

### Local Stdio Servers
```bash
claude mcp add --transport stdio airtable \
  --env AIRTABLE_API_KEY=YOUR_KEY \
  -- npx -y airtable-mcp-server
```

## MCP Scopes
| Scope | File | Use Case |
|-------|------|----------|
| Local (default) | `~/.claude.json` | Private, project-specific |
| Project | `.mcp.json` | Team shared, committed to VCS |
| User | `~/.claude.json` (with `--scope user`) | Cross-project, personal |

## Management Commands
```bash
claude mcp list              # List configured servers
claude mcp get <name>        # Get details
claude mcp remove <name>     # Remove server
/mcp                         # Check status in Claude Code
```

## Key Features
- OAuth 2.0 authentication via `/mcp` command
- Dynamic tool updates (servers notify of changes)
- Tool search auto-enables when MCP tools > 10% of context
- Resources: reference with `@server:protocol://path`
- Prompts: available as `/mcp__servername__promptname`

## Permission Control
```json
{
  "allow": ["mcp__github__*"],
  "deny": ["mcp__dangerous_server"]
}
```
