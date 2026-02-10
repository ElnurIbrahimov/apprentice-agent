# Claude Code — Debugging & Troubleshooting Guide

**Tags:** claude-code, debugging, troubleshooting, errors
**Created:** 2025
**Category:** claude-code

---

## Built-in Debugging Tools

### /doctor — Health Check
```
/doctor
```
Checks:
- Claude Code version
- Node.js version
- API connectivity
- MCP server status
- Extension compatibility
- Shell configuration

### /debug — Session Debug Log
```
/debug
```
Shows detailed debug log for current session.

### /status — Quick Status
```
/status
```
Shows version, model, account, connectivity.

### Debug Flags
```bash
claude --debug              # Full debug output
claude --debug "api,mcp"    # Filter by category
claude --verbose            # Verbose logging
```

---

## Common Issues & Fixes

### MCP Server Issues

**Server not connecting:**
1. Check syntax in `.mcp.json` or `~/.claude.json`
2. Verify npm packages are installed globally
3. Check environment variables are set
4. Try `claude mcp get <name>` for details
5. Use `/mcp` to check status

**Server timeout:**
```bash
export MCP_TIMEOUT=30000    # Increase timeout to 30s
```

**Too many tools slowing down:**
```bash
export ENABLE_TOOL_SEARCH=auto:5   # Search when >5 tools
```

### Extension Issues

**VS Code extension won't install:**
- Requires VS Code 1.98.0+
- Restart IDE after install
- Check for conflicting extensions

**JetBrains not connecting:**
- Settings -> Tools -> Claude Code [Beta]
- Verify Claude Code CLI path
- For WSL: `wsl -d Ubuntu -- bash -lic "claude"`

### Permission Issues

**Tool keeps asking for permission:**
Add to settings:
```json
{"permissions": {"allow": ["Bash(npm run *)"]}}
```

**Can't bypass permissions:**
- Check managed settings aren't blocking it
- `disableBypassPermissionsMode` may be set

### Shell Issues

**Wrong shell detected:**
```bash
export CLAUDE_CODE_SHELL=/bin/bash
```

**Commands failing on Windows:**
- Ensure Git Bash or WSL is available
- Check `CLAUDE_CODE_SHELL` environment variable
- Some Unix commands need Windows equivalents

### Context Issues

**Context getting too large:**
```
/compact focus on [what matters]
```

**Lost important context after compaction:**
- Use CLAUDE.md for persistent instructions
- Put key decisions in CLAUDE.md via `/memory`
- Auto-memory saves patterns automatically

### API Issues

**Rate limiting:**
```
/usage     # Check current rate limit status
```
- Wait and retry
- Switch to a less-used model
- Reduce token usage per request

**Authentication failed:**
```bash
claude logout && claude login
```

### WSL-Specific Issues

**Path translation:**
- Windows paths: `C:\Users\...`
- WSL paths: `/mnt/c/Users/...`
- Claude Code handles translation, but some tools may not

**Performance:**
- WSL 2 is faster than WSL 1
- Keep project files in Linux filesystem (`~/projects/`)
- Avoid `/mnt/c/` for performance-critical work

---

## Diagnostic Checklist
1. Run `/doctor` first
2. Check `/status` for connectivity
3. Try `/debug` for session-specific issues
4. Check `~/.claude/` directory for corrupt configs
5. Try `claude --debug` for verbose startup info
6. Check GitHub issues: github.com/anthropics/claude-code/issues
