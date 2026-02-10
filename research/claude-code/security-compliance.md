# Claude Code — Security & Compliance

**Tags:** claude-code, security, sandboxing, enterprise
**Created:** 2025
**Category:** claude-code

---

## Permission Architecture
- Three-tier: read-only, modification, command execution
- Per-tool granularity
- Pre-session approval
- Deny rules always take precedence

## Sandboxing
- Filesystem isolation
- Network isolation (allowed domains)
- OS-level enforcement
- Protects against prompt injection

## Managed Settings (Enterprise)
**Locations:**
- macOS: `/Library/Application Support/ClaudeCode/`
- Linux: `/etc/claude-code/`
- Windows: `C:\Program Files\ClaudeCode\`

**Enterprise-only Settings:**
- `disableBypassPermissionsMode` — Prevent permission bypass
- `allowManagedPermissionRulesOnly` — Only managed rules
- `allowManagedHooksOnly` — Only managed hooks
- `strictKnownMarketplaces` — Restrict plugin sources

## Data Handling
- Code doesn't train models
- Configurable telemetry
- 30-day data retention
- Audit logging for enterprises
- Session quality surveys (optional)

## Best Practices
1. Use `.env*` deny rules to protect secrets
2. Set up project-level permissions in `.claude/settings.json`
3. Use hooks to validate commands before execution
4. Enable managed settings for team environments
5. Regular audit of allowed tools and permissions
