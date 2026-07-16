# Tool Policy

Every tool — static, MCP, or Bridge — passes through `ToolExecutor` with
a local policy check. Remote annotations (MCP server claims) are hints,
not authority.

## Policy layers (innermost wins)

1. **Built-in `ToolRisk`**: READ_ONLY → REMOTE_COST → WRITE_LOCAL → WRITE_REMOTE → PRIVACY → DESTRUCTIVE
2. **Incognito gate**: when `memoryEnabled=false`, risk >= WRITE_LOCAL is blocked
3. **User tool policy**: enabled/disabled, allowed contexts, minimum confirmation, cost ceiling, app/domain/path scopes
4. **Approval scope**: per-run, per-step, per-tool — never blanket

## Rules

1. Remote MCP risk annotations can only tighten, never loosen, local classification.
2. An expired approval is rejected. "Always for this scope" creates a policy proposal, not an auto-grant.
3. Cost ceilings are enforced before the API call, not after.
4. Shell tools use argv/cwd/env structures — never string interpolation from model output.
5. Accessibility actions require a fresh UI snapshot — stale nodes are rejected.
6. Destructive/irreversible actions always require explicit user confirmation.
7. Policy is stored in DataStore (non-secret) + SecureDataStore (secrets), never in tool definitions.