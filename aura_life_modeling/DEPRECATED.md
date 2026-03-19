# DEPRECATED: Life Modeling Module

**Status:** Not wired into the main Aura agent loop.

## Why This Is Deprecated

1. **Mesa v1 API** — Uses `from mesa.time import RandomActivation` and other v1 imports that are incompatible with Mesa v2+
2. **MCP tools not registered** — The tools defined here are never loaded into the agent's tool registry
3. **No auto-import** — The main codebase does not import this module (guarded behind try/except in agent.py)

## To Revive

1. Upgrade Mesa imports to v2 API
2. Register MCP tools in the agent's tool loader
3. Add integration tests
4. Wire into the agent loop or expose via API route
