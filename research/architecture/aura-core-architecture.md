# AURA Core Architecture

## Overview
AURA (Autonomous Universal Reasoning Agent) is a local-first AI agent built on the Observe-Plan-Act-Evaluate-Remember cognitive loop. It runs entirely on-device using Ollama for LLM inference.

## Core Loop
```
User Input -> Observe -> Plan -> Act -> Evaluate -> Remember -> Response
```

### Phases (via StateMachine)
1. **IDLE** - Waiting for input
2. **OBSERVING** - Processing user input, detecting emotions, gathering context
3. **PLANNING** - LLM decides which tool + action to use
4. **ACTING** - Execute the selected tool
5. **EVALUATING** - Check if the result answers the user's query
6. **REMEMBERING** - Store important information in memory systems
7. **RESPONDING** - Format and return response to user

## Key Files
- `aura/agent.py` - Main agent class, tool dispatch, cognitive loop
- `aura/brain.py` - Ollama LLM integration, prompt engineering, tool selection
- `aura/config.py` - Configuration management
- `aura/identity.py` - Agent personality and identity
- `aura/state_machine.py` - Phase transitions with validation
- `aura/thinking_mode.py` - System 1 (fast) vs System 2 (deep) thinking

## Tool Pattern
Every tool follows this pattern:
```python
class MyTool:
    name = "my_tool"
    description = "What this tool does"

    def execute(self, action: str, **kwargs) -> dict:
        # Dispatch to methods based on action
        return {"success": True, "response": "..."}
```

## Tool Registration (3 places)
1. `agent.py` - Import + instantiate in `self.tools` dict
2. `brain.py` - Add description in `_get_tool_descriptions()` + parse rules
3. `tools/__init__.py` - Export for external use

## Fast Path vs Full Loop
- **Fast path**: Simple queries answered directly by LLM without tool use
- **Full loop**: Complex queries go through plan->act->evaluate cycle
- Controlled by `self.use_fastpath` flag

## Tool Loading Strategy
- **Core tools** (always loaded): filesystem, web_search, clipboard, calendar, etc.
- **Heavy tools** (conditional): vision, browser, voice, image_gen - loaded in `if not fast_init:`
- **Lazy tools**: Loaded on first use via `_ensure_tool()`

## Security
- Custom tools validated via AST analysis before loading
- Blocked imports: os.system, subprocess, eval, exec, etc.
- Shell executor has allowlist + blocklist for commands

## Date Created
2025 (ongoing development)
