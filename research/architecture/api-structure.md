# AURA API Structure

## Overview
FastAPI backend serving the AURA frontend and external integrations.

## Main Entry Points
- `api/main.py` - FastAPI app setup, router registration
- `main.py` - Application launcher (GUI + API server)

## Router Organization
| File | Prefix | Purpose |
|------|--------|---------|
| `api/routes/chat.py` | /api/chat | Chat and conversation |
| `api/routes/tools.py` | /api/tools | Tool management |
| `api/routes/tools_new.py` | /api | Calendar, flashcards, email, shell, etc. |
| `api/routes/features.py` | /api | Feature endpoints |
| `api/routes/thinking.py` | /api | Thinking stream |
| `api/routes/state_machine.py` | /api | State machine control |
| `api/routes/thinking_mode.py` | /api | Thinking mode control |

## Common Patterns
```python
# Lazy agent import
def _get_agent_service():
    from api.services.agent_service import agent_service
    return agent_service

# Async wrapper for sync tool calls
@router.get("/endpoint")
async def my_endpoint():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _sync_helper)
    return result

def _sync_helper() -> dict:
    agent = _get_agent_service().agent
    if "tool_name" in agent.tools:
        return agent.tools["tool_name"].method()
    return {"success": False, "error": "Tool not loaded"}
```

## Services
- `api/services/agent_service.py` - Singleton agent instance management
- Handles agent lifecycle (init, shutdown, state persistence)

## Frontend
- `gui.py` - Tkinter-based desktop GUI
- Web UI served via FastAPI static files
