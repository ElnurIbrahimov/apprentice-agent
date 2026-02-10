# Skill: FastAPI Route Pattern for AURA Tools

## Pattern
```python
from fastapi import APIRouter
from pydantic import BaseModel
import asyncio

router = APIRouter(prefix="/api", tags=["tools"])


def _get_agent_service():
    """Lazy import to avoid circular deps and blocking event loop."""
    from api.services.agent_service import agent_service
    return agent_service


# Request model (for POST/PUT)
class MyRequest(BaseModel):
    param1: str
    param2: int = 10
    optional_param: Optional[str] = None


# GET endpoint
@router.get("/my-tool/action")
async def my_tool_action(query_param: str = "default"):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _my_tool_sync(query_param)
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _my_tool_sync(param: str) -> dict:
    agent = _get_agent_service().agent
    if "my_tool" in agent.tools:
        return agent.tools["my_tool"].some_method(param)
    return {"success": False, "error": "Tool not loaded"}


# POST endpoint
@router.post("/my-tool/create")
async def my_tool_create(request: MyRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _my_tool_create_sync(request)
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Key Rules
1. Always use `run_in_executor` for sync tool calls (don't block event loop)
2. Always use `_get_agent_service()` lazy import
3. Always check `if "tool_name" in agent.tools:` before calling
4. Always wrap in try/except returning error dict
5. Use Pydantic BaseModel for POST/PUT request bodies
6. Use query params for GET endpoints

## Registration
Add to `api/main.py`:
```python
from api.routes import tools_new
app.include_router(tools_new.router)
```
