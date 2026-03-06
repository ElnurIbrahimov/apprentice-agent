# Tool Registration Guide

## How to Add a New Tool to AURA

### Step 1: Create the Tool File
Location: `aura/tools/<tool_name>.py`

```python
"""Tool description."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "<tool_data>.json"


class MyNewTool:
    """One-line description."""

    name = "<tool_name>"
    description = "<what it does>"

    def __init__(self):
        # Setup, create data dirs, etc.
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    def some_method(self, param: str) -> dict:
        """Do something."""
        return {
            "success": True,
            "response": f"Did something with {param}"
        }

    def execute(self, action: str, **kwargs) -> dict:
        """Dispatch actions."""
        action_lower = action.lower().strip()

        if action_lower.startswith("some"):
            return self.some_method(kwargs.get("param", ""))

        return {"success": False, "error": f"Unknown action: {action}"}


# Singleton
my_new_tool = MyNewTool()
```

### Step 2: Update `tools/__init__.py`
```python
from .my_new_tool import MyNewTool
# Add to __all__
"MyNewTool",
```

### Step 3: Update `agent.py`
1. Add to import line (line ~132)
2. Add to core tools dict OR conditional loading block
3. Add to `_lazy_tools` list (for fast_init mode)
4. Add to `_ensure_tool()` method

### Step 4: Update `brain.py`
1. Add description in `_get_tool_descriptions()` (~line 2018)
2. Add TOOL: name normalization in `_parse_action_response()` (~line 2048)
3. Add fallback detection rules (~line 2094)

### Step 5: (Optional) Add API Routes
In `api/routes/tools_new.py`:
```python
@router.get("/my-tool/action")
async def my_tool_action():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _my_tool_sync)
    return result

def _my_tool_sync() -> dict:
    agent = _get_agent_service().agent
    if "my_tool" in agent.tools:
        return agent.tools["my_tool"].some_method()
    return {"success": False, "error": "Tool not loaded"}
```

### Step 6: Verify
```bash
python -m py_compile aura/tools/my_new_tool.py
python -c "from aura.tools import MyNewTool; t = MyNewTool(); print(t.execute('some'))"
```

## Common Patterns
- **Data storage**: JSON files in `data/` directory
- **IDs**: `uuid.uuid4().hex[:8]`
- **Timestamps**: `datetime.now().isoformat()`
- **Lazy model loading**: Module-level singleton with `_load_model()` function
- **Error handling**: Return `{"success": False, "error": "message"}`, never raise
