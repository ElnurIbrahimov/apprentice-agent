# Skill: Creating New AURA Tools

## Pattern
Every AURA tool follows this exact structure:

```python
"""Tool description — what it does in one line."""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "tool_data.json"


class MyTool:
    """One-line description."""

    name = "my_tool"                    # snake_case, unique
    description = "What it does"        # Short description

    def __init__(self):
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> dict:
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"items": []}

    def _save_data(self, data: dict) -> bool:
        try:
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except IOError:
            return False

    def method_one(self, param: str) -> dict:
        return {"success": True, "response": f"Did {param}"}

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()
        if action_lower.startswith("method_one"):
            return self.method_one(kwargs.get("param", ""))
        return {"success": False, "error": f"Unknown: {action}"}


# Singleton
my_tool = MyTool()
```

## Registration Checklist
1. [ ] `tools/__init__.py` — import + __all__
2. [ ] `agent.py` — import, core/conditional dict, _lazy_tools, _ensure_tool
3. [ ] `brain.py` — description, TOOL: normalization, fallback detection
4. [ ] `api/routes/tools_new.py` — API endpoints (optional)
5. [ ] `py_compile` check
6. [ ] Import test
7. [ ] Functional test

## Common IDs
```python
uuid.uuid4().hex[:8]  # "a1b2c3d4"
```

## Common Timestamps
```python
datetime.now().isoformat()  # "2025-01-15T14:30:00.123456"
```
