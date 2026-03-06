# SynapseForge - Dynamic Tool Creation

## Overview
SynapseForge allows AURA to create new tools at runtime by generating Python code. When AURA encounters a task that no existing tool handles, it can synthesize a new one.

## How It Works
1. Agent identifies need for a tool that doesn't exist
2. SynapseForge generates tool code using LLM
3. Code is security-validated (AST analysis, blocked imports/patterns)
4. Tool is saved to `aura/tools/synthesized/`
5. Tool is dynamically imported and registered

## Security Validation
All generated code goes through `validate_custom_tool_code()`:
- **Allowed imports**: typing, json, re, datetime, pathlib, math, etc.
- **Blocked patterns**: os.system, subprocess, eval, exec, pickle, etc.
- **AST walk**: Checks every import and function call against allowlists

## Generated Tool Structure
```python
# aura/tools/synthesized/my_tool.py
"""Auto-generated tool description."""

def execute(action: str, **kwargs) -> dict:
    """Main entry point."""
    return {"success": True, "response": "..."}
```

## Key Classes
- `SynapseForge` - Main generator
- `SynthesizedTool` - Wrapper for generated tools

## Limitations
- No network access (no requests, urllib)
- No file system writes outside data/
- No subprocess execution
- Limited to safe pure-Python operations

## File Location
- `aura/tools/synapseforge.py`
- Generated tools: `aura/tools/synthesized/`
