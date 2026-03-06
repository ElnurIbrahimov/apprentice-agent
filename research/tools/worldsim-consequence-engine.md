# WorldSim - Consequence Simulation

## Overview
Before AURA takes a potentially risky action, WorldSim simulates the consequences to predict outcomes and side effects.

## Risk Levels
- **LOW** - Safe to proceed automatically
- **MEDIUM** - Proceed with caution, log warning
- **HIGH** - Ask user for confirmation
- **CRITICAL** - Block action, require explicit override

## How It Works
1. Action is proposed (e.g., "delete file X", "run command Y")
2. WorldSim evaluates risk level based on action type and context
3. Simulates potential outcomes (success path, failure paths)
4. Returns `SimulationResult` with risk assessment and recommendations

## Quick Check
```python
result = quick_check("rm -rf /tmp/test_dir")
# Returns: SimulationResult with risk=MEDIUM, outcomes=[...]
```

## Use Cases
- File operations (delete, move, overwrite)
- System commands (shell execution)
- Network operations (API calls, email sending)
- Database modifications (DROP, DELETE, UPDATE)

## Key Classes
- `WorldSim` - Main simulation engine
- `SimulationResult` - Outcome prediction
- `RiskLevel` - Enum for risk categorization

## File Location
- `aura/tools/worldsim.py`
