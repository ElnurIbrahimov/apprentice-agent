# State Machine & Thinking Mode

## State Machine (`state_machine.py`)
Controls AURA's cognitive phase transitions with validation.

### Phases
IDLE -> OBSERVING -> PLANNING -> ACTING -> EVALUATING -> REMEMBERING -> IDLE

### Valid Transitions
```python
VALID_TRANSITIONS = {
    Phase.IDLE: [Phase.OBSERVING],
    Phase.OBSERVING: [Phase.PLANNING, Phase.IDLE],
    Phase.PLANNING: [Phase.ACTING, Phase.IDLE],
    Phase.ACTING: [Phase.EVALUATING],
    Phase.EVALUATING: [Phase.REMEMBERING, Phase.PLANNING],  # Can re-plan
    Phase.REMEMBERING: [Phase.IDLE],
}
```

### Features
- Transition hooks (callbacks on enter/exit)
- Phase timing (how long in each phase)
- History tracking (last N transitions)
- Invalid transition prevention

## Thinking Mode (`thinking_mode.py`)
Controls fast vs deep thinking.

### Modes
- **AUTO** - System decides based on query complexity
- **SYSTEM1** - Fast, intuitive responses (simple queries)
- **SYSTEM2** - Slow, deliberate reasoning (complex queries)

### CognitiveLoadState
Tracks current cognitive load to inform mode selection:
- Query complexity score
- Active tool count
- Memory usage
- Current phase

### API Endpoints
- `GET /api/state-machine/current` - Current phase
- `POST /api/state-machine/transition` - Force transition
- `GET /api/state-machine/history` - Transition history
- `GET /api/thinking-mode/current` - Current mode
- `POST /api/thinking-mode/set` - Set mode
