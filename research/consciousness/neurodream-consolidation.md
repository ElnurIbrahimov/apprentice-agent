# NeuroDream - Memory Consolidation Engine

## Overview
Inspired by how human brains consolidate memories during sleep. NeuroDream runs "sleep cycles" that process, connect, and strengthen memories.

## Sleep Phases
1. **LIGHT** - Surface-level memory review
2. **DEEP** - Atomic fact extraction and proposition storage
3. **REM** - Creative connection-making between distant memories
4. **WAKE** - Integration of insights back into active memory

## Key Concepts
- **Dream Triggers**: Events that initiate a sleep cycle (idle time, memory threshold, explicit request)
- **Dream Insights**: Novel connections discovered during REM phase
- **Consolidated Patterns**: Strengthened memory patterns from deep phase
- **Atomic Facts**: Propositions extracted from conversations and stored for spaced repetition

## Integration Points
- Feeds into `SpacedRepetitionTool` for auto-generating flashcards
- Connects to `KnowledgeGraph` for relationship discovery
- Uses `HybridMemory` for storage
- Triggered by `MetacognitiveGuardian` based on cognitive load

## File Location
- `aura/tools/neurodream.py`

## Data Classes
- `SleepSession` - A complete sleep cycle
- `DreamInsight` - A discovered connection
- `ConsolidatedPattern` - A strengthened memory pattern
- `DreamTrigger` - What initiates sleep
