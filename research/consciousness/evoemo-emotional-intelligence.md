# EvoEmo - Evolutionary Emotional System

## Overview
EvoEmo provides real-time emotional awareness by analyzing user messages for emotional tone and maintaining a mood state that influences AURA's responses.

## Architecture
- **Input Analysis**: Detects emotions in user text (joy, sadness, anger, fear, surprise, disgust, neutral)
- **Mood State**: Rolling emotional state that decays over time
- **Output Modulation**: Adjusts response tone based on detected emotions

## Key Components
- `EvoEmoTool` - Main tool class
- `analyze_emotion(text)` - Detect emotions in text
- `get_current_mood()` - Current emotional state
- `get_mood_emoji()` - Visual mood indicator

## Integration Points
- Connected to `InnerMonologue` for emotional thought generation
- Feeds into ALMA engine for deeper emotional processing
- Modulates brain parameters via neuromodulator system
- Influences response style via `get_tone_modifier()` and `build_adaptive_system_prompt()`

## ALMA Engine (Advanced Layer)
The ALMA (Adaptive Limbic Modulation Architecture) engine sits on top of EvoEmo:
- Neuromodulator simulation (dopamine, serotonin, norepinephrine, oxytocin)
- Emotional memory formation
- Mood-dependent parameter tuning
- Located in `aura/emotion/`

## File Locations
- `aura/tools/evoemo.py` - Core emotion detection
- `aura/tools/evoemo_prompts.py` - Tone modifiers and style builders
- `aura/emotion/alma_engine.py` - ALMA emotional engine
- `aura/emotion/integration.py` - Bridge between EvoEmo and ALMA
