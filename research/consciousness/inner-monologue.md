# Inner Monologue System

## Overview
Gives AURA an internal thought stream — observable thoughts that run in parallel with actions. Creates transparency into the agent's "thinking process."

## Thought Types
| Type | Icon | Description |
|------|------|-------------|
| observation | eye | Noticing something in the environment |
| reflection | mirror | Thinking about past interactions |
| planning | compass | Considering next steps |
| emotion | heart | Emotional reaction |
| curiosity | sparkle | Wondering about something |
| concern | warning | Worry or caution |
| insight | lightbulb | Sudden understanding |
| memory | brain | Recalling past information |

## How It Works
1. Agent processes user input
2. Inner monologue generates contextual thoughts
3. Thoughts are timestamped and typed
4. Connected to EvoEmo for emotional thoughts
5. Thoughts visible in UI via API

## Key Functions
- `get_monologue()` - Get singleton instance
- `monologue.add_thought(type, content)` - Record a thought
- `monologue.get_recent(n)` - Get last N thoughts
- `monologue.connect_evoemo(evoemo)` - Link to emotional system

## File Location
- `aura/tools/inner_monologue.py`
