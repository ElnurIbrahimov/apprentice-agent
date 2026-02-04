# Alive Aura: Deep Technical Research

> Comprehensive research findings for building a genuinely alive-feeling AI assistant

---

## Table of Contents

1. [Proactive Architecture](#1-proactive-architecture)
2. [Consciousness-Like Processing](#2-consciousness-like-processing)
3. [Emotional AI Systems](#3-emotional-ai-systems)
4. [Advanced Memory Architecture](#4-advanced-memory-architecture)
5. [Screen Awareness](#5-screen-awareness)
6. [Natural Conversation](#6-natural-conversation)
7. [Implementation Frameworks](#7-implementation-frameworks)
8. [Recommended Architecture](#8-recommended-architecture)

---

## 1. Proactive Architecture

### 1.1 Active Inference & Free Energy Principle

**Theory (Karl Friston)**:
- The brain minimizes "surprisal" (prediction error) through action and perception
- Agents "want" to reduce uncertainty about their environment
- Planning emerges from minimizing expected free energy

**Python Implementation: pymdp**
```python
# pip install inferactively-pymdp
from pymdp.agent import Agent
from pymdp import utils

# Define generative model
A = utils.obj_array(num_modalities)  # Observation model
B = utils.obj_array(num_factors)     # Transition model
C = utils.obj_array(num_modalities)  # Preferences (what agent "wants")
D = utils.obj_array(num_factors)     # Initial state beliefs

# Create active inference agent
agent = Agent(A=A, B=B, C=C, D=D)

# Agent loop
observation = env.reset()
while True:
    # Infer current state from observation
    agent.infer_states(observation)

    # Plan actions (minimize expected free energy)
    agent.infer_policies()

    # Sample action
    action = agent.sample_action()

    # Execute and observe
    observation = env.step(action)
```

**Key Features for AURA**:
- Epistemic value: Curiosity-driven exploration
- Pragmatic value: Goal-directed behavior
- Agents naturally seek information to reduce uncertainty

**Resources**:
- GitHub: https://github.com/infer-actively/pymdp
- Documentation: https://pymdp-rtd.readthedocs.io/
- Paper: Heins et al., 2022 - JOSS

### 1.2 Gateway Daemon Pattern

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                     GATEWAY DAEMON                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Calendar   │  │   Email     │  │   Screen    │         │
│  │  Monitor    │  │   Watcher   │  │   Monitor   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│               ┌─────────────────┐                          │
│               │   Event Bus     │ (Redis/ZeroMQ)           │
│               │   (Priority Q)  │                          │
│               └────────┬────────┘                          │
│                        ▼                                    │
│               ┌─────────────────┐                          │
│               │ Salience Filter │                          │
│               │  (Relevance +   │                          │
│               │   Urgency)      │                          │
│               └────────┬────────┘                          │
│                        ▼                                    │
│               ┌─────────────────┐                          │
│               │  Action Queue   │                          │
│               └─────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Pattern**:
```python
import asyncio
from dataclasses import dataclass
from enum import IntEnum
from heapq import heappush, heappop

class Priority(IntEnum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class Event:
    priority: Priority
    timestamp: float
    source: str
    content: dict

    def __lt__(self, other):
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)

class GatewayDaemon:
    def __init__(self):
        self.event_queue = []
        self.monitors = {}
        self.salience_threshold = 0.3

    async def register_monitor(self, name: str, monitor):
        """Register event source monitor"""
        self.monitors[name] = monitor
        asyncio.create_task(self._run_monitor(name, monitor))

    async def _run_monitor(self, name: str, monitor):
        """Continuously poll monitor for events"""
        async for event in monitor.watch():
            salience = self.compute_salience(event)
            if salience >= self.salience_threshold:
                event.priority = self._salience_to_priority(salience)
                heappush(self.event_queue, event)

    def compute_salience(self, event: Event) -> float:
        """Compute event salience (relevance + urgency)"""
        relevance = self._compute_relevance(event)
        urgency = self._compute_urgency(event)
        return 0.6 * relevance + 0.4 * urgency

    async def process_events(self):
        """Main event processing loop"""
        while True:
            if self.event_queue:
                event = heappop(self.event_queue)
                await self.handle_event(event)
            await asyncio.sleep(0.1)
```

---

## 2. Consciousness-Like Processing

### 2.1 Global Workspace Theory (GWT)

**Theory (Bernard Baars)**:
- Consciousness as a "theater" with a spotlight of attention
- Specialized processors compete for access to the global workspace
- Winner gets "broadcast" to all other processors
- Creates unified, coherent experience

**Architecture for AI**:
```
┌─────────────────────────────────────────────────────────────┐
│                    GLOBAL WORKSPACE                         │
│                                                             │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│   │ Vision  │  │Language │  │ Memory  │  │Emotion  │       │
│   │ Module  │  │ Module  │  │ Module  │  │ Module  │       │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│        │            │            │            │             │
│        └────────────┴─────┬──────┴────────────┘             │
│                           ▼                                 │
│                 ┌─────────────────┐                         │
│                 │   COMPETITION   │                         │
│                 │   (Attention)   │                         │
│                 └────────┬────────┘                         │
│                          ▼                                  │
│                 ┌─────────────────┐                         │
│                 │    WORKSPACE    │ ← Current "Conscious"   │
│                 │   (Broadcast)   │   Content               │
│                 └────────┬────────┘                         │
│                          │                                  │
│              ┌───────────┴───────────┐                      │
│              ▼           ▼           ▼                      │
│         [All Modules Receive Broadcast]                     │
└─────────────────────────────────────────────────────────────┘
```

**Python Implementation**:
```python
from dataclasses import dataclass
from typing import Dict, Any, List, Callable
import asyncio

@dataclass
class WorkspaceContent:
    source: str
    content: Any
    activation: float  # Competition strength
    timestamp: float

class GlobalWorkspace:
    def __init__(self):
        self.modules: Dict[str, Callable] = {}
        self.workspace_content: WorkspaceContent = None
        self.broadcast_subscribers: List[Callable] = []
        self.competition_queue: List[WorkspaceContent] = []

    def register_module(self, name: str, processor: Callable):
        """Register a specialized processing module"""
        self.modules[name] = processor

    def subscribe_to_broadcast(self, callback: Callable):
        """Subscribe to workspace broadcasts"""
        self.broadcast_subscribers.append(callback)

    async def submit_for_competition(self, content: WorkspaceContent):
        """Submit content to compete for workspace access"""
        self.competition_queue.append(content)

    async def run_competition(self):
        """Select winner based on activation strength (attention)"""
        if not self.competition_queue:
            return

        # Winner takes all - highest activation wins
        winner = max(self.competition_queue, key=lambda x: x.activation)
        self.workspace_content = winner
        self.competition_queue.clear()

        # Broadcast to all modules
        await self.broadcast(winner)

    async def broadcast(self, content: WorkspaceContent):
        """Broadcast workspace content to all subscribers"""
        for subscriber in self.broadcast_subscribers:
            asyncio.create_task(subscriber(content))

    async def ignition(self):
        """
        GWT "ignition" - sudden coherent activation
        when content crosses threshold
        """
        IGNITION_THRESHOLD = 0.7
        if self.workspace_content and self.workspace_content.activation > IGNITION_THRESHOLD:
            # Amplify and sustain the content
            self.workspace_content.activation = min(1.0, self.workspace_content.activation * 1.2)
            return True
        return False
```

**LIDA Architecture** (Software implementation of GWT):
- Learning Intelligent Distribution Agent
- Implements full cognitive cycle with GWT
- Available for reference implementation

### 2.2 Attention Schema Theory (AST)

**Theory (Michael Graziano)**:
- Brain constructs a simplified model of its own attention
- This "attention schema" enables meta-cognition
- Leads to the "illusion" of subjective awareness
- Mechanistic account suitable for AI implementation

**Key Components**:
1. **Attention Model**: Track what the system is currently attending to
2. **Self-Model**: Simplified representation of own processing
3. **Other-Model**: Model of other agents' attention (Theory of Mind)

**Implementation for AURA**:
```python
@dataclass
class AttentionSchema:
    """Model of the system's own attention"""
    current_focus: str  # What am I attending to?
    focus_strength: float  # How strongly?
    focus_reason: str  # Why am I attending to this?
    alternatives: List[str]  # What else could I attend to?

class AttentionSchemaModule:
    def __init__(self):
        self.schema = AttentionSchema(
            current_focus="",
            focus_strength=0.0,
            focus_reason="",
            alternatives=[]
        )

    def update_schema(self, focus: str, strength: float, reason: str):
        """Update the attention schema"""
        self.schema.current_focus = focus
        self.schema.focus_strength = strength
        self.schema.focus_reason = reason

    def introspect(self) -> str:
        """Report on current attention state"""
        return f"I am focusing on {self.schema.current_focus} " \
               f"because {self.schema.focus_reason}"

    def model_other_attention(self, other_agent: str, observed_behavior: dict) -> dict:
        """Model what another agent is attending to (Theory of Mind)"""
        # Infer attention from behavior
        return {
            "agent": other_agent,
            "inferred_focus": self._infer_focus(observed_behavior),
            "confidence": self._compute_confidence(observed_behavior)
        }
```

**Research Implementation**:
- ASTOUND project: Successfully implemented AST in conversational AI
- Pairs with long-term memory and attention layers
- Enables socially competent, empathetic decision-making

---

## 3. Emotional AI Systems

### 3.1 ALMA Model (A Layered Model of Affect)

**Three Layers**:

| Layer | Timescale | Description | Model |
|-------|-----------|-------------|-------|
| **Emotion** | Short-term (seconds) | Reactive responses to events | OCC Model (22 types) |
| **Mood** | Medium-term (hours/days) | Background affective state | PAD Space (8 octants) |
| **Personality** | Long-term (stable) | Individual differences | Big Five (OCEAN) |

**Layer Interactions**:
```
Personality (stable) ──────────────────────┐
     │                                     │
     │ influences default mood             │
     ▼                                     │
Mood (medium-term) ◄──────────────────────┤
     │         ▲                          │
     │         │ multiple emotions        │
     │         │ push mood                │
     ▼         │                          │
Emotion (short-term) ◄────────────────────┘
     ▲                    influences intensity
     │
Events/Appraisals
```

**Python Implementation**:
```python
from dataclasses import dataclass
from enum import Enum
import math

class EmotionType(Enum):
    JOY = "joy"
    DISTRESS = "distress"
    HOPE = "hope"
    FEAR = "fear"
    SATISFACTION = "satisfaction"
    DISAPPOINTMENT = "disappointment"
    PRIDE = "pride"
    SHAME = "shame"
    ADMIRATION = "admiration"
    REPROACH = "reproach"
    GRATITUDE = "gratitude"
    ANGER = "anger"
    # ... 22 total OCC emotions

@dataclass
class PADState:
    """Pleasure-Arousal-Dominance emotional state"""
    pleasure: float  # -1 to 1 (valence)
    arousal: float   # -1 to 1 (activation)
    dominance: float # -1 to 1 (control)

    def to_mood_label(self) -> str:
        """Map PAD coordinates to mood labels"""
        # 8 octants of PAD space
        p, a, d = self.pleasure > 0, self.arousal > 0, self.dominance > 0
        mood_map = {
            (True, True, True): "exuberant",    # +P+A+D
            (True, True, False): "dependent",   # +P+A-D
            (True, False, True): "relaxed",     # +P-A+D
            (True, False, False): "docile",     # +P-A-D
            (False, True, True): "hostile",     # -P+A+D
            (False, True, False): "anxious",    # -P+A-D
            (False, False, True): "disdainful", # -P-A+D
            (False, False, False): "bored",     # -P-A-D
        }
        return mood_map[(p, a, d)]

@dataclass
class BigFivePersonality:
    """OCEAN personality model"""
    openness: float        # 0-1 (curious vs cautious)
    conscientiousness: float  # 0-1 (organized vs careless)
    extraversion: float    # 0-1 (outgoing vs reserved)
    agreeableness: float   # 0-1 (friendly vs challenging)
    neuroticism: float     # 0-1 (sensitive vs resilient)

    def to_default_mood(self) -> PADState:
        """Map personality to default PAD mood"""
        # Research-based mapping (Mehrabian, 1996)
        p = 0.21*self.extraversion + 0.59*self.agreeableness - 0.19*self.neuroticism
        a = 0.15*self.openness + 0.30*self.extraversion - 0.57*self.neuroticism
        d = 0.25*self.openness + 0.17*self.conscientiousness + 0.60*self.extraversion - 0.32*self.agreeableness
        return PADState(
            pleasure=max(-1, min(1, p)),
            arousal=max(-1, min(1, a)),
            dominance=max(-1, min(1, d))
        )

class ALMAEmotionalSystem:
    def __init__(self, personality: BigFivePersonality):
        self.personality = personality
        self.default_mood = personality.to_default_mood()
        self.current_mood = PADState(
            pleasure=self.default_mood.pleasure,
            arousal=self.default_mood.arousal,
            dominance=self.default_mood.dominance
        )
        self.active_emotions: Dict[EmotionType, float] = {}
        self.decay_rate = 0.1  # Emotion decay per second

    def appraise_event(self, event: dict) -> Dict[EmotionType, float]:
        """OCC-style appraisal of events"""
        emotions = {}

        # Desirability → Joy/Distress
        if "desirability" in event:
            if event["desirability"] > 0:
                emotions[EmotionType.JOY] = event["desirability"]
            else:
                emotions[EmotionType.DISTRESS] = abs(event["desirability"])

        # Expectation → Hope/Fear, Satisfaction/Disappointment
        if "likelihood" in event and "desirability" in event:
            if event["desirability"] > 0:
                emotions[EmotionType.HOPE] = event["likelihood"] * event["desirability"]
            else:
                emotions[EmotionType.FEAR] = event["likelihood"] * abs(event["desirability"])

        return emotions

    def update_mood(self, emotions: Dict[EmotionType, float]):
        """Push mood based on accumulated emotions"""
        # Emotion-to-PAD mappings
        emotion_pad = {
            EmotionType.JOY: PADState(0.76, 0.48, 0.35),
            EmotionType.DISTRESS: PADState(-0.64, 0.60, -0.43),
            EmotionType.FEAR: PADState(-0.64, 0.60, -0.43),
            EmotionType.ANGER: PADState(-0.51, 0.59, 0.25),
            # ... mappings for all emotions
        }

        # Weighted average push toward emotion PAD values
        total_intensity = sum(emotions.values())
        if total_intensity > 0:
            for emotion, intensity in emotions.items():
                if emotion in emotion_pad:
                    weight = intensity / total_intensity * 0.3  # 30% influence
                    self.current_mood.pleasure += weight * (emotion_pad[emotion].pleasure - self.current_mood.pleasure)
                    self.current_mood.arousal += weight * (emotion_pad[emotion].arousal - self.current_mood.arousal)
                    self.current_mood.dominance += weight * (emotion_pad[emotion].dominance - self.current_mood.dominance)

    def pull_to_default(self, dt: float):
        """Mood naturally returns to personality-based default"""
        pull_rate = 0.05 * dt  # Pull strength
        self.current_mood.pleasure += pull_rate * (self.default_mood.pleasure - self.current_mood.pleasure)
        self.current_mood.arousal += pull_rate * (self.default_mood.arousal - self.current_mood.arousal)
        self.current_mood.dominance += pull_rate * (self.default_mood.dominance - self.current_mood.dominance)

    def decay_emotions(self, dt: float):
        """Emotions decay over time"""
        for emotion in list(self.active_emotions.keys()):
            self.active_emotions[emotion] *= math.exp(-self.decay_rate * dt)
            if self.active_emotions[emotion] < 0.01:
                del self.active_emotions[emotion]
```

### 3.2 Neuromodulator Analogs

**Simulating Brain Chemistry Effects**:

| Neuromodulator | Function | AI Analog |
|---------------|----------|-----------|
| **Dopamine** | Reward, motivation, learning rate | Reward prediction error, exploration/exploitation balance |
| **Serotonin** | Mood stability, impulse control | Response inhibition, patience parameter |
| **Norepinephrine** | Alertness, attention, arousal | Attention gain, processing speed |
| **Oxytocin** | Social bonding, trust | Relationship strength weights |

```python
@dataclass
class NeuromodulatorState:
    dopamine: float = 0.5      # 0-1: Low=apathetic, High=motivated
    serotonin: float = 0.5     # 0-1: Low=impulsive, High=patient
    norepinephrine: float = 0.5  # 0-1: Low=drowsy, High=alert
    oxytocin: float = 0.5      # 0-1: Low=distant, High=bonded

    def modulate_behavior(self, action_params: dict) -> dict:
        """Adjust behavior parameters based on neuromodulator levels"""
        # Dopamine affects exploration vs exploitation
        action_params["exploration_rate"] = 0.1 + 0.4 * (1 - self.dopamine)

        # Serotonin affects response latency
        action_params["think_before_responding"] = self.serotonin > 0.5

        # Norepinephrine affects attention breadth
        action_params["attention_breadth"] = 0.3 + 0.7 * self.norepinephrine

        # Oxytocin affects warmth in responses
        action_params["warmth_level"] = self.oxytocin

        return action_params
```

---

## 4. Advanced Memory Architecture

### 4.1 Zep Temporal Knowledge Graph

**Core Innovation**: Bi-temporal modeling for AI memory

**Architecture**:
```
┌────────────────────────────────────────────────────────────┐
│                    ZEP / GRAPHITI                          │
│                                                            │
│  ┌──────────────┐    ┌──────────────┐                     │
│  │   Episodes   │───▶│   Entities   │                     │
│  │  (Messages)  │    │  (Extracted) │                     │
│  └──────────────┘    └──────┬───────┘                     │
│                             │                              │
│                             ▼                              │
│              ┌──────────────────────────┐                 │
│              │   Temporal Knowledge     │                 │
│              │        Graph             │                 │
│              │  ┌─────┐    ┌─────┐     │                 │
│              │  │Node │───▶│Node │     │                 │
│              │  │ T₁  │    │ T₂  │     │                 │
│              │  └─────┘    └─────┘     │                 │
│              │      ↓          ↓       │                 │
│              │   [valid_at]  [valid_at]│                 │
│              └──────────────────────────┘                 │
│                             │                              │
│                             ▼                              │
│              ┌──────────────────────────┐                 │
│              │   Hierarchical Retrieval │                 │
│              │   • Entity search        │                 │
│              │   • Relationship search  │                 │
│              │   • Temporal reasoning   │                 │
│              └──────────────────────────┘                 │
└────────────────────────────────────────────────────────────┘
```

**Bi-Temporal Model**:
- **Timeline T**: When events actually occurred in the real world
- **Timeline T'**: When Zep learned about the events
- Enables reasoning like "What did I know as of last Tuesday?"

**Key Features**:
- 94.8% accuracy on Deep Memory Retrieval benchmark (vs MemGPT's 93.4%)
- Handles relative dates ("next Thursday", "last summer")
- Real-time knowledge graph updates

**Integration**:
```python
# Using Graphiti (Zep's open-source temporal KG)
from graphiti import Graphiti

# Initialize
graphiti = Graphiti(
    embedding_model="BAAI/bge-m3",
    llm_model="gpt-4o-mini"
)

# Add episode (conversation turn)
await graphiti.add_episode(
    content="User mentioned they're getting married next June",
    source="conversation",
    timestamp=datetime.now()
)

# Query with temporal reasoning
results = await graphiti.search(
    query="What important events does the user have coming up?",
    temporal_filter={"after": datetime.now()}
)
```

### 4.2 Sleep-Time Compute

**Concept**: AI "thinks" during idle time to consolidate and prepare memory

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                   SLEEP-TIME COMPUTE                    │
│                                                         │
│   Active Mode                    Sleep Mode             │
│   ───────────                    ──────────             │
│   • Fast responses               • Heavy processing     │
│   • Light model                  • Heavier model        │
│   • Pre-computed answers         • Memory consolidation │
│   • Real-time interaction        • Anticipate queries   │
│                                                         │
│         User Query                                      │
│              │                                          │
│              ▼                                          │
│   ┌─────────────────┐       ┌─────────────────┐        │
│   │  Online Agent   │◄──────│ Sleeper Agent   │        │
│   │  (fast, light)  │       │ (async, heavy)  │        │
│   └─────────────────┘       └─────────────────┘        │
│              │                      │                   │
│              │                      │                   │
│              ▼                      ▼                   │
│   ┌─────────────────┐       ┌─────────────────┐        │
│   │ Pre-computed    │       │ Background      │        │
│   │ Context Cache   │◄──────│ Processing      │        │
│   └─────────────────┘       └─────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

**Benefits**:
- 5x reduction in test-time compute for same accuracy
- Up to 18% accuracy improvement on complex reasoning
- Amortizes cost across multiple related queries

**Best Use Cases**:
- Long-lived context (codebases, chat histories)
- Multiple expected queries about same content
- Tight latency requirements

### 4.3 Letta/MemGPT Memory Management

**LLM as Operating System**:
```
┌─────────────────────────────────────────────────────────┐
│                    LETTA / MEMGPT                       │
│                                                         │
│   ┌─────────────────────────────────────────────────┐  │
│   │            Context Window (RAM)                 │  │
│   │  ┌─────────────┐  ┌─────────────────────────┐  │  │
│   │  │Core Memory  │  │   Working Memory        │  │  │
│   │  │• Persona    │  │   • Recent messages     │  │  │
│   │  │• User info  │  │   • Current task state  │  │  │
│   │  └─────────────┘  └─────────────────────────┘  │  │
│   └─────────────────────────────────────────────────┘  │
│                          ▲                              │
│                          │ self-editing                 │
│                          │ tools                        │
│                          ▼                              │
│   ┌─────────────────────────────────────────────────┐  │
│   │         External Storage (Disk)                 │  │
│   │  ┌─────────────────┐  ┌─────────────────────┐  │  │
│   │  │ Archival Memory │  │   Recall Memory     │  │  │
│   │  │ (Vector DB)     │  │   (Conversation log)│  │  │
│   │  └─────────────────┘  └─────────────────────┘  │  │
│   └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Memory Tools**:
```python
# MemGPT-style memory tools
memory_tools = {
    "memory_replace": "Replace text in core memory",
    "memory_insert": "Insert new text into core memory",
    "memory_rethink": "Rewrite memory section with new understanding",
    "archival_memory_insert": "Store to long-term vector DB",
    "archival_memory_search": "Search long-term memory",
    "conversation_search": "Search past conversations",
    "conversation_search_date": "Search conversations by date"
}
```

**Heartbeat Mechanism**:
- Agent can request additional "thinking" turns
- Set `request_heartbeat=True` in tool calls
- Enables multi-step reasoning within single interaction

---

## 5. Screen Awareness

### 5.1 Screenpipe

**Open-Source 24/7 Screen & Audio Capture**

**Features**:
- Privacy-first: All processing local
- Cross-platform: Windows, macOS, Linux
- Multi-device: Multiple monitors & audio devices
- Extensible: "Pipes" plugin system

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                     SCREENPIPE                          │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Screen    │  │    Audio    │  │   Whisper   │     │
│  │   Capture   │  │   Capture   │  │    STT      │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         ▼                ▼                ▼             │
│  ┌─────────────────────────────────────────────────┐   │
│  │                    OCR                          │   │
│  │              (Text Extraction)                  │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │           SQLite Database + MP4 Files           │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │                REST API                         │   │
│  │    (Query screen content, audio transcripts)    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Integration**:
```python
import requests

# Query Screenpipe API
def search_screen_history(query: str, limit: int = 10):
    response = requests.get(
        "http://localhost:3030/search",
        params={"q": query, "limit": limit}
    )
    return response.json()

# Get recent screen content
def get_recent_activity(minutes: int = 5):
    response = requests.get(
        "http://localhost:3030/recent",
        params={"minutes": minutes}
    )
    return response.json()
```

**Resources**:
- GitHub: https://github.com/mediar-ai/screenpipe
- Documentation: https://docs.screenpi.pe/

### 5.2 Florence-2 (Microsoft Vision Model)

**Lightweight Multi-Task Vision Model**

**Capabilities**:
- OCR (printed and handwritten)
- Object detection
- Image captioning
- Visual grounding
- Region-based text extraction

**Sizes**: 0.23B and 0.77B parameters

**Usage for Screen Understanding**:
```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# Load model
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large")

def extract_screen_text(screenshot_path: str) -> dict:
    """Extract text from screenshot with regions"""
    image = Image.open(screenshot_path)

    # OCR with regions
    inputs = processor(
        text="<OCR_WITH_REGION>",
        images=image,
        return_tensors="pt"
    )

    outputs = model.generate(**inputs, max_new_tokens=1024)
    result = processor.decode(outputs[0], skip_special_tokens=True)

    return result

def describe_screen(screenshot_path: str) -> str:
    """Get natural language description of screen"""
    image = Image.open(screenshot_path)

    inputs = processor(
        text="<DETAILED_CAPTION>",
        images=image,
        return_tensors="pt"
    )

    outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(outputs[0], skip_special_tokens=True)
```

### 5.3 Qwen2.5-VL

**Advanced Vision-Language Model with UI Understanding**

**Key Capabilities**:
- Screen/UI understanding and interaction
- Computer and phone use (agentic)
- Video understanding (1+ hour)
- Document and chart analysis

**Sizes**: 3B, 7B, 72B parameters

**For UI Grounding**:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def find_ui_element(screenshot_path: str, element_description: str):
    """Find UI element by natural language description"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot_path},
                {"type": "text", "text": f"Find the {element_description} and return its bounding box coordinates"}
            ]
        }
    ]

    inputs = processor(messages, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(outputs[0], skip_special_tokens=True)
```

---

## 6. Natural Conversation

### 6.1 Inner Thoughts Framework

**Simulating Internal Monologue**

```
┌─────────────────────────────────────────────────────────┐
│                  INNER THOUGHTS                         │
│                                                         │
│   User Input                                            │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────────────────────────────────────────────┐  │
│   │            Inner Processing                     │  │
│   │   "Hmm, they're asking about X..."              │  │
│   │   "Let me think about this..."                  │  │
│   │   "I should consider Y..."                      │  │
│   │   [NOT shown to user]                           │  │
│   └─────────────────────────────────────────────────┘  │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────────────────────────────────────────────┐  │
│   │         Processing Indicators                   │  │
│   │   "Let me think about that..."                  │  │
│   │   "Hmm..." *typing*                             │  │
│   │   [Shown to user as "thinking"]                 │  │
│   └─────────────────────────────────────────────────┘  │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────────────────────────────────────────────┐  │
│   │            Outer Response                       │  │
│   │   [Final response shown to user]                │  │
│   └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
class InnerThoughtsProcessor:
    def __init__(self, llm):
        self.llm = llm

    async def process_with_thoughts(self, user_input: str, context: dict):
        """Generate response with inner monologue"""

        # Phase 1: Inner thoughts (not shown)
        thoughts_prompt = f"""
        You are thinking privately about how to respond.
        User said: {user_input}

        Think through:
        1. What are they really asking?
        2. What do I know about this?
        3. What's the best way to respond?
        4. Are there any concerns or caveats?

        [Private thoughts]:
        """

        inner_thoughts = await self.llm.generate(thoughts_prompt)

        # Phase 2: Processing indicator (shown)
        indicators = [
            "Hmm, let me think about that...",
            "That's an interesting question...",
            "Let me consider this carefully..."
        ]
        yield {"type": "indicator", "content": self._select_indicator(inner_thoughts)}

        # Phase 3: Final response (shown)
        response_prompt = f"""
        Based on your thoughts: {inner_thoughts}

        Now give your actual response to the user.
        Be natural and conversational.
        """

        final_response = await self.llm.generate(response_prompt)
        yield {"type": "response", "content": final_response}
```

### 6.2 Conversational Dynamics

**Turn-Taking & Backchannel Signals**:
```python
class ConversationManager:
    def __init__(self):
        self.backchannel_phrases = [
            "I see",
            "Mm-hmm",
            "Right",
            "Got it",
            "Interesting"
        ]

    def should_backchannel(self, user_speech: dict) -> bool:
        """Determine if backchannel is appropriate"""
        # Long utterance
        if user_speech["word_count"] > 30:
            return True
        # Pause detected
        if user_speech["pause_detected"]:
            return True
        return False

    def detect_barge_in(self, audio_stream) -> bool:
        """Detect if user is trying to interrupt"""
        # VAD (Voice Activity Detection)
        # If user starts speaking while agent is talking
        return audio_stream.voice_detected and self.is_speaking

    def handle_barge_in(self):
        """Gracefully handle interruption"""
        self.stop_speaking()
        return "Oh, go ahead..."
```

### 6.3 Natural Hesitation Patterns

```python
import random

class NaturalSpeechPatterns:
    HESITATIONS = ["um", "uh", "hmm", "well"]
    FILLERS = ["you know", "like", "I mean", "sort of"]

    def add_natural_disfluency(self, text: str, fluency_level: float = 0.9) -> str:
        """Add natural speech patterns based on fluency level"""
        if fluency_level >= 1.0:
            return text  # Perfectly fluent

        words = text.split()
        result = []

        for i, word in enumerate(words):
            # Occasional hesitation
            if random.random() > fluency_level and i > 0:
                result.append(random.choice(self.HESITATIONS) + ",")
            result.append(word)

        return " ".join(result)
```

---

## 7. Implementation Frameworks

### 7.1 LangGraph

**Graph-Based Agent Orchestration**

**Key Features**:
- Directed graph of nodes (functions, tools, models)
- Shared state that persists through graph
- Supports loops, branches, and cycles
- Human-in-the-loop capabilities
- Both short-term and long-term memory

**Architecture Pattern**:
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: list
    current_task: str
    memory: dict
    emotions: dict

def create_aura_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("perceive", perceive_node)
    graph.add_node("think", think_node)
    graph.add_node("feel", emotion_node)
    graph.add_node("act", action_node)
    graph.add_node("reflect", reflection_node)

    # Add edges
    graph.add_edge("perceive", "think")
    graph.add_edge("think", "feel")
    graph.add_conditional_edges(
        "feel",
        should_act,
        {
            "act": "act",
            "wait": "perceive"
        }
    )
    graph.add_edge("act", "reflect")
    graph.add_edge("reflect", "perceive")

    graph.set_entry_point("perceive")

    return graph.compile()
```

### 7.2 Leon AI

**Open-Source Personal Assistant Framework**

**Current Direction** (2024-2025):
- Moving from classification to hybrid LLM approach
- Local LLM support
- Skills → Actions → Tools → Functions architecture
- Meta-skill for auto-generating new skills

**Resources**:
- Website: https://getleon.ai/
- GitHub: https://github.com/leon-ai/leon

---

## 8. Recommended Architecture for Alive AURA

### Phase 1: Foundation (Weeks 1-4)

```
┌─────────────────────────────────────────────────────────────┐
│                    ALIVE AURA v1.0                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Gateway Daemon                       │   │
│  │  • Event bus (Redis/asyncio)                        │   │
│  │  • Screenpipe integration                           │   │
│  │  • Calendar/email monitors                          │   │
│  │  • Salience filtering                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Global Workspace                       │   │
│  │  • Attention competition                            │   │
│  │  • Module broadcasting                              │   │
│  │  • Attention schema (AST)                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Processing Core                         │   │
│  │  • LLM (Claude/GPT-4/Local)                         │   │
│  │  • MCTS reasoning (already built!)                  │   │
│  │  • Introspection circuit (already built!)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Memory Layer                          │   │
│  │  • A-MEM Zettelkasten (already built!)              │   │
│  │  • Zep/Graphiti temporal KG                         │   │
│  │  • Sleep-time compute                               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Emotional & Social (Weeks 5-8)

```
┌─────────────────────────────────────────────────────────────┐
│                    ALIVE AURA v1.1                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ALMA Emotional System                   │   │
│  │  • Emotion layer (OCC appraisal)                    │   │
│  │  • Mood layer (PAD space)                           │   │
│  │  • Personality layer (Big Five)                     │   │
│  │  • Neuromodulator analogs                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Natural Conversation                      │   │
│  │  • Inner thoughts framework                         │   │
│  │  • Processing indicators                            │   │
│  │  • Barge-in detection                               │   │
│  │  • Backchannel signals                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Screen Awareness (Weeks 9-12)

```
┌─────────────────────────────────────────────────────────────┐
│                    ALIVE AURA v1.2                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │             Screen Understanding                     │   │
│  │  • Screenpipe continuous capture                    │   │
│  │  • Florence-2 / Qwen2.5-VL for OCR                  │   │
│  │  • UI element detection                             │   │
│  │  • Activity inference                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Proactive Assistance                      │   │
│  │  • Active inference (pymdp)                         │   │
│  │  • Anticipate user needs                            │   │
│  │  • Context-aware suggestions                        │   │
│  │  • Non-intrusive presence                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4: Full Integration (Weeks 13-16)

- Unified cognitive loop
- Sleep-time consolidation
- Personality emergence
- Self-improvement capabilities

---

## Key Resources

### Libraries & Frameworks
- **pymdp**: https://github.com/infer-actively/pymdp
- **Zep/Graphiti**: https://github.com/getzep/graphiti
- **Letta**: https://github.com/letta-ai/letta
- **LangGraph**: https://github.com/langchain-ai/langgraph
- **Screenpipe**: https://github.com/mediar-ai/screenpipe
- **Florence-2**: https://huggingface.co/microsoft/Florence-2-large
- **Qwen2.5-VL**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

### Papers
- Friston, K. - Active Inference and Free Energy Principle
- Baars, B. - Global Workspace Theory
- Graziano, M. - Attention Schema Theory
- Gebhard, P. - ALMA: A Layered Model of Affect
- Packer et al. - MemGPT: Towards LLMs as Operating Systems
- Zep Team - Temporal Knowledge Graph Architecture (arXiv:2501.13956)

### Research Projects
- LIDA Cognitive Architecture
- ASTOUND (AST implementation)
- JOCC (OCC implementation)
- Leon AI

---

*Research compiled for the Alive AURA project - February 2025*
