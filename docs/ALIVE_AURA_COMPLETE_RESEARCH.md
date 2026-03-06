# Alive AURA: Complete Technical Research & Implementation Plan

> Master reference document for building a genuinely alive-feeling AI assistant
> Research Date: February 2025
> Project: Apprentice Agent / AURA

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Proactive Architecture](#2-proactive-architecture)
3. [Consciousness-Like Processing](#3-consciousness-like-processing)
4. [Emotional AI Systems](#4-emotional-ai-systems)
5. [Advanced Memory Architecture](#5-advanced-memory-architecture)
6. [Screen Awareness](#6-screen-awareness)
7. [Natural Conversation](#7-natural-conversation)
8. [Implementation Plan](#8-implementation-plan)
9. [File Structure](#9-file-structure)
10. [API Specifications](#10-api-specifications)
11. [Resources & References](#11-resources--references)

---

## 1. Executive Summary

### 1.1 Project Vision

AURA (Autonomous User-Responsive Agent) aims to feel genuinely "alive" rather than being a passive chatbot. This requires:

- **Proactive behavior**: Acting before being asked
- **Emotional depth**: Mood, personality, emotional reactions
- **Continuous awareness**: Understanding user context from screen/environment
- **Natural conversation**: Human-like speech patterns, interruption handling
- **Persistent memory**: Remembering and learning over time

### 1.2 Current AURA Capabilities (Already Built)

| Component | Status | Location |
|-----------|--------|----------|
| MCTS Reasoning Tree | ✅ Complete | `aura/tools/mcts_reasoning.py` |
| Introspection Circuit | ✅ Complete | `aura/tools/introspection_circuit.py` |
| A-MEM Zettelkasten Memory | ✅ Complete | `aura/memory/amem.py` |
| Knowledge Graph | ✅ Complete | `aura/memory/knowledge_graph.py` |
| RAG System | ✅ Complete | `aura/memory/rag.py` |
| Pattern Prophet | ✅ Complete | `aura/patterns/pattern_prophet.py` |
| Heartbeat Monitor | ✅ Complete | `aura/proactive/heartbeat.py` |

### 1.3 Components to Build

| Priority | Component | Complexity | Impact |
|----------|-----------|------------|--------|
| 1 | ALMA Emotional System | Medium | High - Immediate "aliveness" |
| 2 | Gateway Daemon | Medium | High - Proactive foundation |
| 3 | Global Workspace | High | Medium - Unified processing |
| 4 | Sleep-Time Compute | Low | Medium - Background consolidation |
| 5 | Screen Awareness | High | High - Context awareness |
| 6 | Natural Conversation | Medium | Medium - Human-like interaction |

---

## 2. Proactive Architecture

### 2.1 Active Inference & Free Energy Principle

**Theory (Karl Friston)**:
The brain minimizes "surprisal" (prediction error) through action and perception. Agents "want" to reduce uncertainty about their environment.

**Key Library**: `pymdp` (Python Active Inference)
```bash
pip install inferactively-pymdp
```

**Core Implementation Pattern**:
```python
from pymdp.agent import Agent
from pymdp import utils
import numpy as np

class ActiveInferenceProactiveAgent:
    """
    Agent that uses Active Inference for proactive behavior.
    Naturally seeks to reduce uncertainty and achieve preferences.
    """

    def __init__(self, num_states: int, num_observations: int, num_actions: int):
        # A matrix: P(observation | state) - likelihood mapping
        self.A = self._initialize_likelihood(num_observations, num_states)

        # B matrix: P(state' | state, action) - transition dynamics
        self.B = self._initialize_transitions(num_states, num_actions)

        # C vector: Preferred observations (what agent "wants")
        self.C = self._initialize_preferences(num_observations)

        # D vector: Initial state beliefs
        self.D = np.ones(num_states) / num_states

        # Current beliefs about states
        self.beliefs = self.D.copy()

    def infer_states(self, observation: int) -> np.ndarray:
        """Bayesian update: P(s|o) ∝ P(o|s) * P(s)"""
        likelihood = self.A[observation, :]
        posterior = likelihood * self.beliefs
        posterior = posterior / posterior.sum()
        self.beliefs = posterior
        return posterior

    def compute_expected_free_energy(self, action: int) -> float:
        """
        G = pragmatic_value + epistemic_value

        Pragmatic: KL divergence from preferences
        Epistemic: Expected information gain (curiosity)
        """
        # Predict next state
        next_state = self.B[:, :, action] @ self.beliefs

        # Predict observations
        predicted_obs = self.A @ next_state

        # Pragmatic term: achieve preferences
        pragmatic = np.sum(predicted_obs * (np.log(predicted_obs + 1e-8) - self.C))

        # Epistemic term: reduce uncertainty
        epistemic = -np.sum(predicted_obs * np.log(predicted_obs + 1e-8))

        return pragmatic + 0.5 * epistemic

    def select_action(self) -> int:
        """Select action minimizing expected free energy"""
        G_values = []
        for action in range(self.B.shape[2]):
            G = self.compute_expected_free_energy(action)
            G_values.append(G)
        return np.argmin(G_values)

    def should_act_proactively(self) -> tuple[bool, str]:
        """Determine if proactive action is warranted"""
        # High uncertainty triggers proactive information gathering
        entropy = -np.sum(self.beliefs * np.log(self.beliefs + 1e-8))

        if entropy > 1.5:
            return True, "reducing_uncertainty"

        # Compare waiting vs acting
        wait_G = self.compute_expected_free_energy(0)  # Assuming 0 = wait
        best_action_G = min(self.compute_expected_free_energy(a)
                           for a in range(1, self.B.shape[2]))

        if best_action_G < wait_G - 0.1:
            return True, "anticipated_need"

        return False, ""
```

### 2.2 Gateway Daemon Pattern

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                     GATEWAY DAEMON                          │
│                                                             │
│   Event Sources          Event Bus           AI Core        │
│   ┌─────────────┐       ┌─────────┐       ┌─────────────┐  │
│   │ Calendar    │──────▶│         │       │             │  │
│   │ Email       │──────▶│  Redis  │──────▶│ Salience    │  │
│   │ Screen      │──────▶│ Pub/Sub │       │ Filter      │  │
│   │ File System │──────▶│         │       │             │  │
│   └─────────────┘       └─────────┘       └──────┬──────┘  │
│                                                   │         │
│                                           ┌──────▼──────┐  │
│                                           │  Priority   │  │
│                                           │   Queue     │  │
│                                           └──────┬──────┘  │
│                                                   │         │
│                                           ┌──────▼──────┐  │
│                                           │  Active     │  │
│                                           │ Inference   │  │
│                                           │   Agent     │  │
│                                           └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
import asyncio
import redis.asyncio as redis
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
import heapq
import json

class EventPriority(IntEnum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class Event:
    priority: EventPriority
    timestamp: float
    source: str
    event_type: str
    payload: dict

    def __lt__(self, other):
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)

class SalienceFilter:
    """Filter events by computed salience score"""

    def __init__(self):
        self.recency_weight = 0.3
        self.relevance_weight = 0.4
        self.importance_weight = 0.2
        self.novelty_weight = 0.1
        self.seen_events = set()
        self.context_keywords = set()
        self.importance_rules = {
            "urgent_email": 0.9,
            "meeting_reminder": 0.8,
            "calendar_upcoming": 0.6,
            "screen_change": 0.3,
        }

    def compute_salience(self, event: Event) -> float:
        recency = self._recency_score(event.timestamp)
        relevance = self._relevance_score(event.payload)
        importance = self._importance_score(event.event_type)
        novelty = self._novelty_score(event)

        return (self.recency_weight * recency +
                self.relevance_weight * relevance +
                self.importance_weight * importance +
                self.novelty_weight * novelty)

    def _recency_score(self, timestamp: float) -> float:
        age = datetime.now().timestamp() - timestamp
        half_life = 3600  # 1 hour
        import math
        return math.exp(-0.693 * age / half_life)

    def _relevance_score(self, payload: dict) -> float:
        keywords = set(str(payload).lower().split())
        if not self.context_keywords:
            return 0.5
        overlap = len(keywords & self.context_keywords)
        return min(1.0, overlap / max(1, len(self.context_keywords)))

    def _importance_score(self, event_type: str) -> float:
        return self.importance_rules.get(event_type, 0.5)

    def _novelty_score(self, event: Event) -> float:
        event_hash = hash(json.dumps(event.payload, sort_keys=True))
        if event_hash in self.seen_events:
            return 0.1
        self.seen_events.add(event_hash)
        return 1.0

class GatewayDaemon:
    """Main proactive daemon orchestrating all components"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: redis.Redis = None
        self.event_queue: list = []
        self.salience_filter = SalienceFilter()
        self.running = False
        self.salience_threshold = 0.3

    async def start(self):
        self.redis = await redis.from_url(self.redis_url)
        self.running = True

        # Start event collection tasks
        asyncio.create_task(self._subscribe_events())
        asyncio.create_task(self._process_events())

    async def _subscribe_events(self):
        """Subscribe to all event channels"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("calendar", "email", "screen", "file_system")

        async for message in pubsub.listen():
            if message["type"] == "message":
                event_data = json.loads(message["data"])
                event = Event(
                    priority=EventPriority(event_data.get("priority", 3)),
                    timestamp=event_data.get("timestamp", datetime.now().timestamp()),
                    source=message["channel"].decode(),
                    event_type=event_data.get("type", "unknown"),
                    payload=event_data.get("payload", {})
                )

                salience = self.salience_filter.compute_salience(event)
                if salience >= self.salience_threshold:
                    heapq.heappush(self.event_queue, event)

    async def _process_events(self):
        """Process events from priority queue"""
        while self.running:
            if self.event_queue:
                event = heapq.heappop(self.event_queue)
                await self._handle_event(event)
            await asyncio.sleep(0.1)

    async def _handle_event(self, event: Event):
        """Handle a single event"""
        # Route to appropriate handler based on event type
        pass
```

### 2.3 Resources

- **pymdp**: https://github.com/infer-actively/pymdp
- **pymdp Documentation**: https://pymdp-rtd.readthedocs.io/
- **GAIA Proactive Assistant**: https://github.com/theexperiencecompany/gaia
- **Leon AI**: https://github.com/leon-ai/leon

---

## 3. Consciousness-Like Processing

### 3.1 Global Workspace Theory (GWT)

**Theory (Bernard Baars)**:
Consciousness as a "theater" where specialized processors compete for access to a global workspace. Winners are "broadcast" to all other processors.

**Architecture**:
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
│                 │  (Salience +    │                         │
│                 │   Attention)    │                         │
│                 └────────┬────────┘                         │
│                          ▼                                  │
│                 ┌─────────────────┐                         │
│                 │    WORKSPACE    │  ◄── "Conscious"        │
│                 │   (Broadcast)   │      Content            │
│                 └────────┬────────┘                         │
│                          │                                  │
│              ┌───────────┼───────────┐                      │
│              ▼           ▼           ▼                      │
│         [All Modules Receive Broadcast]                     │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable
import asyncio
import time

@dataclass
class Coalition:
    """Information competing for workspace access"""
    content: Any
    source_module: str
    salience: float
    relevance: float
    timestamp: float

    @property
    def activation(self) -> float:
        """Combined activation strength"""
        return 0.6 * self.salience + 0.4 * self.relevance

class SpecializedModule:
    """Base class for processing modules"""

    def __init__(self, name: str):
        self.name = name
        self.input_queue = asyncio.Queue()

    async def process(self, input_data: Any) -> Coalition:
        """Process input and return coalition if relevant"""
        raise NotImplementedError

    async def receive_broadcast(self, content: Any, source: str):
        """Handle broadcast from workspace"""
        await self.input_queue.put((content, source))

class GlobalWorkspace:
    """Central workspace with broadcast mechanism"""

    IGNITION_THRESHOLD = 0.7

    def __init__(self):
        self.modules: Dict[str, SpecializedModule] = {}
        self.current_content: Coalition = None
        self.competition_queue: List[Coalition] = []

    def register_module(self, module: SpecializedModule):
        self.modules[module.name] = module

    async def submit_coalition(self, coalition: Coalition):
        self.competition_queue.append(coalition)

    async def run_competition(self):
        """Winner-take-all competition"""
        if not self.competition_queue:
            return

        # Sort by activation (highest first)
        self.competition_queue.sort(key=lambda c: c.activation, reverse=True)
        winner = self.competition_queue[0]

        # Check for ignition threshold
        if winner.activation >= self.IGNITION_THRESHOLD:
            self.current_content = winner
            await self._broadcast(winner)

        self.competition_queue.clear()

    async def _broadcast(self, coalition: Coalition):
        """Broadcast to all modules"""
        for name, module in self.modules.items():
            if name != coalition.source_module:
                asyncio.create_task(
                    module.receive_broadcast(coalition.content, coalition.source_module)
                )
```

### 3.2 Attention Schema Theory (AST)

**Theory (Michael Graziano)**:
The brain constructs a simplified model of its own attention. This "attention schema" enables meta-cognition and is what we experience as "awareness."

**Implementation**:
```python
import torch
import torch.nn as nn
from typing import Dict, Tuple

class AttentionSchema(nn.Module):
    """
    Model of the system's own attention process.
    Enables "awareness of awareness."
    """

    def __init__(self, hidden_dim: int, schema_dim: int):
        super().__init__()

        # Encode attention state into schema
        self.schema_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, schema_dim),
            nn.ReLU(),
            nn.Linear(schema_dim, schema_dim),
            nn.LayerNorm(schema_dim)
        )

        # Predict future attention
        self.attention_predictor = nn.Sequential(
            nn.Linear(schema_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Control attention based on schema
        self.attention_controller = nn.Sequential(
            nn.Linear(schema_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def encode(self, attention_weights: torch.Tensor,
               attended_content: torch.Tensor) -> torch.Tensor:
        """Create schema representation"""
        combined = torch.cat([
            attention_weights.flatten(),
            attended_content.flatten()
        ])
        return self.schema_encoder(combined)

    def predict(self, schema: torch.Tensor) -> torch.Tensor:
        """Predict where attention should go"""
        return self.attention_predictor(schema)

    def control(self, schema: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Modulate attention using schema"""
        combined = torch.cat([schema, query], dim=-1)
        gate = self.attention_controller(combined)
        return query * gate

    def introspect(self, schema: torch.Tensor) -> Dict[str, Any]:
        """Report on current attention state"""
        return {
            "focus_strength": schema.norm().item(),
            "schema_state": schema.detach().numpy()
        }
```

### 3.3 Higher-Order Thought (HOT) / SOFAI Architecture

**Dual-Process System**:
```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class MetaState:
    """Higher-order representation"""
    content: Any
    confidence: float
    source: str
    reflection_depth: int

class MetacognitiveController:
    """
    SOFAI-style metacognitive controller.
    Selects between fast (System 1) and slow (System 2) processing.
    """

    def __init__(self, fast_solver, slow_solver, confidence_threshold: float = 0.7):
        self.fast_solver = fast_solver
        self.slow_solver = slow_solver
        self.confidence_threshold = confidence_threshold

    async def solve(self, problem: Any) -> Dict[str, Any]:
        # Try fast solver first
        fast_result, fast_conf = await self.fast_solver(problem)

        if fast_conf >= self.confidence_threshold:
            return {
                "result": fast_result,
                "confidence": fast_conf,
                "solver": "fast"
            }

        # Fall back to slow deliberation
        slow_result, slow_conf = await self.slow_solver(problem)

        return {
            "result": slow_result,
            "confidence": slow_conf,
            "solver": "slow"
        }
```

### 3.4 Resources

- **LIDA Cognitive Architecture**: https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture)
- **AST Foundation Paper**: https://www.frontiersin.org/articles/10.3389/frobt.2017.00060/full
- **ASTOUND Project**: https://blogs.upm.es/astound/
- **PyPhi (IIT)**: https://github.com/wmayner/pyphi
- **SOFAI**: https://github.com/ai4society/sofai_tool

---

## 4. Emotional AI Systems

### 4.1 ALMA Model (A Layered Model of Affect)

**Three Layers**:

| Layer | Timescale | Model | Description |
|-------|-----------|-------|-------------|
| Emotion | Seconds-minutes | OCC (22 types) | Reactive responses to events |
| Mood | Hours-days | PAD Space (8 octants) | Background affective state |
| Personality | Stable | Big Five (OCEAN) | Individual differences |

**Layer Interactions**:
```
Personality ──────► Default Mood Baseline
     │
     ▼
   Mood ◄─────────── Emotions push/pull
     │         ▲
     │         │
     ▼         │
Emotion ◄───── Events/Appraisals
Thresholds
```

**BigFive to PAD Transformation Matrix**:
```python
# From Gebhard (2005)
OCEAN_TO_PAD = np.array([
    [0.00, 0.00, 0.21, 0.59, 0.19],   # Pleasure
    [0.15, 0.00, 0.00, 0.30, -0.57],  # Arousal
    [0.25, 0.17, 0.60, -0.32, 0.00]   # Dominance
])

# Usage: PAD = OCEAN_TO_PAD @ personality_vector
# personality_vector = [O, C, E, A, N]
```

**Complete ALMA Implementation**:
```python
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time
import math

class MoodType(Enum):
    EXUBERANT = "exuberant"      # +P, +A, +D
    DEPENDENT = "dependent"      # +P, +A, -D
    RELAXED = "relaxed"          # +P, -A, +D
    DOCILE = "docile"            # +P, -A, -D
    HOSTILE = "hostile"          # -P, +A, +D
    ANXIOUS = "anxious"          # -P, +A, -D
    DISDAINFUL = "disdainful"    # -P, -A, +D
    BORED = "bored"              # -P, -A, -D

@dataclass
class PADState:
    """Pleasure-Arousal-Dominance emotional state"""
    pleasure: float = 0.0    # -1 to 1
    arousal: float = 0.0     # -1 to 1
    dominance: float = 0.0   # -1 to 1

    def __post_init__(self):
        self.pleasure = np.clip(self.pleasure, -1, 1)
        self.arousal = np.clip(self.arousal, -1, 1)
        self.dominance = np.clip(self.dominance, -1, 1)

    def to_array(self) -> np.ndarray:
        return np.array([self.pleasure, self.arousal, self.dominance])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PADState':
        return cls(pleasure=arr[0], arousal=arr[1], dominance=arr[2])

    def distance_to(self, other: 'PADState') -> float:
        return np.linalg.norm(self.to_array() - other.to_array())

    def get_mood_type(self) -> MoodType:
        """Classify into 8 octants"""
        p = self.pleasure > 0
        a = self.arousal > 0
        d = self.dominance > 0

        mood_map = {
            (True, True, True): MoodType.EXUBERANT,
            (True, True, False): MoodType.DEPENDENT,
            (True, False, True): MoodType.RELAXED,
            (True, False, False): MoodType.DOCILE,
            (False, True, True): MoodType.HOSTILE,
            (False, True, False): MoodType.ANXIOUS,
            (False, False, True): MoodType.DISDAINFUL,
            (False, False, False): MoodType.BORED,
        }
        return mood_map[(p, a, d)]

@dataclass
class BigFivePersonality:
    """OCEAN personality traits"""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([
            self.openness, self.conscientiousness, self.extraversion,
            self.agreeableness, self.neuroticism
        ])

    def to_default_mood(self) -> PADState:
        """Transform to default PAD mood"""
        OCEAN_TO_PAD = np.array([
            [0.00, 0.00, 0.21, 0.59, 0.19],
            [0.15, 0.00, 0.00, 0.30, -0.57],
            [0.25, 0.17, 0.60, -0.32, 0.00]
        ])
        pad = OCEAN_TO_PAD @ self.to_array()
        return PADState.from_array(np.clip(pad, -1, 1))

class ALMAEmotionEngine:
    """Complete ALMA implementation"""

    # Emotion PAD coordinates (empirically derived)
    EMOTION_PAD = {
        'joy':         PADState(0.76, 0.48, 0.35),
        'sadness':     PADState(-0.63, -0.27, -0.33),
        'anger':       PADState(-0.51, 0.59, 0.25),
        'fear':        PADState(-0.64, 0.60, -0.43),
        'disgust':     PADState(-0.60, 0.35, 0.11),
        'surprise':    PADState(0.40, 0.67, -0.13),
        'hope':        PADState(0.51, 0.23, 0.14),
        'anxiety':     PADState(-0.51, 0.50, -0.30),
        'boredom':     PADState(-0.30, -0.50, -0.10),
        'excitement':  PADState(0.62, 0.75, 0.38),
        'calm':        PADState(0.40, -0.60, 0.25),
        'frustration': PADState(-0.40, 0.40, -0.20),
        'pride':       PADState(0.55, 0.30, 0.65),
        'shame':       PADState(-0.52, 0.10, -0.55),
        'guilt':       PADState(-0.57, 0.20, -0.35),
        'gratitude':   PADState(0.65, 0.20, -0.15),
        'love':        PADState(0.70, 0.35, 0.10),
        'hate':        PADState(-0.60, 0.45, 0.35),
        'contempt':    PADState(-0.45, 0.10, 0.55),
        'interest':    PADState(0.62, 0.35, 0.20),
        'neutral':     PADState(0.0, 0.0, 0.0),
    }

    def __init__(self, personality: BigFivePersonality):
        self.personality = personality
        self.default_mood = personality.to_default_mood()
        self.current_mood = PADState.from_array(self.default_mood.to_array().copy())
        self.active_emotions: List[Dict] = []

        # Temporal parameters
        self.emotion_decay_rate = 0.1
        self.mood_pull_strength = 0.05
        self.emotion_to_mood_transfer = 0.3

        self.last_update = time.time()

    def trigger_emotion(self, emotion_name: str, intensity: float = 1.0):
        """Trigger an emotional response"""
        if emotion_name not in self.EMOTION_PAD:
            return

        emotion_pad = self.EMOTION_PAD[emotion_name]
        intensity = np.clip(intensity, 0, 1)

        self.active_emotions.append({
            'name': emotion_name,
            'pad': emotion_pad,
            'intensity': intensity,
            'timestamp': time.time()
        })

        # Push mood toward emotion
        self._push_mood(emotion_pad, intensity)

    def _push_mood(self, emotion_pad: PADState, intensity: float):
        """Push mood toward triggered emotion"""
        current = self.current_mood.to_array()
        target = emotion_pad.to_array()
        push_strength = intensity * self.emotion_to_mood_transfer
        new_mood = current + push_strength * (target - current)
        self.current_mood = PADState.from_array(np.clip(new_mood, -1, 1))

    def update(self, dt: float = None):
        """Update emotional state over time"""
        now = time.time()
        if dt is None:
            dt = now - self.last_update
        self.last_update = now

        # Decay active emotions
        still_active = []
        for emotion in self.active_emotions:
            emotion['intensity'] *= math.exp(-self.emotion_decay_rate * dt)
            if emotion['intensity'] > 0.01:
                still_active.append(emotion)
        self.active_emotions = still_active

        # Pull mood toward baseline
        current = self.current_mood.to_array()
        baseline = self.default_mood.to_array()
        pull_factor = 1 - math.exp(-self.mood_pull_strength * dt)
        new_mood = current + pull_factor * (baseline - current)
        self.current_mood = PADState.from_array(np.clip(new_mood, -1, 1))

    def get_dominant_emotion(self) -> Optional[str]:
        """Get currently dominant emotion"""
        if not self.active_emotions:
            return None
        return max(self.active_emotions, key=lambda e: e['intensity'])['name']

    def identify_emotion_from_pad(self) -> tuple[str, float]:
        """Identify closest emotion to current PAD state"""
        min_dist = float('inf')
        closest = 'neutral'

        for name, pad in self.EMOTION_PAD.items():
            dist = self.current_mood.distance_to(pad)
            if dist < min_dist:
                min_dist = dist
                closest = name

        confidence = max(0, 1 - min_dist / 1.73)
        return closest, confidence

    def get_response_modulation(self) -> Dict[str, float]:
        """Get parameters for modulating AI responses"""
        p = self.current_mood.pleasure
        a = self.current_mood.arousal
        d = self.current_mood.dominance

        return {
            'verbosity': 0.5 + 0.3 * a,
            'formality': 0.5 - 0.2 * p,
            'enthusiasm': 0.5 + 0.3 * p + 0.2 * a,
            'assertiveness': 0.5 + 0.3 * d,
            'warmth': 0.5 + 0.3 * p,
            'response_speed': 0.5 + 0.3 * a,
        }

    def get_state(self) -> Dict:
        """Get complete emotional state"""
        emotion, confidence = self.identify_emotion_from_pad()
        return {
            'emotion': emotion,
            'confidence': confidence,
            'mood_type': self.current_mood.get_mood_type().value,
            'pad': {
                'pleasure': self.current_mood.pleasure,
                'arousal': self.current_mood.arousal,
                'dominance': self.current_mood.dominance
            },
            'active_emotions': [
                {'name': e['name'], 'intensity': e['intensity']}
                for e in self.active_emotions
            ],
            'personality': {
                'openness': self.personality.openness,
                'conscientiousness': self.personality.conscientiousness,
                'extraversion': self.personality.extraversion,
                'agreeableness': self.personality.agreeableness,
                'neuroticism': self.personality.neuroticism
            }
        }
```

### 4.2 OCC Model (22 Emotion Types)

**Appraisal Variables**:
```python
@dataclass
class AppraisalVariables:
    # Event-related
    desirability: float = 0.0           # -1 to 1
    desirability_for_other: float = 0.0
    liking_for_other: float = 0.0       # -1 to 1

    # Prospect-related
    likelihood: float = 0.5             # 0 to 1
    effort: float = 0.0                 # 0 to 1
    realization: float = 0.0            # 0 to 1

    # Agent-related
    praiseworthiness: float = 0.0       # -1 to 1
    is_self_agent: bool = True

    # Object-related
    appealingness: float = 0.0          # -1 to 1

    # Modifiers
    arousal: float = 0.5
    unexpectedness: float = 0.0
```

**OCC Emotion Categories**:
```python
OCC_EMOTIONS = {
    # Well-being (self-focused events)
    'joy': {'condition': 'desirability > 0.1'},
    'distress': {'condition': 'desirability < -0.1'},

    # Fortune-of-others
    'happy_for': {'condition': 'des_other > 0 and liking > 0'},
    'resentment': {'condition': 'des_other > 0 and liking < 0'},
    'gloating': {'condition': 'des_other < 0 and liking < 0'},
    'pity': {'condition': 'des_other < 0 and liking > 0'},

    # Prospect-based
    'hope': {'condition': 'desirability > 0 and 0 < likelihood < 1'},
    'fear': {'condition': 'desirability < 0 and 0 < likelihood < 1'},
    'satisfaction': {'condition': 'hope confirmed'},
    'relief': {'condition': 'fear disconfirmed'},
    'fears_confirmed': {'condition': 'fear confirmed'},
    'disappointment': {'condition': 'hope disconfirmed'},

    # Attribution (agent actions)
    'pride': {'condition': 'praiseworthiness > 0 and is_self'},
    'shame': {'condition': 'praiseworthiness < 0 and is_self'},
    'admiration': {'condition': 'praiseworthiness > 0 and not is_self'},
    'reproach': {'condition': 'praiseworthiness < 0 and not is_self'},

    # Compound
    'gratitude': {'condition': 'admiration + joy'},
    'anger': {'condition': 'reproach + distress'},
    'gratification': {'condition': 'pride + joy'},
    'remorse': {'condition': 'shame + distress'},

    # Attraction
    'love': {'condition': 'appealingness > 0'},
    'hate': {'condition': 'appealingness < 0'},
}
```

### 4.3 Neuromodulator System

**Lovheim Cube Mapping**:
```python
LOVHEIM_CUBE = {
    # (Serotonin, Dopamine, Norepinephrine)
    'shame':      (0.0, 0.0, 0.0),
    'distress':   (0.0, 0.0, 1.0),
    'fear':       (0.0, 1.0, 0.0),
    'anger':      (0.0, 1.0, 1.0),
    'contempt':   (1.0, 0.0, 0.0),
    'surprise':   (1.0, 0.0, 1.0),
    'joy':        (1.0, 1.0, 0.0),
    'excitement': (1.0, 1.0, 1.0),
}
```

**Implementation**:
```python
@dataclass
class NeuromodulatorState:
    dopamine: float = 0.5       # Reward, motivation
    serotonin: float = 0.5      # Mood stability
    norepinephrine: float = 0.5 # Alertness
    oxytocin: float = 0.5       # Social bonding

class NeuromodulatorSystem:
    STIMULUS_EFFECTS = {
        'reward': {'dopamine': 0.3, 'serotonin': 0.1},
        'threat': {'norepinephrine': 0.4, 'dopamine': 0.2},
        'social_positive': {'oxytocin': 0.3, 'serotonin': 0.15},
        'social_negative': {'oxytocin': -0.2, 'serotonin': -0.1},
        'novelty': {'norepinephrine': 0.25, 'dopamine': 0.15},
        'success': {'dopamine': 0.35, 'serotonin': 0.2},
        'failure': {'dopamine': -0.2, 'serotonin': -0.15},
    }

    DECAY_RATES = {
        'dopamine': 0.1,
        'serotonin': 0.02,
        'norepinephrine': 0.15,
        'oxytocin': 0.05
    }

    def apply_stimulus(self, stimulus_type: str, intensity: float = 1.0):
        effects = self.STIMULUS_EFFECTS.get(stimulus_type, {})
        for modulator, change in effects.items():
            current = getattr(self.state, modulator)
            new_value = np.clip(current + change * intensity, 0, 1)
            setattr(self.state, modulator, new_value)

    def get_behavioral_modifiers(self) -> Dict[str, float]:
        return {
            'motivation': self.state.dopamine,
            'impulse_control': self.state.serotonin,
            'alertness': self.state.norepinephrine,
            'trust': self.state.oxytocin,
            'risk_tolerance': 0.3 + 0.4 * self.state.dopamine,
            'patience': self.state.serotonin,
            'focus': min(self.state.norepinephrine,
                        1 - abs(self.state.norepinephrine - 0.6)),
            'empathy': self.state.oxytocin,
        }
```

### 4.4 Resources

- **ALMA Paper**: https://www.semanticscholar.org/paper/ALMA:-a-layered-model-of-affect-Gebhard/
- **ALMA Official**: http://alma.dfki.de/
- **PAD Model**: https://en.wikipedia.org/wiki/PAD_emotional_state_model
- **OCC Model**: https://people.idsia.ch/~steunebrink/Publications/KI09_OCC_revisited.pdf
- **Lovheim Cube**: https://en.wikipedia.org/wiki/L%C3%B6vheim_Cube_of_Emotions

---

## 5. Advanced Memory Architecture

### 5.1 Zep Temporal Knowledge Graph

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
│              │                          │                 │
│              │   Bi-temporal model:     │                 │
│              │   T  = event time        │                 │
│              │   T' = ingestion time    │                 │
│              └──────────────────────────┘                 │
│                             │                              │
│              ┌──────────────┴───────────┐                 │
│              │   Hybrid Retrieval       │                 │
│              │   • Cosine similarity    │                 │
│              │   • BM25 keyword         │                 │
│              │   • Graph traversal      │                 │
│              │   • Reciprocal Rank Fusion│                │
│              └──────────────────────────┘                 │
└────────────────────────────────────────────────────────────┘
```

**Performance**: 94.8% on Deep Memory Retrieval benchmark (vs MemGPT 93.4%)

**Integration**:
```python
from graphiti_core import Graphiti
from graphiti_core.edges import EpisodeType
from datetime import datetime

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    llm_client=OpenAIClient(model="gpt-4o-mini")
)

# Add episode
await graphiti.add_episode(
    name="Conversation",
    episode_body="User mentioned moving to Seattle next month",
    episode_type=EpisodeType.text,
    reference_time=datetime.now(),
    source_description="chat"
)

# Search with temporal reasoning
results = await graphiti.search(
    query="What events does the user have coming up?",
    num_results=5
)
```

### 5.2 Sleep-Time Compute

**Concept**: AI "thinks" during idle time to consolidate memory.

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                   SLEEP-TIME COMPUTE                    │
│                                                         │
│   Online Agent (fast)        Sleeper Agent (heavy)     │
│   ┌─────────────────┐       ┌─────────────────┐        │
│   │ Handle queries  │       │ Process context │        │
│   │ Use pre-computed│◄──────│ Anticipate Qs   │        │
│   │ Low latency     │       │ Consolidate     │        │
│   └─────────────────┘       └─────────────────┘        │
│                                     │                   │
│                              ┌──────▼──────┐           │
│                              │ Pre-computed│           │
│                              │   Cache     │           │
│                              └─────────────┘           │
└─────────────────────────────────────────────────────────┘
```

**Benefits**:
- 5x reduction in test-time compute
- Up to 18% accuracy improvement
- 2.5x cost reduction per query

**Implementation Pattern**:
```python
class SleepTimeCompute:
    async def sleep_cycle(self):
        """Execute during idle time"""
        # Phase 1: Light sleep - surface processing
        await self._process_recent_interactions()

        # Phase 2: Deep sleep - consolidation
        await self._consolidate_memories()

        # Phase 3: REM - pattern extraction
        await self._extract_insights()

    async def _consolidate_memories(self):
        """Merge related memories"""
        clusters = await self.memory.find_similar_clusters(threshold=0.85)
        for cluster in clusters:
            consolidated = await self.llm.summarize_cluster(cluster)
            await self.memory.replace_cluster(cluster, consolidated)

    async def _extract_insights(self):
        """Find patterns across memories"""
        facts = await self.memory.get_all_facts()
        insights = await self.llm.generate(
            f"Analyze and extract patterns: {facts}"
        )
        await self.memory.add_insights(insights)
```

### 5.3 Letta/MemGPT Architecture

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
│   │  │• User info  │  │   • Current task        │  │  │
│   │  └─────────────┘  └─────────────────────────┘  │  │
│   └─────────────────────────────────────────────────┘  │
│                          ▲                              │
│                          │ self-editing tools           │
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

### 5.4 Memory Retrieval with Forgetting Curve

```python
class ForgettingCurveMemory:
    """Ebbinghaus forgetting curve: R = e^(-t/S)"""

    def get_retention(self, memory_id: str) -> float:
        memory = self.memories[memory_id]
        time_hours = (datetime.now() - memory.last_accessed).total_seconds() / 3600
        return math.exp(-time_hours / (memory.stability * 24)) * memory.importance

    def retrieve_with_decay(self, query_embedding, top_k: int = 10):
        results = []
        for memory_id, memory in self.memories.items():
            retention = self.get_retention(memory_id)
            if retention < 0.1:
                continue
            similarity = cosine_similarity(query_embedding, memory.embedding)
            score = similarity * retention
            results.append({'id': memory_id, 'score': score})
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
```

### 5.5 Resources

- **Zep/Graphiti**: https://github.com/getzep/graphiti
- **Zep Paper**: https://arxiv.org/abs/2501.13956
- **Letta/MemGPT**: https://github.com/letta-ai/letta
- **Sleep-Time Compute**: https://www.letta.com/blog/sleep-time-compute
- **Mem0**: https://github.com/mem0ai/mem0

---

## 6. Screen Awareness

### 6.1 Screenpipe

**Open-source 24/7 screen & audio capture**

**Features**:
- Privacy-first: 100% local processing
- Cross-platform: Windows, macOS, Linux
- OCR text extraction
- Audio transcription
- REST API for queries

**API Integration**:
```python
import requests

class ScreenpipeClient:
    def __init__(self, base_url: str = "http://localhost:3030"):
        self.base_url = base_url

    def search(self, query: str = None, content_type: str = None,
               app_name: str = None, limit: int = 10) -> dict:
        params = {"limit": limit}
        if query:
            params["q"] = query
        if content_type:
            params["content_type"] = content_type
        if app_name:
            params["app_name"] = app_name

        response = requests.get(f"{self.base_url}/search", params=params)
        return response.json()

    def get_recent_activity(self, minutes: int = 5) -> dict:
        from datetime import datetime, timedelta
        end = datetime.utcnow()
        start = end - timedelta(minutes=minutes)
        return self.search(
            start_time=start.isoformat() + "Z",
            end_time=end.isoformat() + "Z",
            limit=50
        )
```

**GitHub**: https://github.com/mediar-ai/screenpipe

### 6.2 Florence-2 (Microsoft)

**Lightweight vision model for OCR**

**Sizes**: 0.23B, 0.77B parameters

**Capabilities**:
- OCR with region detection
- Object detection
- Image captioning
- Visual grounding

**Usage**:
```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large")

def extract_screen_text(image: Image.Image) -> dict:
    inputs = processor(text="<OCR_WITH_REGION>", images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1024)
    return processor.decode(outputs[0], skip_special_tokens=True)
```

**HuggingFace**: https://huggingface.co/microsoft/Florence-2-large

### 6.3 Qwen2.5-VL

**Advanced vision-language model**

**Sizes**: 3B, 7B, 72B parameters

**Capabilities**:
- Computer use / Phone use
- GUI grounding (find elements)
- Video understanding (1+ hour)
- Document analysis

**Usage for UI Understanding**:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def find_ui_element(screenshot, description: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": screenshot},
            {"type": "text", "text": f"Find the {description} and return bounding box"}
        ]
    }]
    inputs = processor(messages, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(outputs[0], skip_special_tokens=True)
```

**HuggingFace**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

### 6.4 Resources

- **Screenpipe**: https://github.com/mediar-ai/screenpipe
- **Florence-2**: https://huggingface.co/microsoft/Florence-2-large
- **Qwen2.5-VL**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **OmniParser**: https://github.com/microsoft/OmniParser

---

## 7. Natural Conversation

### 7.1 Inner Thoughts Framework

**Dual-Track Processing**:
```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│         Inner Processing                │
│   "Hmm, they're asking about X..."      │
│   "Let me think about this..."          │
│   [NOT shown to user]                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│      Processing Indicators              │
│   "Let me think about that..."          │
│   [Shown to user as "thinking"]         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         Outer Response                  │
│   [Final response to user]              │
└─────────────────────────────────────────┘
```

**MIRROR Architecture (2025)**:
- **Talker (System 1)**: Immediate response generation
- **Thinker (System 2)**: Asynchronous background reasoning
- Inner Monologue Manager maintains separate conversation history
- 21% improvement on safety benchmarks

### 7.2 Barge-In Detection

**Requirements**:
- Sub-100ms response time
- Voice Activity Detection (VAD)
- Acoustic Echo Cancellation (AEC)
- Context synchronization after interruption

**State Machine**:
```
States: LISTENING, PROCESSING, SPEAKING, INTERRUPTED

LISTENING:
  - Monitor VAD
  - Check turn-completion signals
  - Threshold crossing → PROCESSING

PROCESSING:
  - Generate response
  - Show thinking indicator
  - Response ready → SPEAKING

SPEAKING:
  - Output TTS with timestamps
  - Monitor for barge-in
  - Barge-in → INTERRUPTED
  - Complete → LISTENING

INTERRUPTED:
  - Stop TTS immediately
  - Sync context to last heard word
  - → LISTENING
```

### 7.3 Full-Duplex Voice: Moshi

**First real-time full-duplex spoken LLM**

**Specs**:
- 160ms theoretical latency, 200ms practical
- Inner monologue predicts text tokens as prefix to audio
- 7B-parameter Temporal Transformer
- Trained on 7M hours of audio

**GitHub**: https://github.com/kyutai-labs/moshi

### 7.4 Voice Agent Frameworks

**Pipecat** (Daily):
- Voice-first with STT, TTS, conversation handling
- Ultra-low latency via WebSockets/WebRTC
- Python framework

**GitHub**: https://github.com/pipecat-ai/pipecat

**LiveKit Agents**:
- Open-source WebRTC infrastructure
- Semantic turn detection
- Telephony integration

**GitHub**: https://github.com/livekit/agents

### 7.5 Natural Speech Patterns

```python
class NaturalSpeechPatterns:
    HESITATIONS = ["um", "uh", "hmm", "well"]
    FILLERS = ["you know", "like", "I mean"]

    def add_disfluency(self, text: str, fluency: float = 0.9) -> str:
        if fluency >= 1.0:
            return text

        words = text.split()
        result = []
        for i, word in enumerate(words):
            if random.random() > fluency and i > 0:
                result.append(random.choice(self.HESITATIONS) + ",")
            result.append(word)
        return " ".join(result)
```

### 7.6 Resources

- **MIRROR Paper**: https://arxiv.org/html/2506.00430v1
- **Moshi**: https://github.com/kyutai-labs/moshi
- **Pipecat**: https://github.com/pipecat-ai/pipecat
- **LiveKit**: https://github.com/livekit/agents
- **Deepgram Flux**: https://deepgram.com/learn/introducing-flux

---

## 8. Implementation Plan

### 8.1 Phase 1: Emotional Foundation (Weeks 1-4)

**Goal**: Give AURA emotional depth and personality

**Files to Create**:
```
aura/
├── emotions/
│   ├── __init__.py
│   ├── alma_engine.py       # Main ALMA implementation
│   ├── pad_space.py         # PAD emotional space
│   ├── occ_appraisal.py     # OCC emotion appraisal
│   ├── neuromodulators.py   # Neuromodulator simulation
│   └── personality.py       # Big Five personality
```

**API Endpoints**:
```
POST /api/emotions/trigger     # Trigger emotion
GET  /api/emotions/state       # Get current state
PUT  /api/emotions/personality # Update personality
GET  /api/emotions/modulation  # Get response modulation
```

**Frontend Components**:
```
web/src/components/
├── EmotionPanel.tsx          # Emotion visualization
├── MoodIndicator.tsx         # PAD space display
└── PersonalityEditor.tsx     # Personality configuration
```

**Integration Points**:
- Agent chat flow uses emotional modulation
- Memory stores emotional context
- UI shows current mood state

### 8.2 Phase 2: Proactive System (Weeks 5-8)

**Goal**: Enable AURA to act proactively

**Files to Create**:
```
aura/
├── proactive/
│   ├── __init__.py
│   ├── gateway_daemon.py     # Main daemon
│   ├── event_bus.py          # Redis pub/sub
│   ├── salience_filter.py    # Event filtering
│   ├── active_inference.py   # pymdp integration
│   └── monitors/
│       ├── calendar.py
│       ├── email.py
│       └── screen.py
```

**Services**:
- Redis for event bus
- Background daemon process
- Event source monitors

### 8.3 Phase 3: Consciousness Framework (Weeks 9-12)

**Goal**: Unified cognitive processing

**Files to Create**:
```
aura/
├── consciousness/
│   ├── __init__.py
│   ├── global_workspace.py   # GWT implementation
│   ├── attention_schema.py   # AST implementation
│   ├── metacognition.py      # HOT/SOFAI
│   └── cognitive_cycle.py    # Main processing loop
```

### 8.4 Phase 4: Screen Awareness (Weeks 13-16)

**Goal**: Understand user's screen context

**Files to Create**:
```
aura/
├── screen/
│   ├── __init__.py
│   ├── screenpipe_client.py  # Screenpipe integration
│   ├── vision_models.py      # Florence-2/Qwen2.5-VL
│   ├── ui_understanding.py   # UI element detection
│   └── context_inference.py  # Activity inference
```

**Dependencies**:
- Screenpipe running locally
- Florence-2 or Qwen2.5-VL model

### 8.5 Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 2 | ALMA Core | Working emotion engine |
| 4 | Emotional UI | Visualization + API |
| 6 | Gateway Daemon | Event collection working |
| 8 | Proactive Actions | AURA initiates interactions |
| 10 | Global Workspace | Unified processing |
| 12 | Attention Schema | Meta-cognition |
| 14 | Screen Capture | Screenpipe integrated |
| 16 | Full Integration | All systems connected |

---

## 9. File Structure

### 9.1 Proposed Directory Structure

```
aura/
├── __init__.py
├── agent.py                    # Main agent (existing)
├── config.py                   # Configuration
│
├── emotions/                   # NEW: Emotional AI
│   ├── __init__.py
│   ├── alma_engine.py          # ALMA 3-layer model
│   ├── pad_space.py            # PAD emotional space
│   ├── occ_appraisal.py        # OCC model
│   ├── neuromodulators.py      # Neuromodulator system
│   ├── personality.py          # Big Five
│   └── emotional_memory.py     # Emotion-tagged memory
│
├── consciousness/              # NEW: Consciousness framework
│   ├── __init__.py
│   ├── global_workspace.py     # GWT
│   ├── attention_schema.py     # AST
│   ├── metacognition.py        # HOT/SOFAI
│   └── cognitive_cycle.py      # Main loop
│
├── proactive/                  # Enhanced proactive
│   ├── __init__.py
│   ├── heartbeat.py            # (existing)
│   ├── gateway_daemon.py       # NEW
│   ├── event_bus.py            # NEW
│   ├── salience_filter.py      # NEW
│   ├── active_inference.py     # NEW
│   └── monitors/               # NEW
│       ├── __init__.py
│       ├── calendar.py
│       ├── email.py
│       └── screen.py
│
├── screen/                     # NEW: Screen awareness
│   ├── __init__.py
│   ├── screenpipe_client.py
│   ├── vision_models.py
│   ├── ui_understanding.py
│   └── context_inference.py
│
├── conversation/               # NEW: Natural conversation
│   ├── __init__.py
│   ├── inner_thoughts.py
│   ├── turn_taking.py
│   ├── speech_patterns.py
│   └── barge_in.py
│
├── memory/                     # Enhanced memory
│   ├── __init__.py
│   ├── amem.py                 # (existing)
│   ├── knowledge_graph.py      # (existing)
│   ├── rag.py                  # (existing)
│   ├── temporal_kg.py          # NEW: Zep/Graphiti
│   ├── sleep_compute.py        # NEW
│   └── forgetting_curve.py     # NEW
│
├── tools/                      # Existing tools
│   ├── __init__.py
│   ├── mcts_reasoning.py       # (existing)
│   ├── introspection_circuit.py # (existing)
│   └── ...
│
└── patterns/                   # (existing)
    ├── __init__.py
    └── pattern_prophet.py
```

### 9.2 API Routes Structure

```
api/routes/
├── __init__.py
├── agent.py                    # (existing)
├── memory.py                   # (existing)
├── reasoning_tree.py           # (existing)
├── introspection.py            # (existing)
│
├── emotions.py                 # NEW
├── consciousness.py            # NEW
├── proactive.py                # NEW
├── screen.py                   # NEW
└── conversation.py             # NEW
```

### 9.3 Frontend Components

```
web/src/components/
├── ChatPanel.tsx               # (existing)
├── MemoryPanel.tsx             # (existing)
├── ReasoningTreePanel.tsx      # (existing)
├── IntrospectionPanel.tsx      # (existing)
│
├── EmotionPanel.tsx            # NEW: Emotion visualization
├── MoodIndicator.tsx           # NEW: Current mood display
├── PersonalityEditor.tsx       # NEW: Personality config
├── ConsciousnessPanel.tsx      # NEW: Workspace visualization
├── ProactivePanel.tsx          # NEW: Event monitoring
└── ScreenContextPanel.tsx      # NEW: Screen awareness
```

---

## 10. API Specifications

### 10.1 Emotions API

```yaml
# POST /api/emotions/trigger
request:
  emotion: string              # e.g., "joy", "fear"
  intensity: float             # 0.0 to 1.0
  source: string               # e.g., "user_message", "event"
response:
  success: boolean
  state: EmotionalState

# GET /api/emotions/state
response:
  emotion: string              # Current dominant emotion
  confidence: float
  mood_type: string            # e.g., "exuberant", "anxious"
  pad:
    pleasure: float
    arousal: float
    dominance: float
  active_emotions:
    - name: string
      intensity: float
  neuromodulators:
    dopamine: float
    serotonin: float
    norepinephrine: float
    oxytocin: float

# PUT /api/emotions/personality
request:
  openness: float              # 0.0 to 1.0
  conscientiousness: float
  extraversion: float
  agreeableness: float
  neuroticism: float
response:
  success: boolean
  default_mood: PADState

# GET /api/emotions/modulation
response:
  verbosity: float
  formality: float
  enthusiasm: float
  assertiveness: float
  warmth: float
  response_speed: float
```

### 10.2 Proactive API

```yaml
# GET /api/proactive/events
query:
  limit: int
  priority_min: int
response:
  events:
    - id: string
      type: string
      source: string
      priority: int
      salience: float
      timestamp: string
      payload: object

# POST /api/proactive/context
request:
  keywords: string[]
response:
  success: boolean

# GET /api/proactive/status
response:
  daemon_running: boolean
  event_queue_size: int
  active_monitors: string[]
  last_proactive_action: string
```

### 10.3 Consciousness API

```yaml
# GET /api/consciousness/workspace
response:
  current_content:
    source: string
    activation: float
    content: object
  competing_coalitions:
    - source: string
      activation: float
  attention_focus: string
  meta_state:
    confidence: float
    reflection_depth: int

# POST /api/consciousness/focus
request:
  target: string
  priority: float
response:
  success: boolean
  new_focus: object
```

---

## 11. Resources & References

### 11.1 Core Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| pymdp | Active Inference | `pip install inferactively-pymdp` |
| graphiti-core | Temporal KG | `pip install graphiti-core` |
| letta | MemGPT memory | `pip install letta` |
| transformers | Vision models | `pip install transformers` |

### 11.2 Key Papers

1. **Active Inference**: Friston, K. - Free Energy Principle
2. **Global Workspace**: Baars, B. - Theater of Consciousness
3. **Attention Schema**: Graziano, M. - AST Foundation Paper
4. **ALMA**: Gebhard, P. - A Layered Model of Affect
5. **OCC**: Ortony, Clore, Collins - Cognitive Structure of Emotions
6. **Zep**: arXiv:2501.13956 - Temporal Knowledge Graph
7. **MemGPT**: Packer et al. - LLMs as Operating Systems

### 11.3 GitHub Repositories

- pymdp: https://github.com/infer-actively/pymdp
- Graphiti: https://github.com/getzep/graphiti
- Letta: https://github.com/letta-ai/letta
- Screenpipe: https://github.com/mediar-ai/screenpipe
- Moshi: https://github.com/kyutai-labs/moshi
- Pipecat: https://github.com/pipecat-ai/pipecat
- LiveKit: https://github.com/livekit/agents
- GAIA: https://github.com/theexperiencecompany/gaia
- Leon AI: https://github.com/leon-ai/leon

### 11.4 Documentation

- pymdp: https://pymdp-rtd.readthedocs.io/
- Zep: https://www.getzep.com/
- Letta: https://docs.letta.com/
- Screenpipe: https://docs.screenpi.pe/
- Florence-2: https://huggingface.co/microsoft/Florence-2-large
- Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

---

## Appendix A: Quick Start Commands

```bash
# Install core dependencies
pip install inferactively-pymdp graphiti-core transformers torch

# Install optional dependencies
pip install redis asyncio-redis letta

# Start Redis (for event bus)
docker run -d -p 6379:6379 redis:latest

# Start Screenpipe (for screen awareness)
# Download from https://github.com/mediar-ai/screenpipe/releases
screenpipe start

# Download vision models
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; AutoProcessor.from_pretrained('microsoft/Florence-2-large'); AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large')"
```

---

## Appendix B: AURA Personality Presets

```python
PERSONALITY_PRESETS = {
    "default": BigFivePersonality(
        openness=0.7,
        conscientiousness=0.6,
        extraversion=0.65,
        agreeableness=0.75,
        neuroticism=0.35
    ),
    "helpful_assistant": BigFivePersonality(
        openness=0.6,
        conscientiousness=0.8,
        extraversion=0.5,
        agreeableness=0.9,
        neuroticism=0.2
    ),
    "creative_partner": BigFivePersonality(
        openness=0.95,
        conscientiousness=0.5,
        extraversion=0.7,
        agreeableness=0.7,
        neuroticism=0.4
    ),
    "analytical_expert": BigFivePersonality(
        openness=0.6,
        conscientiousness=0.9,
        extraversion=0.3,
        agreeableness=0.6,
        neuroticism=0.25
    ),
}
```

---

*Document Version: 1.0*
*Last Updated: February 2025*
*Project: Apprentice Agent / Alive AURA*
