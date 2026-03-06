# AURA Real-Time 3D Neural Visualization: Research & Architecture Document

**Date:** 2026-02-10
**Subject:** Making AURA's cognition visible -- turning a GitHub repo into a shareable visual experience
**Codebase:** `C:\Users\asus\apprentice-agent`

---

## Table of Contents

1. [Technology Stack Research](#1-technology-stack-research)
2. [What to Visualize: Mapping AURA to Visual Elements](#2-what-to-visualize)
3. [Architecture Design](#3-architecture-design)
4. [WebSocket Event Protocol](#4-websocket-event-protocol)
5. [React Component Skeletons](#5-react-component-skeletons)
6. [Python Integration](#6-python-integration)
7. [Implementation Phases](#7-implementation-phases)

---

## 1. Technology Stack Research

### 1.1 React Three Fiber (R3F) -- Core 3D Framework

React Three Fiber is a React renderer for Three.js that allows you to build 3D scenes declaratively using React components. It is the clear choice for this project because AURA's frontend is already React-based.

**Key libraries:**

| Package | Purpose | URL |
|---------|---------|-----|
| `@react-three/fiber` | React renderer for Three.js | https://github.com/pmndrs/react-three-fiber |
| `@react-three/drei` | Helpers (OrbitControls, Text, Html, Billboard, etc.) | https://github.com/pmndrs/drei |
| `@react-three/postprocessing` | Bloom, god rays, selective glow | https://github.com/pmndrs/react-postprocessing |
| `r3f-forcegraph` | Force-directed graph as R3F component | https://github.com/vasturiano/r3f-forcegraph |
| `d3-force-3d` | 3D force simulation engine | https://github.com/vasturiano/d3-force-3d |
| `three-forcegraph` | Three.js force-directed graph object | https://github.com/vasturiano/three-forcegraph |
| `leva` | Live GUI controls for tuning parameters | https://github.com/pmndrs/leva |
| `zustand` | Lightweight state management | https://github.com/pmndrs/zustand |

**Best practices (2025-2026):**
- Mutate in `useFrame`, never in React state. Use `useRef` for all per-frame updates to avoid React re-renders on every animation frame.
- Reuse objects with `useMemo` -- never allocate `new THREE.Vector3()` inside a render loop.
- Use InstancedMesh for repeating geometry (nodes, particles). A single `InstancedMesh` with 10,000 instances = 1 draw call.
- Bloom is selective by default when you set `luminanceThreshold={1}` on the Bloom effect -- only materials with `emissiveIntensity > 1` glow. No need for the heavier `SelectiveBloom`.
- WebGPU renderer is now supported across all major browsers (Safari 26 shipped Sept 2025). Three.js `WebGPURenderer` falls back to WebGL 2 automatically.

**References:**
- [R3F Documentation](https://r3f.docs.pmnd.rs/)
- [R3F Examples Gallery](https://r3f.docs.pmnd.rs/getting-started/examples)
- [100 Three.js Best Practices (2026)](https://www.utsubo.com/blog/threejs-best-practices-100-tips)
- [R3F vs Three.js in 2026](https://graffersid.com/react-three-fiber-vs-three-js/)
- [Scaling Performance Guide](https://r3f.docs.pmnd.rs/advanced/scaling-performance)

### 1.2 Shader Techniques for Neural Visualization

**Glow / Pulse / Heat Effects:**

The core visual language for neural activity is glow + pulse. Use emissive materials combined with Bloom post-processing.

```glsl
// Vertex shader for pulsing glow on nodes
uniform float uTime;
uniform float uActivation; // 0.0 = dormant, 1.0 = fully active
varying float vGlow;

void main() {
    float pulse = sin(uTime * 3.0 + position.x * 2.0) * 0.5 + 0.5;
    vGlow = uActivation * (0.6 + 0.4 * pulse);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
```

```glsl
// Fragment shader for node glow
uniform vec3 uColor;
uniform float uActivation;
varying float vGlow;

void main() {
    float glow = vGlow * 1.5; // Push above 1.0 for bloom pickup
    vec3 color = uColor * glow;
    gl_FragColor = vec4(color, 1.0);
}
```

**Flow Particles along Edges (Neuromodulator Streams):**

Use GPGPU compute to move thousands of particles along edge paths. Each particle has a position stored in a Float32 texture, updated in a compute shader that reads the edge path and advances the particle along it.

```glsl
// GPGPU fragment shader for particle flow along bezier curves
uniform sampler2D uPositions;    // Previous positions
uniform sampler2D uEdgePaths;    // Control points for curves
uniform float uSpeed;
uniform float uTime;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec4 pos = texture2D(uPositions, uv);

    float t = fract(pos.w + uSpeed * 0.01); // Advance along curve
    // Cubic bezier evaluation
    vec3 p0 = texture2D(uEdgePaths, vec2(pos.z, 0.0)).xyz;
    vec3 p1 = texture2D(uEdgePaths, vec2(pos.z, 0.33)).xyz;
    vec3 p2 = texture2D(uEdgePaths, vec2(pos.z, 0.66)).xyz;
    vec3 p3 = texture2D(uEdgePaths, vec2(pos.z, 1.0)).xyz;

    vec3 newPos = pow(1.0-t,3.0)*p0 + 3.0*pow(1.0-t,2.0)*t*p1
                + 3.0*(1.0-t)*t*t*p2 + t*t*t*p3;

    gl_FragColor = vec4(newPos, t);
}
```

**Heat Map Shader (for confidence gradients):**

```glsl
uniform float uConfidence; // 0.0 to 1.0
varying vec3 vPosition;

vec3 heatmap(float t) {
    // Blue -> Cyan -> Green -> Yellow -> Red
    vec3 cold = vec3(0.0, 0.2, 1.0);
    vec3 warm = vec3(0.0, 1.0, 0.3);
    vec3 hot  = vec3(1.0, 0.8, 0.0);
    vec3 fire = vec3(1.0, 0.1, 0.0);
    if (t < 0.33) return mix(cold, warm, t / 0.33);
    if (t < 0.66) return mix(warm, hot, (t - 0.33) / 0.33);
    return mix(hot, fire, (t - 0.66) / 0.34);
}

void main() {
    vec3 color = heatmap(uConfidence);
    float glow = uConfidence > 0.7 ? uConfidence * 2.0 : uConfidence;
    gl_FragColor = vec4(color * glow, 1.0);
}
```

**References:**
- [Shader Glow (Three.js)](https://stemkoski.github.io/Three.js/Shader-Glow.html)
- [ShaderParticleEngine](https://github.com/squarefeet/ShaderParticleEngine)
- [GPGPU Dreamy Particle Effect (Codrops)](https://tympanus.net/codrops/2024/12/19/crafting-a-dreamy-particle-effect-with-three-js-and-gpgpu/)
- [GPGPU Flow Field Particles (Three.js Journey)](https://threejs-journey.com/lessons/gpgpu-flow-field-particles-shaders)
- [TSL & WebGPU GPGPU Particles (Wawa Sensei)](https://wawasensei.dev/courses/react-three-fiber/lessons/tsl-gpgpu)
- [Galaxy Simulation with WebGPU Compute Shaders](https://threejsroadmap.com/blog/galaxy-simulation-webgpu-compute-shaders)
- [Bloom Effect (React Postprocessing)](https://react-postprocessing.docs.pmnd.rs/effects/bloom)

### 1.3 Force-Directed Graph Layouts in 3D

The knowledge graph and MCTS tree both need spatial layout algorithms.

**`r3f-forcegraph`** is a native React Three Fiber component that wraps `three-forcegraph` and uses `d3-force-3d` or `ngraph` for physics simulation.

```jsx
import { ForceGraph } from 'r3f-forcegraph';
import { Canvas, useFrame } from '@react-three/fiber';
import { useRef } from 'react';

function KnowledgeGraphScene({ graphData }) {
    const fgRef = useRef();
    useFrame(() => fgRef.current?.tickFrame());

    return (
        <ForceGraph
            ref={fgRef}
            graphData={graphData}
            nodeRelSize={4}
            nodeColor={node => node.type === 'concept' ? '#4fc3f7' : '#ff8a65'}
            linkDirectionalParticles={2}
            linkDirectionalParticleSpeed={0.005}
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
        />
    );
}
```

**`d3-force-3d`** extends d3-force to three dimensions. Key forces:
- `forceCenter()` -- centers the graph
- `forceManyBody()` -- charge repulsion between nodes
- `forceLink()` -- spring force on edges
- `forceCollide()` -- prevents overlap

**`ngraph`** alternative: ngraph.forcelayout3d is faster for large graphs (10,000+ nodes) but less configurable.

**References:**
- [r3f-forcegraph](https://github.com/vasturiano/r3f-forcegraph)
- [3d-force-graph](https://github.com/vasturiano/3d-force-graph)
- [d3-force-3d](https://github.com/vasturiano/d3-force-3d)
- [three-forcegraph](https://github.com/vasturiano/three-forcegraph)
- [react-force-graph](https://github.com/vasturiano/react-force-graph)

### 1.4 WebSocket / SSE Real-Time Streaming

AURA already has a WebSocket endpoint at `/api/chat/stream` in `api/routes/chat.py` using FastAPI. The visualization system should use a **second WebSocket** endpoint dedicated to visualization events.

**Why WebSocket over SSE:**
- Bidirectional: the frontend can send camera position, focus requests, and filter commands back to the server.
- AURA already uses WebSocket infrastructure.
- Lower latency than SSE for high-frequency events (targeting 10-30 events/second during active reasoning).

**FastAPI pattern:**

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

@router.websocket("/api/viz/stream")
async def viz_stream(websocket: WebSocket):
    await websocket.accept()
    viz_queue = asyncio.Queue()
    VizEventBus.subscribe(viz_queue)
    try:
        while True:
            event = await viz_queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        VizEventBus.unsubscribe(viz_queue)
```

**References:**
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [Real-time data streaming using FastAPI and WebSockets](https://stribny.name/blog/2020/07/real-time-data-streaming-using-fastapi-and-websockets/)
- [Real-Time Features in FastAPI](https://python.plainenglish.io/real-time-features-in-fastapi-websockets-event-streaming-and-push-notifications-fec79a0a6812)

### 1.5 Existing Neural / Brain Visualization Projects

| Project | Description | URL |
|---------|-------------|-----|
| **TensorSpace** | 3D neural network visualization in browsers using Three.js | https://github.com/tensorspace-team/tensorspace |
| **Neural Networks in 3D** | Interactive MLP visualization | https://arogozhnikov.github.io/3d_nn/ |
| **TensorFlow Playground** | Classic interactive NN visualization | https://playground.tensorflow.org/ |
| **BrainViz** | Interactive 3D brain in browser | https://github.com/matanmazor/BrainViz |
| **ReasonGraph** | Visualization of LLM reasoning paths | https://arxiv.org/html/2503.03979v1 |
| **Landscape of Thoughts** | t-SNE trajectories of reasoning methods | https://arxiv.org/abs/2503.22165 |
| **Interactive Reasoning (UIST 2025)** | Chain-of-thought tree visualization with editing | https://arxiv.org/html/2506.23678v1 |

### 1.6 Key Academic References

- **Interactive Reasoning (UIST 2025):** Transforms chain-of-thought outputs into interactive tree representations where users can edit reasoning steps and provide feedback. Directly relevant to visualizing MCTS and CognitiveTheater.
- **Landscape of Thoughts (2025):** Uses t-SNE to project reasoning trajectories into 2D. Could be adapted to 3D for showing how different reasoning methods explore the solution space.
- **ReasonGraph (2025):** Provides graph-based visualization of reasoning paths, making scoring mechanisms and path selection visually clear.
- **Tree of Thoughts (NeurIPS 2023):** The algorithmic foundation behind AURA's MCTS, where thoughts form a tree and search algorithms explore it. Visual representation is natural as a literal tree.

---

## 2. What to Visualize: Mapping AURA to Visual Elements

### 2.1 Architecture-to-Visual Mapping

Based on reading the codebase, here is the complete mapping from AURA subsystems to visual elements:

#### CognitiveTheater (Multi-Perspective Reasoning)
**Source:** `aura/tools/cognitive_theater.py`
**Data structures:** `Deliberation` dataclass with `perspectives: Dict[str, str]` (advocate, critic, analyst, integrator) and `synthesis`, `confidence`

| Element | Visual Representation |
|---------|----------------------|
| Advocate actor | Blue sphere node, pulsing when generating |
| Critic actor | Red sphere node |
| Analyst actor | Yellow sphere node |
| Integrator actor | Green sphere node |
| Dialogue between actors | Curved edges with flowing text particles |
| Synthesis | Central bright node where edges converge |
| Confidence | Size and glow intensity of synthesis node |

**Layout:** Four nodes arranged in a square around a central synthesis point. Edges light up sequentially as each perspective generates. Text snippets flow along edges as particle-text sprites.

#### MCTS Tree Search
**Source:** `aura/tools/mcts_reasoning.py`
**Data structures:** `MCTSNode` with `thought: Thought`, `children: List[MCTSNode]`, `visits`, `value`, `avg_value`, `state: NodeState`, `depth`. Callbacks: `on_node_created`, `on_node_evaluated`, `on_iteration_complete`.

| Element | Visual Representation |
|---------|----------------------|
| Root node | Large sphere at top |
| Child nodes | Smaller spheres branching downward |
| UCB1 score | Node brightness (exploitation) + halo size (exploration) |
| Visit count | Node size (more visits = larger) |
| avg_value | Node color on heatmap scale (red=bad to green=good) |
| NodeState.EXPLORING | Pulsing animation |
| NodeState.PRUNED | Fading to transparent |
| NodeState.TERMINAL + successful | Gold glow |
| Best path | Highlighted edge chain with particle flow |
| Reflection | Orange annotation nodes attached to failed branches |

**Layout:** Top-down tree layout. The `depth` field maps to Y-axis. Branching spreads on X/Z axes. The `branching_factor` (default 5) keeps the tree manageable.

#### Knowledge Graph
**Source:** `aura/tools/knowledge_graph.py`
**Data structures:** NetworkX graph. `Node` dataclass with `id`, `type`, `label`, `confidence`, `access_count`, `valid_from`, `valid_to`. 16 node types (concept, entity, person, project, tool, event, emotion, skill, etc.). 16 edge types (relates_to, causes, solves, etc.).

| Element | Visual Representation |
|---------|----------------------|
| Node types | Different geometry/color per type (concept=icosahedron/blue, entity=box/orange, person=sphere/pink, etc.) |
| Node confidence | Opacity (low confidence = translucent) |
| Node access_count | Size scaling |
| Edge types | Line color coding |
| Edge weight/strength | Line thickness |
| Bi-temporal validity | Nodes with `valid_to` set appear desaturated/ghostly |
| Ebbinghaus decay | Node glow intensity fades over time |
| Active query path | Highlighted edges with flowing particles |

**Layout:** 3D force-directed using `r3f-forcegraph` with `d3-force-3d`. Cluster by node type using custom force.

#### Episodic Memory
**Source:** `aura_episodic_memory/` (Qdrant-based)
**Data:** Episodes with temporal context, importance, emotional valence, decay curves

| Element | Visual Representation |
|---------|----------------------|
| Episode node | Sphere with timeline position |
| Memory retrieval | Node briefly flares bright with expanding ring |
| Temporal distance | Horizontal position on time axis |
| Importance | Vertical position (more important = higher) |
| Emotional valence | Color (warm=positive, cool=negative) |
| Decay/forgetting | Gradual transparency reduction over time |
| Consolidation (NeuroDream) | Pulsing connection lines during sleep |

**Layout:** Timeline arrangement with importance on Y-axis. A cylindrical "memory corridor" the camera can fly through.

#### Emotional Neuromodulators (ALMA Engine)
**Source:** `aura/emotion/alma_engine.py`
**Data structures:** `PADState` (Pleasure, Arousal, Dominance each -1 to +1). Neuromodulators: dopamine, serotonin, norepinephrine, oxytocin (each 0.0 to 1.0). 24 emotion types, 8 mood types.

| Element | Visual Representation |
|---------|----------------------|
| PAD state | A glowing orb positioned in 3D PAD space |
| Dopamine level | Orange/gold particle stream density |
| Serotonin level | Blue/teal ambient glow intensity |
| Norepinephrine level | Red/white electrical spark frequency |
| Oxytocin level | Pink/warm haze radius |
| Emotion triggers | Burst particle effects from source direction |
| Mood (slow drift) | Background gradient color shift |
| Personality (Big Five) | Subtle geometry deformation of the brain shape |

**Layout:** A translucent "brain" volume with neuromodulator streams flowing through it. The PAD position could be visualized as a floating indicator within this volume.

#### Truth Spine Verification
**Source:** `aura/truth_spine.py`
**Data structures:** `Artifact` (types: FILE, STDOUT, JSON, NONE) with `content_hash`, `is_valid`. `MemoryTier` (FACT, BELIEF, SPECULATION). `VerificationCheck` subclasses (FileExistsCheck, HashMatchCheck, etc.). Contract: ACTION -> ARTIFACT -> VERIFICATION -> MEMORY_TIER.

| Element | Visual Representation |
|---------|----------------------|
| Verification chain | Horizontal spine of linked checkpoints |
| FACT tier | Green checkpoint, solid |
| BELIEF tier | Yellow checkpoint, semi-transparent |
| SPECULATION tier | Red checkpoint, dashed outline |
| Artifact | Small icon (file, terminal, JSON bracket) at checkpoint |
| Verification pass | Green pulse traveling along spine |
| Verification fail | Red X flash at failed checkpoint |
| Content hash | Tiny hex text label |

**Layout:** A horizontal chain (like a DNA backbone) running along the bottom of the scene. Each verification step is a vertebra.

#### Multi-Agent Orchestrator
**Source:** `aura/multi_agent/orchestrator.py`
**Data structures:** Specialists: ResearchAgent, CoderAgent, AnalystAgent, CreativeAgent. `CollaborationMode`: SINGLE, SEQUENTIAL, PARALLEL, DEBATE. `AgentMessage` with sender, recipient, context.

| Element | Visual Representation |
|---------|----------------------|
| Each specialist agent | Distinct colored brain-lobe region |
| ResearchAgent | Blue lobe (left) |
| CoderAgent | Green lobe (right) |
| AnalystAgent | Yellow lobe (front) |
| CreativeAgent | Purple lobe (back) |
| Orchestrator routing | Beam from center to active agent(s) |
| PARALLEL mode | Multiple beams simultaneously |
| SEQUENTIAL mode | Beam moves from one to next |
| DEBATE mode | Bidirectional beams between debating agents |
| Message passing | Particle streams between lobes |

**Layout:** Brain-hemisphere layout with four distinct regions around a central orchestrator point.

#### Global Workspace (Consciousness)
**Source:** `aura/consciousness/global_workspace.py`
**Data structures:** `WorkspaceContent` with `source_module`, `content_type`, `summary`, `activation`, `salience`, `effective_activation`. `ConsciousState` with `broadcast_content`, `secondary_content`, `attention_focus`, `attention_intensity`. 8 codelets competing for broadcast. Cycle: ~300ms.

| Element | Visual Representation |
|---------|----------------------|
| Workspace center | Central glowing sphere (the "spotlight of consciousness") |
| 8 codelets | Ring of smaller spheres around center |
| Activation level | Codelet sphere brightness |
| Competition | Height of codelet (rising = competing) |
| Broadcast winner | Beam from winning codelet to center, center flares |
| Secondary content | Dimmer connections to center |
| Attention focus label | 3D text at center |
| Cycle tick | Rhythmic pulse at ~300ms |

**Layout:** A ring of 8 codelet nodes surrounding a central broadcast node. This is the "core" of the brain visualization.

#### Inner Monologue
**Source:** `aura/consciousness/inner_thoughts_engine.py`, `api/routes/thinking.py`

| Element | Visual Representation |
|---------|----------------------|
| Current thought | Scrolling 3D text ribbon above the brain |
| Thought type | Color coding (CONNECTING=blue, RECALLING=purple, ANALYZING=yellow, etc.) |
| Thought intensity | Text size and glow |

**Layout:** A curved text ribbon floating above the main brain scene, new thoughts scroll in from the right and fade out to the left.

#### Strategy/Bandit Selection
**Source:** The MCTS `exploration_weight` (UCB1) and tool selection routing in `brain.py`

| Element | Visual Representation |
|---------|----------------------|
| Strategy options | Vertical bar chart overlay |
| Selection probability | Bar height |
| Currently selected | Highlighted bar with glow |
| Exploration vs exploitation | Bar split coloring (blue=explore, orange=exploit) |

**Layout:** Small 2D overlay panel in the corner, or 3D bars arranged in a semicircle.

---

## 3. Architecture Design

### 3.1 System Architecture Overview

```
  AURA Python Backend                    3D Visualization Frontend
 +-----------------------+              +---------------------------+
 |                       |   WebSocket  |                           |
 |  ApprenticeAgent      |   /api/viz/  |   React Three Fiber App   |
 |  +-----------------+  |   stream     |   +---------------------+ |
 |  | VizEventBus     |--+----------->--+-->| useVizWebSocket()   | |
 |  +-----------------+  |              |   +---------------------+ |
 |    ^  ^  ^  ^  ^      |              |       |                   |
 |    |  |  |  |  |      |              |   +---v-----------------+ |
 |  [hooks into:]        |              |   | Zustand Store       | |
 |  - brain.py           |              |   | (all live state)    | |
 |  - mcts_reasoning.py  |              |   +---+-----------------+ |
 |  - cognitive_theater   |              |       |                   |
 |  - alma_engine.py     |              |   +---v-----------------+ |
 |  - truth_spine.py     |              |   | <BrainScene />      | |
 |  - knowledge_graph.py |              |   |  +- <GlobalWorkspace>| |
 |  - global_workspace   |              |   |  +- <KnowledgeGraph> | |
 |  - multi_agent/       |              |   |  +- <MCTSTree />     | |
 |  - episodic_memory    |              |   |  +- <Emotions />     | |
 |  - inner_thoughts     |              |   |  +- <TruthSpine />   | |
 +-----------------------+              |   |  +- <MultiAgent />   | |
                                        |   |  +- <Particles />    | |
                                        |   +---------------------+ |
                                        +---------------------------+
```

### 3.2 Backend: Python WebSocket Server

**Location:** `api/routes/viz_stream.py` (new file)

The VizEventBus is a singleton asyncio pub/sub system. Each AURA subsystem calls `VizEventBus.emit(event)` at key processing points. The WebSocket endpoint subscribes to this bus and streams events to connected frontends.

```python
import asyncio
import json
import time
import logging
from typing import Any, Dict, List, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/viz", tags=["visualization"])


class VizEventBus:
    """Singleton event bus for visualization events.

    Thread-safe: emit() can be called from any thread.
    Subscribers are asyncio.Queues consumed in the WebSocket handler.
    """
    _instance = None
    _lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None

    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop = None
        self._buffer: List[Dict] = []  # Ring buffer for late joiners
        self._buffer_max = 100

    @classmethod
    def get(cls) -> 'VizEventBus':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def subscribe(self, queue: asyncio.Queue):
        self._subscribers.add(queue)
        # Send buffered recent events to new subscriber
        for event in self._buffer:
            queue.put_nowait(event)

    def unsubscribe(self, queue: asyncio.Queue):
        self._subscribers.discard(queue)

    def emit(self, event_type: str, data: Dict[str, Any]):
        """Thread-safe event emission. Can be called from sync code."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }

        # Add to ring buffer
        self._buffer.append(event)
        if len(self._buffer) > self._buffer_max:
            self._buffer = self._buffer[-self._buffer_max:]

        # Dispatch to subscribers
        for queue in list(self._subscribers):
            try:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(queue.put_nowait, event)
                else:
                    queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop events if consumer is too slow
            except Exception:
                pass


@router.websocket("/stream")
async def viz_websocket(websocket: WebSocket):
    """Stream visualization events to the 3D frontend."""
    await websocket.accept()
    bus = VizEventBus.get()
    bus.set_loop(asyncio.get_event_loop())
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    bus.subscribe(queue)
    logger.info("[VizStream] Client connected")

    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("[VizStream] Client disconnected")
    except Exception as e:
        logger.error(f"[VizStream] Error: {e}")
    finally:
        bus.unsubscribe(queue)
```

### 3.3 Frontend: React Three Fiber App

**Component Hierarchy:**

```
<App>
  <VizWebSocketProvider url="ws://localhost:8000/api/viz/stream">
    <Canvas>
      <SceneSetup />           {/* Lights, environment, fog */}
      <CameraController />     {/* OrbitControls + preset viewpoints */}
      <PostProcessing />       {/* Bloom, vignette */}

      <BrainScene>
        <GlobalWorkspaceViz />  {/* Central consciousness ring */}
        <KnowledgeGraphViz />   {/* Force-directed graph region */}
        <MCTSTreeViz />         {/* Reasoning tree region */}
        <CognitiveTheaterViz /> {/* Perspective actors */}
        <EmotionViz />          {/* Neuromodulator flows */}
        <TruthSpineViz />       {/* Verification chain */}
        <MultiAgentViz />       {/* Agent brain regions */}
        <EpisodicMemoryViz />   {/* Memory timeline */}
        <ParticleFlowSystem />  {/* Global particle system */}
      </BrainScene>

      <InnerMonologueOverlay /> {/* 3D text ribbon */}
    </Canvas>

    <HUD>
      <StrategyBanditPanel />   {/* 2D overlay: probability bars */}
      <SystemStatsPanel />      {/* FPS, event rate, node count */}
      <ViewSwitcher />          {/* Camera preset buttons */}
      <TimelineControls />      {/* Playback speed, pause */}
    </HUD>
  </VizWebSocketProvider>
</App>
```

### 3.4 State Management with Zustand

```typescript
import { create } from 'zustand';

interface VizState {
    // Global Workspace
    consciousState: ConsciousState | null;
    codelets: CodeletState[];

    // Knowledge Graph
    kgNodes: KGNode[];
    kgEdges: KGEdge[];
    activeKGPath: string[];  // highlighted node ids

    // MCTS
    mctsRoot: MCTSNodeViz | null;
    mctsActiveNode: string | null;  // currently exploring
    mctsBestPath: string[];

    // CognitiveTheater
    theaterState: TheaterState | null;
    activePerspective: string | null;

    // Emotions
    padState: { pleasure: number; arousal: number; dominance: number };
    neuromodulators: {
        dopamine: number;
        serotonin: number;
        norepinephrine: number;
        oxytocin: number;
    };
    emotionTriggers: EmotionTrigger[];

    // Truth Spine
    verificationChain: VerificationStep[];

    // Multi-Agent
    activeAgents: string[];
    collaborationMode: string;
    agentMessages: AgentMessageViz[];

    // Episodic Memory
    activeMemories: MemoryNode[];
    memoryFlashes: MemoryFlash[];  // brief highlights on retrieval

    // Inner Monologue
    thoughts: ThoughtBubble[];

    // Actions
    handleVizEvent: (event: VizEvent) => void;
}

export const useVizStore = create<VizState>((set, get) => ({
    // ... initial state ...

    handleVizEvent: (event: VizEvent) => {
        switch (event.type) {
            case 'consciousness.broadcast':
                set({ consciousState: event.data });
                break;
            case 'mcts.node_created':
                // Insert into tree
                break;
            case 'mcts.node_evaluated':
                // Update node color/size
                break;
            case 'emotion.update':
                set({
                    padState: event.data.pad,
                    neuromodulators: event.data.neuromodulators,
                });
                break;
            case 'kg.node_activated':
                set(s => ({
                    activeKGPath: [...s.activeKGPath, event.data.node_id]
                }));
                break;
            case 'truth.verification_step':
                set(s => ({
                    verificationChain: [...s.verificationChain, event.data]
                }));
                break;
            case 'theater.perspective':
                set({
                    activePerspective: event.data.perspective,
                    theaterState: event.data,
                });
                break;
            case 'agent.message':
                set(s => ({
                    agentMessages: [...s.agentMessages.slice(-50), event.data]
                }));
                break;
            case 'memory.recall':
                set(s => ({
                    memoryFlashes: [...s.memoryFlashes, {
                        ...event.data,
                        flashStart: Date.now(),
                    }]
                }));
                break;
            case 'thought.inner':
                set(s => ({
                    thoughts: [...s.thoughts.slice(-20), event.data]
                }));
                break;
            // ... more event handlers
        }
    },
}));
```

### 3.5 Camera Controls and Viewpoints

```typescript
const CAMERA_PRESETS = {
    overview: {
        position: [0, 30, 50],
        target: [0, 0, 0],
        fov: 60,
        label: 'Full Brain Overview',
    },
    consciousness: {
        position: [0, 10, 15],
        target: [0, 5, 0],
        fov: 45,
        label: 'Global Workspace (Close)',
    },
    reasoning: {
        position: [-20, 15, 10],
        target: [-15, 0, 0],
        fov: 50,
        label: 'MCTS Reasoning Tree',
    },
    knowledge: {
        position: [20, 10, 15],
        target: [15, 0, 0],
        fov: 55,
        label: 'Knowledge Graph',
    },
    emotions: {
        position: [0, -5, 20],
        target: [0, -5, 0],
        fov: 40,
        label: 'Emotional Core',
    },
    memory: {
        position: [0, 0, 30],
        target: [0, 0, -20],
        fov: 50,
        label: 'Memory Timeline',
    },
    truth: {
        position: [0, -10, 15],
        target: [0, -12, 0],
        fov: 45,
        label: 'Truth Spine',
    },
};
```

### 3.6 Performance Considerations

| Concern | Strategy | Limit |
|---------|----------|-------|
| Node count | InstancedMesh for all graph nodes (1 draw call) | 5,000 nodes max |
| Edge rendering | InstancedBufferGeometry for lines | 10,000 edges max |
| Particle count | GPGPU compute shader for positions | 50,000 particles |
| Event rate | Throttle VizEventBus to max 30 events/sec | Configurable |
| LOD (Level of Detail) | Reduce node detail at distance; hide labels > 50 units away | Automatic |
| Graph physics | Run d3-force in Web Worker, update positions to main thread | 60fps target |
| Memory | Ring buffers for events; prune old nodes from scene | 1,000 events |
| Textures | Use shared atlas for node type icons | 1 texture |
| Post-processing | Single Bloom pass with `luminanceThreshold=1` | Selective |
| React re-renders | All animation in `useFrame`; state updates batched | Never per-frame |

---

## 4. WebSocket Event Protocol

### 4.1 Base Event Envelope

```json
{
    "type": "string (event_type)",
    "timestamp": 1707580800.123,
    "session_id": "string (optional, for reasoning sessions)",
    "data": { }
}
```

### 4.2 Consciousness Events

```json
{
    "type": "consciousness.broadcast",
    "timestamp": 1707580800.123,
    "data": {
        "winner": {
            "source_module": "alma_emotion",
            "content_type": "emotion_shift",
            "summary": "Feeling curious about the user's question",
            "activation": 0.85,
            "salience": 0.72,
            "effective_activation": 0.798,
            "pad_signature": { "pleasure": 0.3, "arousal": 0.6, "dominance": 0.1 }
        },
        "runners_up": [
            {
                "source_module": "episodic_memory",
                "content_type": "relevant_memory",
                "summary": "Recalled similar conversation from yesterday",
                "activation": 0.65,
                "salience": 0.80
            }
        ],
        "attention_focus": "user_query",
        "attention_intensity": 0.85,
        "cycle_number": 1247
    }
}
```

### 4.3 MCTS Reasoning Events

```json
// mcts.search_started
{
    "type": "mcts.search_started",
    "timestamp": 1707580800.123,
    "session_id": "mcts_abc123",
    "data": {
        "problem": "What is the best approach to optimize this database?",
        "config": {
            "max_iterations": 30,
            "max_depth": 10,
            "branching_factor": 5,
            "exploration_weight": 1.414
        }
    }
}

// mcts.node_created
{
    "type": "mcts.node_created",
    "timestamp": 1707580800.200,
    "session_id": "mcts_abc123",
    "data": {
        "node_id": "a1b2c3d4",
        "parent_id": "root_0000",
        "depth": 1,
        "thought_type": "reasoning",
        "thought_content": "Consider indexing strategy for frequent queries",
        "state": "pending"
    }
}

// mcts.node_evaluated
{
    "type": "mcts.node_evaluated",
    "timestamp": 1707580800.500,
    "session_id": "mcts_abc123",
    "data": {
        "node_id": "a1b2c3d4",
        "reward": 0.72,
        "visits": 1,
        "avg_value": 0.72,
        "ucb1": 2.31,
        "state": "evaluated",
        "is_terminal": false
    }
}

// mcts.iteration_complete
{
    "type": "mcts.iteration_complete",
    "timestamp": 1707580801.0,
    "session_id": "mcts_abc123",
    "data": {
        "iteration": 5,
        "total_nodes": 23,
        "best_path_ids": ["root_0000", "a1b2c3d4", "e5f6g7h8"],
        "best_value": 0.85
    }
}

// mcts.search_complete
{
    "type": "mcts.search_complete",
    "timestamp": 1707580810.0,
    "session_id": "mcts_abc123",
    "data": {
        "success": true,
        "best_answer": "Use composite index on (user_id, created_at)...",
        "confidence": 0.87,
        "iterations": 28,
        "nodes_explored": 97,
        "time_taken": 9.877
    }
}
```

### 4.4 CognitiveTheater Events

```json
// theater.deliberation_started
{
    "type": "theater.deliberation_started",
    "timestamp": 1707580800.0,
    "data": {
        "question": "Should we use PostgreSQL or MongoDB for this project?",
        "perspectives": ["advocate", "critic", "analyst", "integrator"]
    }
}

// theater.perspective_generated
{
    "type": "theater.perspective_generated",
    "timestamp": 1707580802.0,
    "data": {
        "perspective": "advocate",
        "content": "PostgreSQL offers ACID compliance, robust joins, and mature ecosystem...",
        "sequence": 1
    }
}

// theater.synthesis
{
    "type": "theater.synthesis",
    "timestamp": 1707580808.0,
    "data": {
        "synthesis": "Given the relational data model and need for transactions...",
        "confidence": 0.85,
        "all_perspectives": {
            "advocate": "...",
            "critic": "...",
            "analyst": "...",
            "integrator": "..."
        }
    }
}
```

### 4.5 Emotion Events

```json
// emotion.update
{
    "type": "emotion.update",
    "timestamp": 1707580800.0,
    "data": {
        "pad": { "pleasure": 0.3, "arousal": 0.6, "dominance": 0.2 },
        "neuromodulators": {
            "dopamine": 0.72,
            "serotonin": 0.55,
            "norepinephrine": 0.68,
            "oxytocin": 0.45
        },
        "dominant_emotion": "curious",
        "mood": "engaged",
        "trigger": "user_asked_interesting_question"
    }
}

// emotion.trigger
{
    "type": "emotion.trigger",
    "timestamp": 1707580800.5,
    "data": {
        "emotion": "satisfaction",
        "intensity": 0.7,
        "pad_delta": { "pleasure": 0.2, "arousal": 0.1, "dominance": 0.05 },
        "cause": "task_completed_successfully"
    }
}
```

### 4.6 Knowledge Graph Events

```json
// kg.node_activated
{
    "type": "kg.node_activated",
    "timestamp": 1707580800.0,
    "data": {
        "node_id": "concept_python_debugging",
        "node_type": "concept",
        "label": "Python Debugging",
        "confidence": 0.92,
        "access_count": 15
    }
}

// kg.edge_traversed
{
    "type": "kg.edge_traversed",
    "timestamp": 1707580800.1,
    "data": {
        "source_id": "concept_python_debugging",
        "target_id": "tool_pdb",
        "edge_type": "uses",
        "weight": 0.85
    }
}

// kg.path_found
{
    "type": "kg.path_found",
    "timestamp": 1707580800.5,
    "data": {
        "path": ["concept_python_debugging", "tool_pdb", "skill_breakpoint_setting"],
        "edge_types": ["uses", "requires"],
        "total_weight": 1.65,
        "query": "How to debug this error?"
    }
}

// kg.snapshot (periodic full state for initialization)
{
    "type": "kg.snapshot",
    "timestamp": 1707580800.0,
    "data": {
        "nodes": [
            { "id": "...", "type": "concept", "label": "...", "x": 0, "y": 0, "z": 0, "confidence": 0.9 }
        ],
        "edges": [
            { "source": "...", "target": "...", "type": "relates_to", "weight": 0.8 }
        ],
        "total_nodes": 342,
        "total_edges": 891
    }
}
```

### 4.7 Truth Spine Events

```json
// truth.verification_started
{
    "type": "truth.verification_started",
    "timestamp": 1707580800.0,
    "data": {
        "action": "file_write",
        "description": "Write config.json with updated settings",
        "chain_id": "verify_xyz789"
    }
}

// truth.check_result
{
    "type": "truth.check_result",
    "timestamp": 1707580801.0,
    "data": {
        "chain_id": "verify_xyz789",
        "check_name": "file_exists",
        "passed": true,
        "reason": "File exists at /path/to/config.json"
    }
}

// truth.tier_assigned
{
    "type": "truth.tier_assigned",
    "timestamp": 1707580801.5,
    "data": {
        "chain_id": "verify_xyz789",
        "tier": "FACT",
        "confidence": 0.98,
        "checks_passed": 3,
        "checks_total": 3
    }
}
```

### 4.8 Memory Events

```json
// memory.recall
{
    "type": "memory.recall",
    "timestamp": 1707580800.0,
    "data": {
        "source": "episodic",
        "episode_id": "ep_20260209_1423",
        "summary": "Discussed Python async patterns yesterday afternoon",
        "relevance": 0.87,
        "emotional_valence": 0.3,
        "age_hours": 22.5,
        "decay_score": 0.91
    }
}

// memory.consolidation (during NeuroDream)
{
    "type": "memory.consolidation",
    "timestamp": 1707580800.0,
    "data": {
        "phase": "deep_sleep",
        "action": "pattern_extracted",
        "pattern": "User frequently asks about async/await patterns",
        "memories_consolidated": 5,
        "strength_delta": 0.15
    }
}
```

### 4.9 Multi-Agent Events

```json
// agent.routing
{
    "type": "agent.routing",
    "timestamp": 1707580800.0,
    "data": {
        "query": "Write a web scraper for news articles",
        "selected_agents": ["coder", "research"],
        "collaboration_mode": "sequential",
        "confidence": 0.82,
        "all_scores": {
            "research": 0.65,
            "coder": 0.92,
            "analyst": 0.30,
            "creative": 0.15
        }
    }
}

// agent.message
{
    "type": "agent.message",
    "timestamp": 1707580802.0,
    "data": {
        "sender": "coder",
        "recipient": "research",
        "content": "Need URL patterns for target news sites",
        "message_type": "request"
    }
}
```

### 4.10 Inner Thought Events

```json
// thought.inner
{
    "type": "thought.inner",
    "timestamp": 1707580800.0,
    "data": {
        "thought_type": "connecting",
        "content": "This question relates to the database optimization we discussed earlier...",
        "intensity": 0.7,
        "source": "inner_thoughts_engine"
    }
}
```

---

## 5. React Component Skeletons

### 5.1 VizWebSocketProvider

```tsx
// src/providers/VizWebSocketProvider.tsx
import React, { useEffect, useRef, createContext, useContext } from 'react';
import { useVizStore } from '../store/vizStore';

interface VizWSContextValue {
    connected: boolean;
    eventRate: number;
}

const VizWSContext = createContext<VizWSContextValue>({
    connected: false,
    eventRate: 0,
});

export const useVizConnection = () => useContext(VizWSContext);

interface Props {
    url: string;
    children: React.ReactNode;
}

export function VizWebSocketProvider({ url, children }: Props) {
    const wsRef = useRef<WebSocket | null>(null);
    const handleEvent = useVizStore(s => s.handleVizEvent);
    const [connected, setConnected] = React.useState(false);
    const [eventRate, setEventRate] = React.useState(0);
    const eventCountRef = useRef(0);

    useEffect(() => {
        let ws: WebSocket;
        let reconnectTimer: ReturnType<typeof setTimeout>;
        let rateTimer: ReturnType<typeof setInterval>;

        function connect() {
            ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                setConnected(true);
                console.log('[VizWS] Connected');
            };

            ws.onmessage = (evt) => {
                try {
                    const event = JSON.parse(evt.data);
                    handleEvent(event);
                    eventCountRef.current++;
                } catch (e) {
                    console.error('[VizWS] Parse error:', e);
                }
            };

            ws.onclose = () => {
                setConnected(false);
                reconnectTimer = setTimeout(connect, 2000);
            };

            ws.onerror = () => ws.close();
        }

        connect();

        // Track events per second
        rateTimer = setInterval(() => {
            setEventRate(eventCountRef.current);
            eventCountRef.current = 0;
        }, 1000);

        return () => {
            clearTimeout(reconnectTimer);
            clearInterval(rateTimer);
            ws?.close();
        };
    }, [url, handleEvent]);

    return (
        <VizWSContext.Provider value={{ connected, eventRate }}>
            {children}
        </VizWSContext.Provider>
    );
}
```

### 5.2 BrainScene (Root 3D Component)

```tsx
// src/components/BrainScene.tsx
import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { GlobalWorkspaceViz } from './GlobalWorkspaceViz';
import { KnowledgeGraphViz } from './KnowledgeGraphViz';
import { MCTSTreeViz } from './MCTSTreeViz';
import { CognitiveTheaterViz } from './CognitiveTheaterViz';
import { EmotionViz } from './EmotionViz';
import { TruthSpineViz } from './TruthSpineViz';
import { MultiAgentViz } from './MultiAgentViz';
import { ParticleFlowSystem } from './ParticleFlowSystem';

// Brain region positions (world space)
const REGIONS = {
    globalWorkspace: [0, 5, 0] as const,    // Top center
    knowledgeGraph:  [18, 0, 0] as const,    // Right
    mctsTree:        [-18, 0, 0] as const,   // Left
    theater:         [0, 0, -15] as const,    // Back
    emotions:        [0, -5, 0] as const,     // Below center
    truthSpine:      [0, -12, 0] as const,    // Bottom
    multiAgent:      [0, 0, 15] as const,     // Front
    memory:          [0, 0, -30] as const,    // Far back (timeline)
};

export function BrainScene() {
    const groupRef = useRef<THREE.Group>(null);

    // Gentle rotation of entire brain
    useFrame((_, delta) => {
        if (groupRef.current) {
            groupRef.current.rotation.y += delta * 0.02;
        }
    });

    return (
        <group ref={groupRef}>
            {/* Translucent brain shell */}
            <mesh>
                <sphereGeometry args={[25, 32, 32]} />
                <meshPhysicalMaterial
                    color="#1a1a3e"
                    transparent
                    opacity={0.05}
                    roughness={0.8}
                    wireframe
                />
            </mesh>

            {/* Brain regions */}
            <group position={REGIONS.globalWorkspace}>
                <GlobalWorkspaceViz />
            </group>

            <group position={REGIONS.knowledgeGraph}>
                <KnowledgeGraphViz />
            </group>

            <group position={REGIONS.mctsTree}>
                <MCTSTreeViz />
            </group>

            <group position={REGIONS.theater}>
                <CognitiveTheaterViz />
            </group>

            <group position={REGIONS.emotions}>
                <EmotionViz />
            </group>

            <group position={REGIONS.truthSpine}>
                <TruthSpineViz />
            </group>

            <group position={REGIONS.multiAgent}>
                <MultiAgentViz />
            </group>

            {/* Global particle system for neuromodulator flows */}
            <ParticleFlowSystem />
        </group>
    );
}
```

### 5.3 GlobalWorkspaceViz

```tsx
// src/components/GlobalWorkspaceViz.tsx
import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { useVizStore } from '../store/vizStore';

const CODELET_NAMES = [
    'alma_emotion', 'episodic_memory', 'pattern_prophet',
    'kg_reasoning', 'reflexion', 'inner_thoughts',
    'active_inference', 'screen_awareness'
];

const CODELET_COLORS = [
    '#ff6b6b', '#4ecdc4', '#ffe66d', '#a8e6cf',
    '#ff8b94', '#6c5ce7', '#fd79a8', '#00cec9'
];

export function GlobalWorkspaceViz() {
    const consciousState = useVizStore(s => s.consciousState);
    const centerRef = useRef<THREE.Mesh>(null);

    // Arrange codelets in a ring
    const codeletPositions = useMemo(() =>
        CODELET_NAMES.map((_, i) => {
            const angle = (i / CODELET_NAMES.length) * Math.PI * 2;
            const radius = 6;
            return new THREE.Vector3(
                Math.cos(angle) * radius,
                0,
                Math.sin(angle) * radius
            );
        })
    , []);

    // Animate center glow based on broadcast
    useFrame((state) => {
        if (centerRef.current) {
            const mat = centerRef.current.material as THREE.MeshStandardMaterial;
            const targetIntensity = consciousState?.attention_intensity ?? 0.3;
            const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.3 + 0.7;
            mat.emissiveIntensity = THREE.MathUtils.lerp(
                mat.emissiveIntensity,
                targetIntensity * pulse * 3, // >1 for bloom pickup
                0.05
            );
        }
    });

    const winnerModule = consciousState?.broadcast_content?.source_module;

    return (
        <group>
            {/* Central broadcast sphere */}
            <mesh ref={centerRef}>
                <sphereGeometry args={[1.5, 32, 32]} />
                <meshStandardMaterial
                    color="#ffffff"
                    emissive="#4fc3f7"
                    emissiveIntensity={1.5}
                    toneMapped={false}
                />
            </mesh>

            {/* Attention focus label */}
            {consciousState && (
                <Billboard position={[0, 3, 0]}>
                    <Text fontSize={0.5} color="#4fc3f7" anchorX="center">
                        {consciousState.attention_focus}
                    </Text>
                </Billboard>
            )}

            {/* Codelet ring */}
            {codeletPositions.map((pos, i) => {
                const isWinner = CODELET_NAMES[i] === winnerModule;
                return (
                    <group key={CODELET_NAMES[i]} position={pos}>
                        <mesh>
                            <sphereGeometry args={[isWinner ? 0.8 : 0.5, 16, 16]} />
                            <meshStandardMaterial
                                color={CODELET_COLORS[i]}
                                emissive={CODELET_COLORS[i]}
                                emissiveIntensity={isWinner ? 2.5 : 0.3}
                                toneMapped={false}
                            />
                        </mesh>
                        <Billboard position={[0, 1.2, 0]}>
                            <Text fontSize={0.3} color="#aaa" anchorX="center">
                                {CODELET_NAMES[i].replace('_', '\n')}
                            </Text>
                        </Billboard>
                    </group>
                );
            })}
        </group>
    );
}
```

### 5.4 EmotionViz (Neuromodulators)

```tsx
// src/components/EmotionViz.tsx
import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { useVizStore } from '../store/vizStore';

const NEUROMOD_CONFIG = {
    dopamine:        { color: '#ffa726', label: 'Dopamine',        angle: 0 },
    serotonin:       { color: '#42a5f5', label: 'Serotonin',       angle: Math.PI / 2 },
    norepinephrine:  { color: '#ef5350', label: 'Norepinephrine',  angle: Math.PI },
    oxytocin:        { color: '#ec407a', label: 'Oxytocin',        angle: 3 * Math.PI / 2 },
};

export function EmotionViz() {
    const pad = useVizStore(s => s.padState);
    const neuro = useVizStore(s => s.neuromodulators);
    const padOrbRef = useRef<THREE.Mesh>(null);

    // PAD orb position maps to 3D space
    useFrame((state) => {
        if (padOrbRef.current) {
            padOrbRef.current.position.x = THREE.MathUtils.lerp(
                padOrbRef.current.position.x, pad.pleasure * 3, 0.05
            );
            padOrbRef.current.position.y = THREE.MathUtils.lerp(
                padOrbRef.current.position.y, pad.arousal * 3, 0.05
            );
            padOrbRef.current.position.z = THREE.MathUtils.lerp(
                padOrbRef.current.position.z, pad.dominance * 3, 0.05
            );

            const mat = padOrbRef.current.material as THREE.MeshStandardMaterial;
            const pulse = Math.sin(state.clock.elapsedTime * (1 + pad.arousal * 3)) * 0.3 + 0.7;
            mat.emissiveIntensity = (0.5 + Math.abs(pad.arousal)) * pulse * 2;
        }
    });

    const padColor = useMemo(() => {
        const p = pad.pleasure;
        if (p < 0) return new THREE.Color(1.0, 0.3 + p * 0.3, 0.3);
        return new THREE.Color(0.3, 0.5 + p * 0.5, 1.0 - p * 0.7);
    }, [pad.pleasure]);

    return (
        <group>
            <Billboard position={[0, 5, 0]}>
                <Text fontSize={0.6} color="#ec407a">Emotional Core</Text>
            </Billboard>

            {/* PAD space wireframe cube */}
            <mesh>
                <boxGeometry args={[6, 6, 6]} />
                <meshBasicMaterial color="#333" wireframe transparent opacity={0.15} />
            </mesh>

            {/* PAD orb */}
            <mesh ref={padOrbRef}>
                <sphereGeometry args={[0.6, 24, 24]} />
                <meshStandardMaterial
                    color={padColor}
                    emissive={padColor}
                    emissiveIntensity={1.5}
                    toneMapped={false}
                    transparent
                    opacity={0.9}
                />
            </mesh>

            {/* Neuromodulator level bars */}
            {Object.entries(NEUROMOD_CONFIG).map(([key, cfg]) => {
                const level = neuro[key as keyof typeof neuro] ?? 0.5;
                const radius = 5;
                const x = Math.cos(cfg.angle) * radius;
                const z = Math.sin(cfg.angle) * radius;
                return (
                    <group key={key} position={[x, -3, z]}>
                        <mesh position={[0, 2, 0]}>
                            <cylinderGeometry args={[0.15, 0.15, 4, 8]} />
                            <meshBasicMaterial color="#222" transparent opacity={0.3} />
                        </mesh>
                        <mesh position={[0, level * 2, 0]}>
                            <cylinderGeometry args={[0.2, 0.2, level * 4, 8]} />
                            <meshStandardMaterial
                                color={cfg.color}
                                emissive={cfg.color}
                                emissiveIntensity={level * 2}
                                toneMapped={false}
                            />
                        </mesh>
                        <Billboard position={[0, 5, 0]}>
                            <Text fontSize={0.3} color={cfg.color}>{cfg.label}</Text>
                        </Billboard>
                    </group>
                );
            })}
        </group>
    );
}
```

### 5.5 PostProcessing + SceneSetup

```tsx
// src/components/PostProcessing.tsx
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';

export function PostProcessing() {
    return (
        <EffectComposer>
            <Bloom
                luminanceThreshold={1.0}
                luminanceSmoothing={0.4}
                intensity={1.5}
                mipmapBlur
            />
            <Vignette offset={0.3} darkness={0.7} />
        </EffectComposer>
    );
}

// src/components/SceneSetup.tsx
import { OrbitControls, Stars } from '@react-three/drei';

export function SceneSetup() {
    return (
        <>
            <color attach="background" args={['#050510']} />
            <fog attach="fog" args={['#050510', 40, 100]} />
            <ambientLight intensity={0.15} />
            <pointLight position={[0, 20, 0]} intensity={0.5} color="#4fc3f7" />
            <pointLight position={[0, -10, 0]} intensity={0.3} color="#ec407a" />
            <Stars radius={100} depth={50} count={3000} factor={3} fade speed={0.5} />
            <OrbitControls
                enableDamping
                dampingFactor={0.05}
                minDistance={5}
                maxDistance={80}
            />
        </>
    );
}
```

---

## 6. Python Integration

### 6.1 Design Principle: Non-Blocking Event Emission

The visualization hooks must **never slow down AURA's reasoning**. All event emission is fire-and-forget via a thread-safe `VizEventBus.emit()` that puts events on an asyncio queue. If no frontend is connected, events are buffered (ring buffer, last 100) and otherwise discarded.

### 6.2 Hook Points in Existing Code

Each hook is a 1-3 line addition. The `viz_emit` helper function ensures safe import and graceful degradation.

```python
# aura/viz_hooks.py  (new file, <30 lines)
"""Visualization event emission hooks. Zero overhead when no frontend connected."""

import logging

logger = logging.getLogger(__name__)

def viz_emit(event_type: str, data: dict):
    """Fire-and-forget event emission. Thread-safe. No-op if viz not running."""
    try:
        from api.routes.viz_stream import VizEventBus
        VizEventBus.get().emit(event_type, data)
    except Exception:
        pass  # Never crash AURA for visualization
```

### 6.3 Hook Locations

#### brain.py (reasoning, model calls)
```python
from .viz_hooks import viz_emit

# After neuromodulator levels are read:
viz_emit("emotion.neuro_read", {
    "dopamine": levels["dopamine"],
    "serotonin": levels["serotonin"],
    "norepinephrine": levels["norepinephrine"],
    "oxytocin": levels.get("oxytocin", 0.5),
})
```

#### mcts_reasoning.py (tree search events)
The MCTS already has callbacks (`on_node_created`, `on_node_evaluated`, `on_iteration_complete`). Wire them to viz_emit:

```python
from aura.viz_hooks import viz_emit

mcts.on_node_created = lambda node: viz_emit("mcts.node_created", {
    "node_id": node.id,
    "parent_id": node.parent.id if node.parent else None,
    "depth": node.depth,
    "thought_content": node.thought.content[:200],
    "state": node.state.value,
})
mcts.on_node_evaluated = lambda node, reward: viz_emit("mcts.node_evaluated", {
    "node_id": node.id,
    "reward": round(reward, 3),
    "visits": node.visits,
    "avg_value": round(node.avg_value, 3),
    "state": node.state.value,
})
```

#### cognitive_theater.py
```python
from aura.viz_hooks import viz_emit

# After each perspective is parsed:
viz_emit("theater.perspective_generated", {
    "perspective": name,
    "content": content[:200],
    "sequence": sequence_number,
})

# After synthesis:
viz_emit("theater.synthesis", {
    "synthesis": deliberation.synthesis[:200],
    "confidence": deliberation.confidence,
})
```

#### alma_engine.py (emotional state changes)
```python
from aura.viz_hooks import viz_emit

# After PAD state update:
viz_emit("emotion.update", {
    "pad": {"pleasure": self.pad.pleasure, "arousal": self.pad.arousal, "dominance": self.pad.dominance},
    "neuromodulators": self.get_neuromodulator_levels(),
    "dominant_emotion": self.get_dominant_emotion(),
    "mood": self.current_mood.value if self.current_mood else "neutral",
})
```

#### truth_spine.py
```python
from aura.viz_hooks import viz_emit

viz_emit("truth.check_result", {
    "chain_id": chain_id,
    "check_name": check.name,
    "passed": passed,
    "reason": reason,
})

viz_emit("truth.tier_assigned", {
    "chain_id": chain_id,
    "tier": tier.value,
    "confidence": confidence,
})
```

#### knowledge_graph.py
```python
from aura.viz_hooks import viz_emit

viz_emit("kg.node_activated", {
    "node_id": node.id,
    "node_type": node.type,
    "label": node.label,
    "confidence": node.confidence,
})

viz_emit("kg.path_found", {
    "path": [n.id for n in path_nodes],
    "edge_types": edge_types,
    "query": query_text[:100],
})
```

#### global_workspace.py
```python
from aura.viz_hooks import viz_emit

viz_emit("consciousness.broadcast", {
    "winner": winner.to_dict() if winner else None,
    "runners_up": [r.to_dict() for r in runners_up[:2]],
    "attention_focus": self._state.attention_focus,
    "attention_intensity": self._state.attention_intensity,
    "cycle_number": self._state.cycle_number,
})
```

### 6.4 KG Snapshot Endpoint (REST)

```python
@router.get("/kg-snapshot")
async def get_kg_snapshot():
    """Return full knowledge graph for initial 3D layout."""
    try:
        from aura.tools import get_knowledge_graph
        kg = get_knowledge_graph()
        graph = kg.graph

        nodes = [{"id": nid, "type": d.get("type", "concept"), "label": d.get("label", nid),
                  "confidence": d.get("confidence", 0.8)} for nid, d in graph.nodes(data=True)]
        edges = [{"source": s, "target": t, "type": d.get("type", "relates_to"),
                  "weight": d.get("weight", 0.5)} for s, t, d in graph.edges(data=True)]

        return {"nodes": nodes, "edges": edges, "total_nodes": len(nodes), "total_edges": len(edges)}
    except Exception as e:
        return {"error": str(e), "nodes": [], "edges": []}
```

---

## 7. Implementation Phases

### Phase 1: Skeleton + Emotions (1-2 days) -- MAXIMUM VISUAL IMPACT

**Goal:** A rotating translucent brain with a glowing emotional core that responds to real AURA emotional state. This alone is shareable.

**Tasks:**
1. Create React app with R3F (`npx create-react-app aura-viz --template typescript`)
2. Install: `@react-three/fiber`, `@react-three/drei`, `@react-three/postprocessing`, `zustand`, `leva`
3. Build: `SceneSetup` (dark background, stars, fog, bloom)
4. Build: `BrainScene` with translucent wireframe sphere
5. Build: `EmotionViz` with PAD orb and 4 neuromodulator bars
6. Add `viz_hooks.py` to Python backend
7. Add `VizEventBus` and WebSocket endpoint at `/api/viz/stream`
8. Hook `alma_engine.py` to emit `emotion.update` events
9. Connect frontend to WebSocket, animate PAD orb in real-time

**Why first:** Emotions change continuously (even when idle, due to autonomous drift and circadian rhythms). The visualization will always be alive. The glowing orb + bloom + neuromodulator bars look stunning with minimal code.

### Phase 2: Global Workspace Ring (1 day)

**Goal:** Add the consciousness ring -- 8 codelet nodes competing for broadcast, with the winner beaming to center.

**Tasks:**
1. Build `GlobalWorkspaceViz` component (ring of 8 spheres + center)
2. Hook `global_workspace.py` to emit `consciousness.broadcast`
3. Animate winning codelet (grow, brighten, beam to center)
4. Show attention_focus as 3D text label

**Why second:** The global workspace cycles at ~300ms, creating constant rhythmic activity. The "spotlight of consciousness" metaphor is visually compelling.

### Phase 3: Knowledge Graph (2 days)

**Goal:** Force-directed 3D knowledge graph with real nodes and edges.

**Tasks:**
1. Add REST endpoint `/api/viz/kg-snapshot`
2. Integrate `r3f-forcegraph` or custom force layout with `d3-force-3d`
3. Color-code nodes by type (16 types)
4. Add node activation glow on `kg.node_activated` events
5. Highlight traversed paths on `kg.path_found` events

### Phase 4: MCTS Reasoning Tree (1-2 days)

**Goal:** Live tree growth during deep reasoning sessions.

**Tasks:**
1. Build `MCTSTreeViz` with instanced nodes
2. Wire existing MCTS callbacks
3. Animate node creation (pop in), evaluation (color change), best path highlighting
4. UCB1 score as halo radius (exploration) + brightness (exploitation)

### Phase 5: CognitiveTheater + Inner Monologue (1 day)

**Goal:** Four perspective actors debating, plus scrolling inner thoughts.

### Phase 6: Truth Spine + Multi-Agent (1 day)

**Goal:** Verification chain as DNA backbone, agent brain regions.

### Phase 7: Particle Systems + Polish (2-3 days)

**Goal:** Neuromodulator particle streams, edge flow particles, camera presets, HUD.

### Phase 8: Optimization + Deployment (1-2 days)

**Goal:** Production-ready performance.

---

## Estimated Total Timeline

| Phase | Description | Days | Cumulative |
|-------|-------------|------|------------|
| 1 | Skeleton + Emotions (shareable!) | 1-2 | 1-2 |
| 2 | Global Workspace Ring | 1 | 2-3 |
| 3 | Knowledge Graph | 2 | 4-5 |
| 4 | MCTS Reasoning Tree | 1-2 | 5-7 |
| 5 | CognitiveTheater + Inner Monologue | 1 | 6-8 |
| 6 | Truth Spine + Multi-Agent | 1 | 7-9 |
| 7 | Particle Systems + Polish | 2-3 | 9-12 |
| 8 | Optimization + Deployment | 1-2 | 10-14 |

**Minimum Viable Demo (Phases 1-2): 2-3 days** -- A glowing brain with emotional state and consciousness broadcast cycling. Already shareable.

**Full Feature Set (All Phases): ~2 weeks** -- Complete neural visualization with all AURA subsystems represented.

---

## References

### React Three Fiber
- [R3F Documentation](https://r3f.docs.pmnd.rs/)
- [R3F GitHub](https://github.com/pmndrs/react-three-fiber)
- [100 Three.js Best Practices (2026)](https://www.utsubo.com/blog/threejs-best-practices-100-tips)
- [Scaling Performance](https://r3f.docs.pmnd.rs/advanced/scaling-performance)

### Force-Directed Graphs
- [r3f-forcegraph](https://github.com/vasturiano/r3f-forcegraph)
- [3d-force-graph](https://github.com/vasturiano/3d-force-graph)
- [d3-force-3d](https://github.com/vasturiano/d3-force-3d)

### Shaders & Particles
- [GPGPU Dreamy Particle Effect (Codrops)](https://tympanus.net/codrops/2024/12/19/crafting-a-dreamy-particle-effect-with-three-js-and-gpgpu/)
- [GPGPU Flow Field Particles (Three.js Journey)](https://threejs-journey.com/lessons/gpgpu-flow-field-particles-shaders)
- [Galaxy Simulation with WebGPU Compute Shaders](https://threejsroadmap.com/blog/galaxy-simulation-webgpu-compute-shaders)
- [Shader Glow (Three.js)](https://stemkoski.github.io/Three.js/Shader-Glow.html)

### Post-Processing
- [Bloom Effect](https://react-postprocessing.docs.pmnd.rs/effects/bloom)
- [React Postprocessing](https://github.com/pmndrs/react-postprocessing)

### Real-Time Streaming
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)

### Neural/Brain Visualization
- [TensorSpace](https://github.com/tensorspace-team/tensorspace)
- [Neural Networks in 3D](https://arogozhnikov.github.io/3d_nn/)
- [BrainViz](https://github.com/matanmazor/BrainViz)

### AI Reasoning Visualization (Academic)
- [Interactive Reasoning (UIST 2025)](https://arxiv.org/html/2506.23678v1)
- [Landscape of Thoughts](https://arxiv.org/abs/2503.22165)
- [ReasonGraph](https://arxiv.org/html/2503.03979v1)
