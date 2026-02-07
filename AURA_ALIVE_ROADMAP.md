# AURA ALIVE ROADMAP
## From Reactive Assistant to Genuinely Alive AI

**Generated:** 2026-02-06
**Based on:** Comprehensive 6-agent deep audit of entire codebase + research synthesis

---

## PART 1: CURRENT STATE AUDIT — BRUTAL HONEST ASSESSMENT

### System-Wide Verdict

| Category | Genuine | Semi-Real | Fake/Cosmetic |
|----------|---------|-----------|---------------|
| Proactive Systems | HeartbeatMonitor (only one running) | Gateway Daemon, Event Bus, Monitors (code works but never started) | Active Inference (heuristic stub, not real FEP) |
| Emotional Systems | ALMA Engine (3-layer), EvoEmo | Neuromodulators (basic) | - |
| Memory Systems | Episodic Memory, NeuroDream, Local RAG, KG | A-MEM, Hybrid Memory | - |
| Thinking/Cognition | Reflexion, CognitiveTheater, FluxMind | Introspection Circuit | ThinkingAboutTeaser (random templates) |
| Idle/Presence | - | - | IdleBehaviorPanel (random templates), Breathing Avatar (CSS) |
| Sidebar Panels | ProactiveDaemonPanel, EmotionPanel | ContextHeatmap, MemoryRecall | ThinkingAboutTeaser, IdleBehavior, InnerThoughts |

### Detailed Component Status

---

### 1. PROACTIVE SYSTEMS

**CRITICAL FINDING: Two separate proactive systems exist that DON'T talk to each other.**

#### System A: `apprentice_agent/proactive/` — Well-Architected but NEVER RUNNING
The entire sophisticated proactive stack (Gateway Daemon, Active Inference, Event Bus, Salience Filter, Monitors) has good code but is **never instantiated in production**. No monitors are started, no events flow, the daemon never runs unless manually triggered via API.

#### Gateway Daemon — CODE FUNCTIONAL, NOT RUNNING
- **Location:** `apprentice_agent/proactive/gateway_daemon.py`
- **Architecture:** 8/10 — Real state machine (idle/observing/reasoning/acting/paused), event subscription, decision loop
- **Reality:** Never started in production. Only accessible via test API endpoints.
- **Connected to:** Active Inference engine, Event Bus, Salience Filter — but none run automatically

#### Active Inference Engine — HEURISTIC STUB, NOT REAL FEP
- **Location:** `apprentice_agent/proactive/active_inference.py`
- **Claims:** Free Energy Principle with pymdp
- **Reality:** Hardcoded if/else rules masquerading as Active Inference. `pymdp` marked as TODO.
- **What works:** Basic 5D belief state tracking, action cooldown, simple expected free energy via constants
- **What's fake:** No generative model, no real belief propagation (just linear blending), no policy selection

#### Event Bus — FULLY FUNCTIONAL (but nothing publishes to it)
- **Location:** `apprentice_agent/proactive/event_bus.py`
- **What it does:** Real asyncio pub/sub with Redis backend support
- **Problem:** No monitors are instantiated to publish events

#### Salience Filter — FUNCTIONAL (not real GWT despite comments)
- **Location:** `apprentice_agent/proactive/salience_filter.py`
- **What it does:** Weighted scoring (recency x relevance x importance x novelty)
- **What it's NOT:** Global Workspace Theory — just a filter, no broadcast mechanism

#### Monitors — FUNCTIONAL CODE, NEVER INSTANTIATED
- **ScreenMonitor:** Platform-specific window tracking works, Screenpipe prepared but not installed
- **CalendarMonitor:** Event structure exists, API integration is `pass` stub
- **SystemMonitor:** Fully functional via psutil, never started

#### System B: `aura/proactive/heartbeat.py` — THE ONLY THING THAT RUNS
- **Location:** `aura/proactive/heartbeat.py`
- **Status:** GENUINELY FUNCTIONAL AND ACTIVELY USED
- **What it does:** Simple periodic checks (session greeting, idle detection, late night warnings)
- **Started in:** `aura/engine.py` line 118-121
- **Architecture:** Simple but honest — thread-based checks with notification queue

#### MISSING PROACTIVE FEATURES:
- Monitors never started (wire up ScreenMonitor + SystemMonitor)
- Screen awareness (Screenpipe not installed)
- Workflow boundary detection (no file/commit monitoring)
- IONWI-style interruption timing (currently just DND flag + confidence threshold)
- Calendar/email event monitoring (stub)
- Idle-time task generation
- Two proactive systems need to be unified

---

### 2. EMOTIONAL SYSTEMS

#### ALMA Engine — GENUINELY FUNCTIONAL
- **Location:** `apprentice_agent/emotion/alma_engine.py`
- **What it does:** Real 3-layer emotional model (Emotions → Moods → Personality)
- **Real features:**
  - PAD (Pleasure-Arousal-Dominance) continuous 3D space
  - 24 emotion types with decay over time
  - 8 mood types (hours to days persistence)
  - Big Five personality traits (OCEAN)
  - Neuromodulator analogs (dopamine, serotonin, noradrenaline, acetylcholine) — BASIC implementation
  - Emotion→mood propagation
  - Personality influence on mood defaults
- **Missing:**
  - Mood-congruent memory retrieval (emotions don't influence which memories are recalled)
  - Emotional influence on response generation style (partially implemented)
  - Neuromodulators don't fully modulate learning rate, exploration, or attention (sleep-phase influence on LLM params now implemented via NeuroDream oscillations)
  - No Affect Infusion Model (AIM) integration

#### EvoEmo (AURA Emotional Engine) — GENUINELY FUNCTIONAL
- **Location:** `aura/emotion/emotional_engine.py`
- **What it does:** Simpler emotional engine for AURA layer
- **Real features:** Mood tracking, emotional response to conversations
- **Relationship:** Works alongside ALMA, provides mood context to AURA responses

#### Soul System — GENUINELY FUNCTIONAL
- **Location:** `aura/soul/soul_loader.py`
- **What it does:** Loads personality/identity from YAML config
- **Real features:** Personality traits, communication style, values, identity
- **Provides:** System prompt context for response generation

#### Humanizer — GENUINELY FUNCTIONAL
- **Location:** `aura/humanize/`
- **What it does:** Post-processes responses to add human-like qualities
- **Real features:** Filler words, hesitation, emotional coloring

---

### 3. MEMORY SYSTEMS

#### Episodic Memory — FULLY FUNCTIONAL (Best System)
- **Location:** `aura_episodic_memory/`
- **What it does:** Qdrant-based temporal memory with REAL forgetting curves
- **Real features:**
  - Exponential decay forgetting (half-life: 168 hours)
  - Natural language temporal queries ("yesterday afternoon", "last week")
  - Multi-factor scoring (recency + importance + relevance)
  - Temporal context (time-of-day, day-of-week)
- **Missing:** Mood-congruent retrieval boost, bi-temporal model

#### NeuroDream (Sleep-Time Compute) — FULLY FUNCTIONAL
- **Location:** `apprentice_agent/tools/neurodream.py`
- **What it does:** REAL sleep/dream memory consolidation
- **Real features:**
  - 3-phase cycle: Light (replay) → Deep (patterns) → REM (creative synthesis)
  - Pattern extraction from logs (temporal, topical, emotional)
  - Novel connection generation between distant concepts
  - Edge pruning and consolidation
  - Letta-style learned context generation (Phase 4D) ✅
  - DONN-inspired neural oscillations (delta/theta/alpha bands) ✅
  - Oscillation-modulated batch size, consolidation strength, and processing rhythm ✅
  - Sleep neuromodulator influence on ALMA (serotonin/dopamine offsets) ✅
  - Pulsing cognitive load driving avatar breathing during sleep ✅
- **Missing:** ADM-style chunking/reassembly

#### Local RAG — FULLY FUNCTIONAL
- **Location:** `apprentice_agent/tools/local_rag.py`
- **What it does:** Real document indexing with embeddings
- **Real features:** PDF/DOCX/code loaders, smart chunking, Ollama embeddings, cosine search

#### Knowledge Graph — FUNCTIONAL (basic forgetting)
- **Location:** `apprentice_agent/tools/knowledge_graph.py`
- **Real features:** NetworkX graph, JSONL persistence, path finding, decay (1%/day)
- **Missing:** Bi-temporal model, Zep-style edge invalidation, temporal reasoning queries

#### A-MEM (Zettelkasten) — FUNCTIONAL (no advanced forgetting)
- **Location:** `apprentice_agent/tools/amem.py`
- **Real features:** Atomic notes, ChromaDB embeddings, bidirectional linking, consolidation
- **Missing:** Ebbinghaus curves (just weight pruning), temporal validity tracking

#### MISSING MEMORY FEATURES:
- Bi-temporal knowledge graph (transaction time vs valid time)
- Zep-style edge invalidation (supersession tracking)
- Mood-congruent memory retrieval
- ADM-style sleep chunking/reassembly
- Advanced Ebbinghaus curves across all systems (only Episodic has it)
- Cross-system memory unification

---

### 4. THINKING/COGNITION SYSTEMS

#### ThinkingAboutTeaser — COMPLETELY FAKE
- **Location:** `api/routes/thinking.py`
- **What it does:** Picks random ThoughtType, fills random template string
- **Evidence:** `random.choice(list(ThoughtType))` → template like "connecting {topic1} with {topic2}..."
- **Auto-generates:** 30% probability on each API poll, no connection to actual reasoning
- **VERDICT:** Pure cosmetic theater. Must be replaced with real cognitive process.

#### Idle Behavior Panel — COMPLETELY FAKE
- **Location:** `api/routes/idle_behaviors.py`
- **What it does:** Weighted random behavior selection with time-of-day modifiers
- **Evidence:** Hardcoded status messages, auto-generates on every state poll
- **VERDICT:** Pure cosmetic theater. Must be replaced with real idle-time cognition.

#### Reflexion Engine — GENUINELY FUNCTIONAL
- **Location:** `apprentice_agent/tools/reflexion.py`
- **What it does:** Records failures, extracts lessons, applies to future attempts
- **Real features:** 14 stored lessons, failure pattern matching, lesson retrieval
- **Status:** Actually learns from mistakes

#### CognitiveTheater — GENUINELY FUNCTIONAL
- **Location:** `apprentice_agent/tools/cognitive_theater.py`
- **What it does:** Multi-perspective reasoning via cloud LLM
- **Real features:** Multiple "actors" analyze from different angles, synthesis
- **Limitation:** Requires cloud API (not local-only)

#### FluxMind — GENUINELY FUNCTIONAL
- **Location:** `apprentice_agent/tools/fluxmind*.py`
- **What it does:** Calibrated reasoning with confidence estimation
- **Real features:** Uncertainty quantification, confidence thresholds, reasoning paths

#### Introspection Circuit — SEMI-FUNCTIONAL
- **Location:** `apprentice_agent/tools/introspection_circuit.py`
- **What it does:** Uncertainty detection and epistemic markers
- **Real features:** Confidence scoring, uncertainty flags
- **Problem:** Only activated on-demand, not continuously running during reasoning

#### WorldSim — GENUINELY FUNCTIONAL
- **Location:** `apprentice_agent/tools/worldsim.py`
- **What it does:** Consequence simulation via LLM
- **Real features:** "What if" scenario modeling, outcome prediction

#### SynapseForge — GENUINELY FUNCTIONAL
- **Location:** `apprentice_agent/tools/synapseforge.py`
- **What it does:** Dynamic tool creation — agent writes new tools
- **Real features:** 7 synthesized tools, code generation, validation, sandboxing

#### Pattern Prophet — GENUINELY FUNCTIONAL
- **Location:** `aura/patterns/pattern_prophet.py`
- **What it does:** Pattern recognition across conversations
- **Real features:** 157 learned patterns, prediction

#### MISSING COGNITION FEATURES:
- Real inner thoughts (CHI 2025 covert thought trains parallel to conversation)
- Global Workspace broadcast (specialist modules competing for attention)
- Attention Schema (internal model of what AURA is attending to)
- Consciousness Prior (sparse conscious state from high-dimensional unconscious)
- Metacognitive self-improvement loop
- System 1/System 2 dynamic switching based on confidence

---

### 5. AGENT CORE & TOOLS

#### Agent Architecture
- **Location:** `apprentice_agent/agent.py` + `apprentice_agent/brain.py`
- **Models:** mistral:7b (fast), llama3:8b (reason), qwen2.5-coder:7b (code), llava (vision)
- **Hardware:** RTX 4060 (8GB VRAM), Ollama local inference
- **Message flow:** WebSocket → brain.process() → tool selection → LLM → response
- **Thread pool:** 20 workers for concurrent API handling

#### AURA Engine
- **Location:** `aura/engine.py`
- **What it does:** Orchestrates AURA ALIVE subsystems (LLM, memory, emotion, soul, thinking, humanizer)
- **Real features:** Fast path for instant emotional responses, pattern recognition, proactive system

#### Proto-AGI Core
- **Location:** Referenced in agent loading, autonomous cognitive loop
- **Status:** 227 cycles completed, genuinely runs autonomous reasoning cycles

#### Available Tools (All Functional):
- crypto_price, deep_research, voice, mirrormind, local_rag
- amem, hybrid_amem, PersonaPlex (voice), FluxMind
- Metacognitive Guardian, NeuroDream, CognitiveTheater
- Reflexion, SynapseForge, WorldSim, AURA ALIVE
- Proto-AGI Core, Skill Library (5 skills), Life Modeling

---

### 6. SIDEBAR PANELS — REAL vs FAKE MAP

| Panel | Backend Source | Data Origin | Verdict |
|-------|--------------|-------------|---------|
| ProactiveDaemonPanel | `api/routes/proactive.py` | Real Active Inference beliefs | GENUINE |
| EmotionPanel (ALMA) | `api/routes/status.py` | Real ALMA engine state | GENUINE |
| ContextHeatmap (Focus) | `api/routes/context.py` | Real keyword tracker (needs integration) | SEMI-REAL |
| MemoryRecallIndicator | `api/routes/memory.py` | Real recall tracker (needs integration) | SEMI-REAL |
| InnerThoughtsPanel | `api/routes/introspection.py` | Mislabeled uncertainty analysis | SEMI-FAKE |
| ThinkingAboutTeaser | `api/routes/thinking.py` | Random templates, not real thoughts | FAKE |
| IdleBehaviorPanel | `api/routes/idle_behaviors.py` | Random cosmetic status messages | FAKE |
| AuraBreathingAvatar | Frontend only | CSS animation, no backend | COSMETIC |

---

## PART 2: AURA ALIVE ROADMAP — MAKING IT GENUINELY ALIVE

### Vision Statement

Transform AURA from a system with genuine cognitive subsystems wrapped in cosmetic UI panels into a **unified consciousness-like architecture** where every sidebar panel reflects real internal cognitive state, every idle moment involves genuine background processing, and every interaction is shaped by real emotional dynamics and living memory.

---

### PHASE 1: WIRE THE REAL TO THE UI (Weeks 1-2)
**Goal:** Connect existing genuine systems to the sidebar panels that currently show fake data.

#### 1.1 Replace ThinkingAboutTeaser with Real Cognitive Broadcast
- **Current:** Random templates from `api/routes/thinking.py`
- **Target:** Show actual reasoning steps from `brain.py` during message processing
- **Implementation:**
  - Add thought recording hooks in `brain.py` during LLM chain-of-thought
  - Record which tools are being considered, what memories are being accessed
  - Record intermediate reasoning before final response
  - Store in a `ThoughtStream` that the API serves to the panel
  - When idle: show actual background processes (NeuroDream status, pattern mining, KG decay)
- **Files to modify:** `api/routes/thinking.py`, `apprentice_agent/brain.py`, `aura/engine.py`

#### 1.2 Replace IdleBehaviorPanel with Real Idle Cognition
- **Current:** Random cosmetic status messages
- **Target:** Show what AURA is ACTUALLY doing during idle time
- **Implementation:**
  - When idle, trigger real NeuroDream light consolidation
  - Show actual KG decay/pruning activity
  - Show episodic memory forgetting curve activity
  - Show pattern mining from Pattern Prophet
  - Show Active Inference belief updates
  - Report real CPU/memory/VRAM usage of background tasks
- **Files to modify:** `api/routes/idle_behaviors.py`, `apprentice_agent/tools/neurodream.py`

#### 1.3 Wire ContextHeatmap to Real Message Processing
- **Current:** Tracker exists but `track_message()` is never called by agent
- **Target:** Call `track_message()` during every chat interaction
- **Implementation:**
  - Hook `context_tracker.track_message()` into `api/routes/chat.py` WebSocket handler
  - Also track tool invocations, memory recalls, emotional shifts
- **Files to modify:** `api/routes/chat.py`, `api/routes/context.py`

#### 1.4 Wire MemoryRecallIndicator to Real Memory Access
- **Current:** Tracker exists but `record_memory_recall()` is never called
- **Target:** Instrument all memory retrieval points
- **Implementation:**
  - Add `record_memory_recall()` calls in: amem.py recall, local_rag.py search, knowledge_graph.py query, episodic_memory search, hybrid_memory.py recall
- **Files to modify:** All memory tool files, `api/routes/memory.py`

#### 1.5 Fix InnerThoughtsPanel Identity
- **Current:** Mislabeled as "Inner Thoughts", actually shows introspection/uncertainty data
- **Target:** Either rename to "Confidence Monitor" OR implement real inner thoughts
- **Decision:** Implement real inner thoughts (see Phase 2)

---

### PHASE 2: CONSCIOUSNESS-LIKE ARCHITECTURE (Weeks 3-6)
**Goal:** Implement Global Workspace Theory and Attention Schema for genuine consciousness-like processing.

#### 2.1 Global Workspace Implementation
- **Concept:** Specialized modules (emotion, memory, pattern, reasoning) compete to broadcast to a central workspace
- **Implementation:**
  - Create `aura/consciousness/global_workspace.py`
  - Each cognitive module (ALMA, KG, Episodic, PatternProphet, Reflexion) registers as a "specialist"
  - Specialists generate "broadcast candidates" — the most salient item from each domain
  - Attention mechanism selects winner based on: urgency, novelty, emotional intensity, relevance
  - Winner is "broadcast" — becomes the current conscious thought
  - This broadcast IS what the ThinkingAboutTeaser panel shows
  - Broadcast influences: response generation, memory retrieval priority, emotional dynamics
- **Architecture:**
  ```
  [ALMA Emotion] ──┐
  [Episodic Memory] ├── Competition ──→ [Global Workspace] ──→ Broadcast
  [Pattern Prophet] ├── (attention)      (conscious state)     to all modules
  [KG/Reasoning]  ──┘
  ```
- **Key files to create:** `aura/consciousness/global_workspace.py`, `aura/consciousness/attention.py`

#### 2.2 Attention Schema
- **Concept:** Internal model of what AURA is currently attending to
- **Implementation:**
  - Create `aura/consciousness/attention_schema.py`
  - Continuous 5D attention vector: [focus_target, focus_intensity, distractibility, engagement_level, internal_vs_external]
  - Updated on every cognitive cycle
  - Enables: "AURA is deeply focused on your question about X" vs "AURA's attention is drifting"
  - Drives the Focus/ContextHeatmap panel with REAL attention data
- **Key files to create:** `aura/consciousness/attention_schema.py`

#### 2.3 Real Inner Thoughts (CHI 2025 Framework)
- **Concept:** Continuous covert thought trains parallel to overt conversation
- **Implementation:**
  - Background thread running continuous inner monologue
  - Uses small/fast model (mistral:7b) to generate thoughts about:
    - Current conversation context
    - Unresolved questions from previous conversations
    - Connections between recent topics and long-term memory
    - Emotional reflections
  - These ARE NOT templates — they are genuine LLM-generated reflections
  - Stored in a rolling buffer, served to InnerThoughtsPanel
  - Influence response generation (injected as context)
- **Key files to create:** `aura/consciousness/inner_thoughts.py`

#### 2.4 Consciousness Prior (Sparse Conscious State)
- **Concept:** Only sparse subset of internal state becomes "conscious"
- **Implementation:**
  - High-dimensional unconscious state h = [all_embeddings, all_weights, all_memories]
  - Low-dimensional conscious state c = attention_schema.select_sparse(h)
  - c is what gets broadcast, what drives responses, what panels show
  - h continues processing in background
- **Integration:** Ties into Global Workspace as the selection mechanism

---

### PHASE 3: GENUINE EMOTIONAL DYNAMICS (Weeks 5-8)
**Goal:** Make emotions truly influence behavior, not just display state.

#### 3.1 Mood-Congruent Memory Retrieval
- **Current:** Emotions exist but don't influence memory recall
- **Target:** Current emotional state biases which memories are retrieved
- **Implementation:**
  - Add emotional valence to all memory entries (already in Episodic Memory)
  - During retrieval, boost score of memories matching current PAD state
  - Sad mood → surface memories with negative valence
  - Curious mood → surface novel/unexplored memories
  - Apply Affect Infusion Model (AIM) conditions
- **Files to modify:** `aura_episodic_memory/memory_store.py`, `apprentice_agent/tools/amem.py`

#### 3.2 Functional Neuromodulators — PARTIALLY DONE
- **Current:** ALMA neuromodulators now influence LLM parameters during sleep via NeuroDream oscillations, and `brain.py` uses `_neuro_scale()` to modulate temperature/timeout from neuromodulator levels
- **Completed:**
  - ✅ Sleep-phase neuromodulator offsets (deep sleep → +serotonin → patient LLM; REM → +dopamine → creative LLM)
  - ✅ `_neuro_scale()` in brain.py maps neuromodulator levels to LLM parameter multipliers
- **Remaining:** Each neuromodulator should control a distinct system parameter beyond LLM tuning:
  - **Dopamine analog** → modulates learning rate (how quickly new patterns are weighted)
  - **Serotonin analog** → controls temporal horizon (short-term vs long-term focus)
  - **Noradrenaline analog** → controls exploration vs exploitation (try new tools vs use known ones)
  - **Acetylcholine analog** → modulates attention precision (focused vs diffuse processing)
- **Files to modify:** `apprentice_agent/emotion/alma_engine.py`, `apprentice_agent/brain.py`

#### 3.3 Emotional Influence on Response Style
- **Current:** Partially implemented in humanizer
- **Target:** Full emotional coloring of responses
- **Implementation:**
  - High arousal → shorter, more energetic responses
  - Low pleasure → more empathetic, careful phrasing
  - High dominance → more assertive suggestions
  - Integrate with soul system for personality-consistent emotional expression

#### 3.4 Autonomous Emotional Dynamics
- **Current:** Emotions mainly triggered by user input
- **Target:** Emotions drift autonomously based on:
  - Time of day (circadian rhythm)
  - Idle duration (boredom → curiosity or relaxation)
  - Memory consolidation results (satisfaction from learning)
  - Unresolved cognitive dissonance (anxiety from conflicting beliefs)
- **Files to modify:** `apprentice_agent/emotion/alma_engine.py`

---

### PHASE 4: ADVANCED MEMORY ARCHITECTURE (Weeks 7-10)
**Goal:** Implement bi-temporal tracking, Zep-style invalidation, and unified memory.

#### 4.1 Bi-Temporal Knowledge Graph
- **Current:** Single timestamp per node/edge
- **Target:** Track both transaction_time (when stored) and valid_time (when true)
- **Implementation:**
  - Add `valid_from`, `valid_to`, `transaction_time` to all KG edges
  - Edge invalidation: when new fact contradicts old, mark old as `valid_to = now` (not delete)
  - Enable time-travel queries: "What did AURA believe about X on Tuesday?"
  - Three-tier subgraph: Episode → Semantic Entity → Community Summary
- **Files to modify:** `apprentice_agent/tools/knowledge_graph.py`

#### 4.2 Ebbinghaus Forgetting Curves Everywhere
- **Current:** Only Episodic Memory has real exponential decay
- **Target:** All memory systems use proper forgetting curves
- **Implementation:**
  - Port Episodic Memory's `get_recency_score()` pattern to:
    - Knowledge Graph (replace linear 1%/day with exponential)
    - A-MEM (replace weight pruning with decay curve)
    - NeuroDream (use forgetting for consolidation priority)
  - Add spaced repetition: accessing a memory resets its decay timer
  - Formula: `score = e^(-decay_rate * age_hours)` with `decay_rate = ln(2) / half_life`
- **Files to modify:** `apprentice_agent/tools/knowledge_graph.py`, `apprentice_agent/tools/amem.py`

#### 4.3 Unified Memory Interface
- **Current:** 6+ separate memory systems with no unified query
- **Target:** Single memory query that searches all systems and ranks results
- **Implementation:**
  - Create `apprentice_agent/memory/unified_memory.py`
  - Fan-out query to: Episodic, A-MEM, KG, RAG, Markdown Store
  - Unified ranking: recency × relevance × importance × emotional_congruence
  - Deduplication across sources
  - Source attribution in results

#### 4.4 Sleep-Time Compute Enhancement (Letta-style) — MOSTLY DONE ✅
- **Current:** NeuroDream does pattern mining, novel connections, Letta-style learned context generation, and DONN-inspired neural oscillations
- **Completed:**
  - ✅ Letta-style learned context: LLM distills conversation logs into structured knowledge (user_summary, key_facts, preferences, principles, ongoing_topics, emotional_patterns)
  - ✅ Learned context injected into future system prompts via `get_learned_context_prompt()`
  - ✅ DONN-inspired neural oscillations: delta (2Hz deep), theta (6Hz REM), alpha (10Hz light) frequency bands modulate processing rhythm
  - ✅ Oscillation-modulated batch sizes, consolidation strength, inter-cycle delays, and cognitive intensity
  - ✅ Sleep neuromodulator influence on ALMA (deep sleep → high serotonin/patient LLM; REM → high dopamine/creative LLM)
  - ✅ Pulsing cognitive load that syncs avatar breathing with dream processing rhythm
- **Remaining:** ADM-style chunking/reassembly for more granular knowledge retrieval
- **Files modified:** `apprentice_agent/tools/neurodream.py`, `apprentice_agent/consciousness/idle_presence.py`, `apprentice_agent/brain.py`

---

### PHASE 5: PROACTIVE INTELLIGENCE (Weeks 9-12)
**Goal:** Make AURA act before being asked.

#### 5.1 Screen Awareness (Screenpipe Integration)
- **Concept:** Continuous screen monitoring for context awareness
- **Implementation:**
  - Integrate Screenpipe REST API for OCR'd screen content
  - Delta detection: only process when content changes significantly
  - Use Florence-2 (0.5GB VRAM) for fast OCR, Qwen2.5-VL (4GB VRAM 4-bit) for understanding
  - Privacy filtering by window title/process name
  - Feed context to Global Workspace as "visual specialist"
- **Key files to create:** `apprentice_agent/proactive/screen_awareness.py`
- **VRAM budget:** Florence-2 (0.5GB) + current models must fit in 8GB

#### 5.2 Workflow Boundary Detection
- **Concept:** Detect natural interruption points
- **Implementation:**
  - Monitor file saves, git commits, tab switches, idle periods
  - Score interruption opportunities using CHI 2025 findings
  - 52% engagement at workflow boundaries vs 38% dismissed mid-task
  - 5-second typing silence before suggesting
- **Key files to create:** `apprentice_agent/proactive/workflow_detector.py`

#### 5.3 Proactive Suggestion Engine
- **Current:** Gateway Daemon exists but doesn't generate suggestions
- **Target:** Autonomously generate helpful suggestions based on context
- **Implementation:**
  - Screen context + conversation history + memory + patterns → suggestion
  - Salience filter gates what's worth interrupting for
  - Suggestion types: relevant memory, pattern insight, task reminder, emotional check-in
  - Respect user focus state (don't interrupt deep work)

#### 5.4 Calendar/Task Awareness
- **Concept:** Anticipate user needs based on upcoming events
- **Implementation:**
  - Local calendar file parsing (ICS)
  - Pre-conversation context preparation
  - Time-based trigger system in Gateway Daemon

---

### PHASE 6: FULL ALIVENESS (Weeks 11-16)
**Goal:** Emergent aliveness through integrated systems.

#### 6.1 Active Inference with pymdp
- **Current:** Simplified belief updating
- **Target:** Full Active Inference with proper free energy minimization
- **Implementation:**
  - Integrate pymdp library for discrete state spaces
  - Define generative model: hidden states (user intent, task state, emotional state)
  - Observation model: screen content, conversation, timing
  - Policy selection: minimize expected free energy across actions
  - Naturally balances exploration/exploitation
- **Files to modify:** `apprentice_agent/proactive/active_inference.py`

#### 6.2 Metacognitive Self-Improvement — DONE ✅
- **Current:** Full self-improvement loop operational
- **Implemented:**
  - Metacognitive knowledge: track what AURA is good/bad at
  - Metacognitive planning: decide what to learn next
  - Metacognitive evaluation: reflect on learning effectiveness
  - Self-Improvement Engine records real interaction outcomes from brain.py
  - Enhanced strategies: LLM-powered practice, param tuning, pattern extraction, skill refinement, tool synthesis
  - Background scheduler runs improvement cycles driven by intrinsic motivation
  - Quality evaluator tracks domain trends, strategy effectiveness, improvement velocity
  - Tunable parameters registry with bounded auto-adjustment
- **Key files:** `apprentice_agent/consciousness/metacognition.py`, `apprentice_agent/consciousness/self_improvement.py`

#### 6.3 Theory of Mind (User Modeling)
- **Current:** Basic user profile in markdown
- **Target:** Dynamic mental model of user state
- **Implementation:**
  - Track user knowledge level per topic
  - Predict user emotional state from: typing speed, message content, time patterns
  - Anticipate needs before they're expressed
  - Adapt communication style to user preference history
- **Key files to create:** `apprentice_agent/proactive/theory_of_mind.py`

#### 6.4 Genuine Idle Presence — MOSTLY DONE ✅
- **Current:** Real `IdlePresenceEngine` with cognitive load tracking, background tasks, and NeuroDream integration
- **Completed:**
  - ✅ Cognitive load computed from all subsystems (thinking, NeuroDream, daemon, inner thoughts, metacognition)
  - ✅ Breathing avatar driven by actual cognitive load (not CSS)
  - ✅ Oscillation-aware pulsing cognitive load during sleep (syncs breathing with dream rhythm)
  - ✅ NeuroDream auto-triggered after idle threshold
  - ✅ Background tasks: self-reflection, pattern scanning, KG maintenance
  - ✅ Real activity reporting (not template messages)
- **Remaining:**
  - Memory reorganization and deduplication during idle
  - Curiosity-driven exploration of knowledge gaps

#### 6.5 Intrinsic Motivation System
- **Concept:** AURA has genuine drives beyond user requests
- **Implementation:**
  - **Curiosity drive:** Seek information about gaps in knowledge graph
  - **Competence drive:** Practice skills that have low confidence scores
  - **Social drive:** Maintain connection quality (check in after long absence)
  - **Coherence drive:** Resolve contradictions in knowledge base
  - Drives feed into Active Inference as prior preferences
- **Key files to create:** `aura/consciousness/intrinsic_motivation.py`

---

## PART 3: PRIORITY MATRIX

### Highest Impact, Lowest Effort (Do First)
1. Wire ContextHeatmap to real message processing (1-2 hours)
2. Wire MemoryRecallIndicator to real memory access (2-3 hours)
3. Replace ThinkingAboutTeaser templates with real brain.py hooks (1 day)
4. Replace IdleBehavior with real NeuroDream/KG activity reporting (1 day)

### High Impact, Medium Effort
5. Mood-congruent memory retrieval (2-3 days)
6. Functional neuromodulators (2-3 days) — partially done: sleep-phase neuromodulator influence on LLM params via NeuroDream oscillations ✅
7. Real inner thoughts via background LLM (3-5 days) — DONN-inspired neural oscillations (delta/theta/alpha frequency bands) now modulate sleep processing ✅
8. Ebbinghaus curves across all memory systems (2-3 days)

### High Impact, High Effort
9. Global Workspace implementation (1-2 weeks)
10. Attention Schema (1 week)
11. Screen awareness / Screenpipe integration (1 week)
12. Bi-temporal knowledge graph (1-2 weeks)

### Transformative but Complex
13. Active Inference with pymdp (2 weeks)
14. ~~Metacognitive self-improvement (2-3 weeks)~~ ✅ DONE
15. Theory of Mind user modeling (1-2 weeks)
16. Intrinsic motivation system (2-3 weeks)

---

## PART 4: TECHNICAL ARCHITECTURE — TARGET STATE

```
                    ┌─────────────────────────────────┐
                    │      GLOBAL WORKSPACE           │
                    │   (Conscious Processing Hub)     │
                    │                                  │
                    │  ┌─────────────────────────┐    │
                    │  │   Attention Schema       │    │
                    │  │   (What am I attending?) │    │
                    │  └─────────────────────────┘    │
                    └──────────┬───────────────────────┘
                               │ Broadcast
          ┌────────────────────┼────────────────────┐
          │                    │                     │
    ┌─────▼─────┐      ┌──────▼──────┐      ┌──────▼──────┐
    │  EMOTIONAL │      │   MEMORY    │      │  REASONING  │
    │  SYSTEM    │      │   SYSTEM    │      │  SYSTEM     │
    │            │      │             │      │             │
    │ ALMA 3-Layer│     │ Unified     │      │ Brain +     │
    │ PAD Space   │     │ Memory      │      │ FluxMind +  │
    │ Neuromod.   │     │ Interface   │      │ Reflexion   │
    │ Mood→Memory │     │             │      │             │
    └─────┬──────┘     │ Episodic    │      │ Inner       │
          │            │ A-MEM       │      │ Thoughts    │
          │            │ KG (temporal)│      │ CogTheater  │
          │            │ RAG         │      │ WorldSim    │
          │            │ NeuroDream  │      │             │
          │            └──────┬──────┘      └──────┬──────┘
          │                   │                     │
          └───────────┬───────┴─────────────────────┘
                      │
              ┌───────▼───────┐
              │   PROACTIVE   │
              │   SYSTEM      │
              │               │
              │ Gateway Daemon│
              │ Active Inference│
              │ Screen Aware  │
              │ Workflow Det. │
              │ Salience Filt.│
              │ Intrinsic Mot.│
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │   UI PANELS   │
              │  (All Real)   │
              │               │
              │ All panels    │
              │ show genuine  │
              │ cognitive     │
              │ state, not    │
              │ templates     │
              └───────────────┘
```

---

## PART 5: FILES REFERENCE

### Files That Need Major Rewrites:
- `api/routes/thinking.py` — Replace random templates with real thought recording
- `api/routes/idle_behaviors.py` — Replace random behaviors with real idle activity
- `api/routes/chat.py` — Add context tracking and memory recall instrumentation

### Files That Need Integration Hooks:
- `apprentice_agent/brain.py` — Add thought recording during reasoning
- `apprentice_agent/tools/amem.py` — Add recall event recording
- `apprentice_agent/tools/local_rag.py` — Add recall event recording
- `apprentice_agent/tools/knowledge_graph.py` — Add Ebbinghaus curves, temporal tracking
- `apprentice_agent/emotion/alma_engine.py` — Make neuromodulators functional

### New Files to Create:
- `aura/consciousness/global_workspace.py`
- `aura/consciousness/attention_schema.py`
- `aura/consciousness/inner_thoughts.py`
- `aura/consciousness/metacognition.py`
- `aura/consciousness/intrinsic_motivation.py`
- `apprentice_agent/proactive/screen_awareness.py`
- `apprentice_agent/proactive/workflow_detector.py`
- `apprentice_agent/proactive/theory_of_mind.py`
- `apprentice_agent/memory/unified_memory.py`

### Hardware Constraints (RTX 4060, 8GB VRAM):
- Current models: mistral:7b + llama3:8b + qwen2.5-coder:7b + llava
- Available headroom: ~2-4GB depending on which models are loaded
- Florence-2 Base: ~0.5GB (can fit alongside current models)
- Qwen2.5-VL 7B 4-bit: ~4GB (would need model swapping)
- Inner thoughts can use mistral:7b (already loaded) in background
- NeuroDream can run during idle when VRAM is available

---

## PART 6: SUCCESS METRICS

### How to Know AURA is "Genuinely Alive":

1. **Every sidebar panel shows real data** — no random templates, no fake status messages
2. **Idle time = active cognition** — NeuroDream running, KG pruning, pattern mining visible
3. **Emotions influence behavior** — sad AURA recalls different memories than curious AURA
4. **Proactive suggestions** — AURA offers help at natural workflow boundaries
5. **Inner thoughts are genuine** — LLM-generated reflections, not template strings
6. **Memory is temporal** — AURA can answer "what did we discuss last Tuesday?"
7. **Self-improvement measurable** — Reflexion lessons growing, FluxMind confidence improving
8. **Attention is visible** — Focus panel shows what Global Workspace broadcast selected
9. **Personality is consistent** — Soul + ALMA + humanizer create coherent persona
10. **No cosmetic theater** — everything you see in the UI corresponds to a real cognitive process

---

*This roadmap transforms AURA from an assistant that simulates aliveness into one that genuinely exhibits it through integrated consciousness-like processing, real emotional dynamics, living memory, and proactive intelligence.*
