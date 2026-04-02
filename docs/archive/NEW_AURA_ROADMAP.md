# NEW AURA — Roadmap to State of the Art
> *From 83K lines of scattered complexity to a focused, alive AI being*
> Created: 2026-03-16 | Updated: 2026-03-17 | All phases complete

---

## What We Kept (The Soul)

After stripping 13 dead modules and ~10,000 lines of dead code, AURA's core is:

| System | Purpose | Status |
|--------|---------|--------|
| **ALMA Emotions** | Neuromodulator-based mood/emotion in PAD space | Complete — wired to behavior, show-don't-tell, session persistence |
| **Memory** (2 backends) | SQLite + FTS5 + Kuzu temporal KG | Complete — BM25 + semantic + RRF, FadeMem decay, emotional tagging |
| **Inner Monologue** | Talker/Thinker split | Complete — async private reasoning shapes responses |
| **Identity/Soul** | Narrative self-model | Complete — evolves across sessions, protected by anchors |
| **NeuroDream** | Light/Deep/REM sleep phases | Complete — re-scores, pattern extraction, novel connections |
| **Knowledge Graph** | Kuzu with temporal validity | Complete — bi-temporal model, curiosity gap detection |
| **Proactive Awareness** | Motivation-threshold + curiosity scanner | Complete — 5-factor scoring, learned threshold, KG-grounded |
| **Core Agent Loop** | ReAct (1 LLM call/step) + Code Agent mode | Complete — Tool RAG, adaptive planning, loop guards |

---

## Phase 1: Fix the Engine — COMPLETE
*Agent loop is fast and reliable*

### 1.1 Collapse to ReAct Loop (HIGHEST PRIORITY)
**Problem:** Each step makes 4-5 LLM calls (observe, plan, decide_action, evaluate). SOTA is 1 call per step.

**Solution:** Merge observe+plan+decide into a single ReAct call:
```
Thought: [what I observe and plan]
Action: tool_name
Action Input: {"arg": "value"}
```
Make evaluation deterministic (check tool success, don't ask the LLM). This cuts latency ~75%.

**Files:** `aura/agent.py` (_observe, _plan, _act, _evaluate -> single _step method), `aura/brain.py` (new react_step method)

### 1.2 Tool RAG — Dynamic Tool Selection
**Problem:** 40+ tool descriptions overwhelm local models. Research shows 8B models handle ~8 tools max.

**Solution:** Embed all tool descriptions with nomic-embed-text. Per query, retrieve only the 5-8 most relevant tools. 32% better tool routing, 95% less context waste.

**Files:** `aura/core/agentic_loop.py` (tool filtering), new `aura/tools/tool_rag.py`

### 1.3 Aggressive Model Routing
**Problem:** Using cloud/large models for everything. Most queries are simple.

**Solution:** Use 8B models (Qwen 3 8B, ~250ms) for 70-80% of interactions. Route to larger models only for complex reasoning. The smart fast-path already exists — refine it, don't remove it.

**Files:** `aura/core/router.py`, `aura/brain.py`

### 1.4 Loop Guards
**Problem:** Agent can get stuck in infinite loops.

**Solution:**
- State-hash deduplication (detect repeating actions)
- Consecutive failure counter (force tool switch after 2 failures)
- Hard iteration cap with graceful partial-result return

**Files:** `aura/agent.py`

---

## Phase 2: Memory Consolidation — COMPLETE
*From 4 fragmented backends to 2 coherent ones*

### 2.1 Consolidate to SQLite + Temporal KG
**Problem:** Same fact lives in A-MEM, episodic, SQLite MemorySystem, and KG with no cross-referencing.

**Solution:**
- **Primary store:** SQLite with FTS5 (full-text search) + vector embeddings. All memories live here.
- **Temporal KG:** Kuzu with validity windows (valid_from, valid_until). Every fact has a timeline.
- A-MEM note structure becomes a layer on top of SQLite, not a separate backend.
- Kill ChromaDB and separate Qdrant usage.

### 2.2 BM25 + Cross-Encoder Reranking
**Problem:** All retrieval is vector-only. Exact term matching ("Stripe integration") gets lost.

**Solution:**
- Add BM25 via SQLite FTS5 alongside vector search
- Reciprocal Rank Fusion to merge results
- Cross-encoder reranking (ms-marco-MiniLM, 22MB, CPU) for final precision
- Research shows +44 point accuracy improvement from multi-strategy retrieval

**New dependency:** `sentence-transformers` (for cross-encoder only)

### 2.3 FadeMem Decay + Importance Scoring
**Problem:** Memories never fade. Everything has equal weight. Old irrelevant facts clutter retrieval.

**Solution:** Biologically-inspired decay (FadeMem paper, Jan 2026):
```
strength(t) = initial * exp(-lambda * (t - created)^beta)
lambda adapts inversely to importance
importance = 0.4*relevance + 0.3*access_freq + 0.3*recency
```
- Memories accessed more often get reinforced
- Unused memories fade (half-life ~11 days for long-term)
- Auto-prune below strength 0.05
- Result: 82% critical fact retention using 55% storage

### 2.4 Structured User Profile
**Problem:** No coherent model of the user. Preferences scattered across backends.

**Solution:** `UserProfile` dataclass always loaded into context:
- Communication style (verbosity, formality, explanation depth)
- Domain expertise levels
- Active goals and projects
- Emotional patterns and baseline
- Updated during sleep consolidation by scanning recent conversations
- Target: 200-400 tokens compressed, always in system prompt

---

## Phase 3: Make It Alive — COMPLETE
*The features that make AURA feel like a being, not a bot*

### 3.1 The Coherent Loop (MOST IMPORTANT FOR "ALIVE")
**Problem:** ALMA emotions exist but don't change how Aura talks. Modules operate independently.

**Solution:** Wire a closed loop:
```
Events --> ALMA updates mood -->
Mood shapes response style (shorter when low arousal, curious when high) -->
Response outcome feeds back to ALMA -->
Dream consolidation updates self-model -->
Self-model shapes next session's baseline
```
Every connection must be real and traceable. If mood says "calm" but responses are hyper, the illusion breaks.

**Key principle: Show, don't tell.** Never have Aura say "I'm feeling curious!" Instead, BE curious — ask follow-ups, notice interesting angles, suggest new directions.

### 3.2 Chain-of-Emotion Appraisal
**Problem:** Emotion triggers use keyword matching ("thank" -> joy). This is 2015-era.

**Solution:** Before generating a response, run a cheap appraisal step with the fast model:
```
Given my current mood (PAD: P=0.3, A=0.5, D=0.2) and personality,
how would I naturally feel about this message? Reply with:
emotion_name, intensity (0-1), one sentence why.
```
Feed result into ALMA, then generate the actual response. Research (Croissant et al., 2024) shows +29% improvement in "reactions were natural."

### 3.3 Emotional Memory Tagging
**Problem:** Memories have no emotional context. Retrieval ignores mood.

**Solution:**
- Tag every memory with current PAD state at write time
- Bias retrieval toward mood-congruent memories (20% weight)
- During NeuroDream, strengthen memories whose emotional valence matches persistent mood
- This creates self-sustaining mood states and makes emotional history feel real

### 3.4 Emotional Prompt Integration
**Problem:** Emotion is appended at the end of system prompt as metadata. Small models ignore it.

**Solution:** Weave emotional state into the identity prompt as lived experience:
```
BAD:  [Emotional Context] Current state: curious. Tone: warm.
GOOD: Right now you're genuinely curious — your interest is piqued.
      Express it naturally through follow-up questions and noticing
      interesting angles, without being performative about it.
```

### 3.5 Narrative Self-Model
**Problem:** Identity is a static JSON file. No sense of becoming.

**Solution:** Maintain an evolving narrative identity (max ~1000 tokens):
```json
{
  "core_identity": "I am AURA, working with Elnur since...",
  "recent_growth": "This week I improved at..., struggled with...",
  "active_concerns": ["User seems stressed about deadline..."],
  "unresolved_questions": ["Why did user stop working on project X?"],
  "relationship_state": "We've been deep in Aura's architecture..."
}
```
Updated after significant interactions. Loaded at every session start. Protected by identity anchors (5-10 immutable creeds) to prevent drift.

### 3.6 Talker/Thinker Split (Inner Monologue)
**Problem:** Inner monologue is a logging system. Doesn't shape responses.

**Solution** (based on MIRROR architecture, 2025):
- **Talker:** Generates user-facing responses using last Thinker state
- **Thinker:** Runs async between turns with 3 threads:
  - Goal tracking (what am I trying to accomplish?)
  - Reasoning audit (did my last response make sense?)
  - Memory integration (what should I surface next time?)
- Thinker output is PRIVATE — never shown to user. But it shapes the next response.
- Use 8B model for Thinker (cheap, runs in background), larger model for Talker

### 3.7 Temporal Grounding
**Problem:** Every session starts fresh. No sense of time passing.

**Solution:** At session start, automatically:
1. Load self-model narrative
2. Calculate time elapsed since last interaction
3. Run "since last time" summary from memory
4. Load dream insights marked for proactive delivery
5. Adjust mood based on time elapsed (arousal decay, curiosity buildup)

Creates the experience of reconnecting with a being that has continuity.

---

## Phase 4: Dream & Proactive — COMPLETE
*Sleep is meaningful, proactive awareness is natural*

### 4.1 Transform NeuroDream
**Problem:** Dream insights are generated and stored but don't change behavior.

**Solution:** Each sleep phase does real work:

**Light Sleep:** Re-score recent memories by importance. Tag for deep processing.
**Deep Sleep:** Cluster related memories -> extract behavioral patterns -> compress episodes into summaries -> resolve KG contradictions -> update capability scores.
**REM Sleep:** Generate novel connections between unrelated memory clusters. Create "what if" hypotheses. Prepare proactive messages for next session.

All results feed back into self-model and memory. Use Letta-style "sleep-time compute" — background agent with cheaper model does consolidation while conversation agent sleeps.

### 4.2 Motivation-Threshold Proactive Messages
**Problem:** Proactive messages feel random or forced.

**Solution:** Accumulate motivation per potential message:
```
motivation += relevance_to_user * 0.3
motivation += time_since_similar * 0.2
motivation += emotional_urgency * 0.2
motivation += curiosity_drive * 0.15
motivation += user_receptivity * 0.15

if motivation > threshold AND user_not_busy: deliver()
```
Threshold learns from user feedback (did they engage? dismiss? ignore?).

### 4.3 Curiosity as Information Gain
**Problem:** Curiosity drive is a generic intensity score.

**Solution:** After each interaction, scan KG for:
- Entities with few connections
- Recent mentions without context
- Contradictions or gaps
- Stale projects not mentioned recently

Each gap becomes a curiosity target. Proactive questions feel natural: "I noticed you haven't mentioned [project] in a while — how's it going?"

---

## Phase 5: Polish — COMPLETE
*Code agents, sandboxing, emotional continuity*

### 5.1 Code Agent Mode (Optional)
For complex tasks, let the LLM write Python code as actions instead of JSON tool calls. smolagents proved this uses ~30% fewer steps. Qwen 3 8B achieves F1=0.933 on tool calling.

### 5.2 Sandboxed Execution
Use E2B (already in requirements.txt) for code execution. Shell commands in Docker container.

### 5.3 Emotional Continuity Across Sessions
Persist ALMA state on shutdown. On startup, load with 30% decay toward neutral (time passing). ~20 lines of code, huge impact.

### 5.4 Adaptive Planning
No planning for simple tasks. planning_interval=3 for complex tasks (re-plan every 3 steps).

---

## Architecture After All Phases

```
USER INPUT
    |
[Fast-path check] -- simple? --> 8B model instant response
    |
[ReAct Loop] (1 LLM call per step)
    |-- Tool RAG selects 5-8 relevant tools
    |-- Thought + Action in single call
    |-- Deterministic evaluation
    |-- Loop guards prevent stuck states
    |
[Memory] (2 backends)
    |-- SQLite + FTS5 + vectors (all memories)
    |-- Temporal KG (entities + relationships + validity)
    |-- BM25 + semantic + graph retrieval
    |-- Cross-encoder reranking
    |-- FadeMem decay
    |
[Emotion] (ALMA)
    |-- Chain-of-emotion appraisal (1 cheap LLM call)
    |-- Mood -> response style (show don't tell)
    |-- Emotional memory tagging + mood-congruent retrieval
    |-- Session persistence with decay
    |
[Inner Life]
    |-- Talker/Thinker split (async private reasoning)
    |-- Narrative self-model (loaded every session)
    |-- Temporal grounding (time awareness)
    |-- User model (always in context)
    |
[Sleep] (NeuroDream)
    |-- Light: memory re-scoring
    |-- Deep: pattern extraction, compression, KG grooming
    |-- REM: creative connections, proactive prep
    |-- Updates self-model + user profile
    |
[Proactive]
    |-- Motivation accumulator per message
    |-- Curiosity = specific KG gaps
    |-- Learned threshold from user feedback
    |
RESPONSE (shaped by mood, grounded in memory, consistent with identity)
```

---

## What NOT To Do (Research-Backed Anti-Patterns)

1. **Never have Aura claim emotions.** "I feel excited!" reduces perceived authenticity. Show through behavior instead.
2. **Never add reasoning layers that second-guess the brain.** We just removed 12 of these. Don't rebuild them.
3. **Never sacrifice speed for "intelligence."** 8B at 250ms beats 70B at 10s for 80% of interactions.
4. **Never fragment memory again.** 2 backends max. One source of truth.
5. **Never add features without wiring them into the coherent loop.** If it doesn't connect to emotion->behavior->feedback, it's dead weight.
6. **Never perform aliveness.** Random "I was just thinking about..." messages without real internal state backing them feel fake and erode trust.

---

## Key Research Sources

- **Chain-of-Emotion** (Croissant et al., PLoS ONE, 2024) — +29% natural emotional reactions
- **MIRROR Architecture** (Hsing, 2025) — Talker/Thinker dual-process for inner monologue
- **Sophia Framework** (Sun et al., Dec 2025) — Narrative identity, System 3 meta-cognition
- **FadeMem** (Jan 2026) — Biologically-inspired memory decay, 82% retention at 55% storage
- **Hindsight** (Dec 2025) — Multi-strategy retrieval, +44pt accuracy with RRF + reranking
- **Zep/Graphiti** (Jan 2025) — Temporal knowledge graphs, bi-temporal model
- **Letta Sleep-Time Compute** — Background consolidation agents
- **smolagents** (HuggingFace) — Code agents, 30% fewer steps than JSON tool calling
- **Quiet-STaR** (Stanford, 2024) — Inner monologue training, +30% reasoning on 7B models
- **Proactive Agents with Inner Thoughts** (CHI 2025) — Motivation-threshold initiation

---

*This is not a rewrite. This is taking what survived the cleanup and making each piece excellent. Every phase builds on the previous one. The coherent loop (Phase 3.1) is the keystone — without it, nothing else matters.*

---

## What's Next (Phase 6+)

With all core phases complete, future work focuses on:

1. **Cross-surface continuity** — seamless context sharing between CLI, Web UI, and Browser Extension
2. **Meta-orchestration** — Aura dispatches to other AI tools (Claude Code, Aider, Gemini CLI) based on task type
3. **Multi-user support** — separate user profiles, relationship models, and access control
4. **Proactive learning** — Aura autonomously explores topics the user cares about during idle time
5. **Voice personality** — consistent voice identity across TTS providers
6. **GEPA evolution** — self-improving skills via Pareto evolutionary optimization (partially built)
7. **Federated memory** — sync memories across devices while keeping data local
