# Aura Android — MoA Integration Plan

> **For Hermes:** Plan + execute in this session. One atomic commit per item. Verify with `./gradlew :app:assembleDebug` per commit, full test suite every 3 commits. No "should I continue?" between items.

**Goal:** Integrate Mixture of Agents into Aura Android as a first-class product feature — visible, selectable, and adaptive. Not a hidden quality knob.

**Current state:** MoA preset `moa/default` is configured in Hermes config (deepseek-v4-pro aggregator + glm-5.2 + kimi-k2.7-code references). It works as a Hermes virtual provider. Aura Android needs to wire it in.

**MoA preset reference:**
- Aggregator: `deepseek:deepseek-v4-pro` — writes responses, emits tool calls
- Reference 1: `ollama-cloud:glm-5.2` — broad reasoning, semantic analysis
- Reference 2: `ollama-cloud:kimi-k2.7-code` — code analysis, technical precision

---

## Pre-execution verification

- [ ] Grep `moa`, `mixture.of.agents`, `deepMode` across `android/` — confirm nothing exists yet
- [ ] Check `ProviderRegistry` for `moa` provider registration pattern
- [ ] Verify gradle builds clean before any changes

---

## Item 1: MoA Provider Registration in Aura Android

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/providers/MoaProvider.kt`
- Modify: `android/aura-core/src/main/kotlin/com/aura/providers/ProviderModule.kt`

**What:** Register a `moa` provider that appears in the model picker. The MoA provider delegates to the Hermes MoA runtime (or, for v1, reimplements the reference-model → aggregator pattern natively). 

**Design decision:** For v1, MoA is a **client-side pattern** — the Android app itself calls reference models in parallel, then calls the aggregator with reference outputs appended. This avoids a Hermes server dependency and works fully on-device.

**MoaProvider architecture:**
```
MoaProvider implements Provider {
  - referenceModels: List<ProviderRef>  // ollama-cloud:glm-5.2, ollama-cloud:kimi-k2.7-code
  - aggregatorModel: ProviderRef        // deepseek:deepseek-v4-pro
  - aggregatorProvider: Provider        // resolved from ProviderRegistry
  
  chat(model, messages, options, tools):
    1. Strip tool schemas from reference messages (save tokens, avoid strict-provider rejections)
    2. Call reference models IN PARALLEL via coroutineScope { async {...} }
    3. Collect reference outputs as text strings
    4. Build aggregator messages: original messages + last user message amended with:
       "[MoA Reference Analysis]\nModel A (glm-5.2): {output}\nModel B (kimi-k2.7-code): {output}"
    5. Call aggregator provider with FULL tool schemas
    6. Stream aggregator response as normal
    7. On failure of one reference, include error in context, continue
}
```

**Commit:** `feat(android): MoA provider — parallel reference models + aggregator synthesis`

---

## Item 2: Deep Mode Chip in ChatScreen

**Files:**
- Modify: `android/app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt`
- Modify: `android/app/src/main/kotlin/com/aura/ui/viewmodel/ChatViewModel.kt`

**What:** A chip/button next to the model picker that temporarily enables MoA for the next turn only.

**ChatUiState additions:**
```kotlin
val deepModeEnabled: Boolean = false,  // true = next turn uses MoA
val deepModeActive: Boolean = false,   // true = current turn IS MoA
```

**ChatViewModel additions:**
```kotlin
fun toggleDeepMode() {
    _state.update { it.copy(deepModeEnabled = !it.deepModeEnabled) }
}
// In the send path: if deepModeEnabled, use "moa:default" model,
// then after turn completes, set deepModeActive = false, deepModeEnabled = false
```

**ChatScreen UI:**
- FilterChip next to model picker: "🚀 Deep" with gradient/purple tint when enabled
- When `deepModeActive`: show animated thinking indicator — 3 small pulsing dots with model labels (glm / kimi / deepseek)
- When response arrives with MoA: subtle "✨ 3 models contributed" badge under the last message bubble
- Tapping the badge shows a bottom sheet: "This response was synthesized from 3 AI models — glm-5.2 (reasoning), kimi-k2.7-code (code analysis), deepseek-v4-pro (final response)."

**Commit:** `feat(android): Deep Mode chip — one-shot MoA with live thinking indicator`

---

## Item 3: MoA Thinking Visibility Indicator

**Files:**
- Modify: `android/app/src/main/kotlin/com/aura/ui/screens/ChatScreen.kt`
- Create: `android/app/src/main/kotlin/com/aura/ui/components/MoaThinkingIndicator.kt`

**What:** A composable that renders during MoA streaming:
- 3 small circular avatars with model initials (G/K/D)
- Pulsing/animated borders while thinking
- Checkmark overlay when each model completes
- All 3 checkmarked = "Synthesizing..." then aggregator response streams normally

**MoaThinkingIndicator composable:**
```kotlin
@Composable
fun MoaThinkingIndicator(
    referenceStates: List<ModelThinkingState>,  // Thinking, Done, Failed
    aggregatorState: AggregatorState,           // Waiting, Streaming, Done
)
```

**States:**
- REFERENCE PHASE: "Consulting 3 models..." with 3 pulsing avatars
- AGGREGATOR PHASE: "Synthesizing..." with 1 avatar + streaming text
- DONE: small "✨ MoA" badge, tappable for detail

**Commit:** `feat(android): MoA thinking indicator — live model status visualization`

---

## Item 4: Adaptive MoA Escalation

**Files:**
- Modify: `android/aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt`

**What:** The agent loop detects struggle patterns and auto-escalates to MoA.

**Detection heuristics:**
1. 3+ consecutive tool call failures (ToolResult.Error)
2. User sends corrective message ("no", "wrong", "that's not right", "try again") on the last turn
3. Model emits same tool call twice in a row (tool call loop detection)
4. Model response contains uncertainty markers ("I'm not sure", "I'm uncertain", "let me try a different approach")

**When any heuristic fires:**
```
escalationReason = "3 consecutive tool failures" (or whichever)
log("MoA escalation triggered: $escalationReason")
modelForNextTurn = "moa:default"
deepModeActive = true  // show thinking indicator
```

**After MoA turn:**
- If MoA turn succeeds → revert to normal model, log improvement
- If MoA turn also fails → surface error to user with escalation context

**Commit:** `feat(android): adaptive MoA escalation on detected struggle`

---

## Item 5: Morning Brief MoA

**Files:**
- Modify: `android/aura-core/src/main/kotlin/com/aura/proactive/MorningBriefWorker.kt`

**What:** MorningBriefWorker uses MoA instead of the first configured solo provider.

**Change:** Instead of `provider.chat(model, ...)`, construct a `MoaProvider` instance (or inject it) and call `moaProvider.chat(...)`. The MoA provider handles parallel reference calls + aggregator synthesis.

**Optional enhancement:** Add a `useMoa: Boolean` flag to the worker's input data so it can be toggled via WorkManager constraints.

**Commit:** `feat(android): morning brief via MoA for richer daily summaries`

---

## Item 6: MoA Onboarding Moment

**Files:**
- Modify: `android/app/src/main/kotlin/com/aura/ui/screens/OnboardingScreen.kt`

**What:** On the final onboarding page ("All set!"), add a section about MoA.

**Content:**
- Visual: 3 small model icons (glm, kimi, deepseek) arranged in a triangle
- Text: "Aura can use MoA — 3 AI models working together on hard problems."
- [Try Demo] button: sends a quick query through MoA and shows the response inline
- Skip link: "I'll try it later"

**Commit:** `feat(android): MoA introduction in onboarding flow`

---

## Item 7: MoA for Memory Consolidation & KG Extraction

**Files:**
- Modify: `android/aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt` (consolidation path)
- Modify: `android/aura-core/src/main/kotlin/com/aura/kg/ConversationKgExtractor.kt`

**What:** Periodically use MoA for memory consolidation and KG extraction.

**Memory consolidation MoA:**
- When the loop triggers consolidation (every N turns), use MoA for the summarization call
- MoA is better at: identifying which memories to merge, writing accurate summaries, recognizing when a memory is obsolete

**KG extraction MoA:**
- `ConversationKgExtractor` already has debounce (5s) and throttling
- When extraction fires, use MoA for richer triple extraction:
  - glm-5.2: semantic relationships, topics, concepts
  - kimi-k2.7-code: code entities, file references, technical terms
  - deepseek: merge into graph edges, resolve conflicts

**Commit:** `feat(android): MoA for memory consolidation and KG extraction`

---

## Item 8: SpecialistRouter MoA for Task Decomposition

**Files:**
- Modify: `android/aura-core/src/main/kotlin/com/aura/agent/SpecialistRouter.kt`

**What:** When routing a complex task, use MoA for decomposition quality.

**When to use MoA for decomposition:**
- Task requires 3+ subtasks
- Task spans multiple specialist types (code + research + planning)
- User explicitly asked for "plan this" or "break this down"

**Implementation:**
- `SpecialistRouter.route(task)` checks complexity heuristics
- If complex → `moaProvider.chat(decompositionPrompt, ...)` 
- Parse the MoA output into a structured task list
- Route individual tasks to specialists

**Commit:** `feat(android): MoA-powered specialist task decomposition`

---

## Execution Order

```
Commit 1: Item 1 — MoA Provider (foundation, everything depends on it)
Commit 2: Item 2 — Deep Mode chip (highest user impact per LOC)
Commit 3: Item 3 — MoA thinking indicator (visual polish)
Commit 4: Item 5 — Morning brief MoA (quick win, background feature)
Commit 5: Item 6 — Onboarding MoA moment (user discovery)
Commit 6: Item 4 — Adaptive escalation (intelligent automation)
Commit 7: Item 7 — Memory + KG MoA (deep integration)
Commit 8: Item 8 — SpecialistRouter MoA (multi-agent enhancement)
```

**Verification per commit:**
- `./gradlew :app:assembleDebug` — must build
- Every 3 commits: `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest` — all green

---

## What's NOT in this plan (deferred)

| Item | Why deferred |
|------|-------------|
| MoA preset creator UI | Needs SettingsScreen redesign, non-trivial |
| MoA cost-aware toggle | Needs cost calculation API, not yet available |
| MoA fallback chain | Aggregator failure is rare with DeepSeek; add reactively |
| MoA fact-checking | Research-grade feature, needs accuracy measurement first |
| MoA personality blending | Depends on OCEAN engine port, v3 |
| MoA voice correction | Separate feature, needs STT pipeline changes |
| Per-specialist MoA config | Depends on specialist model override UI (not built) |
