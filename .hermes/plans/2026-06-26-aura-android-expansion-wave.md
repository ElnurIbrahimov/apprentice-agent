# Aura Android Expansion Wave — Implementation Plan

> **For Hermes:** Use subagent-driven-development + direct execution. Plan+execute in same session. No "should I proceed?" prompts between items.

**Goal:** Expand the cloud-only Aura Android superapp with 5 major capability areas: more providers, semantic memory, better research tools, multimodal I/O, and multi-agent specialists + hands.

**Constraints:**
- Pure cloud. No on-device model. No server.
- Phone-only app. No laptop dependency.
- Single-user, sideload v1.
- All new code in `:aura-core` library; UI in `:app`.
- One atomic commit per major item.

**Architecture approach:**
- Provider SDK: extend the existing `Provider` interface; add Gemini (multimodal), Groq (fast), OpenRouter (aggregator).
- Memory: add cloud `Embedder` calling Ollama Cloud / OpenAI embedding endpoint, with local deterministic fallback cached in Room. Rerank using vector + BM25 + recency + decay.
- Research tools: add `BraveSearchTool`, `TavilySearchTool`, `FirecrawlFetchTool`, `DeepResearchTool`.
- Multimodal: add `VisionTool` (camera/gallery → Gemini/GPT-4o), `ImageGenerationTool` (DALL-E/Stable Diffusion API), `AudioTranscriberTool` (cloud Whisper API).
- Multi-agent: add `Orchestrator`, specialist system prompts, chip row in chat.
- Hands: add `Hand` base class + `HandManager` + user-defined automation store.

**Verification pipeline:**
- `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest` — must be green
- `./gradlew :app:assembleDebug` — must build APK
- 60+ tests at end

---

## Pre-execution verification (MANDATORY)

Before writing code for each item, grep the target location to confirm it does not already exist. The codebase may have partial stubs. If found, note "already shipped" and skip.

---

## Phase 1: More cloud providers

### Task 1.1: Add Gemini provider

**Objective:** Implement `GeminiProvider` for text + image input via `generativelanguage.googleapis.com`.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/providers/GeminiProvider.kt`
- Modify: `android/aura-core/src/main/kotlin/com/aura/providers/ProviderModule.kt` (bind it)
- Modify: `android/aura-core/src/main/kotlin/com/aura/providers/ProviderKeys.kt` (add key field)
- Modify: `android/app/src/main/java/com/aura/android/di/AppModule.kt` (if provider list is hardcoded there)
- Test: `android/aura-core/src/test/kotlin/com/aura/providers/GeminiProviderTest.kt`

**Implementation notes:**
- Use OkHttp. Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={apiKey}`
- Map `ProviderMessage` to Gemini `Content` with `role: user/model`.
- For images, accept inline base64 in `ProviderMessage.imageData` (add optional field).
- Hardcoded model list: `gemini-1.5-flash`, `gemini-1.5-pro`.
- Prefix: `gemini:`.

**Test:** mock server with MockWebServer, assert streaming chunks.

**Commit:** `feat(android): add Gemini cloud provider`

---

### Task 1.2: Add Groq provider

**Objective:** Implement `GroqProvider` as an OpenAI-compatible provider at `https://api.groq.com/openai/v1/`.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/providers/GroqProvider.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/providers/OpenAiCompatProvider.kt` if not already a reusable base
- Modify: `ProviderModule.kt`, `ProviderKeys.kt`, app provider list
- Test: `android/aura-core/src/test/kotlin/com/aura/providers/GroqProviderTest.kt`

**Implementation notes:**
- Reuse existing OpenAI-compatible streaming parser if possible (`OllamaCloudProvider` is already OpenAI-compatible).
- Create a generic `OpenAiCompatProvider` with `baseUrl`, `apiKey`, `displayName`, `prefix`, `defaultModels`.
- Convert `OllamaCloudProvider`, `DeepSeekProvider`, `OpenAIProvider` to use this base if they duplicate code.
- Groq prefix: `groq:`. Models: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`.

**Test:** mock streaming response with tool_calls.

**Commit:** `feat(android): add Groq provider + generic OpenAI-compat base`

---

### Task 1.3: Add OpenRouter provider

**Objective:** Implement `OpenRouterProvider` as OpenAI-compatible aggregator.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/providers/OpenRouterProvider.kt`
- Modify: `ProviderModule.kt`, `ProviderKeys.kt`, app provider list
- Test: `android/aura-core/src/test/kotlin/com/aura/providers/OpenRouterProviderTest.kt`

**Implementation notes:**
- Base URL: `https://openrouter.ai/api/v1`.
- Required extra header: `HTTP-Referer` and `X-Title`.
- Prefix: `openrouter:`. Models can be a small curated list: `openrouter:gpt-4o`, `openrouter:claude-3.5-sonnet`, `openrouter:deepseek-v3`.
- Fetch model list from `/api/v1/models` optionally.

**Test:** mock server, assert extra headers sent.

**Commit:** `feat(android): add OpenRouter provider`

---

### Task 1.4: Update Settings UI for new providers

**Objective:** Add key input fields and model pickers for Gemini, Groq, OpenRouter.

**Files:**
- Modify: `android/app/src/main/java/com/aura/android/features/settings/SettingsScreen.kt`
- Modify: `android/app/src/main/java/com/aura/android/features/settings/SettingsViewModel.kt`
- Modify: `android/app/src/main/java/com/aura/android/features/chat/ChatViewModel.kt` (model picker source)

**Implementation notes:**
- Add 3 new key rows in Settings (collapsible or below existing).
- Model picker should pull from `ProviderRegistry.all()` + `listModels()`.
- No UI redesign — add rows to existing list.

**Test:** existing Settings UI tests should still pass; update if needed.

**Commit:** `feat(android): settings UI for Gemini/Groq/OpenRouter`

---

## Phase 2: Semantic memory via cloud embeddings

### Task 2.1: Add cloud Embedder interface + Ollama embedding

**Objective:** Replace deterministic embedder with cloud embedding from Ollama Cloud.

**Files:**
- Modify: `android/aura-core/src/main/kotlin/com/aura/memory/Embedder.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/memory/CloudEmbedder.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/memory/LocalEmbedder.kt` (move existing deterministic logic)
- Modify: `android/aura-core/src/main/kotlin/com/aura/memory/MemoryModule.kt` (Hilt binding)
- Test: `android/aura-core/src/test/kotlin/com/aura/memory/CloudEmbedderTest.kt`

**Implementation notes:**
- `interface Embedder { suspend fun embed(text: String): FloatArray }`
- `CloudEmbedder` calls `POST https://api.ollama.com/api/embeddings` with `model: nomic-embed-text`.
- Cache result keyed by text hash in Room or memory LRU to avoid re-embedding.
- Fallback to `LocalEmbedder` if no network or key missing.
- Add `embeddingModel` to ProviderKeys / Settings (default `nomic-embed-text`).

**Test:** mock Ollama embedding endpoint, assert vector dim and caching.

**Commit:** `feat(android): cloud semantic embedder with local fallback`

---

### Task 2.2: Improve retrieval ranking (RRF)

**Objective:** Port Aura's retrieval fusion to Android.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/memory/Retrieval.kt`
- Modify: `android/aura-core/src/main/kotlin/com/aura/memory/MemoryStore.kt` (use `Retrieval` in `query()`)
- Modify: `android/aura-core/src/main/kotlin/com/aura/memory/VectorIndex.kt`
- Test: `android/aura-core/src/test/kotlin/com/aura/memory/RetrievalTest.kt`

**Implementation notes:**
- Score candidates by: vector similarity (cosine), BM25-like text match, recency, decayScore, importance.
- Use RRF: `score = sum(1 / (k + rank_i))` across ranking signals.
- Keep existing `touch()` behavior.
- No schema change; all signals already in `MemoryEntity`.

**Test:** seed memories, query, assert correct ranking.

**Commit:** `feat(android): RRF retrieval fusion for memory`

---

## Phase 3: Better web/research tools

### Task 3.1: Add Brave web search tool

**Objective:** `BraveSearchTool` using Brave API (or HTML scrape fallback like desktop `brave_search.py`).

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/tools/BraveSearchTool.kt`
- Modify: `android/aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt` (register)
- Test: `android/aura-core/src/test/kotlin/com/aura/tools/BraveSearchToolTest.kt`

**Implementation notes:**
- Primary: `https://api.search.brave.com/api/summarizer/search` or web search endpoint. Requires `BRAVE_API_KEY`.
- Fallback: HTML scrape DuckDuckGo if no key (existing behavior).
- Return results as markdown list with titles, URLs, snippets.

**Test:** mock Brave API and fallback.

**Commit:** `feat(android): Brave web search tool`

---

### Task 3.2: Add Tavily search tool

**Objective:** `TavilySearchTool` for research-grade search.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/tools/TavilySearchTool.kt`
- Modify: `ToolsModule.kt`
- Test: `android/aura-core/src/test/kotlin/com/aura/tools/TavilySearchToolTest.kt`

**Implementation notes:**
- Endpoint: `https://api.tavily.com/search`.
- Support `include_answer`, `max_results`, `search_depth`.
- Return structured result with `answer` + sources.

**Commit:** `feat(android): Tavily research search tool`

---

### Task 3.3: Add Firecrawl fetch tool

**Objective:** `FirecrawlFetchTool` extracts clean markdown from URLs.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/tools/FirecrawlFetchTool.kt`
- Modify: `ToolsModule.kt`
- Test: `android/aura-core/src/test/kotlin/com/aura/tools/FirecrawlFetchToolTest.kt`

**Implementation notes:**
- Endpoint: `https://api.firecrawl.dev/v1/scrape`.
- Input: `url`. Output: markdown string.
- SSRF guard: only `http/https`, block private IPs, block localhost.

**Commit:** `feat(android): Firecrawl web fetch tool`

---

### Task 3.4: Add deep research tool

**Objective:** `DeepResearchTool` orchestrates search + fetch + LLM synthesis.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/tools/DeepResearchTool.kt`
- Modify: `ToolsModule.kt`
- Test: `android/aura-core/src/test/kotlin/com/aura/tools/DeepResearchToolTest.kt`

**Implementation notes:**
- Steps: (1) Tavily/Brave search for query, (2) Firecrawl top results, (3) chunk + embed, (4) LLM synthesize answer with citations.
- Use `ProviderRegistry` for synthesis step.
- Emit progress events via `ToolContext` extension or return intermediate result.
- Citation format: `[1] Title — url`.

**Commit:** `feat(android): deep research tool with citations`

---

### Task 3.5: Add citations panel to chat UI

**Objective:** Show sources when agent uses web/research tools.

**Files:**
- Create: `android/app/src/main/java/com/aura/android/features/chat/CitationsPanel.kt`
- Modify: `android/app/src/main/java/com/aura/android/features/chat/MessageBubble.kt`
- Modify: `android/app/src/main/java/com/aura/android/features/chat/ChatViewModel.kt` (parse citations from tool result)

**Implementation notes:**
- Tool results can include `citations: List<Citation>` in metadata.
- Message model gets `citations: List<Citation>`.
- Bubble shows a small "Sources (3)" chip that expands into bottom sheet.
- Minimal design — reuse existing Material 3 components.

**Commit:** `feat(android): citations panel in chat`

---

## Phase 4: Multimodal

### Task 4.1: Add image support to ProviderMessage

**Objective:** Allow messages to carry inline images for Gemini/GPT-4o.

**Files:**
- Modify: `android/aura-core/src/main/kotlin/com/aura/providers/ProviderMessage.kt`
- Modify: `android/aura-core/src/main/kotlin/com/aura/providers/AnthropicProvider.kt` (support image blocks)
- Modify: `android/aura-core/src/main/kotlin/com/aura/providers/OpenAiCompatProvider.kt` or relevant providers

**Implementation notes:**
- Add `val imageData: ByteArray? = null` and `val mimeType: String? = null` to `ProviderMessage`.
- Anthropic: `image` block with `source.type = "base64"`.
- OpenAI/Gemini: inline base64 data URI or Gemini `inlineData`.

**Commit:** `feat(android): inline image payloads in provider messages`

---

### Task 4.2: Add Vision tool

**Objective:** `VisionTool` captures or picks image, sends to multimodal provider.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/tools/VisionTool.kt`
- Modify: `ToolsModule.kt`
- Modify: `android/app/src/main/java/com/aura/android/features/chat/ChatScreen.kt` (image attach button)
- Test: `android/aura-core/src/test/kotlin/com/aura/tools/VisionToolTest.kt`

**Implementation notes:**
- Inputs: `source` (camera/gallery), `prompt`.
- Use existing camera capture and photo library tools to get image bytes.
- Convert to `ProviderMessage.imageData` and call provider via `ProviderRegistry`.
- Default model: `gemini-1.5-flash` if configured, else `gpt-4o` via OpenAI-compatible.

**Commit:** `feat(android): vision tool for image understanding`

---

### Task 4.3: Add image generation tool

**Objective:** `ImageGenerationTool` generates images via DALL-E / Stable Diffusion API.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/tools/ImageGenerationTool.kt`
- Modify: `ToolsModule.kt`
- Modify: `android/app/src/main/java/com/aura/android/features/chat/ChatScreen.kt` (display generated image)
- Test: `android/aura-core/src/test/kotlin/com/aura/tools/ImageGenerationToolTest.kt`

**Implementation notes:**
- Use OpenAI DALL-E 3 (`https://api.openai.com/v1/images/generations`) as primary.
- Save result to app cache, return file path/URI for UI.
- Fallback: Pollinations AI (`https://image.pollinations.ai/prompt/{prompt}`) if no OpenAI key.

**Commit:** `feat(android): image generation tool`

---

### Task 4.4: Add audio transcription tool

**Objective:** `AudioTranscriberTool` records or picks audio, sends to Whisper API.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/tools/AudioTranscriberTool.kt`
- Modify: `ToolsModule.kt`
- Modify: `AndroidManifest.xml` (no new permission; audio recording already granted)
- Test: `android/aura-core/src/test/kotlin/com/aura/tools/AudioTranscriberToolTest.kt`

**Implementation notes:**
- Use OpenAI Whisper (`https://api.openai.com/v1/audio/transcriptions`) or Groq Whisper.
- Accept `audioBytes` (PCM/WAV) or file URI.
- Return transcript string.
- Reuse existing mic capture if available.

**Commit:** `feat(android): cloud audio transcription tool`

---

### Task 4.5: Multimodal UI wiring

**Objective:** Chat supports image attachments, generated image preview, audio message send.

**Files:**
- Modify: `android/app/src/main/java/com/aura/android/features/chat/ChatScreen.kt`
- Modify: `android/app/src/main/java/com/aura/android/features/chat/MessageInput.kt` (if exists)
- Modify: `android/app/src/main/java/com/aura/android/features/chat/MessageBubble.kt`

**Implementation notes:**
- Add attach image button next to mic.
- Show thumbnail preview above input before sending.
- Generated images render inline in chat bubble.
- Audio messages: hold mic → record → send as audio attachment to Vision/Transcriber path.

**Commit:** `feat(android): multimodal chat UI wiring`

---

## Phase 5: Multi-agent specialists + hands

### Task 5.1: Add multi-agent orchestrator

**Objective:** Port `aura/multi_agent/` to Kotlin.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/multi_agent/AgentMessage.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/multi_agent/AgentResult.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/multi_agent/CollaborationMode.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/multi_agent/Specialist.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/multi_agent/Orchestrator.kt`
- Modify: `android/aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt` (accept specialist override)
- Test: `android/aura-core/src/test/kotlin/com/aura/multi_agent/OrchestratorTest.kt`

**Implementation notes:**
- `Orchestrator` selects specialist based on message content + tools needed.
- Each `Specialist` has name, description, system prompt, allowed tools, preferred model.
- Modes: SINGLE, SEQUENTIAL, PARALLEL, DEBATE.
- Single mode: just swap system prompt + tool subset.

**Commit:** `feat(android): multi-agent orchestrator + specialists`

---

### Task 5.2: Add specialist chip row in chat

**Objective:** UI to pick or confirm specialist.

**Files:**
- Modify: `android/app/src/main/java/com/aura/android/features/chat/ChatScreen.kt`
- Modify: `android/app/src/main/java/com/aura/android/features/chat/ChatViewModel.kt`
- Create: `android/app/src/main/java/com/aura/android/features/chat/SpecialistChipRow.kt`

**Implementation notes:**
- Chips: General, Research, Code, Creative, OnTheGo.
- Tapping a chip sets `specialist` in chat request.
- Auto-detect from prompt if no chip selected.

**Commit:** `feat(android): specialist chip row UI`

---

### Task 5.3: Add hands system

**Objective:** Port `aura/hands/` for user-defined automations.

**Files:**
- Create: `android/aura-core/src/main/kotlin/com/aura/hands/Hand.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/hands/HandManager.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/hands/HandStore.kt` (Room)
- Create: `android/aura-core/src/main/kotlin/com/aura/hands/MorningBriefingHand.kt`
- Create: `android/aura-core/src/main/kotlin/com/aura/hands/MemoryHand.kt`
- Modify: `android/aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt` (register hands as invokable)
- Modify: `android/app/src/main/java/com/aura/android/features/home/HomeScreen.kt` (hands quick actions)
- Test: `android/aura-core/src/test/kotlin/com/aura/hands/HandManagerTest.kt`

**Implementation notes:**
- A `Hand` = name, description, trigger, templated prompt, tool sequence.
- `HandStore` persists in Room.
- `HandManager.run(handId, context)` executes the hand.
- MorningBriefingHand replaces existing `MorningBriefWorker` logic (or wraps it).
- MemoryHand auto-extracts memories after each conversation turn.

**Commit:** `feat(android): hands automation system`

---

### Task 5.4: Add hands management UI

**Objective:** Screen to list / create / trigger hands.

**Files:**
- Create: `android/app/src/main/java/com/aura/android/features/hands/HandsScreen.kt`
- Create: `android/app/src/main/java/com/aura/android/features/hands/HandsViewModel.kt`
- Modify: `android/app/src/main/java/com/aura/android/navigation/NavGraph.kt` (add route)
- Modify: `android/app/src/main/java/com/aura/android/ui/AppScaffold.kt` (add tab or menu item)

**Implementation notes:**
- List existing hands (built-in + user-created).
- "Create from last message" button.
- Trigger button runs hand immediately.
- Keep UI simple — list + detail sheet.

**Commit:** `feat(android): hands management UI`

---

## Final verification

1. Run full test suite: `./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest`
2. Run assemble: `./gradlew :app:assembleDebug`
3. Check test count >= 80 (was 60).
4. Re-split subtree and push to `aura-android`.

---

## Anti-features (deliberately not in this wave)

- No on-device model.
- No server backend.
- No auth / billing.
- No cross-device sync.
- No full browser automation (Playwright).
- No desktop shell/git/file-system tools.
- No heavy ALMA emotion engine (slim version only).

---

## Source-of-truth mapping

| Android module | Aura source | Notes |
|---|---|---|
| `providers/GeminiProvider.kt` | `aura/providers/gemini_provider.py` | text + image |
| `providers/OpenAiCompatProvider.kt` | `aura/providers/openai_compat.py` | generic base |
| `memory/CloudEmbedder.kt` | `aura/memory/embedding.py` | cloud embedding |
| `memory/Retrieval.kt` | `aura/memory/retrieval.py` | RRF fusion |
| `tools/BraveSearchTool.kt` | `aura/tools/brave_search.py` | search |
| `tools/TavilySearchTool.kt` | `aura/tools/tavily_tool.py` | research search |
| `tools/FirecrawlFetchTool.kt` | `aura/tools/firecrawl_tool.py` | fetch |
| `tools/DeepResearchTool.kt` | `aura/tools/deep_research.py` + `aura/tools/research_tool.py` | orchestration |
| `tools/VisionTool.kt` | `aura/tools/vision.py` | image understanding |
| `tools/ImageGenerationTool.kt` | `aura/tools/image_gen.py` | image gen |
| `tools/AudioTranscriberTool.kt` | `aura/tools/audio_transcriber.py` | transcription |
| `multi_agent/Orchestrator.kt` | `aura/multi_agent/orchestrator.py` + `router.py` | specialists |
| `hands/HandManager.kt` | `aura/hands/manager.py` | automations |
