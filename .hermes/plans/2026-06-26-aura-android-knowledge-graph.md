# Aura Android — Knowledge Graph Implementation Plan

**Goal:** Add a runtime knowledge graph to the Android app that extracts entities and relationships from conversations via cloud LLM, stores them locally, and lets the user browse and query them.

**Architecture:**
- Local graph stored in Room (nodes + edges + properties as JSON).
- Extraction tool calls a cloud LLM with a structured prompt; response is parsed into nodes/edges.
- Injection point: after each assistant turn, run a lightweight extractor that summarizes facts into the KG.
- Retrieval: semantic + text search over nodes; expand neighbors when a node is tapped.
- UI: "Graph" tab in bottom nav with search, node cards, relationship list, and a detail sheet.

**Constraints:** cloud-only LLM, no server, no on-device embedding. SHA-256 hash fallback for vector IDs; embedding via cloud provider if configured.

---

## Task 1: Data model (Room entities + DAO)

**Files:**
- Create `aura-core/src/main/kotlin/com/aura/kg/NodeEntity.kt`
- Create `aura-core/src/main/kotlin/com/aura/kg/EdgeEntity.kt`
- Create `aura-core/src/main/kotlin/com/aura/kg/KnowledgeGraphDao.kt`
- Modify `aura-core/src/main/kotlin/com/aura/memory/MemoryDatabase.kt`

**Implementation:**
- NodeEntity: id (UUID/String PK), label, type (concept/entity/person/project/tool/event/skill/location/file), properties JSON, confidence, sourceTurnId, createdAt, updatedAt, accessCount.
- EdgeEntity: id PK, type, sourceId FK, targetId FK, weight, properties JSON, confidence, sourceTurnId, createdAt.
- Add `upsertNode`, `upsertEdge`, `getNode`, `getNodeByLabel`, `searchNodes`, `edgesFrom`, `edgesTo`, `deleteNode`, `deleteEdgesForNode`, `incrementAccessCount`, `recentNodes`.
- Update MemoryDatabase to include new entities and bump version with destructive migration (v1 KG).

**Tests:**
- Create `aura-core/src/test/kotlin/com/aura/kg/KnowledgeGraphDaoTest.kt` covering upsert, search, and neighbor expansion.

**Commit:** `feat(android): add knowledge graph Room model and DAO`

---

## Task 2: Domain models and repository

**Files:**
- Create `aura-core/src/main/kotlin/com/aura/kg/KgNode.kt`
- Create `aura-core/src/main/kotlin/com/aura/kg/KgEdge.kt`
- Create `aura-core/src/main/kotlin/com/aura/kg/KnowledgeGraphRepository.kt`
- Create `aura-core/src/main/kotlin/com/aura/kg/KnowledgeGraphModule.kt`

**Implementation:**
- Data classes KgNode/KgEdge with `toEntity` / `fromEntity` mapping.
- Repository exposes `saveGraph(nodes, edges, sourceTurnId)`, `search(query)`, `getNode(id)`, `getNeighbors(id)`, `recent(limit)`.
- Use deterministic IDs: `sha256("kg|node|type|label")` so re-extraction of the same fact doesn't duplicate.
- Module provides Repository + DAO via Hilt.

**Tests:**
- Create `aura-core/src/test/kotlin/com/aura/kg/KnowledgeGraphRepositoryTest.kt`.

**Commit:** `feat(android): add knowledge graph domain model and repository`

---

## Task 3: Cloud extraction tool

**Files:**
- Create `aura-core/src/main/kotlin/com/aura/tools/KnowledgeGraphTool.kt`
- Modify `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt`

**Implementation:**
- Tool name `knowledge_graph_extract`. Input: `text` (conversation snippet). Output: JSON with `nodes` and `edges`.
- Prompt asks LLM to extract 5-15 nodes and 5-20 edges from the text, with types drawn from an allowed set.
- Parse response via `Json { ignoreUnknownKeys = true }`; return tool result as JSON string.
- Fall back gracefully on parse failure: log and return empty result.
- Inject provider registry; use first configured provider (or default model via ProviderRegistry.parse) with a cheap model.

**Tests:**
- Create `aura-core/src/test/kotlin/com/aura/tools/KnowledgeGraphToolTest.kt` using a fake provider.

**Commit:** `feat(android): add knowledge graph extraction tool`

---

## Task 4: Hook extraction into agent loop

**Files:**
- Modify `aura-core/src/main/kotlin/com/aura/agent/MemoryAugmentedAgenticLoop.kt`
- Create `aura-core/src/main/kotlin/com/aura/kg/ConversationKgExtractor.kt`

**Implementation:**
- After a successful assistant turn (when `AgentEvent.Done` is emitted), fire a non-blocking `ConversationKgExtractor.extract(turnText)`.
- Extractor joins the last N turns (e.g. last 4) into a single text, calls `KnowledgeGraphTool`, maps result to `KgNode`/`KgEdge`, and saves via repository.
- Use a separate `supervisorJob()` scope so failure doesn't crash the chat stream.
- Guard with repository existence (extractor only runs if repo non-null).

**Tests:**
- Add test in `MemoryAugmentedAgenticLoopTest` if it exists; else create `ConversationKgExtractorTest.kt`.

**Commit:** `feat(android): wire knowledge graph extraction after assistant turns`

---

## Task 5: Graph tab UI — list + search

**Files:**
- Create `app/src/main/kotlin/com/aura/ui/screens/GraphScreen.kt`
- Create `app/src/main/kotlin/com/aura/ui/viewmodel/GraphViewModel.kt`
- Modify `app/src/main/kotlin/com/aura/ui/nav/NavGraph.kt`

**Implementation:**
- Add `Graph` to `TopLevelRoute` and bottom nav (icon `AccountTree` filled/outlined).
- GraphScreen: search bar, list of recent nodes, cards with icon per type, confidence badge.
- Tapping a node opens a detail bottom sheet.
- Search triggers `repository.search(query)` which matches label/type/properties substring.
- ViewModel holds `query`, `nodes`, `selectedNode`, `neighbors`, `loading`, `error`.

**Tests:** None (UI smoke covered by compile).

**Commit:** `feat(android): add graph tab with search and node list`

---

## Task 6: Node detail sheet + neighbor expansion

**Files:**
- Modify `app/src/main/kotlin/com/aura/ui/screens/GraphScreen.kt`

**Implementation:**
- Bottom sheet modal for selected node.
- Shows label, type, properties (pretty-printed JSON), confidence, source turn.
- Lists incoming + outgoing edges grouped by relationship type.
- Tapping a related node navigates the sheet to that node.
- "What connects X and Y?" helper: if exactly two nodes selected, find shortest path via repository BFS and display it.

**Tests:** None.

**Commit:** `feat(android): add knowledge graph node detail and path finder`

---

## Task 7: Manual graph operations in chat

**Files:**
- Create `aura-core/src/main/kotlin/com/aura/tools/KgQueryTool.kt`
- Modify `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt`

**Implementation:**
- Tool name `kg_query`. Input: `query` (e.g. "show my projects", "what do I know about Elnur?", "path between X and Y").
- LLM interprets the query, maps to repository calls (search, getNeighbors, findPath), returns formatted markdown result.
- Provides user a way to ask about the graph directly: "what does my graph know about X?"

**Tests:**
- Create `aura-core/src/test/kotlin/com/aura/tools/KgQueryToolTest.kt`.

**Commit:** `feat(android): add kg_query tool for conversational graph access`

---

## Task 8: Integration and final verification

**Run:**
- `./gradlew :aura-core:testDebugUnitTest :app:compileDebugKotlin :app:assembleDebug --no-daemon`

**Expected:** all tests pass, APK builds.

**Commit:** `chore(android): knowledge graph final verification`

---

## Anti-features (not in this wave)
- No persistent NetworkX-style graph algorithms (centrality, community detection) — local BFS/DFS only.
- No cross-device sync.
- No KG visual graph rendering (force-directed layout) — MVP is list + sheet.
- No public KG marketplace or skill loading.
- No on-device embedding for KG semantic search; use label/text search until cloud embedder is available.

## Prior plan
- Prior: `2026-06-26-aura-android-expansion-wave.md` (executed in commits e2395d3..ceff95c — providers, memory, research tools, multimodal, specialists, hands). This plan extends from that point.
