# Aura Beyond-SOTA Master Implementation Plan

> **For the implementation agent:** planning is complete; execution is not authorized by this file alone. When the user says `begin`, load `software-development:test-driven-development`, `software-development:subagent-driven-development`, and `software-development:requesting-code-review`; then start Phase 0 with real tool calls. Work continuously through the selected milestone without asking between atomic tasks. Use RED → GREEN → REFACTOR and verify every phase before moving on.

**Created:** 2026-07-16 13:04 AZT  
**Project:** `D:\android`  
**Current app:** Aura v0.15.2, personal-use/sideloaded Android app  
**Plan status:** implementation-ready  
**Scope:** complete the unfinished capability fabric, then build a Creative World OS and Personal Action OS sharing one artifact/event/policy substrate  
**Execution shape:** 15 ordered phases, 118 steps / 117 atomic code change sets, six release milestones

---

## 1. Mission

Aura must stop being a chat application with many disconnected features and become two joined operating systems:

1. **Creative World OS** — persistent, versioned, executable, multimodal creative universes.
2. **Personal Action OS** — durable goals that can perceive, plan, act, verify, resume, and learn.

They share:

- a durable artifact and revision graph;
- an append-only run/event ledger;
- user-controlled model/capability routing;
- scoped tool, cost, privacy, and approval policies;
- source-aware memory and temporal beliefs;
- an inspectable workflow runtime;
- one export/restore format;
- explicit verification instead of trusting a model or tool to claim success.

### Flagship acceptance scenario

Given one creative brief, Aura produces and preserves an editable mini-production: research, concept alternatives, outline, character/reference pack, storyboard, generated media, voice and music direction, continuity report, artifact lineage, branches, and exports.

Given one real-world goal, Aura creates a definition of done, decomposes it into a durable workflow, uses local/remote/MCP/computer tools under policy, survives process death, asks only for required approvals, verifies postconditions, and delivers evidence.

---

## 2. Verified baseline — do not re-plan shipped work

Verified directly from the current source tree and latest Gradle output:

- `843` JUnit test cases pass; zero failures or skipped tests.
- Full gate passed: `:aura-core:testDebugUnitTest`, `:app:testDebugUnitTest`, `:app:lintDebug`, `:app:assembleDebug` (`112` tasks).
- `46` tools are registered in `aura-core/src/main/kotlin/com/aura/tools/ToolsModule.kt`.
- `17` chat providers are Hilt-bound in `ProviderModule.kt`, including custom OpenAI-compatible and ChatGPT subscription providers.
- Six non-chat capability providers exist: Exa, Jina, ElevenLabs, Stability, Kling, World Labs.
- Only TTS currently consumes `CapabilityRegistry`; search/image/video/3D infrastructure is not an honest user-facing capability yet.
- Quick Ask already uses `ChatViewModel` and the full agentic path. Do not rebuild it.
- Provider failover warnings, inline citations, conversation/memory/KG provenance, pinned history, Hands run history, approve/resume, grant-permission/resume, creative project CRUD, six creative text modes, custom providers, and provider model catalogs are already present.
- Creative output is still transient ViewModel text; no artifact/revision/job substrate exists.
- `CreativeEngine` calls `ProviderRegistry.chat()` directly; it bypasses specialist/tool/subagent orchestration.
- `WorldBible` is serialized as one project JSON document. Continuity checking is prompt-only, not a typed compiler.
- `MemoryDatabase` is v6; `HandDatabase` v2; `ConversationDatabase` v4; `TaskDatabase` v4; `ProactiveEventDatabase` v3; `UserProfileDatabase` v1; `AuraBackup.SCHEMA_VERSION` is 6.
- Documents are chunked and stored as memories, but chunks are not first-class, page/source-addressable records.
- `LocalEmbedder` is deterministic hash projection, not a semantic model. Cloud embedding fallback exists.
- Screen capture is one-shot and has no Android 14+ MediaProjection foreground-service lifecycle.
- No MCP client, durable goal graph, true subagents, remote sandbox, browser automation, AccessibilityService action loop, temporal belief model, or taste learner exists.
- `D:\android` has no `.git` repository. There is no canonical `D:\Aura\android` tree on this machine. All “commit” names below are atomic change-set names until repository lineage is restored.

### Reconciliation with `2026-07-15_014715-aura-expansion-program.md`

| Prior area | Current status | This plan |
|---|---|---|
| Hands history and approve/resume | Shipped | Extend into durable DAG/checkpoint runtime |
| Provenance | Shipped | Reuse for beliefs, canon, artifacts, runs |
| Quick Ask full agent | Shipped | No duplicate work |
| Provider warning/citations | Shipped | Reuse in every new surface |
| Memory edit history | Shipped | Add visible extraction/knowledge feedback |
| Structured Personality Studio | Open | Implement in Phase 12, integrated with Taste Twin |
| Writer specialist | Shipped | Replace static-only role with real subagents later |
| Worldbuilder/Simulator roles | Not real agents | Implement through Creative Council, not static prompt duplication |
| Creative Studio text kernel | Shipped | Add artifacts, canon, revisions, media, council |
| Backup v6 | Shipped | Evolve into `.aura` archive with assets |
| New chat/capability providers | Infrastructure shipped | Wire capability-backed tools in Phase 4 |
| Generated-media gallery | Open | Phase 3 |
| Per-tool controls | Open | Phase 1 |

This plan extends the shipped expansion program. It does **not** supersede it or reopen completed tasks.

---

## 3. Scope boundaries and anti-features

### Included

- local semantic embeddings and source-addressable document RAG;
- creative artifacts, revisions, branches, jobs, canon, simulations, continuity, dependencies, and exports;
- Exa/Jina/Stability/Kling/World Labs tool wiring;
- durable goal/workflow runtime and postcondition verification;
- official Kotlin MCP client and late tool discovery;
- optional desktop Aura Bridge for files, shell, code, and browser MCP servers;
- opt-in Android computer use and user-started Ground sessions;
- temporal personal world model and opportunity proposals;
- real isolated subagents and Creative Council;
- Taste Twin, multimodal reference packs, cross-media consistency;
- document editor, storyboard, canvas, timeline, and production pipelines;
- safe, evaluated self-improvement proposals.

### Deliberate non-goals

- No Play Store/distribution work. This remains a personal sideloaded app.
- No multi-user accounts, teams, billing, subscriptions, telemetry SaaS, or cloud backend requirement.
- No hardcoded model names. All model IDs come from live provider catalogs and user role preferences.
- No binary media blobs in Room.
- No silent always-on screen/camera/microphone surveillance.
- No unattended payments, destructive file changes, messages, or account/security changes.
- No automatic source-code self-modification. Aura may propose patches only after the eval framework exists.
- No trust in remote MCP risk annotations without local policy classification.
- No new static specialist simply to imitate a real subagent.
- No custom browser automation engine when the official Playwright MCP server can be used through the Bridge.
- No heavy mobile video renderer in the first release; export timelines/EDLs and delegate final rendering to the Bridge.

---

## 4. Target architecture

```text
Compose surfaces
  Chat | Goals/Runs | Creative Studio | Canvas/Timeline | Ground | Tools/MCP | World Model
        |
Application coordinators / ViewModels
        |
+------------------------- Shared kernel --------------------------+
| ModelRoleRouter | CapabilityRouter | PolicyEngine | EventLedger |
| ArtifactStore   | Provenance       | CostBudget   | Verifier    |
+--------------------------+----------------------+----------------+
                           |                      |
             Personal Action OS           Creative World OS
             GoalService                  CreativeDirector
             WorkflowRuntime              CreativeArtifactStore
             SubagentManager              CreativeContextRetriever
             TriggerBus                   CanonRepository
             GroundCoordinator            ContinuityCompiler
             OpportunityEngine            TasteEngine
                           |                      |
                      Tool / capability fabric
          Static tools | MCP tools/resources | Android UI | Bridge
                           |
           Room metadata | app-private asset files | `.aura` archive
```

### Architectural rules

1. **Metadata in Room; bytes in files.** Store hashes and URIs in Room, never large image/video/audio blobs.
2. **Every side effect has a run and step ID.** Tool calls without provenance are rejected from durable workflows.
3. **Every success can be verified.** `ToolResult.Ok` is not proof; postconditions determine final success.
4. **Every generated artifact is versioned.** No text/media output exists only in a ViewModel.
5. **Every dynamic tool is policy-wrapped.** MCP and Bridge tools go through `ToolExecutor` semantics.
6. **Every background operation resumes idempotently.** WorkManager jobs use durable input IDs, not full payloads.
7. **Every model/capability is selected by role and configuration.** Never by hardcoded ID.
8. **Every learned preference is inspectable and reversible.** No hidden profile drift.
9. **Raw Ground frames are ephemeral by default.** Persist only user-approved artifacts or redacted summaries.
10. **Legacy formats remain readable until two full backup/restore releases pass.**

---

## 5. Persistence and migration sequence

Do not combine all schema work into one migration. Each phase owns one independently testable upgrade.

| Database | Current | Planned | Purpose |
|---|---:|---:|---|
| MemoryDatabase | 6 | 7 | embedding metadata + first-class document chunks |
| MemoryDatabase | 7 | 8 | creative artifacts, revisions, branches, generation jobs |
| MemoryDatabase | 8 | 9 | canon facts, simulations, continuity issues, artifact dependencies |
| MemoryDatabase | 9 | 10 | temporal beliefs, evidence, world events, opportunities |
| MemoryDatabase | 10 | 11 | preference signals, style profiles, reference identities, routing outcomes |
| HandDatabase | 2 | 3 | workflow definitions/nodes/edges, runs/steps/events, approvals/checkpoints |
| ProactiveEventDatabase | 3 | 4 | trigger source, goal/run links, action and retention state |
| UserProfileDatabase | 1 | 2 | structured personality profile and adaptation settings |
| ConversationDatabase | 4 | 5 only if queries require it | typed run/project/branch links; otherwise keep metadata to avoid churn |
| AuraBackup | 6 | 7 | creative metadata + archive manifest |
| AuraBackup | 7 | 8 | workflows/runs/policies/world model |
| AuraBackup | 8 | 9 | personality/taste/reference data |

### Core new entities

**Creative v8**

- `CreativeArtifactEntity`: id, projectId, branchId, kind, title, currentRevisionId, previewText, mimeType, storageUri, contentHash, status, metadataJson, createdAt, updatedAt.
- `CreativeRevisionEntity`: id, artifactId, branchId, parentRevisionIdsJson, contentText or storageUri, contentHash, authorKind, providerPrefix, modelId, prompt, settingsJson, createdAt.
- `CreativeBranchEntity`: id, projectId, name, baseBranchId, baseRevisionId, headRevisionId, status, createdAt, updatedAt.
- `CreativeGenerationJobEntity`: id, projectId, branchId, capabilityKind, requestJson, status, progress, providerPrefix, providerOperationId, resultArtifactIdsJson, errorCode, attempts, timestamps.

**Canon v9**

- `CanonFactEntity`: projectId/branchId, subjectType, subjectId, predicate, valueJson, validity window, confidence, sourceRevisionId, status.
- `CreativeSimulationEntity`: project/branch, premise, assumptionsJson, narrative, stateDeltaJson, causalGraphJson, createdAt, canonizedAt.
- `ContinuityIssueEntity`: project/branch/artifact, category, severity, message, evidenceFactIdsJson, suggestedPatchJson, status, timestamps.
- `ArtifactDependencyEntity`: sourceArtifactId, targetArtifactId, relation, invalidationPolicy.

**Automation v3**

- `WorkflowDefinitionEntity`, `WorkflowNodeEntity`, `WorkflowEdgeEntity`.
- `AgentRunEntity`: goal, definition of done, policy, budget, status, conversation/project link, timestamps.
- `AgentStepEntity`: node, kind, state, input/output/error, attempt, postcondition, timestamps.
- `AgentEventEntity`: ordered append-only event ledger.
- `ApprovalRequestEntity`: scope, rationale, decision, expiry.
- `RunCheckpointEntity`: active nodes, variables, pending callbacks, definition version.

**World/taste v10-v11**

- `BeliefEntity`, `EvidenceEntity`, `WorldEventEntity`, `OpportunityEntity`.
- `PreferenceSignalEntity`, `StyleProfileEntity`, `ReferenceIdentityEntity`, `RoutingOutcomeEntity`.

### Migration invariants

- Every `@Database` version bump must ship with exported schema JSON and migration test.
- SQL migrations create only tables/columns/indexes. JSON parsing/backfills run in idempotent application migrators with completion markers.
- Legacy `Hand.steps` and `CreativeProject.worldJson` remain intact until migrated data is verified.
- Backup restore accepts older schemas 6–8 and upgrades in memory; unknown future schemas fail loudly.
- Asset archive restore validates SHA-256, blocks ZIP slip, applies size/count ceilings, and remaps URIs.

---

## 6. Dependency graph and delivery milestones

```text
Phase 0 -> Phase 1 -> Phase 2 -----------------------+
                  -> Phase 3 -> Phase 4 -> Phase 5 --+--> Phase 11 -> Phase 12 -> Phase 13
                  -> Phase 6 -> Phase 7 -> Phase 8 --+
                             -> Phase 9 -> Phase 10 --+
                                                       -> Phase 14
```

Parallel-safe after Phase 1:

- Phase 2 local intelligence;
- Phase 3 creative artifacts;
- Phase 6 durable workflow runtime.

Milestones:

1. **M1 — Honest Foundation:** Phases 0–2.
2. **M2 — Persistent Creative Kernel:** Phases 3–5.
3. **M3 — Durable Agent OS:** Phase 6.
4. **M4 — Open Tool/Perception Fabric:** Phases 7–10.
5. **M5 — Beyond-SOTA Creative System:** Phases 11–13.
6. **M6 — Trustworthy Release:** Phase 14.

---

# PHASED IMPLEMENTATION

## Phase 0 — Execution safety and dependency spikes

### 0.1 Restore repository lineage before code execution

- **Action:** locate or restore the actual remote/`.git`; do not `git init` and fabricate history without explicit user direction.
- **If no repository is available:** continue only as atomic change sets, export patches after each phase, and never claim commits/pushes.
- **Verify:** record clean/dirty baseline and remote in the execution log.

### 0.2 Lock the real baseline

- **Files:** new `aura-core/src/test/kotlin/com/aura/architecture/BeyondSotaBaselineTest.kt`; update README only when features actually ship.
- **RED:** assert current tool names are unique, every tool has category/risk/schema, every capability binding has a distinct prefix/kind, and backup/Room versions match the documented baseline. Do not size-pin exact specialist/tool totals; assert required members plus floors.
- **GREEN:** add only missing metadata discovered by the contract test.
- **Change set:** `test(architecture): lock Aura capability contracts`.

### 0.3 Benchmark real on-device embedding runtimes

- **Candidates:** official MediaPipe Text Embedder versus ONNX Runtime Mobile; use current official releases at execution time.
- **Files:** disposable `app/src/debug/kotlin/com/aura/debug/EmbeddingBenchmarkActivity.kt`; debug-only assets; benchmark report under `.hermes/benchmarks/`.
- **Dataset:** 100 semantic-positive/negative pairs, English plus the user’s real language mix, short facts and 1,800-character document chunks.
- **Acceptance:** deterministic output; genuine semantic separation; p95 under 150 ms/chunk on target phone; memory under 150 MB; acceptable APK delta; offline operation.
- **Decision:** prefer MediaPipe if quality is adequate and Preview risk is acceptable; otherwise ONNX with a bundled tokenizer. Delete losing spike code.
- **Change set:** `spike(memory): select local semantic embedder`.

### 0.4 Validate official Kotlin MCP SDK on Android

- **Dependency:** official `modelcontextprotocol/kotlin-sdk`, verified from its current GitHub release and Maven Central at execution time.
- **Files:** disposable `aura-core/src/test/.../mcp/McpSdkCompatibilityTest.kt`; MockWebServer fixture.
- **Prove:** Android/JVM compatibility, Streamable HTTP or SSE connection, cancellation, `tools/list`, `tools/call`, resource reads, schema serialization, and no forbidden desktop dependency.
- **Fallback:** if umbrella artifact pulls unsuitable transports, depend only on core/client/Ktor modules or implement the small transport adapter against official protocol types.
- **Change set:** `spike(mcp): validate official Kotlin client`.

### 0.5 Fix architecture contracts before feature work

- **New docs:** `docs/architecture/artifacts-and-events.md`, `docs/architecture/tool-policy.md`, `docs/architecture/privacy-boundaries.md`.
- **Define:** stable IDs, event ordering, artifact URI ownership, model/capability routing, policy precedence, redaction, retry/idempotency, and postcondition semantics.
- **Tests:** serialization round trips for every shared sealed type.
- **Change set:** `docs(architecture): define beyond-SOTA kernel contracts`.

**Phase gate:** original 843 tests still pass; both spikes have a written decision; no production dependency from rejected spikes remains.

---

## Phase 1 — Shared routing, policy, and observability

### 1.1 Add `CapabilityRouter`

- **New:** `aura-core/.../capabilities/CapabilityRouter.kt`, `CapabilityRole.kt`.
- **Modify:** `CapabilityRegistry.kt`, `CapabilityProvider.kt`, `UserPreferences.kt`, backup preference models.
- **Behavior:** select configured providers by kind, explicit user preference, health, required operation, cost, and fallback order. Never depend on Hilt map insertion order.
- **RED:** preference wins; unavailable provider falls back; unsupported operation fails honestly; no configured provider returns typed error.
- **Change set:** `feat(capabilities): add user-controlled capability routing`.

### 1.2 Add `ModelRoleRouter`

- **New:** `aura-core/.../providers/ModelRole.kt`, `ModelRoleRouter.kt`.
- **Roles:** conversation, background, deep research, creative draft, creative critic, planner, verifier, embedding.
- **Modify:** `UserPreferences`, Settings model role UI, backup preferences; migrate existing default/background/deep values.
- **Rule:** model comes from live catalog/configuration only; no role owns a static model ID.
- **Change set:** `feat(models): route configured models by task role`.

### 1.3 Persist tool and approval policy

- **New:** `aura-core/.../agent/policy/ToolPolicy.kt`, `ToolPolicyStore.kt`, `PolicyEngine.kt`.
- **Policy:** enabled, allowed contexts, minimum confirmation, max risk, cost ceiling, app/domain/path scopes, approval expiry.
- **Storage:** non-secret policy in DataStore; secrets remain `SecureDataStore`.
- **Modify:** `ToolExecutor.kt`, `ToolContext`, backup preferences.
- **RED:** policy can only tighten built-in risk; incognito remains a hard lower boundary; expired approval is rejected.
- **Change set:** `feat(policy): add scoped tool controls`.

### 1.4 Upgrade Tools UI from catalog to control center

- **Modify:** `app/.../ui/screens/ToolsScreen.kt`, `ToolsViewModel.kt`.
- **Add:** enabled toggle, risk badge, provider/health status, approval behavior, usage count, last failure, configuration CTA.
- **Preserve:** 48dp targets, Aura tokens, search/grouping.
- **Tests:** ViewModel filtering/policy updates; Compose semantics instrumentation.
- **Change set:** `feat(tools): add policy and health controls`.

### 1.5 Add shared trace and event contracts

- **New:** `aura-core/.../agent/runtime/RunContext.kt`, `AgentTraceEvent.kt`, `TraceSink.kt`.
- **Modify:** `MemoryAugmentedAgenticLoop`, `ToolExecutor`, `CreativeEngine`, WorkManager workers.
- **Fields:** correlationId, runId, stepId, conversationId, projectId, parent event, timestamps; never secrets/full private payloads.
- **RED:** parallel tool events remain attributable and ordered per step.
- **Change set:** `feat(runtime): add correlated agent trace events`.

### 1.6 Surface knowledge/profile extraction feedback

- **New:** one-shot event from memory/KG/profile extraction with counts and source links.
- **Modify:** `MemoryAugmentedAgenticLoop`, `ChatViewModel`, chat snackbar/chip surface.
- **UX:** “Knowledge updated · 3 facts” opens exact records; failures are non-blocking but visible in diagnostics.
- **Change set:** `feat(memory): surface learned knowledge feedback`.

**Phase gate:** no hardcoded model IDs added; all 46 existing tools obey policy without behavior regressions; visual screenshot of Tools controls.

---

## Phase 2 — Real local intelligence and source-addressable document RAG

### 2.1 Implement the selected semantic embedder

- **New:** `SemanticLocalEmbedder.kt`, runtime/model loader, tokenizer only if required.
- **Modify:** `MemoryModule`, `CloudEmbedder` fallback chain.
- **Fallback:** cloud configured → real local semantic → deterministic hash only as degraded last resort, visibly labeled.
- **Tests:** golden similarity pairs, cancellation, corrupt/missing model, concurrency.
- **Change set:** `feat(memory): add real on-device semantic embeddings`.

### 2.2 Ship MemoryDatabase 6→7

- **Modify:** `MemoryEntity` with embedding model/version metadata; `DocumentEntity` with indexing status/error; `MemoryDatabase`, `MemoryModule`.
- **New:** `DocumentChunkEntity`, `DocumentChunkDao` with documentId, ordinal, character/page offsets, text, embedding blob/model, hash.
- **Index:** documentId+ordinal, documentId, hash.
- **Tests:** schema export and migration from v6 fixture preserving every existing row.
- **Change set:** `feat(rag): persist versioned document chunks`.

### 2.3 Make document import transactional and resumable

- **Modify:** `DocumentRepository`, `DocumentTextExtractor`, `DocumentChunker`.
- **New:** `DocumentIndexWorker` using durable document ID and progress.
- **Behavior:** extract → chunk → embed in batches → transactionally publish; cancellation leaves resumable status, not half-visible chunks.
- **Legacy:** existing memory-backed document chunks remain searchable until successful reindex, then deduplicate by source/hash.
- **Change set:** `feat(rag): add resumable document indexing`.

### 2.4 Implement hybrid document retrieval

- **New:** `DocumentRetriever.kt`.
- **Ranking:** lexical candidate score + semantic cosine + document/title match + recency; diversity by document and adjacent-chunk expansion.
- **Output:** stable citations with document ID, chunk ordinal, page/offset, URI.
- **Tests:** paraphrase retrieval, exact code tokens, multi-document diversity, no citation drift.
- **Change set:** `feat(rag): add hybrid cited document retrieval`.

### 2.5 Add document agent tools

- **New:** `DocumentSearchTool.kt`, `DocumentReadTool.kt`, `DocumentListTool.kt`.
- **Modify:** `ToolsModule`, categories, specialist allowlists.
- **Limits:** bounded chunks/characters, no arbitrary URI reading, explicit source citations.
- **Change set:** `feat(tools): expose cited document retrieval`.

### 2.6 Add OCR for scanned documents

- **Path:** Android ML Kit text recognition or the smallest current official on-device OCR dependency selected at execution.
- **Modify:** extractor to detect low-text PDFs/images and route pages through OCR with progress/cancel.
- **Do not:** upload scans silently.
- **Tests:** scanned fixture, rotated page, empty scan, 20MB ceiling.
- **Change set:** `feat(documents): add opt-in on-device OCR`.

### 2.7 Complete the Documents UI

- **Modify:** Documents screen/ViewModel.
- **Add:** indexing state, failure/retry, chunk/source preview, reindex, model/version label, open original, cited search.
- **Visual gate:** empty/importing/ready/failed/offline states on emulator.
- **Change set:** `feat(documents): add indexing and citation controls`.

**M1 gate:** semantic regression suite passes offline; imported documents survive restart and cite exact chunks; backup/restore v6 compatibility remains green.

---

## Phase 3 — Creative artifact, revision, branch, and job substrate

### 3.1 Ship MemoryDatabase 7→8 creative schema

- **New:** `CreativeArtifactEntity.kt`, `CreativeRevisionEntity.kt`, `CreativeBranchEntity.kt`, `CreativeGenerationJobEntity.kt`; DAOs.
- **Modify:** `CreativeProjectEntity` with activeBranchId default, `MemoryDatabase`, `MemoryModule`.
- **Constraints:** foreign keys/cascade, indexes on projectId/branchId/status/updatedAt, unique main branch per project enforced in repository.
- **Tests:** v7→v8 migration and clean-create schema identity.
- **Change set:** `feat(creative): add artifact and revision schema`.

### 3.2 Build the artifact and revision stores

- **New:** `CreativeArtifactStore.kt`, `CreativeRevisionStore.kt`, `CreativeBranchStore.kt`.
- **Operations:** create, revise, branch, compare, restore, soft-delete, lineage, dependency query. Implement ancestry as a depth-capped recursive SQLite CTE rather than loading the entire revision table into Kotlin.
- **Concurrency:** project mutex plus compare-and-set head revision to prevent lost updates.
- **Change set:** `feat(creative): add versioned artifact stores`.

### 3.3 Backfill every legacy project with a main branch

- **New:** idempotent `CreativeLegacyMigrator.kt` with DataStore completion marker.
- **Behavior:** preserve `worldJson`; create main branch; never fabricate artifacts from transient output.
- **Tests:** repeated run, interrupted run, empty/invalid world JSON.
- **Change set:** `feat(creative): backfill project branches safely`.

### 3.4 Replace transient generation output with artifacts

- **Modify:** `CreativeEngine` to return `CreativeGenerationResult`; `CreativeStudioViewModel`; `CreativeProjectScreen`.
- **Behavior:** every brainstorm/outline/draft/rewrite/continuity result creates or revises an artifact; ViewModel holds IDs, not canonical text.
- **UX:** Save as new, replace selection, append, create variant, compare, restore.
- **Change set:** `feat(creative): persist every generated result`.

### 3.5 Add secure creative file storage

- **New:** `CreativeAssetFileStore.kt` using app-private files; sanitized extensions, SHA-256, atomic temp+rename, MIME validation, URI grants for share/export.
- **Rules:** no path traversal, no external URI trusted as owned, deduplicate by hash without merging metadata.
- **Tests:** traversal names, duplicate files, interrupted write, oversized file.
- **Change set:** `feat(creative): add integrity-checked asset storage`.

### 3.6 Add durable generation jobs

- **New:** `CreativeGenerationWorker.kt`, `CreativeJobRepository.kt`.
- **State:** queued/running/waiting-provider/succeeded/failed/cancelled; progress and provider operation ID.
- **Behavior:** idempotent polling, retry-after, cancellation, restart recovery, user-visible cost estimate before REMOTE_COST execution.
- **Change set:** `feat(creative): persist media generation jobs`.

### 3.7 Build gallery, artifact detail, lineage, and revision diff

- **New/modify:** creative UI package: `CreativeArtifactGallery.kt`, `CreativeArtifactDetailScreen.kt`, `CreativeRevisionDiffScreen.kt`; NavGraph routes.
- **States:** text/image/audio/video/3D cards, queued/progress/error/retry, parent/child lineage, branch badge.
- **Visual:** emulator screenshots for 360dp and large screen, dark/light.
- **Change set:** `feat(creative): add artifact gallery and revision history`.

### 3.8 Upgrade backup to `.aura` archive schema 7

- **New:** `AuraArchiveManager.kt`, manifest model, asset importer/exporter.
- **Modify:** `AuraBackup`, `BackupManager`, Backup UI.
- **Archive:** ZIP containing `manifest.json` plus `assets/<sha256>.<ext>`; no secrets/embeddings/running jobs.
- **Security:** ZIP-slip protection, size/count ceilings, hashes, unsupported-version failure, partial restore rollback.
- **Change set:** `feat(backup): archive creative artifacts with integrity`.

**Phase gate:** kill the app mid-generation and restore it; output resumes or fails honestly; every successful result exists after restart; archive round-trip preserves lineage and files.

---

## Phase 4 — Wire the existing capability providers into honest tools

### 4.1 Unify web search routing

- **Modify:** `WebSearchTool`, `BraveSearchTool`, `TavilySearchTool`, `DuckDuckGoSearch`, `ExaSearchProvider`.
- **Behavior:** one `web_search` contract with `provider=auto|prefix`, free-first policy configurable, semantic Exa features when available, identical cited result model.
- **Compatibility:** keep old names as hidden aliases for one release; avoid multiple nearly identical schemas in model context.
- **Change set:** `feat(search): route web search across capabilities`.

### 4.2 Add safe `read_url` through Jina

- **New:** `ReadUrlTool.kt`; extend Jina provider if its current search contract conflates reading.
- **Security:** reuse shared SSRF guard, validate every redirect, block credentials/private hosts unless endpoint is explicitly trusted, enforce content/type/size/time limits.
- **Output:** title, canonical URL, clean text, section anchors, truncation marker.
- **Change set:** `feat(web): add safe cited URL reader`.

### 4.3 Replace hardcoded DALL-E image generation

- **Modify:** `ImageGenTool` to use `CapabilityRouter`/`ImageProvider`; remove hardcoded `dall-e-3` and direct endpoint path.
- **Output:** durable generation job and creative artifact ID, not a temporary URL.
- **Tests:** configured Stability, missing provider, failed download, cost approval, cancellation.
- **Change set:** `refactor(image): route generation through capability fabric`.

### 4.4 Add image edit operations

- **Extend:** `ImageProvider` with declared capabilities rather than assuming every provider supports generate/edit/variation/upscale/outpaint.
- **New tools:** `image_edit`, `image_variation`, `image_upscale` registered only when supported.
- **Artifacts:** parent reference and mask/settings lineage.
- **Change set:** `feat(image): add capability-aware revision tools`.

### 4.5 Add Kling video generation

- **New:** `VideoGenTool.kt`; route through `KlingVideoProvider` and durable polling jobs.
- **Inputs:** prompt, optional source artifact/reference, duration/aspect/motion; model/catalog values from provider.
- **Output:** video artifact with provider operation provenance.
- **Change set:** `feat(video): add durable Kling generation`.

### 4.6 Add World Labs 3D generation

- **New:** `World3DGenTool.kt`; support prompt/reference artifact, polling, preview link/file, metadata.
- **UX:** artifact detail opens supported viewer/browser; never claim offline rendering.
- **Change set:** `feat(3d): expose World Labs generation`.

### 4.7 Make TTS and transcription artifact-aware

- **Modify:** `TtsSpeakTool`, transcription paths, ElevenLabs capability.
- **Behavior:** optional save-to-project; voice/model from live provider configuration; audio and transcript linked as parent/child artifacts.
- **Change set:** `feat(audio): preserve generated speech and transcripts`.

### 4.8 Add capability health and budget UI

- **Modify:** Settings/provider/capabilities UI.
- **Show:** configured, live health, supported operations, selected role, estimated charge class, last failure; no invented model catalog.
- **Change set:** `feat(capabilities): add health and budget dashboard`.

**Phase gate:** five previously dormant provider backends have an end-to-end test and visible route; no tool returns an orphan URL; old image behavior is removed or explicitly deprecated.

---

## Phase 5 — Executable canon, continuity compiler, branches, and simulations

### 5.1 Ship MemoryDatabase 8→9 canon schema

- **New:** canon/simulation/continuity/dependency entities and DAOs defined in Section 5.
- **Tests:** migration preserves projects/artifacts/revisions; indexes and foreign keys asserted.
- **Change set:** `feat(canon): add typed creative world schema`.

### 5.2 Build idempotent world-JSON backfill

- **New:** `WorldBibleCanonMigrator.kt`.
- **Map:** characters, locations, factions, rules, timeline, relationships, objects, and current simulations into typed facts without deleting `worldJson`.
- **Conflict:** duplicate facts are recorded as reviewable issues, not silently overwritten.
- **Change set:** `feat(canon): backfill typed facts from world bibles`.

### 5.3 Add `CanonRepository` and typed patch language

- **New:** `CanonPatch`, operations add/update/retire/relation/state-transition, validator, transaction applier.
- **Rules:** subject existence, relationship endpoints, validity intervals, branch scope, confidence/source required.
- **Tools:** replace repeated one-item world edits with bounded `creative_apply_world_patch` after preview/approval.
- **Change set:** `feat(canon): add validated batch world patches`.

### 5.4 Replace flat context truncation with `CreativeContextRetriever`

- **Inputs:** project, branch, mode, prompt, selected artifact, token/character budget.
- **Always include:** identity, locked rules, current relevant character states.
- **Retrieve:** relevant canon facts, artifacts/revisions, prior scenes, research/doc citations, unresolved continuity issues.
- **Return:** context plus provenance map and truncation diagnostics.
- **Change set:** `feat(creative): build relevance-budgeted project context`.

### 5.5 Implement the continuity compiler

- **New:** `ContinuityCompiler.kt`, deterministic rule checkers, optional LLM critic using creative-critic role.
- **Checks:** identity/state/location/time, knowledge chronology, relationships, world rules, unresolved promises, visual/voice references later.
- **Trigger:** after artifact save with debounce; never on every streaming token.
- **Output:** typed `ContinuityIssueEntity` with evidence and suggested patch.
- **Change set:** `feat(creative): add evidence-backed continuity compiler`.

### 5.6 Build continuity review UI

- **New:** issue panel/filter/detail/diff; navigation from artifact and project.
- **Actions:** accept patch, dismiss with reason, mark intentional exception, jump to evidence.
- **One-shot events:** show “3 continuity issues” after save without blocking writing.
- **Change set:** `feat(creative): add continuity review workflow`.

### 5.7 Implement branch operations and three-way merge

- **Operations:** branch from revision, switch, compare, merge, archive.
- **Merge:** text diff plus canon patch merge; conflicts require explicit resolution.
- **Tests:** divergent text and fact changes, deleted artifacts, repeated merge.
- **Change set:** `feat(creative): add branch diff and merge`.

### 5.8 Replace prose-only simulation with typed state deltas

- **Modify:** simulation mode/records to return narrative, assumptions, state delta, causal edges, confidence, second-order effects, contradictions.
- **Store:** `CreativeSimulationEntity`; do not mutate canon.
- **Change set:** `feat(simulation): persist causal world-state deltas`.

### 5.9 Canonize via impact preview and transaction

- **UI:** Git-style fact/relation/timeline diff, affected artifacts, invalidation count.
- **Apply:** approved subset only, one transaction, source simulation link, undo revision.
- **Change set:** `feat(simulation): canonize approved state changes safely`.

### 5.10 Propagate invalidation through artifact dependencies

- **Behavior:** a changed canon/reference fact marks dependent scenes/images/audio as `needs_review`; it never auto-regenerates paid media.
- **UI:** impact inbox with bulk approve/ignore/regenerate.
- **Change set:** `feat(creative): propagate canon changes to dependent assets`.

**M2 gate:** branch a project, simulate a divergence, canonize selected deltas, show exact affected artifacts, archive/restore, and reproduce the same state.

---

## Phase 6 — Hands 2.0 durable goals and workflow runtime

### 6.1 Ship HandDatabase 2→3 runtime schema

- **New:** workflow/run/step/event/approval/checkpoint entities and DAOs.
- **Modify:** `HandDatabase`, `HandsModule`, exported schema.
- **Tests:** migrate v2 fixture with Hands and run history intact.
- **Change set:** `feat(runtime): add durable workflow and run schema`.

### 6.2 Backfill legacy Hands into linear workflow graphs

- **New:** `LegacyHandGraphMigrator.kt`.
- **Mapping:** each existing tool step becomes node; conditions become entry guards; schedule/variables/policies preserved; legacy Hand remains compatibility source.
- **Idempotency:** deterministic IDs and completion marker.
- **Change set:** `feat(hands): migrate macros to workflow graphs`.

### 6.3 Implement graph validation

- **New:** `WorkflowValidator.kt`.
- **Reject:** missing node, invalid edge, unsafe unbounded loop, unreachable required node, cyclic graph without loop policy, incompatible outputs, secret persistence.
- **Change set:** `feat(runtime): validate workflow graphs`.

### 6.4 Implement core workflow executor

- **Node kinds:** tool, agent, parallel, condition, loop, wait, event, human approval, subworkflow, verify, deliver.
- **New:** `WorkflowRuntime.kt`, node executors, variable resolver with typed JSON values.
- **Rule:** run state changes and event append occur transactionally.
- **Change set:** `feat(runtime): execute typed workflow DAGs`.

### 6.5 Add checkpoints, restart recovery, and idempotency

- **New:** `WorkflowRunWorker`, `RunRecoveryCoordinator`.
- **Behavior:** checkpoint after every terminal step; resume active frontier; deduplicate side-effect retries by idempotency key; stale in-flight steps become retry/review, never assumed success.
- **Fault tests:** kill process before/after tool side effect, network loss, reboot, cancelled WorkManager.
- **Change set:** `feat(runtime): resume goals after interruption`.

### 6.6 Add goals, definitions of done, and budgets

- **New:** `GoalSpec`, `DefinitionOfDone`, `RunBudget`, `GoalService`.
- **Fields:** outcome, constraints, deliverables, deadline, tools/apps/domains, token/cost/time ceiling, approval policy, verification criteria.
- **UI:** review editable plan before start for high-risk goals; low-risk goals can begin immediately under policy.
- **Change set:** `feat(goals): add explicit outcomes and budgets`.

### 6.7 Generalize approval and permission resumption

- **Reuse:** existing Hands approve/resume logic.
- **Add:** durable approval requests scoped to one step/tool/resource/action, expiry, deny reason, “always for this scope” policy proposal.
- **Never:** persist secret runtime values in checkpoints.
- **Change set:** `feat(runtime): add scoped durable approvals`.

### 6.8 Implement postcondition verification and compensation

- **New:** `PostconditionVerifier`, verification contracts on tools, compensation descriptors.
- **Examples:** file hash exists, reminder row exists, UI text changed, artifact saved, HTTP resource status.
- **Status:** succeeded only when required postconditions pass; otherwise `needs_review`/failed.
- **Change set:** `feat(runtime): verify outcomes instead of tool claims`.

### 6.9 Build Goals/Runs UI

- **New:** Goals list, run detail timeline, live parallel branches, approvals inbox, retry/cancel/pause, artifacts/evidence, replay from safe checkpoint.
- **Navigation:** Home and Chat deep links; bottom navigation only if visual verification proves the slot belongs there.
- **Change set:** `feat(goals): add inspectable run timeline`.

### 6.10 Add goal/workflow agent tools

- **Tools:** `goal_create`, `goal_status`, `goal_cancel`, `workflow_list`, `workflow_run`; destructive changes require approval.
- **Chat:** converting a successful ad-hoc sequence into a draft Hand is a proposal, never automatic.
- **Change set:** `feat(tools): expose durable goals and workflows`.

**M3 gate:** start a multi-step goal, force-kill after a side effect, reboot, resume without duplicate effects, approve one pending step, and finish only after postconditions pass.

---

## Phase 7 — MCP client, late tool discovery, and dynamic policy wrapping

### 7.1 Add official MCP client dependencies and core abstractions

- **Files:** version catalog, `aura-core/build.gradle.kts`, new `mcp/` package.
- **New:** `McpServerConfig`, `McpConnection`, `McpClientManager`.
- **Transport:** Streamable HTTP first; SSE compatibility if needed; no Android stdio transport.
- **Change set:** `feat(mcp): add official Kotlin client foundation`.

### 7.2 Persist server configuration securely

- **Non-secret:** name, URL, enabled, trusted-local flag, scopes in Room/DataStore.
- **Secret:** bearer/OAuth tokens in `SecureDataStore` keyed by server ID.
- **Validation:** HTTPS by default; HTTP/private endpoint only through explicit trusted-local setup; no embedded URL credentials.
- **Change set:** `feat(mcp): store scoped server connections`.

### 7.3 Implement connection lifecycle and OAuth

- **Behavior:** connect, initialize, capability negotiation, reconnect/backoff, cancellation, token refresh, health status.
- **OAuth:** use current Android AppAuth-compatible flow where server advertises it; no WebView password capture.
- **Change set:** `feat(mcp): add authenticated connection lifecycle`.

### 7.4 Cache tools, resources, and prompts

- **New:** discovery cache with server/version/ETag-like identity and expiry.
- **Limits:** schema size, tool count, resource bytes, prompt count; malformed server cannot exhaust memory.
- **Change set:** `feat(mcp): cache bounded server capabilities`.

### 7.5 Adapt MCP tools through local `ToolExecutor`

- **New:** `McpToolAdapter` implementing Aura Tool contract.
- **Policy:** local risk classifier + user override; remote annotations are hints only; output truncation/redaction; run/step provenance.
- **Change set:** `feat(mcp): wrap remote tools in Aura policy`.

### 7.6 Add semantic late tool search

- **New:** `ToolSearchIndex`, `tool_search` internal mechanism.
- **Behavior:** model sees a small core tool set plus retrieved schemas relevant to the turn/goal; exact tool names always retrievable.
- **Tests:** 1,000 fake tools, adversarial descriptions, deterministic top-k, schema budget.
- **Change set:** `feat(agent): load dynamic tools on demand`.

### 7.7 Build MCP management UI and audit

- **Screens:** add/pair/auth server, tools/resources/prompts, per-tool policy, health/errors, last calls, disconnect/delete credentials.
- **Visual/security:** prominent trust warning for local/private servers; never display tokens.
- **Change set:** `feat(mcp): add server and tool control center`.

**Phase gate:** connect to a fixture server, discover a tool only after search, call it through policy, deny a write action, refresh credentials, and recover from disconnect.

---

## Phase 8 — Aura Bridge, filesystem, shell, code, and browser ecosystem

### 8.1 Add shared protocol and desktop modules

- **Modify:** `settings.gradle.kts` with `:aura-protocol` and `:aura-bridge` only after Phase 7 is stable.
- **New:** pure Kotlin/JVM protocol module and Ktor/JVM bridge application.
- **Do not:** put Android classes in shared protocol.
- **Change set:** `feat(bridge): add shared desktop protocol modules`.

### 8.2 Implement explicit device pairing

- **Flow:** desktop displays one-time QR/code; phone pins bridge public key and receives revocable scoped token.
- **Transport:** TLS/WebSocket or MCP Streamable HTTP; bind localhost by default; remote use requires explicit network interface/Tailscale-style private network.
- **Change set:** `feat(bridge): add pinned device pairing`.

### 8.3 Add scoped workspace filesystem tools

- **Bridge tools:** list/read/write/patch/search/stat within user-approved roots.
- **Security:** canonical path checks, symlink escape prevention, byte ceilings, atomic writes, hash preconditions, no secret-file read by default.
- **Change set:** `feat(bridge): add scoped workspace operations`.

### 8.4 Add approval-gated shell execution

- **Command contract:** argv array, cwd in approved root, timeout, environment allowlist, output cap.
- **Profiles:** read-only, build/test, full; destructive/network commands always elevated.
- **No:** shell string interpolation from model without parsing/review.
- **Change set:** `feat(bridge): add policy-gated process execution`.

### 8.5 Add code sandbox jobs

- **Preferred:** container/sandbox backend when available; fallback to isolated temp worktree/process profile with strict ceilings.
- **Artifacts:** stdout/stderr/files/test report attached to AgentStep.
- **Change set:** `feat(bridge): add isolated code execution jobs`.

### 8.6 Integrate browser automation through existing MCP servers

- **Use:** official Playwright MCP or another user-approved browser MCP server on desktop.
- **Aura work:** connection template, browser tool policy presets, screenshots/download artifact sync, login-profile warning.
- **Do not:** implement a competing browser engine in Android.
- **Change set:** `feat(browser): integrate desktop browser MCP`.

### 8.7 Synchronize artifacts between phone and Bridge

- **Protocol:** chunked upload/download, hash verification, resume, MIME/size policy, temporary URL expiry.
- **Storage:** imported file becomes Aura artifact with source bridge provenance.
- **Change set:** `feat(bridge): sync verified artifacts`.

### 8.8 Build Bridge status and control UI

- **Show:** paired devices, online state, scopes, roots, active jobs, emergency revoke, logs, version mismatch.
- **Change set:** `feat(bridge): add connection and scope controls`.

**Phase gate:** pair, run a test in a scoped workspace, read a file, use Playwright MCP, transfer an artifact, revoke bridge, and prove all subsequent calls fail.

---

## Phase 9 — Android computer use and opt-in Ground sessions

### 9.1 Add computer-use policy and onboarding

- **Manifest/resources:** AccessibilityService declaration and configuration; Settings onboarding.
- **Default deny:** password managers, banking/payment apps, system credential/security settings; per-app allowlist.
- **Personal-use note:** Play policy is not a release constraint, but security boundaries still apply.
- **Change set:** `feat(computer): add explicit accessibility setup`.

### 9.2 Build sanitized UI-tree snapshots

- **New:** `AuraAccessibilityService`, `UiTreeSnapshot`, `UiNodeSanitizer`, holder/gateway.
- **Redact:** password/sensitive nodes, notification/IME secrets, oversized text, disallowed packages.
- **Lifecycle:** no persistence unless attached to an approved run artifact.
- **Change set:** `feat(computer): expose redacted UI structure`.

### 9.3 Add read-only computer tools first

- **Tools:** `ui_observe`, `ui_find`, `ui_wait`.
- **Output:** package/window/node IDs, bounds, roles, text/content descriptions, freshness.
- **Change set:** `feat(computer): add inspectable read-only tools`.

### 9.4 Add action tools under policy

- **Tools:** `ui_tap`, `ui_type`, `ui_scroll`, `ui_back`.
- **Safety:** target must come from fresh snapshot; sensitive fields blocked; messages/deletes/account changes require scoped approval.
- **Change set:** `feat(computer): add verified Android actions`.

### 9.5 Add perception-action verification loop

- **New:** `ComputerUseCoordinator` plans one bounded action at a time, then observes and verifies expected state.
- **Stop:** stale tree, unexpected package, repeated failure, policy boundary, user stop.
- **Change set:** `feat(computer): verify every UI transition`.

### 9.6 Correct MediaProjection for API 34/35

- **Manifest:** `FOREGROUND_SERVICE_MEDIA_PROJECTION` and service type.
- **New:** user-started capture foreground service; one consent/token per session; one `createVirtualDisplay()` per token.
- **Modify:** `CaptureScreenTool`, `ScreenCaptureHolder` to fail honestly without active session.
- **Tests:** service lifecycle where feasible plus device instrumentation.
- **Change set:** `fix(screen): comply with modern MediaProjection lifecycle`.

### 9.7 Create `GroundSessionService`

- **User action:** explicit Start Ground / Stop Ground; persistent notification and emergency stop.
- **Inputs:** selected app screen/UI events, optional camera/mic only when separately enabled.
- **Retention:** raw frames/audio discarded after local/remote inference; summaries tagged with source/time.
- **Change set:** `feat(ground): add visible user-controlled sessions`.

### 9.8 Add Ground event summarization and trigger handoff

- **New:** bounded `GroundEvent` queue with dedupe/rate limit; route relevant events to workflow TriggerBus.
- **Never:** execute writes from raw perception without a run policy and approval.
- **Change set:** `feat(ground): convert perception into safe events`.

### 9.9 Build replay, privacy, and stop controls

- **UI:** current observed app, redaction status, events/actions, approvals, retention, delete session, emergency revoke Accessibility.
- **Visual/device gate:** complete one safe cross-app workflow on emulator/physical device and review every frame/action.
- **Change set:** `feat(ground): add transparent session replay`.

---

## Phase 10 — Temporal world model, triggers, and opportunity engine

### 10.1 Ship MemoryDatabase 9→10 world-model schema

- **New:** beliefs, evidence, world events, opportunities; indexes by subject/predicate/validity/status.
- **Modify:** backup schema later in 10.8.
- **Change set:** `feat(world): add temporal belief and evidence schema`.

### 10.2 Ingest source-aware observations

- **Sources:** user statements, profile/KG extraction, calendar/tasks, notifications, Ground summaries, verified tool results.
- **Each belief:** value, confidence, validFrom/validTo, evidence IDs, privacy class, last verified, supersededBy.
- **Change set:** `feat(world): ingest attributed observations`.

### 10.3 Resolve contradictions without erasing history

- **New:** `BeliefResolver` applying recency/source/confidence and asking user for high-impact ambiguity.
- **UI:** contradiction inbox; current versus superseded values and evidence.
- **Change set:** `feat(world): reconcile temporal contradictions`.

### 10.4 Feed the world model into context safely

- **New:** `WorldContextRetriever`; only relevant, current, policy-allowed beliefs enter prompts.
- **Diagnostics:** source chips and “why included.”
- **Change set:** `feat(agent): ground context in current beliefs`.

### 10.5 Add extensible TriggerBus

- **Adapters:** schedule, Android broadcasts/notifications, Ground events, Bridge webhook, WebSocket; MQTT adapter optional behind dependency/profile.
- **Behavior:** dedupe, debounce, replay protection, signed remote events, trigger-to-workflow mapping.
- **Change set:** `feat(runtime): add event-driven workflow triggers`.

### 10.6 Implement OpportunityEngine

- **Inputs:** goals, tasks/calendar, world events, creative issues, run failures.
- **Output:** ranked proposals with benefit, urgency, confidence, cost/risk, evidence, suggested workflow.
- **Default:** proposal only; auto-run limited to explicitly approved read-only policies.
- **Change set:** `feat(proactive): add evidence-backed opportunities`.

### 10.7 Build world/opportunity UI

- **Screens:** timeline, current beliefs, sources, contradictions, proposed actions, dismiss/snooze/approve, retention controls.
- **Change set:** `feat(world): add inspectable personal state and proposals`.

### 10.8 Ship Proactive DB 3→4 and backup schema 8

- **Fields:** triggerSourceId, goalId, runId, actionState, expiresAt/retention.
- **Backup:** workflows/policies/world model; exclude transient event payloads/raw frames/secrets.
- **Migration tests:** v3 fixture and schema round-trip.
- **Change set:** `feat(backup): preserve durable action and world state`.

**M4 gate:** an external/Android event creates one deduplicated proposal, user approves it, durable workflow executes, computer/Bridge/MCP step verifies outcome, and every source/action is replayable.

---

## Phase 11 — Real subagents and Creative Council

### 11.1 Define isolated subagent contracts

- **New:** `SubagentSpec`, `SubagentTask`, `SubagentResult`, `ContextBundle`, structured result schemas.
- **Fields:** role, objective, context/artifact refs, model role, tool allowlist, budget, deadline, output schema.
- **No:** shared mutable conversation history.
- **Change set:** `feat(agents): define isolated worker contracts`.

### 11.2 Implement `SubagentManager`

- **Behavior:** spawn/cancel/timeout, parallel execution, parent-child budgets, progress events, result validation, no nested unbounded fan-out.
- **Runtime:** each subagent is an AgentStep/child run in Hands 2.0.
- **Change set:** `feat(agents): orchestrate bounded subagent runs`.

### 11.3 Add shared blackboard through artifacts/events

- **Use:** subagents read immutable artifact revisions and append proposals/criticisms; director chooses/merges.
- **Avoid:** passing private hidden reasoning; store structured evidence and decisions.
- **Change set:** `feat(agents): add artifact-backed collaboration`.

### 11.4 Add outcome-based model routing

- **Record:** task role, provider/model, latency, cost class, verifier result, user selection/reaction.
- **Routing:** recommendations first; automatic route only after enough evidence and user opt-in.
- **Change set:** `feat(models): learn role performance without hardcodes`.

### 11.5 Implement Creative Council roles

- **Roles:** Director, Writer, Story Editor, Continuity Editor, World Simulator, Researcher, Art Director, Cinematographer, Sound Designer, Audience Critic.
- **Each:** explicit input/output schema, relevant tools, model role, budget, stop conditions.
- **Existing static Writer/Creative specialist:** compatibility UI only; council uses real subagents.
- **Change set:** `feat(creative): add structured council roles`.

### 11.6 Build council workflows

- **Templates:** concept exploration, scene drafting, continuity review, storyboard, trailer, podcast drama, world simulation.
- **Pattern:** independent proposals → critics → director synthesis → user selection → artifact commit.
- **Change set:** `feat(creative): orchestrate multi-agent production`.

### 11.7 Build proposal/constraint UI

- **UI:** proposal cards, evidence, compare, accept/merge/reject, lock constraints, role progress, budget/cancel.
- **Never:** expose raw chain of thought; show concise rationale and source artifacts.
- **Change set:** `feat(creative): add inspectable council decisions`.

### 11.8 Add adversarial council tests

- **Cases:** one worker fails, invalid schema, contradictory proposals, budget exhaustion, cancellation, prompt injection in research artifact, stale canon.
- **Change set:** `test(creative): harden multi-agent council`.

---

## Phase 12 — Personality Studio, Taste Twin, and multimodal consistency

### 12.1 Ship UserProfileDatabase 1→2 personality schema

- **Add:** structured profile JSON or dedicated entity: preset, identity text, tone, verbosity, initiative, humor, challenge level, explanation style, risk posture, adaptation enabled.
- **Migration:** existing `customIdentity` imported without deletion; no silent personality change.
- **Change set:** `feat(personality): persist structured Aura profiles`.

### 12.2 Build Personality Studio and presets

- **UI:** presets, sliders/selectors, preview conversation, specialist inheritance/override, reset/export.
- **Brain:** compile structured profile plus SOUL identity deterministically; user text takes precedence within safe boundaries.
- **Change set:** `feat(personality): add previewable profile editor`.

### 12.3 Ship MemoryDatabase 10→11 taste schema

- **New:** preference signals, style profiles, reference identities, routing outcomes.
- **Signals:** accepted/rejected variants, edits, reactions, repeated rewrites, palette/camera/voice/music selections.
- **Change set:** `feat(taste): add reversible preference evidence`.

### 12.4 Implement TasteEngine

- **Start:** retrieval + weighted examples + explicit style profile; no fine-tuning.
- **Use:** prompt context, candidate ranking, provider recommendation, critic rubric.
- **Control:** show why, edit/disable/delete signals, project versus global scope.
- **Change set:** `feat(taste): personalize creative ranking transparently`.

### 12.5 Add multimodal reference packs

- **Entities/UI:** character identity, location architecture, props/logos, palette, voice, music motif, cinematography; reference artifacts with locked/optional status.
- **Generation:** providers receive only supported reference types; unsupported constraints shown honestly.
- **Change set:** `feat(creative): add cross-media reference identities`.

### 12.6 Add cross-media consistency analyzers

- **Analyzers:** visual identity, palette/setting, voice identity/metadata, motif usage, timeline/time-of-day; combine metadata/deterministic checks with configured vision/audio critic.
- **Output:** continuity issues with evidence and severity, never silent regeneration.
- **Change set:** `feat(creative): audit identity across media`.

### 12.7 Expand audio/media capability interfaces

- **Kinds:** voice design, speech-to-speech/dubbing, sound effects, music generation/stems; providers added only against verified current APIs/subscriptions.
- **Artifact lineage:** script → voice → mix/stem; cost/rights metadata.
- **Change set:** `feat(media): add extensible audio production capabilities`.

### 12.8 Add dependency-guided regeneration

- **Flow:** changed reference/canon shows impacted artifacts, groups compatible regeneration jobs, estimates cost, lets user select.
- **Change set:** `feat(creative): regenerate only approved dependents`.

**Phase gate:** choose between variants, observe editable Taste signals, change a reference identity, detect affected assets, and regenerate only selected dependents with lineage intact.

---

## Phase 13 — Creative canvas, storyboard, timeline, and production pipelines

### 13.1 Build project canvas model and surface

- **Model:** saved node positions/groups/views separate from artifact graph.
- **UI:** Compose zoom/pan canvas, artifact cards, dependency edges, branch/filter, large-project virtualization.
- **Change set:** `feat(creative): add persistent artifact canvas`.

### 13.2 Add document editor with selection actions

- **Actions:** rewrite/expand/shorten/tone/continue/continuity check/reference; each creates a revision diff.
- **Safety:** never overwrite without revision; streaming draft can cancel and keep partial variant.
- **Change set:** `feat(creative): add revision-safe document editing`.

### 13.3 Add storyboard editor

- **Model:** ordered shots with script beat, frame artifact, duration, camera, location/characters, generation state.
- **UI:** drag reorder, generate variants, compare, lock reference, continuity warnings.
- **Change set:** `feat(creative): add storyboard production surface`.

### 13.4 Add audio/video timeline

- **Model:** tracks, clips, offsets, trim metadata, transitions, voice/music/SFX links.
- **Mobile scope:** planning/preview metadata and lightweight playback; final heavy render delegated to Bridge.
- **Change set:** `feat(creative): add editable production timeline`.

### 13.5 Add pipeline templates

- **Templates:** novel, screenplay, comic, short film, trailer, podcast drama, RPG campaign, visual novel, brand campaign.
- **Implementation:** versioned Hands 2.0 workflow definitions with required artifacts/roles/gates; user can inspect/fork.
- **Change set:** `feat(creative): ship reusable production workflows`.

### 13.6 Add portable exports

- **Outputs:** Markdown/DOCX/EPUB/Fountain/JSON package, storyboard PDF, OpenTimelineIO/FCPXML/EDL where appropriate, `.aura` archive.
- **Bridge:** optional rough-cut render job; exported files return as artifacts.
- **Change set:** `feat(creative): export editable production packages`.

### 13.7 Add Live Director mode

- **Flow:** user speaks/types direction while a production workflow is paused/running; director updates constraints and schedules new variants without corrupting approved artifacts.
- **No:** continuous mic without visible active session.
- **Change set:** `feat(creative): add interactive director sessions`.

### 13.8 Add branch/replay UX across canvas and timeline

- **UI:** branch from any revision/shot, compare timelines, replay council decisions, merge approved elements.
- **Change set:** `feat(creative): unify branch history across surfaces`.

**M5 gate:** run the flagship mini-production scenario from brief to editable/exported project, branch one creative decision, preserve references/continuity, and replay every generated artifact and council decision.

---

## Phase 14 — Safe self-improvement, hardening, and release

### 14.1 Build local evaluation harness

- **New:** `AgentEvalScenario`, fixture tools/providers, deterministic replay, scorecards.
- **Suites:** tool selection, RAG citation, workflow recovery, postconditions, creative continuity, preference ranking, policy refusal.
- **Change set:** `feat(eval): add reproducible agent and creative benchmarks`.

### 14.2 Record outcome evidence for improvement proposals

- **Signals:** verifier results, user corrections, approval/denial, model/tool failures, selected variants, latency/cost class.
- **Privacy:** local only; retention and deletion controls.
- **Change set:** `feat(eval): collect local outcome evidence`.

### 14.3 Propose—not silently apply—improvements

- **Proposal kinds:** workflow/Hand, skill instructions, model routing preference, personality/taste rule, tool policy.
- **Each proposal:** evidence, predicted benefit, diff, eval result, rollback.
- **Change set:** `feat(agent): add evaluated improvement proposals`.

### 14.4 Add approval, activation, and rollback

- **State:** draft → evaluated → approved → active → reverted.
- **No source-code mutation in this milestone.** Bridge may generate a Git patch artifact, but Aura cannot apply it automatically.
- **Change set:** `feat(agent): safely activate reversible improvements`.

### 14.5 Run full security review

- **Cases:** MCP prompt injection/tool spoofing, OAuth/token leakage, SSRF/redirect rebinding, ZIP slip, path/symlink traversal, shell injection, malicious archive, Accessibility sensitive fields, stale UI nodes, Ground retention, approval replay, cost bypass, artifact MIME spoofing.
- **Tools:** static scan plus explicit tests; fix every P0/P1 before release.
- **Change set:** `test(security): harden expanded agent surface`.

### 14.6 Run reliability and performance review

- **Faults:** process kill at every workflow state, reboot, provider timeout/rate limit, bridge/MCP disconnect, URI permission loss, corrupt DB JSON, disk full, cancelled media job.
- **Scale:** 10k memories, 2k document chunks, 1k artifacts, 500 workflow steps/events, 1k dynamic tools.
- **Budgets:** no main-thread I/O; bounded prompts/results/events; gallery/canvas stays responsive.
- **Change set:** `perf(runtime): validate large personal datasets`.

### 14.7 Perform mandatory live visual verification

- **Devices:** API 30 Pixel 5 class and API 35 modern phone; 360dp/large display; dark/light; 1.0x/1.3x font.
- **Flows:** onboarding, Tools policies, Documents RAG, Creative artifact/gallery/canon, Goals run/approval, MCP/Bridge, Ground, Council, Taste, canvas/timeline, archive restore.
- **Evidence:** screenshots under `.hermes/screenshots/beyond-sota/`; no SOTA claim without live rendering.
- **Fix:** duplicate icons, clipped controls, invisible empty states, nav dead ends, back behavior, 48dp targets, keyboard/inset issues.
- **Change set:** `fix(ui): close live beyond-SOTA visual findings`.

### 14.8 Final release gate and artifact

Run fresh, not from cached prior output:

```bash
./gradlew --rerun-tasks \
  :aura-core:testDebugUnitTest \
  :app:testDebugUnitTest \
  :aura-core:connectedDebugAndroidTest \
  :app:connectedDebugAndroidTest \
  :app:lintDebug \
  :app:assembleDebug \
  --continue --console=plain
```

Then:

- verify migrations from every supported schema fixture;
- export → uninstall/reinstall → restore `.aura` archive;
- complete flagship creative and durable-agent acceptance scenarios;
- inspect APK, manifest permissions, network security, Room schemas, and secrets scan;
- if/when a real Git remote is restored: review diff, atomic history, push, watch CI to green, create downloadable GitHub Release APK, verify URL.

**Change set:** `release(android): ship Aura Creative and Action OS milestone`.

---

## 7. TDD and verification protocol for every atomic task

1. **RED:** write the smallest behavior/migration/policy test and run it alone; capture expected failure.
2. **GREEN:** implement the minimum behavior; rerun focused test.
3. **REFACTOR:** remove duplication, preserve public contracts, rerun focused suite.
4. **Phase gate:** full core/app unit tests + lint + assembleDebug.
5. **Migration task:** exported schema + `MigrationTestHelper` from real prior schema.
6. **UI task:** ViewModel test + Compose semantics/instrumentation where behavior matters + emulator screenshot.
7. **Provider task:** MockWebServer contract tests covering headers, body, error, timeout, retry, cancellation, secrets.
8. **Workflow task:** process-death/idempotency fault test.
9. **Security-sensitive task:** negative tests precede happy path.
10. **Claim discipline:** never report “done/green/SOTA” without tool output from that execution turn.

### Per-task commands

```bash
# Focused core test
./gradlew :aura-core:testDebugUnitTest --tests 'com.aura.<TestClass>' --console=plain

# Focused app test
./gradlew :app:testDebugUnitTest --tests 'com.aura.<TestClass>' --console=plain

# Phase gate
./gradlew :aura-core:testDebugUnitTest :app:testDebugUnitTest :app:lintDebug :app:assembleDebug --continue --console=plain

# Device/migration/UI gate
./gradlew :aura-core:connectedDebugAndroidTest :app:connectedDebugAndroidTest --continue --console=plain
```

Do not run `pytest`; this is a Kotlin/Gradle project.

---

## 8. Security and privacy acceptance checklist

- [ ] Hardcoded model-ID scan returns zero production decisions.
- [ ] Secret-pattern scan finds no API keys/tokens in source, Room, archive, logs, traces, or UI state.
- [ ] All provider credentials remain in `SecureDataStore`.
- [ ] Every URL/redirect fetch passes shared SSRF checks unless user explicitly trusts a local MCP/Bridge endpoint.
- [ ] MCP remote risk metadata never lowers local risk.
- [ ] Dynamic schemas and outputs are size-bounded.
- [ ] Bridge paths are canonicalized and symlink-escape tested.
- [ ] Shell tools use argv/cwd/env structures and explicit scopes.
- [ ] Accessibility defaults deny sensitive packages and fields.
- [ ] MediaProjection asks consent for every session and uses proper foreground service.
- [ ] Ground raw frames/audio are ephemeral by default.
- [ ] Costed media/model calls have budget/approval enforcement.
- [ ] Postcondition verification is required for externally visible success.
- [ ] Archive restore blocks ZIP slip, bombs, hash mismatch, unsupported schema, and partial corruption.
- [ ] Self-improvement remains proposal/eval/approval/rollback only.

---

## 9. Atomic delivery strategy

Because this source tree currently lacks Git:

- Treat each numbered task above as one atomic change set.
- At the end of each task, save `git diff`-style patch evidence or a file manifest until repository lineage is restored.
- Once Git is restored, preserve task boundaries; do not squash the whole program into one commit.
- Each eventual commit message states the behavior, not the mechanism, and includes test-count delta when meaningful.
- Never mix schema migration with unrelated UI cleanup.
- Never stage multiple files blindly; verify staged paths before commit.
- Run full build after every 3–4 atomic changes and at every phase boundary.
- No “continue?” prompts between tasks after the user starts a milestone. Pause only for a real external blocker or an explicitly user-owned product decision.

Recommended implementation sessions:

1. M1 Foundation
2. M2 Creative Kernel
3. M3 Durable Runtime
4. MCP + Bridge
5. Computer Use + Ground + World Model
6. Subagents + Council
7. Taste + Multimodal
8. Canvas + Production
9. Hardening + Release

---

## 10. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Scope explodes across 15 phases | permanent half-built surfaces | milestone gates; no later UI before substrate passes |
| Media generation produces orphan URLs | lost work | artifact/job substrate precedes provider tools |
| Workflow retry duplicates side effects | messages/files/actions repeated | idempotency keys + postcondition checks + checkpoints |
| MCP server lies about risk/schema | remote write or prompt injection | local classifier/policy, caps, approval, audit |
| Accessibility becomes a universal bypass | catastrophic privacy/action risk | denylist, fresh-node requirement, one-action loop, confirmations, emergency stop |
| Ground becomes surveillance | user loses control | visible session, explicit sources, ephemeral raw data, delete controls |
| Hash embedder replacement invalidates vectors | recall quality/regression | model/version columns + resumable rebuild + golden benchmark |
| World/creative JSON migration loses canon | project corruption | dual-read/backfill, no deletion, diff verification, archive before migration |
| Canvas/timeline performs poorly | unusable premium UX | virtualization, profiling, large-project fixture, Bridge rendering |
| Taste model silently changes voice | trust loss | evidence UI, global/project scope, reversible signals, opt-in automation |
| Subagents burn cost or fan out forever | cost/latency runaway | parent budgets, depth/fan-out ceilings, cancellation, structured results |
| New plan repeats completed work | plan cascade | baseline reconciliation section is authoritative |
| No Git repository | no safe atomic history | resolve lineage in Phase 0; patches/change manifests until then |

---

## 11. Definition of “beyond SOTA” for Aura

Aura is not beyond SOTA merely because it has more providers or tools. The claim is allowed only when all of these are demonstrably true:

1. **Continuity:** cross-project creative work persists as versioned artifacts and typed canon, not prompt history.
2. **Executability:** world changes can be simulated, diffed, selectively canonized, and propagated.
3. **Editability:** generated text/media has revisions, branches, references, canvas/timeline placement, and portable export.
4. **Agency:** goals survive restarts and finish only after verified postconditions.
5. **Openness:** MCP and Bridge add tools dynamically without bypassing local policy.
6. **Embodiment:** Aura can observe and safely operate Android under visible user control.
7. **Understanding:** temporal beliefs retain evidence, uncertainty, and superseded history.
8. **Collaboration:** real isolated subagents produce structured proposals under budget and cancellation.
9. **Personalization:** Taste Twin learns from reversible evidence rather than hidden prompt drift.
10. **Trust:** every action, decision, artifact, source, approval, and failure is inspectable and replayable.
11. **Verification:** flagship scenarios are exercised on a real emulator/device with screenshots and fault injection.
12. **Portability:** a `.aura` archive can reconstruct projects, workflows, world state, and assets without secrets.

---

## 12. Primary technical references to re-check at execution time

- Official Kotlin MCP SDK: `https://github.com/modelcontextprotocol/kotlin-sdk`
- MCP Kotlin API docs: `https://modelcontextprotocol.github.io/kotlin-sdk/`
- ONNX Runtime Mobile: `https://onnxruntime.ai/docs/tutorials/mobile/`
- MediaPipe Text Embedder Android: `https://ai.google.dev/edge/mediapipe/solutions/text/text_embedder/android`
- Android AccessibilityService: `https://developer.android.com/reference/android/accessibilityservice/AccessibilityService`
- Android MediaProjection: `https://developer.android.com/media/grow/media-projection`
- Android foreground service types: `https://developer.android.com/develop/background-work/services/fgs/service-types`

Versions must be verified when each dependency phase begins; do not copy version numbers from this plan or model memory.

---

## 13. Final completion checklist

- [ ] M1: real local semantic RAG and shared routing/policy are production-ready.
- [ ] M2: every creative result is a persistent versioned artifact; canon/simulation/continuity are typed.
- [ ] M3: durable goals resume after process death and verify outcomes.
- [ ] M4: MCP/Bridge/computer/Ground/world-model paths are policy-safe and replayable.
- [ ] M5: Creative Council, Taste Twin, reference consistency, canvas/timeline, and pipelines complete the flagship production.
- [ ] M6: full security, fault, migration, scale, build, emulator, archive, and downloadable APK gates pass.
- [ ] No old expansion item was duplicated.
- [ ] No code path hardcodes a model ID.
- [ ] No output is declared successful without execution evidence.

When these boxes are green, Aura is no longer “an Android assistant with many features.” It is a durable personal action system and executable creative-world system sharing one trustworthy kernel.
