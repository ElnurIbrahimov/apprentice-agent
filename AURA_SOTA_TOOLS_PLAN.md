# Aura State-of-the-Art Tools Plan

**Created:** 2026-03-22
**Research basis:** 5 parallel agents researching Perplexity, Cursor, LATS, Aider, VOYAGER, and 30+ other systems

---

## Tool 1: Deep Research (SOTA Target: Perplexity-level)

**Current state:** 1,055 LOC. Basic multi-query search + summarize.
**Gap:** No query decomposition, no citation scoring, no contradiction detection, no stopping criteria.

### Architecture: Outline-First Hierarchical RAG

```
User query
    ↓
1. PLANNER: Decompose into perspective-guided sub-queries (STORM pattern)
   - Generate 3-5 perspectives (e.g., "Economist", "Skeptic", "Technical Expert")
   - Each perspective produces 3-5 targeted questions
   - Output: research outline (tree structure)
    ↓
2. RETRIEVER: Hybrid BM25 + dense + MMR per sub-query
   - BM25 via SQLite-FTS5 (already have this in memory store)
   - Dense via nomic-embed-text (already have this)
   - MMR (λ=0.7) for diversity — NEW
   - Cross-encoder reranking (ms-marco-MiniLM — already have this)
    ↓
3. EVALUATOR: Citation quality scoring — NEW
   - Domain authority score (TLD weighting: .edu=1.0, .gov=0.95, .org=0.8)
   - Recency decay: score *= exp(-λ * age_days)
   - Redundancy filtering: deduplicate near-identical sources
    ↓
4. VERIFIER: Multi-hop claim verification (RARR pattern) — NEW
   - Extract claims from each source
   - Cross-verify claims against other sources
   - Flag contradictions with confidence scores
   - Hallucinaton drops from ~18% to ~4% with 3-hop verification
    ↓
5. SYNTHESIZER: Section-by-section generation anchored to citations
   - Each claim tagged with [source_id]
   - Post-filter: reject any claim without citation anchor
    ↓
6. STOPPING: Information saturation — NEW
   - Track new_entities / total_entities per round
   - Stop when < 5% new entities AND < 5% new relations
   - Hard cap: max 5 rounds
```

### Key algorithms to implement:
1. **MMR (Maximal Marginal Relevance):** `score = λ * relevance - (1-λ) * max_similarity_to_selected` — ~50 LOC
2. **Perspective-guided decomposition:** STORM-style persona prompts — ~100 LOC
3. **Citation anchoring + post-filter:** Regex cross-check citations against retrieval set — ~80 LOC
4. **Entity saturation stopping:** Track named entities per round, stop at <5% novelty — ~40 LOC
5. **Contradiction detection:** Compare claims across sources, surface disagreements — ~150 LOC

### Expected improvement:
- Citation precision: 50% → 93%
- Coverage: +34-47% (from outline-first planning)
- Hallucination: 18% → 4% (from multi-hop verification)

---

## Tool 2: Code Intelligence Stack (SOTA Target: Cursor-level)

**Current state:** codebase_index (1,291) + code_intelligence (1,283) + code_search (692) = 3,266 LOC
**Gap:** No incremental indexing, no content-hash caching, no cross-file dependency graph, no AST-aware chunking.

### Architecture: Hybrid tree-sitter + Embeddings + Content-Hash Cache

```
Codebase
    ↓
1. PARSER: tree-sitter AST extraction (already partially have this)
   - Chunk by function/class boundaries (cAST pattern, +4.3% Recall@5)
   - Extract: definitions, imports, exports, call sites
   - Output: chunks[] with metadata (file, start_line, end_line, type, name)
    ↓
2. INDEXER: Content-hash caching (Cursor pattern) — NEW
   - SHA256 of each chunk's content
   - Only re-embed chunks whose hash changed
   - Background async indexing (non-blocking)
   - Store: {content_hash → embedding_vector} in SQLite
    ↓
3. DEPENDENCY GRAPH: Import + call graph — NEW
   - Parse import statements → build directed graph
   - Parse function calls → add call edges
   - Store in NetworkX (reuse existing KG infrastructure)
   - Query: "what depends on this function?" in O(1)
    ↓
4. RETRIEVER: Multi-signal context selection
   - Semantic search (embedding cosine similarity)
   - Structural search (dependency graph traversal)
   - Recency boost (recently edited files weighted higher)
   - Aider-style repo map for compact LLM context
    ↓
5. CONTEXT BUILDER: Smart prompt assembly
   - Budget-aware: fit within model's context window
   - Priority: direct dependencies > semantic matches > recent edits
   - Include: function signatures of related code (not full bodies)
```

### Key algorithms to implement:
1. **cAST chunking:** tree-sitter parse → chunk at function/class boundaries — ~200 LOC
2. **Content-hash cache:** SHA256 → embedding lookup, skip unchanged — ~100 LOC
3. **Import graph builder:** Parse imports → directed graph — ~150 LOC
4. **Incremental re-indexing:** Watch file changes → re-chunk only changed files — ~100 LOC
5. **Aider-style repo map:** Compact symbol listing for LLM context — ~150 LOC

### Expected improvement:
- Index rebuild: minutes → seconds (incremental)
- Retrieval quality: +4.3% Recall@5 (from AST chunking)
- Context relevance: significantly better (dependency-aware selection)

---

## Tool 3: MCTS Reasoning (SOTA Target: LATS-level)

**Current state:** 996 LOC. Implemented but DEAD — never wired into agent (reasoning_tree never assigned).
**Gap:** Not connected to Strategy Bandit, no tool integration, no adaptive depth, basic UCB1.

### Architecture: LATS with Tool Integration + Strategy Bandit Routing

```
Query classified by Strategy Bandit
    ↓ (if MCTS selected)
1. ROOT: Create root node from query + current context
    ↓
2. SELECT: UCT with confidence weighting (Marco-o1 pattern) — UPGRADE
   - UCT = Q/N + C * sqrt(ln(N_parent) / N) * confidence_weight
   - confidence_weight from model's token-level log-probabilities
    ↓
3. EXPAND: Sample n candidate actions from LLM — EXISTING (upgrade)
   - Actions can be: reasoning steps OR tool calls (LATS pattern)
   - Tool calls: web_search, code_executor, calculator, memory_recall
   - Each action creates a new child node
    ↓
4. EVALUATE: Dual scoring — NEW
   - Internal: LLM self-evaluation ("is this path promising?" 0-1)
   - External: Tool execution feedback (if action was a tool call)
   - Process Reward Model: per-step quality (not just final answer)
    ↓
5. BACKPROPAGATE: Update Q-values up the tree — EXISTING
    ↓
6. ADAPTIVE STOPPING: — NEW
   - Stop if: confidence > 0.9 (high certainty)
   - Stop if: Q-value plateau (no improvement in last 10% of budget)
   - Stop if: time budget exhausted
   - Stop if: external verification passes (code runs, proof checks)
    ↓
7. WIRE INTO AGENT: — CRITICAL
   - Assign reasoning_tree in agent.__init__()
   - Strategy Bandit selects MCTS for: MATH, CODE, PLANNING (high branching)
   - Strategy Bandit selects CoT for: ANALYSIS, CREATIVE, DEBUG (linear)
```

### Decision criteria (when MCTS beats CoT):
```
Is problem inherently sequential?
├─ Yes → CoT (faster)
└─ No → Can backtracking help?
    ├─ No → CoT
    └─ Yes → External feedback available?
        ├─ Yes → LATS with tools (best quality)
        └─ No → LATS with LM evaluation (decent)
```

### Key changes to implement:
1. **Wire into agent.py:** Assign `self.reasoning_tree` in `__init__` — ~10 LOC
2. **Tool-integrated expansion:** Actions include tool calls, not just reasoning — ~200 LOC
3. **Dual evaluation:** Internal LM + external tool feedback — ~100 LOC
4. **Adaptive stopping:** Confidence + plateau + budget — ~80 LOC
5. **Strategy Bandit integration:** Route based on problem classification — ~50 LOC

### Expected improvement:
- Complex reasoning: +20-70% pass@1 (from LATS paper)
- Cost: 10-50x CoT (but only used when needed, via Strategy Bandit)

---

## Tool 4: Code Execution Stack (SOTA Target: Aider-level edit-test loop)

**Current state:** code_executor (476) + code_edit (403) + auto_verify (148) = 1,027 LOC
**Gap:** No automatic test-after-edit, no search/replace format, no multi-layer verification, no lost detection.

### Architecture: Edit → Verify → Feedback Loop

```
Agent proposes code change
    ↓
1. EDIT: Search/replace block format (Aider pattern) — UPGRADE
   - LLM outputs SEARCH/REPLACE blocks (proven most reliable format)
   - Fallback chain: exact match → whitespace-flexible → fuzzy match
   - Multi-file support: atomic application (all or nothing)
    ↓
2. VERIFY (multi-layer): — NEW
   Layer 1: Syntax check (AST parse)
   Layer 2: Lint (ruff --check)
   Layer 3: Type check (mypy, if configured)
   Layer 4: Unit tests (detect and run relevant tests)
   Layer 5: Behavioral test (execute edited code path)
    ↓
3. FEEDBACK:
   - If all pass → commit change, update index
   - If fail → rollback (git checkout), report error to agent
   - Agent retries with error context (max 3 attempts)
    ↓
4. LOST DETECTION: — NEW
   - Track which functions were modified
   - Run callers of modified functions (not just tests)
   - Compare output before/after edit
   - Flag behavioral changes even when tests pass
```

### Key changes to implement:
1. **Auto-test-after-edit:** Detect test runner → run after each edit — ~150 LOC
2. **Search/replace format:** Parser for SEARCH/REPLACE blocks with fallback chain — ~200 LOC
3. **Multi-layer verification:** Sequential lint→type→test pipeline — ~100 LOC
4. **Rollback on failure:** Git stash/checkout integration — ~80 LOC
5. **Lost detection:** Track modified function callers, compare behavior — ~200 LOC

---

## Tool 5: Self-Extending Tool Builder (SOTA Target: VOYAGER-level)

**Current state:** tool_builder (723 LOC) + GEPA evolution (1,627 LOC). Separate, not connected.
**Gap:** No VOYAGER-style skill composition, no automatic testing of generated tools, no GEPA integration.

### Architecture: Generate → Test → Evolve → Compose

```
User describes needed tool
    ↓
1. GENERATE: LLM creates tool code (existing tool_builder)
   - Template-based scaffolding
   - Pydantic schema for inputs/outputs
    ↓
2. TEST: Automatic test generation — NEW
   - LLM generates 3-5 test cases for the tool
   - Execute in sandbox (E2B or subprocess)
   - Track: pass rate, execution time, error types
    ↓
3. SIGN & REGISTER: Ed25519 signing → custom tool registry
   - Only if tests pass
   - Reject tools with <80% test pass rate
    ↓
4. EVOLVE: GEPA integration — NEW CONNECTION
   - Periodically run GEPA on tool procedures
   - Reflective mutation based on failure traces
   - Pareto frontier preserves diverse strategies
    ↓
5. COMPOSE: VOYAGER-style skill composition — NEW
   - Index tools by embedding of description
   - When building complex tools, retrieve relevant existing tools
   - New tools can call existing tools (composition)
   - Track composition success rate
    ↓
6. DEPRECATE: Quality-based lifecycle — NEW
   - Usage tracking: tools with <5% use in 30 days flagged
   - Performance decay: periodically re-evaluate on test cases
   - Auto-retire tools that fall below threshold
```

### Key changes to implement:
1. **Auto-test generation:** LLM generates test cases for new tools — ~150 LOC
2. **GEPA integration:** Connect tool_builder output → GEPA input — ~100 LOC
3. **Composition retrieval:** Embed tool descriptions, retrieve for composition — ~100 LOC
4. **Usage tracking:** SQLite table for tool invocation counts — ~80 LOC
5. **Deprecation lifecycle:** Cron job to flag low-use tools — ~60 LOC

---

## Implementation Priority

| Phase | Tool | Effort | Impact | Dependencies |
|-------|------|--------|--------|-------------|
| **Phase 1** | Deep Research | 1 week | Highest — defines Aura as research agent | Existing retrieval + MMR |
| **Phase 2** | MCTS Wiring | 2-3 days | High — unlocks existing dead code | Strategy Bandit (exists) |
| **Phase 3** | Code Intelligence | 1 week | High — core dev tool | tree-sitter (exists) |
| **Phase 4** | Code Execution Loop | 1 week | High — edit-test-verify | auto_verify (exists) |
| **Phase 5** | Tool Builder + GEPA | 1 week | Medium-High — meta-capability | GEPA (exists), tool_builder (exists) |

### Phase 1 is the clear winner:
- Highest user-visible impact
- Builds on existing infrastructure (BM25, embeddings, cross-encoder)
- Most of the algorithms are well-documented (STORM is open-source)
- Competitive benchmark: citation precision ≥93%, hallucination ≤4%

### Phase 2 is the quickest win:
- MCTS is already built (996 LOC), just needs wiring
- Strategy Bandit routing exists
- 2-3 days of work to connect dead code

---

## Research Sources

| Topic | Key Systems Studied |
|-------|-------------------|
| Deep Research | Perplexity, Google Deep Research, OpenAI, STORM (Stanford), Tavily, RARR, FLARE |
| Code Intelligence | Cursor (Turbopuffer), Aider (tree-sitter), Sourcegraph (SCIP), GitHub Copilot Workspace, cAST (CMU) |
| MCTS Reasoning | LATS (Stanford/CMU), ToT (Princeton), Marco-o1 (Alibaba), ReST-MCTS* (THUDM), AlphaProof (DeepMind), RAP |
| Code Execution | Aider (search/replace), Claude Code (Edit tool), Cursor Composer, OpenHands (CodeAct), E2B, Pyodide |
| Self-Extending | VOYAGER (MineDojo), Reflexion, ToolBench/ToolLLM, CrewAI, LangChain, GEPA (Aura) |
