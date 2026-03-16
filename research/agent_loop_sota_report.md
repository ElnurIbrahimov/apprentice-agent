# AURA Agent Loop: State-of-the-Art Research Report
## March 2026

---

## 1. CURRENT STATE ASSESSMENT

### What AURA Does Now

**Loop structure:** OBSERVE → PLAN → ACT → EVALUATE → REMEMBER

Each iteration makes **4-5 separate LLM calls:**
1. `brain.observe()` — analyzes context (LLM call #1)
2. `brain.plan()` — creates action plan (LLM call #2)
3. `brain.decide_action()` — picks tool + parameters (LLM call #3)
4. `brain.evaluate()` — judges result (LLM call #4)
5. `brain.summarize_for_memory()` — creates memory entry (LLM call #5)

**Tool calling:** Text-based parsing (TOOL/ACTION/REASONING format), not native function calling or code generation. The brain dumps all tool descriptions into every `plan()` and `decide_action()` prompt.

**Tool count:** 50+ tools imported in `agent.py` line 140, all exposed to the model simultaneously.

**Model routing:** Separate tiers (fast/reason/code/think/vision/longctx) via Config, using Ollama cloud models (Qwen, Gemini, DeepSeek, etc.) and local models (nomic-embed, glm-ocr).

**Key problems identified:**
- 4-5 LLM roundtrips per step = massive latency (each call 1-5s depending on model)
- OBSERVE is redundant — it just summarizes what the agent already knows
- PLAN and DECIDE_ACTION are nearly the same thing for single-step tasks
- All 50+ tools dumped into context every call = confused model + wasted tokens
- No error recovery — if tool call fails, the evaluate phase just says "retry" with no reflection
- No loop termination guards (relies on max_iterations=5 and LLM saying "complete")
- REMEMBER runs on every iteration, even trivial ones

---

## 2. SOTA RESEARCH FINDINGS

### 2.1 Agent Loop Architecture

**The consensus in 2025-2026 is: ReAct (Reason+Act) in a single LLM call is the standard.**

Every major framework has converged on this:
- **smolagents (HuggingFace):** Single LLM call per step that outputs reasoning + action together. CodeAgent writes Python; ToolCallingAgent writes JSON. ~1000 lines of code total.
- **LangGraph:** Graph-based state machine. Each "node" is one LLM call. The standard ReAct agent is a 2-node graph: `call_model → call_tool → call_model → ...`
- **OpenAI Agents SDK:** Single agent loop with tool calls embedded in the response. One LLM call decides action, runs tool, gets result, decides next action.
- **CrewAI:** Role-based agents, each running ReAct internally.
- **AutoGen:** Multi-agent conversation, but each agent is still ReAct internally.

**Optimal loop: 1 LLM call per step, not 4.**

The SOTA pattern is:
```
while not done:
    response = LLM(system_prompt + conversation_history + tool_results)
    if response.has_tool_call:
        result = execute_tool(response.tool_call)
        conversation_history.append(tool_result)
    else:
        return response.final_answer  # Done
```

This is fundamentally different from AURA's approach. There is no separate observe, plan, decide, evaluate. The LLM does ALL of that in a single generation — it reasons about what it observes, plans its approach, and outputs a tool call, all in one response.

**Key insight from the research:** Separating reasoning into phases was a 2023-era design. By 2025, models are good enough to do observe+plan+act in a single generation, and the conversation history serves as implicit memory of past observations and evaluations.

### 2.2 Code Agents vs JSON Tool Calling

**smolagents demonstrated that code agents (LLM writes Python) outperform JSON tool calling by ~30% fewer steps and higher accuracy on complex benchmarks.**

Why code is better:
- **Composability:** A single code block can call multiple tools, use loops, define variables, nest function calls. JSON tool calling can only do one tool at a time.
- **Object management:** Code can pass results between tool calls naturally. JSON requires the framework to manage intermediate results.
- **Expressiveness:** Code can express `for`, `if`, `while` — complex logic in a single step that would take 5+ JSON tool-calling steps.
- **Training data:** LLMs have been trained on massive amounts of Python code. They are natively good at generating Python.

**Feasibility with local models (8B-70B):**
- **Qwen 3 8B** achieves F1=0.933 on tool calling benchmarks. Competitive with GPT-4o on code repair.
- **Llama 3.1/3.3 8B-70B** have excellent native tool token support for function calling.
- **Qwen 2.5-Coder 32B** is competitive with GPT-4o on Aider code repair benchmark.
- **Code generation is more reliable than JSON for local models** because models are trained on more code than JSON schemas.

**BUT — critical caveat for 8B models:**
- 8B models struggle with large tool sets (more than 10-15 tools)
- They have higher rates of "eager invocation" (calling tools when they shouldn't)
- Invalid arguments and wrong tool selection are common failure modes
- **Recommendation: Use code agents with 8B models only for limited tool sets (<10 tools). Use 70B+ for full tool set.**

### 2.3 Tool Selection at Scale

**The #1 problem with 40+ tools: model confusion and context bloat.**

Research findings:

1. **Dynamic System Instructions & Tool Exposure (Feb 2026 paper):**
   - Dynamically selecting relevant tools per step achieved **95% reduction in per-step context tokens**
   - **32% improvement in correct tool routing accuracy**
   - **70% reduction in end-to-end episode cost**
   - Agents can run **2-20x more loops within context limits**

2. **Dynamic ReAct (Sep 2025 paper):**
   - Proposes "Search and Load" architecture: two meta-tools (`search_tools` and `load_tools`)
   - Agent first searches for relevant tools, then explicitly loads only the ones it needs
   - **Reduces tool loading by 50%** while maintaining accuracy
   - Typically loads **<5 tools per query** vs 10+ with naive approaches

3. **Tool RAG (Red Hat, Nov 2025):**
   - Treat tools like documents in RAG: embed tool descriptions, retrieve top-k per query
   - Anthropic's RAG-MCP boosted tool selection accuracy from **13% to 43%** in large toolsets
   - ToolScope library (Feb 2026) provides production-ready Tool RAG

4. **AutoTool (ICLR 2026):**
   - RL-based dynamic tool selection across 1000+ tools and 100+ tasks
   - Models learn when to select tools vs reason without them
   - Uses dual-phase optimization: SFT for trajectory stabilization, then RL for tool selection

**Practical recommendation for AURA with 50+ tools:**
- **Tier 1 (always available):** 5-7 core tools (web_search, filesystem, code_executor, shell, vision, summarize)
- **Tier 2 (retrieved on demand):** Remaining 40+ tools retrieved via semantic search on user query
- **Max tools for 8B model:** 8-12 with descriptions. Beyond that, accuracy degrades.
- **Max tools for 70B model:** 20-25 with descriptions.

### 2.4 Reliability & Error Recovery

**The Reflection Trap:** Agents get stuck in infinite loops because feedback is ambiguous. Vague signals like "something went wrong" cause the model to retry the same failing approach.

**SOTA solutions:**

1. **Structured Error Classification (Reflect & Retry pattern):**
   - Before retrying, classify the error:
     - **RETRY** — fixable (wrong arguments, format error) → fix and retry
     - **WAIT** — transient (503, timeout) → exponential backoff
     - **ABORT** — fatal (403, wrong tool entirely) → stop and report
   - This prevents blind retries

2. **Practical Loop Guards (must-have):**
   - `max_rounds`: Hard cap on iterations (AURA has this at 5)
   - `no_progress_k`: Stop after k rounds with no measurable progress
   - `state_hash` deduplication: If agent returns to a previous state, break
   - `cost_budget`: Token/time budget per task

3. **UFO (Unary Feedback as Observation):**
   - Minimal feedback ("Try again") paired with multi-turn RL
   - Models learn to self-correct from minimal signal

4. **Tool-Reflection-Bench:**
   - Transforms error-to-correction into a trainable skill
   - Teaches: (1) diagnose error from evidence, (2) propose new valid tool call

5. **Agent-R framework:**
   - Trains self-reflective agents by generating revision trajectories
   - Agents learn to avoid repetitive dead-ends
   - Key: train on error *recoveries*, not just successes

**For AURA specifically:**
- The EVALUATE phase should not require a full LLM call. Use deterministic checks first (did tool return success? is output non-empty?), only call LLM for ambiguous cases.
- Add a `state_hash` to detect loops (hash of tool+action pairs).
- Add `consecutive_failure_count` — after 2 failures with same tool, force tool switch.

### 2.5 Speed Optimization

**Key findings:**

1. **Speculative Actions (Oct 2025 paper):**
   - Use a small "Speculator" model to predict the next action while the main "Actor" processes the current step
   - **19-38% latency reduction** in practice
   - Works by pre-executing likely tool calls in parallel
   - Speculator can be a small model (gpt-5-nano, Gemini Flash) while Actor is large
   - **Theoretical max: 50% latency reduction** when predictions are perfect

2. **Model Size vs Latency (2025 benchmarks):**
   | Model Size | Performance | Inference Time |
   |-----------|-------------|---------------|
   | 7-8B      | 79/100      | ~250ms        |
   | 13B       | 84/100      | ~450ms        |
   | 34B       | 89/100      | ~1.2s         |
   | 70B       | 94/100      | ~2.5s         |

3. **Smart Routing (NVIDIA + Avengers-Pro research):**
   - Most agent actions (routing, classification, tool selection, template expansion) don't need a large model
   - Use SLMs (7-8B) for 70-80% of agentic actions
   - Reserve large models for complex planning and reasoning
   - **Compounding effect:** Agent workflows make dozens of calls. 5x cost savings per call = 100x at workflow level.

4. **Parallel Execution:**
   - Run independent tool calls concurrently (DAG-based execution)
   - **>20% latency reduction** from serial → parallel
   - AURA already uses ThreadPoolExecutor for some operations

5. **Prompt Optimization (incident.io case study):**
   - Achieved **4x speedup** by tuning prompt length and structure
   - Shorter prompts = fewer output tokens = less latency
   - Remove redundant instructions, use structured formats

6. **Caching:**
   - Cache tool descriptions (don't regenerate every call)
   - Cache model responses for repeated queries
   - Cache embedding lookups for memory retrieval
   - **Warning:** Cache hit rate can be low for agent workloads; don't add cache if latency to check cache > saved latency

**For AURA specifically:**
- Biggest win: Reduce from 4-5 LLM calls to 1 per step. This alone cuts latency by 75%.
- Second biggest: Use 8B model for simple tasks, 70B only for complex reasoning.
- Third: Parallel tool execution when the code agent generates multiple independent calls.

### 2.6 Planning Strategy

**Three dominant patterns in 2025-2026:**

1. **ReAct (default, recommended for most cases):**
   - Single-step reasoning: think → act → observe → think → act → ...
   - 1 LLM call per step
   - Best for: tasks with unpredictable steps, interactive workflows
   - Used by: smolagents (default), LangGraph, OpenAI Agents SDK

2. **Plan-and-Execute (for structured, multi-step tasks):**
   - First call: generate full plan
   - Subsequent calls: execute each step
   - Periodic replanning when results diverge from plan
   - Best for: long-horizon tasks, research projects, multi-file code changes
   - **More efficient for long tasks** because plan is decided once, not re-derived each step

3. **smolagents `planning_interval` (hybrid approach):**
   - ReAct by default, but every N steps, insert a planning step
   - `planning_interval=3` means: replan after every 3 action steps
   - The planning step reviews all past actions and results, then creates updated plan
   - **Three planning prompts:** `initial_plan`, `update_plan_pre_messages`, `update_plan_post_messages`
   - Best for: tasks that might take 5-10+ steps

4. **HiPlan (hierarchical, from the latency research):**
   - Master planner creates strategic milestones
   - Subordinate executor handles tactical prompts per milestone
   - Enables reusable milestone components
   - Best for: very long tasks (20+ steps)

5. **Plan-and-Act (Mar 2025 paper):**
   - Separate Planner and Executor roles
   - Dynamic replanning: Planner updates plan after each Executor step
   - Chain-of-thought reasoning before generating plans/actions
   - **Key finding:** Dynamic replanning consistently outperforms static planning

**Recommendation for AURA:**
- **Default:** Pure ReAct (1 LLM call per step, no separate plan phase)
- **For complex tasks (detected via intent classifier):** Plan-and-Execute with replanning every 3 steps
- **Kill the OBSERVE phase entirely.** The context is already in the prompt. The model observes as it reasons.

### 2.7 Sandboxed Execution

**If AURA moves to code agents, sandboxing becomes critical.**

**2026 Landscape:**

| Platform | Isolation | Cold Start | Best For | Pricing |
|----------|-----------|------------|----------|---------|
| **E2B** | Firecracker microVM | ~150ms | AI agent SDKs, fast integration | Free tier + $0.05/hr |
| **Modal** | gVisor | Sub-second | Python ML workloads, GPU | $0.000014/core/sec |
| **Daytona** | Docker (Kata optional) | ~90ms | Full dev environments | $200 free credits |
| **microsandbox** | libkrun microVM | <200ms | Self-hosted, max security | Open source (Apache 2.0) |
| **Docker + E2B** | Partnership | N/A | Enterprise trust | Combined pricing |
| **Local Python** | Process-level | 0ms | Dev/testing only | Free |

**For AURA specifically:**
- **Phase 1 (now):** Use smolagents-style local Python sandbox with restricted imports (AURA already has `ALLOWED_TOOL_IMPORTS` and `FORBIDDEN_PATTERNS` in agent.py)
- **Phase 2 (production):** E2B for cloud, microsandbox for self-hosted
- **E2B integrates directly with Ollama:** Their docs show `ollama.chat()` → `Sandbox()` workflow
- **smolagents supports:** `executor_type` = `'local'`, `'e2b'`, `'modal'`, `'docker'`, `'wasm'`

---

## 3. CONCRETE RECOMMENDATIONS (Ranked by Impact)

### Rank 1: Collapse the Loop to ReAct (CRITICAL — ~75% latency reduction)

**Current:** 4-5 LLM calls per step (observe → plan → decide_action → evaluate → summarize_for_memory)

**Target:** 1 LLM call per step

**How:**
- Merge OBSERVE + PLAN + DECIDE_ACTION into a single LLM call
- The model receives: system prompt + conversation history + last tool result
- The model outputs: reasoning (thinking) + tool call OR final answer
- EVALUATE becomes deterministic: did the tool succeed? Is the result non-empty? Does it answer the user's question? Only call LLM for evaluation on ambiguous cases.
- REMEMBER becomes async and non-blocking: queue memory storage, don't wait for it

**New loop:**
```python
while step < max_steps:
    response = llm(system_prompt, history, tool_descriptions)
    if response.is_final_answer:
        return response.text
    tool_name, args = parse_tool_call(response)
    result = execute_tool(tool_name, args)
    history.append({"role": "tool", "content": result})
    # Deterministic evaluation
    if result.success:
        step += 1
    else:
        error_class = classify_error(result)  # RETRY/WAIT/ABORT
        handle_error(error_class)
```

**Impact:** Reduces per-message latency from ~10-20s (4-5 calls x 2-5s each) to ~3-5s (1 call).

### Rank 2: Implement Tool RAG / Dynamic Tool Selection (HIGH — fixes 50+ tool problem)

**Current:** All 50+ tool descriptions dumped into every prompt.

**Target:** 5-8 relevant tools per prompt, dynamically selected.

**How:**
1. Create tool embeddings: embed each tool's name + description using nomic-embed-text (already available locally)
2. On each user message, embed the query and retrieve top-5 most relevant tools
3. Always include core tools: `web_search`, `filesystem`, `code_executor`, `summarize`
4. Add retrieved tools on top (total: 5-8 per call)
5. If the model says "I need tool X that isn't available," add a `search_more_tools` meta-tool

**Impact:**
- 32% better tool selection accuracy
- 95% less context tokens spent on tool descriptions
- Enables 8B models to work reliably (they can handle 8 tools, not 50)

### Rank 3: Smart Model Routing per Phase (HIGH — cost + speed)

**Current:** Model routing exists but could be more aggressive.

**Target:** Use smallest adequate model for each operation.

**How:**
- **Simple queries** (greetings, factual, single-tool): 8B model (Qwen 3 8B, ~250ms)
- **Medium complexity** (multi-step, 2-3 tools): 30-70B model (~1-2.5s)
- **Complex reasoning** (planning, code generation, analysis): 235B+ cloud model (~3-5s)
- **Intent classification** (to decide which model): Rule-based first (keyword/regex), then 8B classifier as fallback
- **Don't use cloud model for evaluation.** Evaluation is almost always deterministic.

**Impact:** 70-80% of interactions use the fast 8B model. Average latency drops significantly.

### Rank 4: Code Agent Mode (MEDIUM-HIGH — better for complex tasks)

**Current:** Text-based TOOL/ACTION/REASONING parsing.

**Target:** Hybrid mode — code agent for complex tasks, simple tool calling for straightforward ones.

**How:**
1. For simple tasks (single tool call): Keep current approach but use native Ollama function calling (not text parsing)
2. For complex tasks (multi-step, data manipulation, logic): Switch to code agent mode
3. The model generates Python that calls tool functions directly:
   ```python
   # Model generates this:
   results = web_search("Bitcoin price")
   price = extract_price(results)
   final_answer(f"Bitcoin is currently {price}")
   ```
4. Execute in sandboxed Python interpreter with tool functions injected

**Prerequisites:** Tool RAG (Rank 2) must be implemented first to keep tool count manageable.

**Impact:** ~30% fewer steps for complex tasks. More reliable than text parsing.

### Rank 5: Loop Guards & Error Recovery (MEDIUM — prevents stuck agents)

**Current:** max_iterations=5, relies on LLM saying "complete."

**Target:** Multi-layered termination and recovery.

**How:**
1. **State hash deduplication:** Hash (tool_name, action_args) — if repeated, force different approach
2. **Consecutive failure counter:** After 2 failures with same tool, blacklist it for this task
3. **Error classifier:** Before retry, classify error as RETRY/WAIT/ABORT
4. **Progress detector:** If last 2 iterations produced no new information, stop
5. **Token budget:** Track total tokens used; abort if approaching context limit
6. **Time budget:** Hard timeout per task (already exists at 120s, good)

**Impact:** Eliminates infinite loops, reduces wasted compute on doomed tasks.

### Rank 6: Speculative Tool Execution (LOW-MEDIUM — advanced optimization)

**Current:** Sequential execution only.

**Target:** Predict and pre-execute likely next tool calls.

**How:**
1. While the main model processes, use a small model (8B) to predict the next tool call
2. Pre-execute predicted tools in parallel
3. If prediction matches actual request, use cached result (instant response)
4. If not, discard and execute normally
5. Only speculate on idempotent/read-only tools (search, filesystem read, etc.)

**Impact:** 20-38% additional latency reduction on multi-step tasks. But complex to implement — do this last.

### Rank 7: Planning Mode for Long Tasks (LOW-MEDIUM)

**Current:** Plan created every iteration regardless of task complexity.

**Target:** Adaptive planning based on task complexity.

**How:**
1. **Simple tasks (1-2 steps):** No planning. Pure ReAct.
2. **Medium tasks (3-5 steps):** Plan once at start, replan on failure.
3. **Long tasks (5+ steps):** Plan at start, replan every 3 steps (smolagents `planning_interval=3`).
4. **Detect task complexity** via simple heuristics: word count, presence of "and then," "step by step," "multiple," etc.

**Impact:** Saves 1 LLM call for simple tasks. Improves coherence for long tasks.

---

## 4. IMPLEMENTATION NOTES

### Priority Order

1. **Week 1-2:** Collapse loop to ReAct (Rank 1). This is the single biggest improvement. Rewrite the core loop in `agent.py` to use 1 LLM call per step.

2. **Week 2-3:** Implement Tool RAG (Rank 2). Embed tool descriptions, add retrieval. This unblocks using smaller models.

3. **Week 3-4:** Aggressive model routing (Rank 3). Use 8B for simple tasks. This makes the collapsed loop fast.

4. **Week 4-6:** Code agent mode (Rank 4). Add Python generation alongside existing tool calling. Start with a sandboxed local executor.

5. **Ongoing:** Loop guards (Rank 5), speculative execution (Rank 6), adaptive planning (Rank 7).

### Architecture Decision: smolagents vs Custom

**Don't adopt smolagents as a dependency.** It's designed for HuggingFace ecosystem, is ~1000 lines, and AURA already has a more sophisticated architecture (memory systems, emotional engine, knowledge graph, etc.).

**Instead, steal the design patterns:**
- ReAct loop structure (1 call per step)
- Code agent execution model (generate Python, execute in sandbox)
- `planning_interval` concept (periodic replanning for long tasks)
- Tool description format (smolagents' tool schemas are clean)

### Key Files to Modify

- `D:/Aura/aura/agent.py` — Core loop rewrite. Collapse `_observe`, `_plan`, `_act`, `_evaluate` into single `_step()` method.
- `D:/Aura/aura/brain.py` — Merge `observe()`, `plan()`, `decide_action()` into single `step()` method that returns reasoning + tool call.
- `D:/Aura/aura/config.py` — Add tool embedding config, model routing thresholds.
- New file: `aura/tool_rag.py` — Tool retrieval system using nomic-embed-text.
- New file: `aura/code_executor_sandbox.py` — Sandboxed Python execution for code agent mode.

### Model Recommendations for Ollama

| Role | Model | Size | Use When |
|------|-------|------|----------|
| Fast/Simple | Qwen 3 8B or Gemini Flash | 8B | Greetings, single-tool, classification |
| Reasoning | Qwen 3.5 or DeepSeek V3.2 | 70-235B | Multi-step planning, complex logic |
| Code | Qwen 3-Coder or Devstral 2 | 32-123B | Code generation, code agent mode |
| Tool Calling | Qwen 3 8B (fine-tuned) | 8B | Native function calling, tool selection |
| Embedding | nomic-embed-text | N/A | Tool RAG, memory retrieval |

### What NOT to Do

1. **Don't add more LLM calls.** Every additional call adds 1-5s of latency.
2. **Don't use LangGraph/CrewAI/AutoGen as dependencies.** They add complexity without value for a single-agent system. AURA's custom loop is fine — it just needs to be simplified.
3. **Don't try to make 8B models handle 50+ tools.** It won't work. Use Tool RAG.
4. **Don't over-plan.** For 80% of user messages, zero planning is needed. ReAct handles it.
5. **Don't evaluate with an LLM when a boolean check suffices.** `tool_result.success == True` is instant. An LLM call to "evaluate" that takes 2 seconds is waste.

---

## 5. SOURCES

### Frameworks & Documentation
- smolagents (HuggingFace): https://smolagents.org/
- smolagents GitHub: https://github.com/huggingface/smolagents
- LangGraph: https://www.philschmid.de/langgraph-gemini-2-5-react-agent
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- OpenAI Practical Guide to Building Agents: https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/

### Research Papers
- Speculative Actions (Oct 2025): https://arxiv.org/html/2510.04371v1
- Dynamic ReAct — Tool Selection at Scale (Sep 2025): https://arxiv.org/html/2509.20386v1
- Dynamic System Instructions & Tool Exposure (Feb 2026): https://arxiv.org/abs/2602.17046
- AutoTool — Dynamic Tool Selection (ICLR 2026): https://openreview.net/forum?id=52c4trAbmd
- Plan-and-Act (Mar 2025): https://arxiv.org/html/2503.09572v3
- SPAgent — Speculation-based Agent Acceleration (Nov 2025): https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/66ef348c-c150-46d7-b2fc-c6f2afb217a5.pdf
- Avengers-Pro — Test-time Routing: https://arxiv.org/html/2508.12631v2

### Industry Analysis
- AI Agents Stack 2026 Edition: https://medium.com/data-science-collective/the-ai-agents-stack-2026-edition-37fa32db7a56
- State of AI Agents Oct 2025: https://medium.com/@fahey_james/the-state-of-ai-agents-agent-teams-oct-2025-27d7dac01667
- Tool RAG (Red Hat): https://next.redhat.com/2025/11/26/tool-rag-the-next-breakthrough-in-scalable-ai-agents/
- Why AI Agents Fail: https://langcopilot.com/posts/2025-10-17-why-ai-agents-fail-latency-planning
- Docker Local LLM Tool Calling: https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/
- NVIDIA Small Models for Agents: https://medium.com/@avigoldfinger/why-the-next-generation-of-agentic-ai-will-run-on-small-models-smart-routing-and-fine-tuning-b7cd3750aaf4

### Sandboxing
- Best Code Sandboxes 2026: https://fast.io/resources/best-code-execution-sandboxes-ai-agents/
- E2B vs Modal: https://northflank.com/blog/e2b-vs-modal
- How to Sandbox AI Agents 2026: https://substack.com/home/post/p-187330720
- microsandbox: https://github.com/microsandbox/microsandbox

### Comparisons
- smolagents vs LangGraph: https://www.analyticsvidhya.com/blog/2025/01/smolagents-vs-langgraph/
- smolagents vs Other Frameworks: https://mem0.ai/blog/smolagents-vs-langchain-autogen-comparison
- smolagents Planning Explained: https://medium.com/@laurentkubaski/smolagents-planning-explained-f6f827f48573
- Agent Architectures 2025: https://dev.to/sohail-akbar/the-ultimate-guide-to-ai-agent-architectures-in-2025-2j1c
