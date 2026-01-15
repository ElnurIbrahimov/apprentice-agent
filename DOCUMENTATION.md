# Apprentice Agent - Technical Documentation

Complete technical documentation for the Apprentice Agent project.

## Table of Contents

- [Architecture](#architecture)
- [Phase A: Core Agent](#phase-a-core-agent)
- [Phase B: Self-Reflection](#phase-b-self-reflection)
- [API Reference](#api-reference)
- [Changelog](#changelog)

---

## Architecture

### Project Structure

```
apprentice-agent/
├── gui.py                    # Gradio web interface
├── main.py                   # CLI entry point
├── logs/
│   └── metacognition/        # JSONL logs (YYYY-MM-DD.jsonl)
├── data/
│   └── chromadb/             # Memory storage
└── apprentice_agent/
    ├── __init__.py
    ├── agent.py              # Main agent loop
    ├── brain.py              # LLM interface (Ollama)
    ├── memory.py             # Long-term memory system
    ├── config.py             # Configuration
    ├── metacognition.py      # [Phase B] Confidence & logging
    ├── dream.py              # [Phase B] Memory consolidation
    └── tools/
        ├── __init__.py
        ├── filesystem.py     # File operations
        ├── web_search.py     # DuckDuckGo search
        ├── code_executor.py  # Sandboxed Python
        ├── screenshot.py     # Screen capture
        ├── vision.py         # Image analysis
        ├── pdf_reader.py     # PDF extraction
        └── clipboard.py      # Clipboard access
```

### Core Loop

The agent operates on a 5-phase cognitive cycle:

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                           │
├─────────────────────────────────────────────────────────┤
│  1. OBSERVE  → Gather context, recall relevant memories │
│  2. PLAN     → Create strategy based on observations    │
│  3. ACT      → Execute one action from the plan         │
│  4. EVALUATE → Assess result + confidence scoring       │
│  5. REMEMBER → Store experience in long-term memory     │
└─────────────────────────────────────────────────────────┘
         ↓ (repeat until goal achieved or max iterations)
```

### Data Flow

```
User Goal
    │
    ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Brain  │◄──►│  Tools  │◄──►│ Memory  │
│ (Ollama)│    │         │    │(ChromaDB│
└─────────┘    └─────────┘    └─────────┘
    │                              │
    ▼                              ▼
┌─────────────┐            ┌─────────────┐
│Metacognition│            │   Dream     │
│   Logger    │───────────►│    Mode     │
└─────────────┘            └─────────────┘
    │                              │
    ▼                              ▼
  JSONL Logs              Long-term Insights
```

---

## Phase A: Core Agent

### Components

#### OllamaBrain (`brain.py`)

The reasoning engine that interfaces with Ollama for all LLM operations.

**Methods:**
- `think(prompt, system_prompt, use_history, task_type)` - Generate response with model routing
- `observe(context)` - Analyze current state
- `plan(goal, observations, tools)` - Create action plan
- `decide_action(plan, tools)` - Select next action
- `evaluate(action, result, goal)` - Assess outcome with confidence
- `summarize(content, goal)` - Summarize information
- `_select_model(prompt, task_type)` - Route to appropriate model
- `get_last_model_used()` - Get model from last call

#### MemorySystem (`memory.py`)

JSON-based memory storage with text similarity search.

**Methods:**
- `remember(content, memory_type, metadata)` - Store memory
- `recall(query, n_results)` - Retrieve relevant memories
- `count()` - Get total memory count
- `get_recent(n)` - Get recent memories

#### Tools

| Tool | File | Description |
|------|------|-------------|
| `filesystem` | `filesystem.py` | List/read local files |
| `web_search` | `web_search.py` | DuckDuckGo search |
| `code_executor` | `code_executor.py` | Sandboxed Python execution |
| `screenshot` | `screenshot.py` | Screen capture |
| `vision` | `vision.py` | Image analysis (LLaVA) |
| `pdf_reader` | `pdf_reader.py` | PDF text extraction |
| `clipboard` | `clipboard.py` | Clipboard read/write |

---

## Phase B: Self-Reflection

Phase B adds metacognitive capabilities for self-monitoring and learning.

### Confidence Scoring

**Location:** `brain.py` (evaluate method), `agent.py` (display)

The agent now outputs a confidence score (0-100%) with each evaluation, indicating how certain it is that the goal has been achieved.

**Prompt format:**
```
SUCCESS: yes OR no
CONFIDENCE: 0-100 (how confident are you the goal is fully achieved)
PROGRESS: one sentence about progress
NEXT: continue OR complete OR retry
```

**Parsed in:** `_parse_evaluation_response()`

**Output example:**
```
[EVALUATE] Assessing result...
Success: True
Confidence: 85%
Progress: Found relevant search results for the query.
```

### Metacognition Logger

**Location:** `apprentice_agent/metacognition.py`

Logs every action's outcome for later analysis.

**Log format:** JSONL at `logs/metacognition/YYYY-MM-DD.jsonl`

**Log entry structure:**
```json
{
  "timestamp": "2026-01-15T20:19:02.565174",
  "goal": "What is 2 + 2?",
  "iteration": 1,
  "tool": "code_executor",
  "action": "print(2 + 2)",
  "confidence": 100,
  "success": true,
  "retried": false,
  "progress": "Calculation successful.",
  "next_step": "complete",
  "result_summary": "{'success': True, 'output': '4'}"
}
```

**Class: MetacognitionLogger**

| Method | Description |
|--------|-------------|
| `start_goal(goal)` | Begin tracking a new goal |
| `increment_iteration()` | Track iteration count and retries |
| `log_evaluation(...)` | Write log entry to JSONL |
| `get_stats(date)` | Get statistics for a date |

**Statistics returned:**
```python
{
  "date": "2026-01-15",
  "total_actions": 4,
  "successful": 4,
  "success_rate": 100.0,
  "retried": 2,
  "retry_rate": 50.0,
  "avg_confidence": 87.5,
  "tool_usage": {"code_executor": 1, "web_search": 3},
  "model_usage": {"llama3:8b": 3, "qwen2:1.5b": 1}
}
```

### Dream Mode

**Location:** `apprentice_agent/dream.py`

Memory consolidation system that analyzes logs and generates insights.

**CLI usage:**
```bash
python main.py --dream                    # Analyze today's logs
python main.py --dream-date 2026-01-14    # Analyze specific date
```

**Class: DreamMode**

| Method | Description |
|--------|-------------|
| `dream(date)` | Run full consolidation pipeline |
| `_load_logs(date)` | Read metacognition JSONL |
| `_analyze_patterns(logs)` | Extract statistics |
| `_generate_insights(patterns, logs)` | Use LLM to create insights |
| `_store_insights(insights, date)` | Save to long-term memory |
| `recall_insights(query, n)` | Query past insights |
| `get_all_insights()` | Get all stored insights |

**Pattern analysis includes:**
- Tool performance (success rate, avg confidence)
- Confidence distribution (low/medium/high)
- Retry analysis (first attempt vs needed retry)
- Goal completion tracking

**Sample insights generated:**
```
1. Use code_executor for high-confidence calculation tasks
2. Prefer web_search for information retrieval
3. Avoid low-confidence tools when possible
4. Focus on improving first-attempt success rate
```

**Insights stored as:** `memory_type: "dream_insight"` in ChromaDB

### Multi-Model Routing

**Location:** `brain.py` (`_select_model` method), `config.py`

Automatically routes tasks to the most appropriate model based on complexity.

**Models configured:**

| Model | Config Key | Use Case |
|-------|------------|----------|
| `qwen2:1.5b` | `MODEL_FAST` | Simple tasks, greetings, short answers |
| `llama3:8b` | `MODEL_REASON` | Reasoning, planning, code, evaluation |
| `llava` | `MODEL_VISION` | Image/screenshot analysis |

**Task Types (enum):**
```python
class TaskType(Enum):
    SIMPLE = "simple"       # Greetings, short answers
    REASONING = "reasoning" # Planning, evaluation
    CODE = "code"           # Code generation, calculations
    VISION = "vision"       # Image analysis
```

**Auto-detection patterns:**
- **SIMPLE**: "hello", "hi", "thanks", "bye" (short prompts <10 words)
- **CODE**: "calculate", "factorial", "print(", "import", "python"
- **VISION**: "image", "picture", "screenshot", "analyze image"
- **REASONING**: Default for complex tasks

**Usage:**
```python
from apprentice_agent.brain import OllamaBrain, TaskType

brain = OllamaBrain()

# Auto-detect model
response = brain.think("Hello!")  # Uses qwen2:1.5b

# Explicit task type
response = brain.think("Plan a strategy", task_type=TaskType.REASONING)

# Check which model was used
print(brain.get_last_model_used())  # "llama3:8b"
```

**Metacognition logging:**
- Model used is logged in `model_used` field
- `get_stats()` returns `model_usage` breakdown

### GUI Integration

**Stats Tab additions:**
- Metacognition stats table
- Date selector for historical data
- "Run Dream Mode" button
- Dream output display with insights

---

## API Reference

### CLI Commands

```bash
# Run with goal
python main.py "Your goal here"
python main.py "Goal" --max-iterations 5

# Interactive chat
python main.py --chat

# Dream mode
python main.py --dream
python main.py --dream --dream-date 2026-01-14

# GUI
python gui.py
```

### Python API

```python
from apprentice_agent import ApprenticeAgent
from apprentice_agent.metacognition import MetacognitionLogger
from apprentice_agent.dream import DreamMode

# Run agent
agent = ApprenticeAgent()
result = agent.run("Search for Python tutorials")

# Get metacognition stats
logger = MetacognitionLogger()
stats = logger.get_stats()  # Today's stats
stats = logger.get_stats("2026-01-14")  # Specific date

# Run dream mode
dreamer = DreamMode()
result = dreamer.dream()  # Analyze today
insights = dreamer.get_all_insights()  # Get stored insights
```

---

## Changelog

### Phase B - Self-Reflection (2026-01-15)

#### Added
- **Confidence Scoring**
  - Agent now outputs confidence (0-100%) with each evaluation
  - Displayed in CLI and GUI evaluate phase
  - Parsed from LLM response in `_parse_evaluation_response()`

- **Metacognition Logger** (`metacognition.py`)
  - Logs all actions to `logs/metacognition/YYYY-MM-DD.jsonl`
  - Tracks: tool, action, confidence, success, retried, progress
  - `get_stats()` method for analyzing daily performance
  - Integrated into agent evaluation phase

- **Dream Mode** (`dream.py`)
  - Memory consolidation system
  - Analyzes metacognition logs for patterns
  - Generates actionable insights using LLM
  - Stores insights in long-term memory
  - CLI: `python main.py --dream`

- **Multi-Model Routing** (`brain.py`)
  - `_select_model()` method for automatic model selection
  - `TaskType` enum: SIMPLE, REASONING, CODE, VISION
  - Routes simple tasks to `qwen2:1.5b` (fast)
  - Routes reasoning/code to `llama3:8b`
  - Routes vision to `llava`
  - Logs `model_used` in metacognition

- **GUI Enhancements**
  - Stats tab with metacognition statistics
  - "Run Dream Mode" button
  - Dream output display with generated insights

#### Modified
- `config.py`: Added MODEL_FAST, MODEL_REASON, MODEL_VISION settings
- `brain.py`: Added confidence scoring, multi-model routing, `_select_model()`
- `agent.py`: Integrated MetacognitionLogger, displays confidence and model
- `metacognition.py`: Added `model_used` tracking and `model_usage` stats
- `main.py`: Added `--dream` and `--dream-date` CLI flags
- `gui.py`: Added Stats tab, dream mode button, model usage display

### Phase A - Core Agent (Initial Release)

#### Added
- 5-phase agent loop (Observe/Plan/Act/Evaluate/Remember)
- OllamaBrain for LLM reasoning
- MemorySystem with JSON persistence
- Tools: filesystem, web_search, code_executor, screenshot, vision, pdf_reader, clipboard
- Gradio GUI with real-time thinking visualization
- CLI with goal and chat modes

---

## Future Phases

### Phase C - Adaptive Learning (Planned)
- Use dream insights to adjust planning strategies
- Tool selection based on historical success rates
- Confidence-based retry decisions

### Phase D - Multi-Agent (Planned)
- Spawn sub-agents for complex tasks
- Inter-agent communication
- Shared memory pool
