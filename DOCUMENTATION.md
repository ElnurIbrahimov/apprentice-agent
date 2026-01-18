# Apprentice Agent - Technical Documentation

Complete technical documentation for the Apprentice Agent project.

## Table of Contents

- [Architecture](#architecture)
- [Phase A: Core Agent](#phase-a-core-agent)
- [Phase B: Self-Reflection](#phase-b-self-reflection)
- [Phase C: Voice & Image Tools](#phase-c-voice--image-tools)
- [API Reference](#api-reference)
- [Changelog](#changelog)
- [Roadmap](#roadmap)

---

## Architecture

### Project Structure

```
apprentice-agent/
├── gui.py                    # Gradio web interface
├── main.py                   # CLI entry point
├── logs/
│   ├── metacognition/        # JSONL logs (YYYY-MM-DD.jsonl)
│   ├── notifications/        # Notification scheduler logs
│   └── tool_builder/         # Tool creation logs
├── data/
│   ├── chromadb/             # Memory storage
│   ├── scheduled_tasks.json  # Notification tasks
│   └── custom_tools.json     # Custom tool registry
├── generated_images/         # [Phase C] Stable Diffusion outputs
└── apprentice_agent/
    ├── __init__.py
    ├── agent.py              # Main agent loop
    ├── brain.py              # LLM interface (Ollama)
    ├── memory.py             # Long-term memory system
    ├── config.py             # Configuration
    ├── metacognition.py      # [Phase B] Confidence & logging
    ├── dream.py              # [Phase B] Memory consolidation
    ├── scheduler.py          # Notification scheduler daemon
    └── tools/
        ├── __init__.py
        ├── filesystem.py     # File operations
        ├── web_search.py     # DuckDuckGo search
        ├── code_executor.py  # Sandboxed Python
        ├── screenshot.py     # Screen capture
        ├── vision.py         # Image analysis
        ├── pdf_reader.py     # PDF extraction
        ├── clipboard.py      # Clipboard access
        ├── voice.py          # [Phase C] Speech-to-text & TTS
        ├── image_gen.py      # [Phase C] Stable Diffusion
        ├── arxiv_search.py   # arXiv paper search
        ├── browser.py        # Playwright browser automation
        ├── system_control.py # Volume, brightness, apps
        ├── notifications.py  # Reminders & scheduled alerts
        ├── tool_builder.py   # Meta-tool for creating tools
        ├── tool_template.py  # Templates for generated tools
        └── custom/           # Auto-generated custom tools
            ├── __init__.py
            └── tests/        # Custom tool tests
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

#### Tools (14 Total)

| Tool | File | Description |
|------|------|-------------|
| `filesystem` | `filesystem.py` | List/read local files |
| `web_search` | `web_search.py` | DuckDuckGo search |
| `code_executor` | `code_executor.py` | Sandboxed Python execution |
| `screenshot` | `screenshot.py` | Screen capture |
| `vision` | `vision.py` | Image analysis (LLaVA) |
| `pdf_reader` | `pdf_reader.py` | PDF text extraction |
| `clipboard` | `clipboard.py` | Clipboard read/write |
| `voice` | `voice.py` | [Phase C] Speech-to-text (Whisper) & TTS (pyttsx3) |
| `image_gen` | `image_gen.py` | [Phase C] Image generation (Stable Diffusion 1.5) |
| `arxiv_search` | `arxiv_search.py` | arXiv paper search & summarization |
| `browser` | `browser.py` | Playwright browser automation |
| `system_control` | `system_control.py` | Volume, brightness, apps, system info |
| `notifications` | `notifications.py` | Reminders, scheduled alerts, conditional triggers |
| `tool_builder` | `tool_builder.py` | Meta-tool for creating custom tools at runtime |

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

### Fast-Path Responses

**Location:** `agent.py` (`_is_simple_query`, `_fast_path_response`)

Simple conversational queries skip the full 5-phase agent loop and respond directly using the fast model (`qwen2:1.5b`).

**Triggers (Phase C - Expanded):**
- **Greetings**: "hello", "hi", "hey", "thanks", "bye", "good morning", etc.
- **Agent questions**: "who are you", "what can you do", "are you an AI", "what model are you"
- **Yes/no questions**: "can you", "is it", "do you", "should I", "will it"
- **Conversational starters**: "what do you think", "explain", "define", "tell me a joke"
- **General questions**: "what is", "who is", "why is", "how many", "when is"
- **Short queries**: Less than 8 words without tool keywords
- **Medium queries**: 8-12 words starting with question words

**Tool keywords that REQUIRE full agent loop:**
```
search the web, search online, google, take a screenshot, capture screen,
read file, open file, list files, run code, execute python, calculate,
compute, factorial, fibonacci, read pdf, clipboard, analyze image,
current weather, weather in, news about, stock price, bitcoin price
```

**Example:**
```bash
# Fast-path (direct response)
python main.py "Hello, how are you?"
# Output: [FAST-PATH] Using qwen2:1.5b

# Disable fast-path
python main.py --no-fastpath "Hello!"
# Output: Runs full 5-phase loop
```

**How it works:**
1. `_is_simple_query(goal)` checks if goal is conversational
2. If true, `_fast_path_response(goal)` responds using `qwen2:1.5b`
3. Logs to metacognition with `tool="fast_path"`
4. Returns immediately without observe/plan/act/evaluate

**Result structure for fast-path:**
```python
{
    "goal": "Hello!",
    "completed": True,
    "iterations": 0,
    "fast_path": True,
    "response": "Hi there! How can I help you?",
    "final_evaluation": {"success": True, "confidence": 100, ...}
}
```

### GUI Integration

**Stats Tab additions:**
- Metacognition stats table
- Date selector for historical data
- "Run Dream Mode" button
- Dream output display with insights

---

## Phase C: Voice & Image Tools

Phase C adds voice interaction and image generation capabilities.

### Voice Interface

**Location:** `apprentice_agent/tools/voice.py`

**Classes:**
- `VoiceTool` - Core voice functionality
- `VoiceConversation` - Continuous voice chat loop

**VoiceTool Methods:**

| Method | Description |
|--------|-------------|
| `listen(duration, silence_threshold, silence_duration, max_duration)` | Record and transcribe speech |
| `speak(text, block)` | Convert text to speech |
| `set_voice(voice_id, rate, volume)` | Configure TTS settings |
| `get_voices()` | List available TTS voices |
| `list_audio_devices()` | List microphone devices |

**Usage:**
```python
from apprentice_agent.tools import VoiceTool

voice = VoiceTool(whisper_model="base")

# Listen and transcribe
result = voice.listen()
print(result["text"])  # Transcribed speech

# Speak text
voice.speak("Hello, how can I help you?")
```

**CLI Voice Mode:**
```bash
python main.py --voice
```

**GUI Voice Mode:**
1. Check "Voice Mode" checkbox
2. Click "Speak" button
3. Speak into microphone
4. View transcription in status area

**Dependencies:**
- `openai-whisper` - Speech-to-text (local)
- `pyttsx3` - Text-to-speech (offline)
- `sounddevice` - Audio recording
- `numpy` - Audio processing

### Image Generation

**Location:** `apprentice_agent/tools/image_gen.py`

**Class:** `ImageGenerationTool`

**Methods:**

| Method | Description |
|--------|-------------|
| `generate(prompt, negative_prompt, width, height, steps, guidance_scale, seed, save)` | Generate image from prompt |
| `generate_variations(prompt, num_images)` | Generate multiple variations |
| `get_device_info()` | Check GPU/CPU status |

**Usage:**
```python
from apprentice_agent.tools import ImageGenerationTool

# Initialize (downloads ~4GB model on first use)
tool = ImageGenerationTool()

# Check device
print(tool.get_device_info())
# {'device': 'cuda', 'cuda_available': True, ...}

# Generate image
result = tool.generate(
    prompt="a cat sitting on a rainbow",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    num_inference_steps=50,
    guidance_scale=7.5
)

print(result["image_path"])  # generated_images/20260116_123456_a_cat_sitting.png
```

**Quick function:**
```python
from apprentice_agent.tools import generate_image

result = generate_image("a futuristic city at sunset")
```

**Performance:**
- GPU (CUDA): ~10-30 seconds per image
- CPU: ~5-10 minutes per image

**Note:** Python 3.14 does not yet have PyTorch CUDA wheels. Use Python 3.11/3.12 for GPU acceleration.

---

## API Reference

### CLI Commands

```bash
# Run with goal
python main.py "Your goal here"
python main.py "Goal" --max-iterations 5

# Fast-path (auto for simple queries)
python main.py "Hello!"              # Uses fast-path
python main.py --no-fastpath "Hello" # Forces full loop

# Interactive chat
python main.py --chat

# Voice conversation mode [Phase C]
python main.py --voice

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
from apprentice_agent.tools import VoiceTool, ImageGenerationTool

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

# Voice interface [Phase C]
voice = VoiceTool()
result = voice.listen()  # Record and transcribe
voice.speak("Hello!")    # Text-to-speech

# Image generation [Phase C]
img_tool = ImageGenerationTool()
result = img_tool.generate("a sunset over mountains")
print(result["image_path"])
```

---

## Changelog

### Self-Tool-Creation - Tool #14 (2026-01-18)

#### Added
- **Tool Builder** (`tools/tool_builder.py`)
  - Meta-tool enabling the agent to create new tools at runtime
  - `create_tool(name, description, functions_spec)` — generates Python file from specification
  - `test_tool(name)` — runs auto-generated tests in isolated subprocess
  - `enable_tool(name)` — activates tool for agent use
  - `disable_tool(name)` — deactivates tool without deletion
  - `rollback_tool(name)` — deletes tool and removes from registry
  - `list_custom_tools()` — shows all agent-created tools with status

- **Tool Templates** (`tools/tool_template.py`)
  - Class templates for generated tools
  - Method templates with proper typing and docstrings
  - Test templates for auto-generated unit tests
  - Blocked patterns list for security scanning

- **Custom Tools Infrastructure**
  - Custom tools saved in `tools/custom/`
  - Tests saved in `tools/custom/tests/`
  - Registry in `data/custom_tools.json`
  - Logs in `logs/tool_builder/`

- **Safety Features**
  - Code scanning blocks dangerous imports: `os.system`, `subprocess`, `eval`, `exec`, `__import__`, `compile`, `open`, `socket`, `requests`
  - Function bodies scanned before file generation
  - Isolated test execution in subprocess

- **Natural Language Tool Spec Generation** (`agent.py`)
  - `_generate_tool_spec(user_request)` — uses LLM to parse natural language into structured JSON spec
  - Enables commands like "Create a tool that calculates BMI"

- **Custom Tool Detection** (`agent.py`)
  - `custom_tool_keywords` dict maps keywords to tool names
  - `_detect_custom_tool()` routes goals directly to custom tools
  - Keywords auto-generated from tool name, description, and functions
  - Agent bypasses planning phase for clear custom tool matches

- **Example Custom Tools Created**
  - `bmi_calculator` — calculates BMI from height/weight (metric & imperial)
  - `temperature_converter` — converts between Celsius and Fahrenheit

#### Modified
- `agent.py`: Added custom tool loading, keyword detection, direct routing
- `brain.py`: Added tool_builder keywords and routing
- `tools/__init__.py`: Added ToolBuilderTool export
- `README.md`: Updated to 14 tools, added Tool Builder section

#### Usage
```python
from apprentice_agent.tools import ToolBuilderTool

builder = ToolBuilderTool()

# Create a custom tool
builder.create_tool(
    name='bmi_calculator',
    description='Calculate BMI from height and weight',
    functions_spec=[{
        'name': 'calculate_bmi',
        'params': ['weight_kg', 'height_m'],
        'description': 'Calculate BMI',
        'body': 'bmi = float(weight_kg) / (float(height_m) ** 2)\nreturn {"success": True, "bmi": round(bmi, 1)}'
    }]
)

# Test and enable
builder.test_tool('bmi_calculator')
builder.enable_tool('bmi_calculator')

# Use via agent
# "Calculate my BMI - height 1.75m, weight 70kg" → bmi_calculator
```

#### Registry Format (`data/custom_tools.json`)
```json
{
    "tools": [{
        "name": "bmi_calculator",
        "class_name": "BmiCalculatorTool",
        "description": "Calculate BMI from height and weight",
        "status": "active",
        "created": "2026-01-18T18:34:03",
        "file": "tools/custom/bmi_calculator.py",
        "test_file": "tools/custom/tests/test_bmi_calculator.py",
        "functions": ["calculate_bmi", "calculate_bmi_imperial"],
        "keywords": ["bmi", "body mass index", "height and weight"]
    }]
}
```

---

### Phase C - Voice, Vision & Fast-Path (2026-01-16)

#### Added
- **Improved Fast-Path Detection** (`agent.py`)
  - Agent self-questions: "who are you", "what can you do", "are you an AI"
  - Yes/no question detection: "can you", "is it", "should I"
  - Conversational starters: "explain", "define", "what do you think"
  - Increased word threshold (8 words default, 12 for question words)
  - More specific tool keywords to avoid false positives
  - Better system prompt with agent identity

- **Voice Interface** (`tools/voice.py`)
  - `VoiceTool` class with `listen()` and `speak()` methods
  - Speech-to-text using OpenAI Whisper (local, "base" model)
  - Text-to-speech using pyttsx3 (offline, no API needed)
  - Silence detection for automatic recording stop
  - Direct numpy array to Whisper (bypasses ffmpeg dependency)
  - `VoiceConversation` class for continuous voice chat
  - GUI integration: Voice Mode toggle, Speak button
  - CLI: `python main.py --voice` for voice conversation mode

- **Image Generation** (`tools/image_gen.py`)
  - `ImageGenerationTool` using Stable Diffusion 1.5
  - `generate(prompt)` method with customizable parameters
  - Options: negative_prompt, width, height, steps, guidance_scale, seed
  - `generate_variations(prompt, num_images)` for multiple outputs
  - Auto-saves to `generated_images/` folder
  - Lazy model loading (~4GB download on first use)
  - GPU/CPU auto-detection

#### Modified
- `agent.py`: Expanded `_is_simple_query()` with comprehensive patterns
- `agent.py`: Improved `_fast_path_response()` system prompt
- `tools/__init__.py`: Added VoiceTool, VoiceConversation, ImageGenerationTool exports
- `gui.py`: Added Voice Mode checkbox, Speak button, voice status display
- `main.py`: Added `--voice` CLI flag for voice conversation mode

#### Known Limitations
- **Python 3.14 + CUDA**: PyTorch CUDA wheels not yet available for Python 3.14
  - Image generation runs on CPU only (slow: ~5-10 min/image)
  - Workaround: Use Python 3.11/3.12 venv for GPU acceleration
- **Voice on Windows**: Requires working microphone; pyttsx3 uses Windows SAPI

---

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

- **Fast-Path Responses** (`agent.py`)
  - `_is_simple_query()` detects greetings and short questions
  - `_fast_path_response()` responds directly using fast model
  - Skips full 5-phase loop for conversational queries
  - CLI flag `--no-fastpath` to disable

- **GUI Enhancements**
  - Stats tab with metacognition statistics
  - "Run Dream Mode" button
  - Dream output display with generated insights

#### Modified
- `config.py`: Added MODEL_FAST, MODEL_REASON, MODEL_VISION settings
- `brain.py`: Added confidence scoring, multi-model routing, `_select_model()`
- `agent.py`: Integrated MetacognitionLogger, fast-path, displays confidence and model
- `metacognition.py`: Added `model_used` tracking and `model_usage` stats
- `main.py`: Added `--dream`, `--dream-date`, `--no-fastpath` CLI flags
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

## Roadmap

### Completed Features
- **Phase A** - Core Agent (5-phase loop, 7 tools, memory, GUI)
- **Phase B** - Self-Reflection (confidence scoring, metacognition, dream mode, multi-model routing)
- **Phase C** - Voice & Image Tools (Whisper STT, pyttsx3 TTS, Stable Diffusion)
- **Tool #14** - Self-Tool-Creation (tool_builder meta-tool, custom tool registry, auto-detection)

### Phase D - Adaptive Learning (Planned)
- Use dream insights to adjust planning strategies
- Tool selection based on historical success rates
- Confidence-based retry decisions
- Auto-tune model routing based on task performance

### Phase E - Multi-Agent (Planned)
- Spawn sub-agents for complex tasks
- Inter-agent communication
- Shared memory pool
- Task delegation and coordination
