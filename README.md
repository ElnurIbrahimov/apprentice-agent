# Apprentice Agent

An AI agent with memory and reasoning capabilities, powered by local LLMs via Ollama. **Monthly cost: $0** - runs entirely on your hardware.

## Features

- **15 Integrated Tools** - Web search, browser automation, code execution, vision, voice, PDF reading, system control, notifications, tool builder, plugin marketplace, and more
- **4-Model Routing** - Automatically selects the best model for each task type
- **Observe-Plan-Act-Evaluate-Remember Loop** - Structured reasoning cycle for achieving goals
- **Fast-Path Responses** - Instant replies for conversational queries without full agent loop
- **Long-Term Memory** - ChromaDB-powered memory system for learning from past experiences
- **Dream Mode** - Memory consolidation and pattern analysis from metacognition logs
- **Voice Interface** - Whisper STT and pyttsx3 TTS for hands-free interaction
- **Confidence Scoring** - Each action includes confidence levels for transparency
- **Metacognition Logging** - Detailed logs in `logs/metacognition/` for analysis
- **Gradio GUI** - Modern web interface with real-time thinking process visualization

## Requirements

- Python 3.14+
- [Ollama](https://ollama.ai/) running locally
- Required models (see Installation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ElnurIbrahimov/apprentice-agent.git
cd apprentice-agent
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

3. Pull the required Ollama models:
```bash
ollama pull qwen2:1.5b          # Fast responses
ollama pull llama3:8b           # Reasoning
ollama pull deepseek-coder:6.7b # Code tasks
ollama pull llava               # Vision
```

4. (Optional) Install browser automation:
```bash
pip install playwright
playwright install chromium
```

5. (Optional) Copy and configure environment:
```bash
cp .env.example .env
```

## Model Routing

The agent automatically selects the optimal model based on task type:

| Task Type | Model | Use Cases |
|-----------|-------|-----------|
| Simple | `qwen2:1.5b` | Greetings, short answers, basic queries |
| Code | `deepseek-coder:6.7b` | Code generation, debugging, scripts, algorithms |
| Reasoning | `llama3:8b` | Planning, evaluation, complex decisions, browser tasks |
| Vision | `llava` | Image analysis, screenshot description, OCR |

## Usage

### Goal-based execution

Run the agent with a specific goal:

```bash
python main.py "Search the web for latest news about AI and summarize it"
```

Limit iterations:
```bash
python main.py "List files in ./src" --max-iterations 3
```

### Interactive chat mode

```bash
python main.py --chat
```

Chat mode commands:
- `/goal <text>` - Run the agent loop with a goal
- `/recall <query>` - Search agent's memory
- `/clear` - Clear conversation history
- `/quit` - Exit

### Voice mode

```bash
python main.py --voice
```

Hands-free interaction using:
- **Whisper** for speech-to-text
- **pyttsx3** for text-to-speech

### Dream mode

Consolidate memories and analyze patterns from the day's activity:

```bash
python main.py --dream
```

Analyzes `logs/metacognition/` to generate insights about tool effectiveness and learning opportunities.

### GUI mode

Launch the Gradio web interface:

```bash
python gui.py
```

Opens at `http://127.0.0.1:7860` with:

- **Chat panel** - Send messages and view agent responses
- **Thinking Process tab** - Watch observe/plan/act/evaluate phases in real-time
- **Tool Usage tab** - Monitor tool invocations with timestamps
- **Memory tab** - Search past experiences and view memory stats
- **Settings tab** - Adjust max iterations

## Available Tools

| Tool | Description | Example Action |
|------|-------------|----------------|
| `web_search` | Search the internet via DuckDuckGo | `AI news 2024` |
| `filesystem` | List/read local files | `list C:/Users/project` |
| `code_executor` | Run Python code (sandboxed) | `print(math.factorial(50))` |
| `screenshot` | Capture screen images | `capture` |
| `vision` | Analyze images with LLaVA | `analyze screenshot.png` |
| `pdf_reader` | Read and search PDF files | `read document.pdf pages 1-5` |
| `clipboard` | Read/write system clipboard | `read` or `write "text"` |
| `voice` | Speech-to-text and text-to-speech | `speak "Hello world"` |
| `image_gen` | Generate images with Stable Diffusion | `a sunset over mountains` |
| `arxiv_search` | Search academic papers on arXiv | `transformer attention mechanism` |
| `browser` | Automate web browser with Playwright | `open github.com` |
| `system_control` | Volume, brightness, apps, system info | `set volume 50` |
| `notifications` | Reminders, scheduled alerts, conditional triggers | `remind me in 30 minutes` |
| `tool_builder` | Create, test, enable, disable custom tools | `list custom tools` |
| `marketplace` | Browse, install, publish, rate plugins | `browse plugins` |

### Code Executor Safety

The code executor runs Python code in a sandboxed subprocess with:
- **Blocked imports**: `os`, `subprocess`, `sys`, `socket`, `requests`, etc.
- **No file access**: `open()`, `file()` are blocked
- **Timeout protection**: 30 second default limit
- **Isolated execution**: Runs in temp directory
- **Escaped newline handling**: Converts LLM output `\n` to actual newlines

### Browser Safety

The browser tool blocks navigation to sensitive URLs containing:
- `login`, `signin`, `checkout`, `payment`, `bank`, `password`

### System Control Safety

The system control tool uses a strict allowlist for launching applications:
- **Allowed apps**: `notepad`, `calculator`, `browser`, `chrome`, `firefox`, `explorer`, `vscode`, `terminal`, `cmd`, `powershell`
- Volume and brightness controls are clamped to 0-100 range
- Lock screen requires no parameters (immediate action)

### Notifications

The notifications tool supports three types of alerts:

| Type | Description | Example |
|------|-------------|---------|
| **Reminders** | One-time notifications after a delay | "Remind me to take a break in 30 minutes" |
| **Scheduled** | Recurring notifications at specific times | "Notify me every day at 9 AM for standup" |
| **Conditional** | System threshold alerts | "Alert me when CPU exceeds 80%" |

**Scheduler Daemon**: To receive notifications, run the background scheduler:

```bash
python -m apprentice_agent.scheduler
```

The scheduler checks every 30 seconds and sends Windows toast notifications via `winotify`. Logs are stored in `logs/notifications/`.

### Tool Builder (Self-Extension)

The tool builder allows the agent to create new tools dynamically:

| Method | Description |
|--------|-------------|
| `create_tool(name, description, functions_spec)` | Generate a new tool from specification |
| `test_tool(name)` | Run auto-generated tests |
| `enable_tool(name)` | Activate tool for use |
| `disable_tool(name)` | Deactivate tool |
| `rollback_tool(name)` | Delete tool and remove from registry |
| `list_custom_tools()` | List all custom tools with status |

**Example - Creating a BMI Calculator:**

```python
from apprentice_agent.tools.tool_builder import ToolBuilderTool
builder = ToolBuilderTool()

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
builder.enable_tool('bmi_calculator')
```

**Safety:** Generated code is scanned for dangerous patterns (`eval`, `exec`, `subprocess`, `os.system`, etc.) before saving. Custom tools are stored in `tools/custom/` and registered in `data/custom_tools.json`.

### Plugin Marketplace

The marketplace allows browsing, installing, and sharing plugins from a remote registry:

| Method | Description |
|--------|-------------|
| `browse(category, sort_by)` | List plugins by category, sorted by downloads/rating/newest |
| `search(query)` | Search plugins by keyword |
| `get_info(plugin_id)` | Get full plugin details |
| `install(plugin_id)` | Download, scan, and enable a plugin |
| `uninstall(plugin_id)` | Remove an installed plugin |
| `publish(tool_name)` | Package a custom tool for sharing |
| `rate(plugin_id, stars)` | Rate a plugin 1-5 stars |
| `my_plugins()` | List installed plugins |
| `update(plugin_id)` | Check for and install updates |

**Example - Using the Marketplace:**

```python
from apprentice_agent.tools.marketplace import MarketplaceTool
mp = MarketplaceTool()

# Browse health plugins
mp.browse(category="health", sort_by="rating")

# Install a plugin
mp.install("bmi_calculator")

# Publish your custom tool
mp.publish("my_custom_tool")
```

**Natural Language:**
```
"Browse plugins in the marketplace"
"Install the weather_tool plugin"
"Publish my temperature_converter to the marketplace"
```

**Safety:** Downloaded plugins are scanned for dangerous patterns before installation. Logs are stored in `logs/marketplace/`.

**Registry:** Plugins are hosted at `github.com/ElnurIbrahimov/aura-plugins`

## Configuration

Edit `apprentice_agent/config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `MODEL_FAST` | `qwen2:1.5b` | Model for simple tasks |
| `MODEL_CODE` | `deepseek-coder:6.7b` | Model for code tasks |
| `MODEL_REASON` | `llama3:8b` | Model for reasoning |
| `MODEL_VISION` | `llava` | Model for vision tasks |
| `CHROMADB_PATH` | `./data/chromadb` | Memory storage location |

## Architecture

```
apprentice-agent/
├── gui.py                    # Gradio web interface
├── main.py                   # CLI entry point
└── apprentice_agent/
    ├── agent.py              # Main agent loop (observe/plan/act/evaluate/remember)
    ├── brain.py              # OllamaBrain - LLM interface and 4-model routing
    ├── memory.py             # ChromaDB-powered long-term memory
    ├── config.py             # Configuration settings
    ├── metacognition.py      # Confidence scoring and action logging
    ├── dream.py              # Memory consolidation and pattern analysis
    ├── scheduler.py          # Background daemon for notifications
    └── tools/
        ├── web_search.py     # DuckDuckGo search
        ├── filesystem.py     # File operations
        ├── code_executor.py  # Sandboxed Python execution
        ├── screenshot.py     # Screen capture with mss
        ├── vision.py         # Image analysis with LLaVA
        ├── pdf_reader.py     # PDF text extraction with PyMuPDF
        ├── clipboard.py      # System clipboard access
        ├── voice.py          # Whisper STT + pyttsx3 TTS
        ├── image_gen.py      # Stable Diffusion image generation
        ├── arxiv_search.py   # arXiv paper search and summarization
        ├── browser.py        # Playwright browser automation
        ├── system_control.py # Volume, brightness, apps, system info
        ├── notifications.py  # Reminders, scheduled, conditional alerts
        ├── tool_builder.py   # Meta-tool for creating custom tools
        ├── tool_template.py  # Templates for generated tools
        ├── marketplace.py    # Plugin marketplace
        └── custom/           # Auto-generated custom tools
```

## License

MIT
