# Apprentice Agent

An AI agent with memory and reasoning capabilities, powered by local LLMs via Ollama. **Monthly cost: $0** - runs entirely on your hardware.

## Features

- **19 Integrated Tools** - Web search, browser automation, code execution, vision, voice, PDF reading, system control, notifications, tool builder, plugin marketplace, FluxMind, regex builder, git, Clawdbot messaging, and more
- **5-Model Routing** - Automatically selects the best model for each task type (including FluxMind for calibrated reasoning)
- **Observe-Plan-Act-Evaluate-Remember Loop** - Structured reasoning cycle for achieving goals
- **Fast-Path Responses** - Instant replies for conversational queries without full agent loop
- **Long-Term Memory** - ChromaDB-powered memory system for learning from past experiences
- **Dream Mode** - Memory consolidation and pattern analysis from metacognition logs
- **Voice Interface** - Whisper STT + pyttsx3 TTS or PersonaPlex real-time duplex voice
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
| Calibrated | `FluxMind v0.75.1` | Uncertainty-aware reasoning, OOD detection, confidence checks |

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
| `fluxmind` | Calibrated reasoning with uncertainty awareness | `FluxMind status` |
| `regex_builder` | Build, test, and explain regular expressions | `build regex for email` |
| `git` | Git repository management with natural language | `what branch am I on?` |
| `personaplex` | Real-time full-duplex voice with NVIDIA PersonaPlex | `start personaplex` |
| `clawdbot` | Send/receive messages via WhatsApp, Telegram, Discord | `send "Hello" to +1234567890` |

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

### FluxMind (Calibrated Reasoning)

FluxMind v0.75.1 is a calibrated uncertainty-aware reasoning engine that **knows when it doesn't know**:

| Capability | Description |
|------------|-------------|
| **Calibrated Confidence** | Real confidence scores (not LLM hallucinations) |
| **OOD Detection** | 1664x confidence drop on unfamiliar inputs |
| **Sub-ms Inference** | <1ms vs 500ms+ for LLMs |
| **Compositional Programs** | Mix reasoning strategies mid-sequence |

**Performance:**
- 99.86% accuracy on in-distribution inputs
- 0.06% confidence on out-of-distribution inputs

**Commands:**

```bash
# Check status
python main.py "FluxMind status"

# Execute a reasoning step
python main.py "FluxMind step [5,3,7,2] op 0 context 0"

# Check confidence on a state
python main.py "Ask FluxMind about state [5,3,7,2]"

# Test OOD detection (should show low confidence)
python main.py "How confident is FluxMind about [25,25,25,25]?"
```

**Example Output:**
```
FluxMind Step Result:
  Input: [5, 3, 7, 2]
  Next State: [6, 3, 7, 2]
  Confidence: 99.86%
  Should Trust: True
```

**Training:** The model is pre-trained and included at `models/fluxmind_v0751.pt`. To retrain:
```python
from tools.fluxmind import train_fluxmind
train_fluxmind("models/fluxmind_v0751.pt")
```

### Regex Builder

The regex builder tool creates, tests, and explains regular expressions using natural language:

| Method | Description |
|--------|-------------|
| `build(description)` | Natural language to regex pattern (26 common patterns) |
| `test(pattern, test_string)` | Test pattern with matches, groups, positions |
| `explain(pattern)` | Human-readable breakdown of regex components |
| `find_all(pattern, text)` | Find all matches with positions and highlighting |
| `replace(pattern, text, replacement)` | Regex substitution with count |
| `validate(pattern)` | Check if pattern is syntactically valid |
| `common_patterns()` | Get 26 pre-built patterns (email, url, phone, ip, date, uuid, etc.) |

**Example - Building and Testing Patterns:**

```python
from apprentice_agent.tools.regex_builder import RegexBuilderTool
regex = RegexBuilderTool()

# Build from natural language
result = regex.build("match email addresses")
# → pattern: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}

# Test a pattern
result = regex.test(r'\d+', 'abc 123 def 456')
# → matches: ['123', '456'], positions: [{start: 4, end: 7}, ...]

# Explain a pattern
result = regex.explain(r'^[a-z]+$')
# → "Matches lowercase letters from start to end of string"

# Get common patterns
patterns = regex.common_patterns()
# → email, url, phone, ip_address, date_iso, uuid, hex_color, ...
```

**Natural Language:**
```
"Build a regex for phone numbers"
"Test pattern \d+ against 'abc123def456'"
"Explain this regex: ^[a-zA-Z0-9]+$"
"What common regex patterns are available?"
```

### Git Tool

The git tool provides repository management with natural language support and fast-path routing that bypasses the LLM to prevent hallucination:

| Method | Description |
|--------|-------------|
| `status(repo_path)` | Branch, staged, unstaged, untracked files |
| `log(repo_path, count)` | Commit history with hash, author, message, date |
| `diff(repo_path, file)` | Changes with summary and diff content |
| `branch(repo_path)` | Local/remote branches with current marked |
| `add(repo_path, files)` | Stage files for commit |
| `commit(repo_path, message)` | Create commit with message |
| `push(repo_path, remote, branch)` | Push to remote repository |
| `pull(repo_path, remote, branch)` | Pull from remote repository |
| `clone(url, destination)` | Clone a repository |
| `stash(repo_path, action, message)` | Stash operations (push/pop/list/drop/clear) |

**Example - Using Git Tool:**

```python
from apprentice_agent.tools.git_tool import GitTool
git = GitTool()

# Get status
result = git.status('.')
# → ACTUAL GIT STATUS: Branch: main, Staged: 1, Unstaged: 2...

# View recent commits
result = git.log('.', count=5)
# → ACTUAL GIT LOG (5 commits): abc1234 Fix bug (2 hours ago)...

# Show branches
result = git.branch('.')
# → ACTUAL GIT BRANCHES: Current branch: main
```

**Natural Language (Fast-Path):**
```
"what branch am I on?"      → Shows current branch
"show recent commits"       → Shows git log
"any unstaged files?"       → Shows git status
"what changed?"             → Shows git status
"show staged files"         → Shows git status
```

**Note:** Git commands use fast-path routing with "ACTUAL GIT" prefixed output to ensure real data is displayed verbatim, not hallucinated by the LLM.

### PersonaPlex (Real-time Voice)

PersonaPlex provides real-time full-duplex speech-to-speech conversations using NVIDIA's PersonaPlex model. It replaces the traditional Whisper+pyttsx3 pipeline for natural voice interactions.

| Method | Description |
|--------|-------------|
| `status()` | Check if PersonaPlex server is running |
| `start_server(voice, persona)` | Launch the voice server |
| `stop_server()` | Shutdown the server |
| `set_voice(voice_id)` | Change voice (16 options) |
| `set_persona(prompt)` | Set AI personality/role |
| `list_voices()` | Show available voices |
| `reset_to_defaults()` | Reset to Aura persona |

**Available Voices (16 total):**

| Category | Voice IDs |
|----------|-----------|
| Natural Female | NATF0, NATF1, NATF2, NATF3 |
| Natural Male | NATM0, NATM1 (default), NATM2, NATM3 |
| Variety Female | VARF0, VARF1, VARF2, VARF3, VARF4 |
| Variety Male | VARM0, VARM1, VARM2, VARM3, VARM4 |

**Setup:**

```bash
# 1. Install opus codec (required)
sudo apt install libopus-dev  # Ubuntu/Debian
brew install opus             # macOS

# 2. Set HuggingFace token (required)
export HF_TOKEN=<your_huggingface_token>

# 3. Accept the PersonaPlex license on HuggingFace model card
```

**Example - Using PersonaPlex:**

```python
from apprentice_agent.tools.personaplex import PersonaPlexTool
pp = PersonaPlexTool()

# Check status
pp.status()

# Start with default Aura persona
pp.start_server()

# Start with custom voice and persona
pp.start_server(voice="NATF2", persona="You are a helpful coding assistant.")

# Change voice while running (requires restart)
pp.set_voice("VARM1")

# Stop server
pp.stop_server()
```

**Natural Language:**
```
"start personaplex"              → Launch with defaults (NATM1 voice, Aura persona)
"personaplex status"             → Check if running
"list personaplex voices"        → Show 16 available voices
"set voice to NATF2"             → Change to Natural Female 2
"set persona to helpful teacher" → Update AI personality
"stop personaplex"               → Shutdown server
```

**Default Aura Persona:**
> "You are Aura, an intelligent personal AI assistant. You are wise, helpful, and occasionally witty with subtle sarcasm."

**Web Interface:** Once started, access the voice interface at `https://localhost:8998` (accept the self-signed certificate).

**Safety:** Requires `HF_TOKEN` environment variable. Server does not auto-start.

### Clawdbot (Multi-Platform Messaging)

Clawdbot enables Aura to send and receive messages via WhatsApp, Telegram, Discord, Signal, and iMessage through a unified gateway.

| Method | Description |
|--------|-------------|
| `send_message(to, message, channel)` | Send message to phone number or username |
| `get_status()` | Check gateway status |
| `list_channels()` | List connected messaging channels |
| `start_gateway(port)` | Start the Clawdbot gateway |
| `stop_gateway()` | Stop the gateway |
| `pair_channel(channel)` | Pair a new channel (WhatsApp QR, etc.) |

**Setup:**

```bash
# 1. Install Clawdbot CLI
npm install -g clawdbot@latest

# 2. Run initial setup
clawdbot setup
clawdbot config set gateway.mode local

# 3. Enable WhatsApp plugin
clawdbot plugins enable whatsapp

# 4. Start gateway
clawdbot gateway --port 18789

# 5. Pair WhatsApp (scan QR code)
clawdbot channels login
```

**Example - Sending Messages:**

```python
from apprentice_agent.tools.clawdbot import ClawdbotTool
cb = ClawdbotTool()

# Check status
cb.get_status()

# Send WhatsApp message
cb.send_message("+1234567890", "Hello from Aura!", "whatsapp")

# Send Telegram message
cb.send_message("@username", "Hello!", "telegram")
```

**Natural Language:**
```
"send whatsapp message 'Meeting at 3pm' to +1234567890"
"text John on telegram saying I'll be late"
"what's the clawdbot status?"
"start the clawdbot gateway"
```

**Aura-Clawdbot Bridge:** For two-way communication (receiving messages and auto-responding), run the bridge:

```bash
python clawdbot_bridge.py
```

This connects the gateway to Aura, allowing incoming messages to trigger agent responses.

**Supported Channels:**
- WhatsApp (via WhatsApp Web)
- Telegram (bot token)
- Discord (bot token)
- Signal (signal-cli)
- iMessage (macOS only)

**GUI Integration:** The Aura GUI includes a Clawdbot panel in the sidebar for quick status checks and message sending.

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
| `PERSONAPLEX_ENABLED` | `true` | Enable PersonaPlex voice tool |

## Architecture

```
apprentice-agent/
├── gui.py                    # Gradio web interface
├── main.py                   # CLI entry point
├── clawdbot_bridge.py        # Aura-Clawdbot message bridge
├── models/
│   └── fluxmind_v0751.pt     # Trained FluxMind model (1.5MB)
├── tools/
│   └── fluxmind/             # FluxMind calibrated reasoning engine
│       ├── fluxmind_core.py  # Core model (393K params)
│       └── fluxmind_tool.py  # Aura integration wrapper
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
        ├── regex_builder.py  # Regex pattern building and testing
        ├── git_tool.py       # Git repository management
        ├── personaplex/      # NVIDIA PersonaPlex real-time voice
        │   └── personaplex_tool.py
        ├── clawdbot.py       # Multi-platform messaging (WhatsApp, Telegram, etc.)
        └── custom/           # Auto-generated custom tools
```

## License

MIT
