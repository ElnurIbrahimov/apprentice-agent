# Apprentice Agent

An AI agent with memory and reasoning capabilities, powered by local LLMs via Ollama. **Monthly cost: $0** - runs entirely on your hardware.

## Features

- **30 Integrated Tools** - Web search, browser automation, code execution, vision, voice, PDF reading, system control, notifications, tool builder, plugin marketplace, FluxMind, regex builder, git, Clawdbot messaging, EvoEmo emotional tracking, Inner Monologue, Knowledge Graph Memory, Metacognitive Guardian, NeuroDream sleep/dream memory consolidation, MirrorMind self-critique, CognitiveTheater multi-perspective reasoning, Reflexion learning-from-mistakes, SynapseForge dynamic tool creation, WorldSim consequence simulation, **AURA v3.0 ALIVE emotional presence system**, and more
- **5-Model Routing** - Automatically selects the best model for each task type (including FluxMind for calibrated reasoning)
- **Observe-Plan-Act-Evaluate-Remember Loop** - Structured reasoning cycle for achieving goals
- **Fast-Path Responses** - Instant replies for conversational queries without full agent loop
- **Long-Term Memory** - ChromaDB-powered memory system for learning from past experiences
- **Knowledge Graph** - Relationship-based memory with semantic understanding (nodes, edges, paths)
- **Dream Mode** - Memory consolidation and pattern analysis from metacognition logs
- **Voice Interface** - Whisper STT + Sesame CSM 1B TTS (human-quality) or pyttsx3 fallback
- **Confidence Scoring** - Each action includes confidence levels for transparency
- **Metacognition Logging** - Detailed logs in `logs/metacognition/` for analysis
- **Gradio GUI** - Modern web interface with real-time thinking process visualization
- **Inner Monologue** - Real-time visualization and vocalization of Aura's thought process (perceive, recall, reason, decide, execute, reflect)

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
- **Sesame CSM 1B** for high-quality text-to-speech (or pyttsx3 fallback)

### Sesame CSM 1B (Human-Quality TTS)

Aura uses [Sesame CSM 1B](https://github.com/SesameAILabs/csm), an open-source conversational speech model that produces remarkably human-like voice output.

**Features:**
- Human-quality conversational TTS
- ~4.5GB VRAM on CUDA
- Pipeline mode for low latency
- Falls back to pyttsx3 if unavailable

**Setup:**

```bash
# 1. Clone Sesame CSM
git clone https://github.com/SesameAILabs/csm ~/sesame-csm
cd ~/sesame-csm && pip install -e .

# 2. Set HuggingFace token (required for model access)
export HF_TOKEN=<your_huggingface_token>

# 3. Accept model licenses on HuggingFace:
#    - sesame/csm-1b
#    - meta-llama/Llama-3.2-1B
```

**GUI Usage:**

1. Launch `python gui.py`
2. Click **Load Sesame** in the sidebar (takes ~30s first time)
3. Enable **Voice** checkbox
4. Aura now speaks with human-quality voice

**Note:** On Windows, Sesame requires CUDA. If unavailable, Aura automatically uses pyttsx3 as fallback.

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
| `sesame_tts` | High-quality TTS using Sesame CSM 1B (open-source) | GUI: Load Sesame button |
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
| `evoemo` | Emotional state tracking and adaptive responses | `my mood` or `mood history` |
| `inner_monologue` | Real-time thought visualization and Think Aloud | `show thoughts` or `why did you do that?` |
| `knowledge_graph` | Relationship-based memory with semantic understanding | `what do you know about X?` |
| `metacog_guardian` | Self-aware failure prediction and proactive intervention | `guardian stats` or `set guardian level high` |
| `mirrormind` | Self-critique system that evaluates and improves responses | `enable mirrormind` or set `MIRRORMIND_ENABLED=true` |
| `cognitive_theater` | Multi-perspective reasoning for decision questions | `Should I use X or Y?` or `Compare A vs B` |
| `reflexion` | Learn from mistakes - retry failed tasks with accumulated lessons | Automatic for code execution failures |
| `synapseforge` | Dynamic tool creation - synthesizes new tools at runtime | Automatic when capability gaps detected |
| `worldsim` | Consequence simulation - previews risky actions before execution | Automatic for dangerous commands |
| `aura` | AURA v3.0 ALIVE - emotionally present AI with memory, mood, patterns | `aura status` or `aura mood` |

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

### EvoEmo (Emotional State Tracking)

EvoEmo is Aura's emotional intelligence system that detects user emotional states and adapts responses accordingly.

| Method | Description |
|--------|-------------|
| `analyze_text(text)` | Detect emotion from text input |
| `get_current_mood()` | Get most recent emotional reading |
| `get_session_summary()` | Get current session emotion stats |
| `get_daily_summary()` | Get today's emotion summary |
| `get_history(days)` | Get mood history for last N days |
| `get_patterns()` | Analyze emotional patterns over time |
| `clear_history()` | Clear all mood data (privacy) |
| `set_enabled(bool)` | Enable/disable tracking |

**Detected Emotions:**

| Emotion | Indicators | Response Adaptation |
|---------|------------|---------------------|
| `calm` | Polite language, proper sentences | Normal responses |
| `focused` | Direct questions, "specifically", "exactly" | Concise, no fluff |
| `stressed` | "urgent", "ASAP", exclamation marks | Supportive, step-by-step |
| `frustrated` | "ugh", "doesn't work", ALL CAPS | Acknowledge, solution-focused |
| `excited` | "awesome", "can't wait", "!" | Match energy, enthusiastic |
| `tired` | "exhausted", short responses, "meh" | Brief, gentle, suggest breaks |
| `curious` | Questions, "how", "why", "explain" | Detailed, educational |

**Example - Emotion Detection:**

```python
from apprentice_agent.tools.evoemo import EvoEmoTool
evoemo = EvoEmoTool()

# Analyze text
reading = evoemo.analyze_text("Ugh, this STILL isn't working!!")
# → emotion: frustrated, confidence: 85%

# Get current mood
mood = evoemo.get_current_mood()
print(f"{evoemo.get_mood_emoji()} {mood.emotion}")
# → 😤 frustrated
```

**Natural Language:**
```
"what's my mood?"           → Shows current emotional state
"mood history"              → Shows 7-day emotion distribution
"mood patterns"             → Analyzes stress hours, dominant emotions
"clear mood history"        → Deletes all mood data
"disable mood tracking"     → Turns off emotion detection
```

**Adaptive Responses:**

When EvoEmo detects emotional states with high confidence (>50%), Aura automatically adjusts:

1. **System Prompt** - Adds tone modifiers to guide LLM responses
2. **Response Length** - Shorter for tired/frustrated, detailed for curious
3. **Voice Parameters** - Slower/calmer for stressed, energetic for excited
4. **Acknowledgments** - "I understand this is frustrating" for negative emotions

**GUI Integration:**

The Aura GUI shows a real-time mood indicator in the sidebar with:
- Emoji representation of current emotion
- Confidence percentage
- Color-coded indicator (green=calm, red=frustrated, etc.)
- Expandable panel for mood history and patterns

**Privacy:**

- All processing is local (no data sent externally)
- History stored in `data/evoemo/mood_history.jsonl`
- Users can disable tracking or clear history at any time
- Rolling 7-day history by default

### Inner Monologue (Thought Visualization)

The Inner Monologue system makes Aura's thinking visible and audible in real-time, providing transparency into the agent's reasoning process.

| Method | Description |
|--------|-------------|
| `start_session()` | Begin a new monologue session |
| `think(type, content, confidence)` | Emit a thought |
| `get_recent_thoughts(n)` | Get last N thoughts |
| `get_reasoning_chain()` | Get formatted reasoning for "why?" queries |
| `set_verbosity(level)` | Set detail level (0=silent, 3=debug) |
| `end_session()` | Finalize and save session |

**Thought Types:**

| Type | Icon | Description |
|------|------|-------------|
| `perceive` | 🔍 | Understanding user input |
| `recall` | 💾 | Memory retrieval |
| `reason` | 🧠 | Logic and planning |
| `decide` | ⚡ | Tool/action selection |
| `execute` | 🔧 | Running tool |
| `reflect` | 🪞 | Evaluating result |
| `uncertain` | ❓ | Low confidence moment |
| `eureka` | 💡 | Insight/breakthrough |

**Example - Using Inner Monologue:**

```python
from apprentice_agent.tools.inner_monologue import get_monologue

monologue = get_monologue()
monologue.start_session()

# Emit thoughts during agent execution
monologue.think("perceive", "Received query about weather", confidence=95)
monologue.think("reason", "User wants current conditions. Should use web_search.", confidence=85)
monologue.think("decide", "Selected tool: web_search", confidence=90)

# Get reasoning chain
chain = monologue.get_reasoning_chain()
print(chain)

monologue.end_session()
```

**Natural Language:**
```
"show thoughts"           → Display recent thoughts
"why did you do that?"    → Show reasoning chain
"think aloud on"          → Enable spoken thoughts
"verbosity 3"             → Set debug level
"export thoughts"         → Save session to file
```

**Think Aloud Mode:**

When enabled, Aura speaks her thoughts aloud using TTS (Sesame or pyttsx3 fallback):

1. Enable **Voice** in the GUI sidebar
2. Check **Think Aloud** in the Inner Monologue panel
3. Send a query that triggers the agent loop
4. Hear Aura narrate: "Perceiving... Received user query" → "Reasoning..." → "Deciding..."

**GUI Integration:**

The Aura GUI includes an "Aura's Thoughts" accordion panel with:
- Real-time thought stream with emoji icons and confidence scores
- Verbosity slider (0=silent to 3=debug)
- Think Aloud toggle
- Refresh, Clear, Export, and Why? buttons
- "Aura's Recent Reasoning" chain display

**Session Logging:**

All thoughts are persisted to `logs/inner_monologue/`:
- `sessions/YYYY-MM-DD_session_<id>.jsonl` - Individual session thoughts
- `summaries/YYYY-MM-DD_summary.json` - Daily aggregate statistics

**Integration with Agent Loop:**

The Inner Monologue is automatically integrated into the agent's OBSERVE → PLAN → ACT → EVALUATE → REMEMBER cycle:

| Phase | Thought Type |
|-------|--------------|
| OBSERVE | `perceive` - User input received |
| OBSERVE | `recall` - Memory search results |
| PLAN | `reason` - Planning approach |
| PLAN | `decide` - Tool selection |
| ACT | `execute` - Running tool |
| EVALUATE | `reflect` - Result assessment |
| EVALUATE | `eureka` or `uncertain` - Confidence-based |

### Knowledge Graph (Relationship Memory)

The Knowledge Graph provides semantic memory that stores entities and relationships, allowing Aura to answer questions like "what do you know about X?" and "how is X related to Y?"

| Method | Description |
|--------|-------------|
| `add_node(type, label, properties)` | Add entity/concept to graph |
| `add_edge(source, target, type)` | Create relationship |
| `find_nodes(query, type)` | Search for nodes |
| `get_related(node_id, depth)` | Get neighborhood |
| `find_path(source, target)` | Find connection path |
| `query(question)` | Natural language graph query |
| `consolidate()` | Dream mode: merge similar, prune weak |
| `get_stats()` | Graph statistics |

**Node Types (10):**

| Type | Icon | Description |
|------|------|-------------|
| `concept` | 💡 | Ideas, topics, domains |
| `entity` | 📌 | Specific things |
| `person` | 👤 | People |
| `project` | 📁 | Projects |
| `tool` | 🔧 | Aura's tools |
| `event` | 📅 | Things that happened |
| `emotion` | 💚 | Emotional associations |
| `skill` | ⚡ | Learned capabilities |
| `location` | 📍 | Places |
| `file` | 📄 | Files |

**Edge Types (16):**

`relates_to`, `is_a`, `part_of`, `causes`, `solves`, `created_by`, `uses`, `triggers`, `learned_from`, `preceded_by`, `followed_by`, `conflicts_with`, `strengthens`, `weakens`, `knows`, `works_on`, `located_at`

**Example - Using Knowledge Graph:**

```python
from apprentice_agent.tools.knowledge_graph import get_knowledge_graph

kg = get_knowledge_graph()

# Add nodes
aura = kg.add_node("entity", "Aura", {"role": "AI assistant"})
python = kg.add_node("concept", "Python", {"type": "language"})

# Add relationship
kg.add_edge(aura.id, python.id, "knows", weight=0.9)

# Query
results = kg.query("what do you know about Aura?")
# → [💡 Aura [100%], 🔧 web_search [80%], ...]

# Find path
path = kg.find_path("Aura", "Python")
# → Aura --knows--> Python
```

**Natural Language:**
```
"what do you know about X?"     → Search for X and related concepts
"how is X related to Y?"        → Find connection path
"remember that X uses Y"        → Add relationship
"show graph stats"              → Display node/edge counts
"consolidate memory"            → Run dream mode optimization
```

**GUI Integration:**

The Aura GUI includes a Knowledge Graph panel with:
- Interactive vis.js graph visualization
- Search and center on node
- Depth slider for neighborhood expansion
- Add Knowledge form (type, label, related to)
- Find Path between concepts
- Consolidate button for dream mode

**Persistence:**

- Nodes stored in `data/knowledge_graph/nodes.jsonl`
- Edges stored in `data/knowledge_graph/edges.jsonl`
- Stats in `data/knowledge_graph/stats.json`
- Append-only format for durability

**Hybrid Memory:**

The Knowledge Graph integrates with ChromaDB for hybrid retrieval:

```python
from apprentice_agent.tools.hybrid_memory import create_hybrid_memory

memory = create_hybrid_memory(chromadb=my_chromadb)

# Store with relationships
memory.remember("Aura is an AI assistant", relations=[
    {"target": "Elnur", "type": "created_by"}
])

# Recall with both vector search and graph traversal
results = memory.recall("AI assistant", use_graph=True, use_vectors=True)
```

**Seeding Initial Knowledge:**

```python
from apprentice_agent.tools.knowledge_graph import get_knowledge_graph, seed_initial_knowledge

kg = get_knowledge_graph()
result = seed_initial_knowledge(kg)
# → {"nodes_created": 18, "edges_created": 15, "status": "seeded"}
```

This seeds core knowledge about Aura, Elnur, tools, and foundational concepts.

### Metacognitive Guardian (Self-Aware Failure Prediction)

The Metacognitive Guardian is a meta-layer that monitors Aura's reasoning and predicts failures BEFORE they happen, enabling proactive intervention rather than reactive error handling.

| Method | Description |
|--------|-------------|
| `assess_risk(task, tool, context)` | Predict failure probability for task/tool |
| `execute_intervention(prediction)` | Execute recommended intervention |
| `record_outcome(prediction_id, success)` | Learn from actual outcomes |
| `get_stats()` | Get guardian statistics |
| `set_monitoring_level(level)` | Adjust sensitivity (low/medium/high/critical) |

**Failure Types (9):**

| Type | Description | Intervention |
|------|-------------|--------------|
| `knowledge_gap` | Missing information | Request clarification |
| `tool_mismatch` | Wrong tool for task | Suggest alternative |
| `ambiguous_request` | Unclear user intent | Ask for clarification |
| `confidence_drop` | Sudden confidence loss | Pause and explain |
| `emotional_mismatch` | Tone doesn't match mood | Emotional adjustment |
| `skill_boundary` | Task beyond capabilities | Human handoff |
| `hallucination_risk` | High fabrication risk | Confidence warning |
| `context_overflow` | Too much context | Summarize/truncate |
| `loop_detected` | Repeated failed attempts | Abort with explanation |

**Monitoring Levels:**

| Level | Warning | Intervention | Abort | Use Case |
|-------|---------|--------------|-------|----------|
| `low` | 50% | 75% | 95% | Autonomous operation |
| `medium` | 30% | 60% | 90% | Balanced (default) |
| `high` | 20% | 45% | 80% | Learning/unfamiliar tasks |
| `critical` | 10% | 30% | 60% | High-stakes operations |

**Example - Using Metacognitive Guardian:**

```python
from apprentice_agent.tools.metacog_guardian import get_guardian, GuardianConfig

guardian = get_guardian(config=GuardianConfig(monitoring_level="medium"))

# Assess risk before action
prediction = guardian.assess_risk(
    task="delete all files in directory",
    tool="filesystem",
    context={"user_mood": "frustrated"}
)

if prediction and prediction.probability > 0.6:
    intervention = guardian.execute_intervention(prediction)
    print(intervention["message"])
    # → "I want to make sure I understand correctly..."

# Record outcome for learning
guardian.record_outcome(prediction.id, was_successful=True)
```

**Natural Language:**
```
"guardian stats"              → Show monitoring level, predictions, interventions
"guardian status"             → Same as above
"set guardian level high"     → Increase sensitivity
"show predictions"            → Recent failure predictions
"failure patterns"            → Learned patterns from history
"reset guardian"              → Clear session data
```

**Integration with Agent Loop:**

The Guardian automatically monitors every task in `agent.run()`:

1. **Pre-check**: Before processing, assess risk of the incoming task
2. **Tool Selection**: Detect tool mismatches before execution
3. **Confidence Monitoring**: Track confidence drops during reasoning
4. **Post-execution**: Record outcomes to improve future predictions

**GUI Integration:**

The Aura GUI includes a Metacognitive Guardian panel with:
- Guardian status indicator (Active/Inactive)
- Monitoring level dropdown (low/medium/high/critical)
- Stats display (interventions, patterns learned, predictions)
- Threshold configuration accordion
- Recent predictions accordion
- Helpful/Not Helpful feedback buttons

**Learning System:**

The Guardian learns from outcomes:
- Stores failure patterns in `data/metacog_guardian/patterns/`
- Records outcomes in `data/metacog_guardian/outcomes/`
- Improves prediction accuracy over time
- Patterns are keyed by failure type for targeted learning

**Integration Points:**

| System | Integration |
|--------|-------------|
| Inner Monologue | Monitors thought confidence levels |
| EvoEmo | Detects emotional mismatch risks |
| Knowledge Graph | Checks for knowledge gaps |
| FluxMind | Leverages calibrated confidence |

### NeuroDream (Sleep/Dream Memory Consolidation)

NeuroDream is a sleep/dream memory consolidation system inspired by biological sleep cycles and the research paper "NeuroDream: A Biologically-Inspired Sleep Architecture for Lifelong Learning" (which showed 38% reduction in catastrophic forgetting and 17.6% increase in zero-shot transfer).

**Sleep Phases:**

| Phase | Duration | Purpose |
|-------|----------|---------|
| Light | 30-60s | Recent memory replay, working memory transfer |
| Deep | 60-120s | Pattern abstraction, knowledge graph strengthening |
| REM | 30-90s | Creative synthesis, novel connections, insight generation |

**Triggers:**

- **Scheduled**: Configurable intervals (default: every 4 hours)
- **Idle Detection**: When user is inactive (default: 30 minutes)
- **Manual**: User-initiated via commands
- **Memory Threshold**: When memory buffer reaches capacity

**Commands:**

```
# Enter sleep mode
"go to sleep"
"sleep now"
"enter sleep mode"

# Check sleep status
"dream status"
"sleep status"
"neurodream status"

# Wake up (interrupt sleep)
"wake up"
"stop sleeping"

# View dream journal
"dream journal"
"show dreams"
"recent dreams"

# View insights
"dream insights"
"show insights"

# View consolidated patterns
"sleep patterns"
"consolidated patterns"
```

**Example Session:**

```
User: go to sleep
Aura: Entering sleep mode...

  Phase 1: LIGHT SLEEP
  - Replaying 15 recent memories
  - Strengthening 8 connections

  Phase 2: DEEP SLEEP
  - Abstracting patterns from 47 memories
  - Found 3 new patterns
  - Consolidated 12 memory clusters

  Phase 3: REM SLEEP
  - Creative synthesis in progress
  - Generated 2 novel insights
  - Connected 4 distant concepts

Sleep cycle complete!
Duration: 3m 24s
Insights generated: 2
Patterns consolidated: 3
```

**Safety Features:**

- **Interruptible**: User activity or commands wake the system
- **Resource-Limited**: Caps VRAM usage (default: 2GB)
- **Read-Only Operations**: Only writes to memory stores
- **Scheduled Downtime**: Respects user activity patterns

**GUI Integration:**

The Aura GUI includes a NeuroDream panel with:
- Sleep status indicator (Awake/Dreaming with phase)
- Session and insight counters
- Sleep Now / Wake Up buttons
- Dream Journal accordion (recent dream entries)
- Dream Insights accordion (generated insights)
- Consolidated Patterns accordion (pattern summaries)

**Storage:**

- Dream journal: `data/neurodream/dream_journal.jsonl`
- Insights: `data/neurodream/insights.jsonl`
- Consolidated patterns: `data/neurodream/consolidated/`
- Session logs: `data/neurodream/sessions/`

**Integration Points:**

| System | Integration |
|--------|-------------|
| ChromaDB | Replays and consolidates vector memories |
| Knowledge Graph | Strengthens edges, finds new paths, abstracts patterns |
| EvoEmo | Emotional tagging of memories influences consolidation priority |
| Inner Monologue | Dreams appear as special "dream" thought type |
| Metacognitive Guardian | Learns from sleep-discovered patterns |

### MirrorMind (Self-Critique System)

MirrorMind is a self-critique system that evaluates Aura's responses and improves them before showing to the user. It implements an iterative refinement loop that scores responses on multiple dimensions.

**Flow:**
```
User Query → Generate Response → Critique → Score Too Low? → Improve → Repeat
                                      ↓
                               Score Good? → Return Final Response
```

**Critique Dimensions (weighted):**

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Accuracy | 25% | Is the information correct? |
| Completeness | 25% | Does it fully answer the question? |
| Clarity | 20% | Is it easy to understand? |
| Actionable | 15% | Does it provide clear next steps? |
| Tone | 15% | Is the tone appropriate? |

**Configuration:**

| Setting | Default | Description |
|---------|---------|-------------|
| `MIRRORMIND_ENABLED` | `false` | Enable self-critique (adds latency) |
| `MIRRORMIND_THRESHOLD` | `0.75` | Minimum quality score to accept |
| `MIRRORMIND_MAX_ITERATIONS` | `2` | Maximum improvement rounds |

**Example - Before/After:**

```
Query: "How do I center a div in CSS?"

BEFORE (Score: 0.38):
  "Use margin auto."

AFTER (Score: 0.93):
  "To center a div in CSS, use the `position` property
   with either `relative`, `absolute`, or `fixed`.
   Set its margins to auto:

   #myDiv {
       margin-left: auto;
       margin-right: auto;
   }

   This will center your div on both sides of the page."
```

**Enable:**

```bash
# Environment variable
export MIRRORMIND_ENABLED=true

# Or in .env file
MIRRORMIND_ENABLED=true
MIRRORMIND_THRESHOLD=0.75
MIRRORMIND_MAX_ITERATIONS=2
```

**Python API:**

```python
from apprentice_agent.tools.mirrormind import MirrorMind

mirror = MirrorMind(quality_threshold=0.75, max_iterations=2)

# Refine a weak response
result = mirror.refine("How do I X?", "Just do Y.")

print(f"Original: {result.original}")
print(f"Improved: {result.improved}")
print(f"Score: {result.quality_score}")
print(f"Iterations: {result.iterations}")
print(f"Was Improved: {result.was_improved()}")
```

**Note:** MirrorMind adds latency (1-2 LLM calls per iteration) and is disabled by default. Enable it for high-quality responses when latency is acceptable.

### CognitiveTheater (Multi-Perspective Reasoning)

CognitiveTheater analyzes questions from multiple perspectives in a SINGLE LLM call, then synthesizes a balanced recommendation. It automatically activates for decision-type questions.

**Flow:**
```
User Question → Single LLM Call → 4 Perspectives → Synthesis → Balanced Answer
```

**Four Perspectives:**

| Perspective | Role | Focus |
|-------------|------|-------|
| **Advocate** | Argues IN FAVOR | Benefits, opportunities, why it's a good idea |
| **Critic** | Argues AGAINST | Risks, flaws, what could go wrong |
| **Analyst** | Neutral analysis | Data, facts, evidence-based assessment |
| **Integrator** | Synthesizes all | Balanced recommendation considering all views |

**Auto-Detection Keywords:**

Questions containing these phrases automatically trigger CognitiveTheater:
- "should I", "should we"
- "compare", "pros and cons"
- "vs", "or" (in context of choices)
- "decide", "decision"
- "worth it", "good idea", "recommend"

**Example:**

```
User: Should I use microservices or monolith?

**Multi-Perspective Analysis**

**Pro:** Microservices scale teams independently, enable polyglot
persistence, and allow independent deployments...

**Con:** But you're a 2-person team - complexity will kill velocity.
You'll spend more time on infrastructure than features...

**Analysis:** Data shows startups under 10 engineers ship 3x faster
with monoliths. Netflix didn't start with microservices...

**Recommendation:** Start with a well-structured monolith. Extract
services only when you hit real scale problems or team boundaries.

Confidence: 85%
```

**Configuration:**

| Setting | Default | Description |
|---------|---------|-------------|
| `COGNITIVE_THEATER_ENABLED` | `true` | Enable multi-perspective reasoning |

**Python API:**

```python
from apprentice_agent.tools.cognitive_theater import CognitiveTheater

theater = CognitiveTheater()

# Full deliberation with structured output
result = theater.deliberate("Should I use PostgreSQL or MongoDB?")
print(result.perspectives["advocate"])
print(result.perspectives["critic"])
print(result.synthesis)
print(result.confidence)

# Quick formatted string for chat
output = theater.quick_debate("Should I learn Rust or Go?")
print(output)
```

**Disable:**

```bash
export COGNITIVE_THEATER_ENABLED=false
```

### Reflexion (Learn From Mistakes)

Reflexion enables Aura to learn from task failures by reflecting on what went wrong, storing lessons, and using past lessons to improve future attempts. It's automatically integrated with code execution.

**Flow:**
```
Task → Check Past Lessons → Attempt → Failed? → Reflect → Store Lesson → Retry
                                         ↓
                                    Success? → Done (with lessons used)
```

**Example - Code Execution:**

```
Original code (broken):
  def greet(name)      # Missing colon
      print(f"Hello!")

Attempt 1: Execute → SyntaxError: expected ':'
Reflect: "Missing colon after function definition. Always check syntax."
Store lesson.

Attempt 2: Fixed code with colon → SUCCESS
Output: "Hello!"
Reflexion metadata: {attempts: 2, learned: true, lessons_used: 1}
```

**Configuration:**

| Setting | Default | Description |
|---------|---------|-------------|
| `REFLEXION_ENABLED` | `true` | Enable learning from mistakes |
| `REFLEXION_MAX_ATTEMPTS` | `3` | Maximum retry attempts |

**How It Works:**

1. **Past Lessons**: Before attempting a task, Reflexion searches for relevant lessons using keyword matching
2. **Attempt with Context**: Generates attempt using past lessons as context
3. **Evaluate**: Uses an evaluator function to check success/failure
4. **Reflect on Failure**: If failed, generates a reflection on what went wrong
5. **Store Lesson**: Saves the lesson to `data/reflexion/memories.jsonl`
6. **Retry**: Next attempt includes the new lesson

**Memory Storage (JSONL):**

```json
{"task": "parse CSV", "attempt": "used read_csv", "outcome": "failure", "feedback": "encoding error", "reflection": "Always detect encoding first for CSVs", "timestamp": "2026-01-30T10:00:00"}
```

**Built-in Evaluators:**

| Evaluator | Checks |
|-----------|--------|
| `code_syntax_evaluator` | Valid Python syntax |
| `function_evaluator` | Has `def` + `return` statement |
| `json_evaluator` | Valid JSON output |
| `answer_completeness_evaluator` | Response length and completeness |

**Python API:**

```python
from apprentice_agent.tools.reflexion import ReflexionEngine, code_syntax_evaluator

engine = ReflexionEngine(model="llama3:8b", max_attempts=3)

# Custom evaluator
def my_evaluator(task, output):
    if "expected_output" in output:
        return True, "Success"
    return False, "Missing expected output"

result = engine.execute("Generate a greeting", my_evaluator)

print(f"Success: {result.success}")
print(f"Attempts: {result.attempts}")
print(f"Lessons used: {result.reflections_used}")
print(f"New lesson: {result.new_reflection}")
```

**Integration with Agent:**

Reflexion is automatically used when code execution fails:

```python
# In agent.py - automatic integration
result = agent._execute_code_with_reflexion(code, tool)

# Result includes reflexion metadata
{
    "success": True,
    "output": "15",
    "reflexion": {
        "attempts": 2,
        "lessons_used": ["Check variable initialization"],
        "learned": True
    }
}
```

**View Stored Lessons:**

```python
engine = ReflexionEngine()
print(engine.get_lessons_summary())
# Total lessons: 8
# Failures learned from: 6
# Successes recorded: 2
```

**Disable:**

```bash
export REFLEXION_ENABLED=false
```

### SynapseForge (Dynamic Tool Creation)

SynapseForge enables AURA to create NEW tools at runtime when capability gaps are detected. Tools are generated as Python code, validated in a sandbox, and stored for future use.

**Flow:**
```
Capability Gap Detected -> Generate Tool Code -> Validate in Sandbox -> Store Tool -> Use It
```

**Example:**
```
User: "Convert 100 celsius to fahrenheit"

AURA: "I don't have a temperature tool... let me create one."

SynapseForge:
|- Capability needed: "temperature conversion"
|- Checking registry... not found
|- Generating tool code...
|- Validation passed
|- Saved to tools/synthesized/celsius_to_fahrenheit.py
|- Executing: 100C = 212F

Next time: Tool already exists, just use it!
```

**Configuration:**

| Setting | Default | Description |
|---------|---------|-------------|
| `SYNAPSEFORGE_ENABLED` | `true` | Enable dynamic tool creation |

**How It Works:**

1. **Capability Detection**: When a tool isn't found, SynapseForge is triggered
2. **Registry Check**: Searches existing synthesized tools for a match
3. **Code Generation**: Uses LLM to generate Python code for the capability
4. **Sandbox Validation**: Runs code in subprocess with 10s timeout
5. **Auto-Fix**: If validation fails, attempts one fix via LLM
6. **Storage**: Saves valid tools to `tools/synthesized/` and registry

**Generated Tool Structure:**

```python
"""
Tool: celsius_to_fahrenheit
Description: Convert Celsius to Fahrenheit
Auto-generated by SynapseForge
"""

def execute(celsius: float) -> dict:
    try:
        result = (celsius * 9/5) + 32
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Safety Constraints:**

- Standard library only (no pip installs)
- Subprocess sandbox (isolated execution)
- 10 second timeout (kills runaway code)
- No file system write access
- No network access

**Python API:**

```python
from apprentice_agent.tools.synapseforge import SynapseForge

forge = SynapseForge()

# Synthesize a new tool
tool = forge.synthesize("convert miles to kilometers")

if tool and tool.test_passed:
    result = forge.execute_tool(tool.name, miles=10)
    print(result)  # {'success': True, 'result': 16.0934}

# Find existing tool
existing = forge.find_tool("temperature conversion")

# List all synthesized tools
for t in forge.list_tools():
    print(f"{t['name']}: {t['description']} (used {t['uses']}x)")
```

**Registry Storage:**

Tools are persisted to `data/synapseforge/registry.json`:

```json
{
  "celsius_to_fahrenheit": {
    "name": "celsius_to_fahrenheit",
    "description": "Convert Celsius to Fahrenheit",
    "code": "...",
    "created_at": "2026-01-30T12:00:00",
    "test_passed": true,
    "usage_count": 5
  }
}
```

**Disable:**

```bash
export SYNAPSEFORGE_ENABLED=false
```

### WorldSim (Consequence Simulation)

WorldSim simulates consequences BEFORE executing risky actions. Shows what could happen, whether it's reversible, and suggests safer alternatives.

**Flow:**
```
Action Detected -> Check Risk Level -> Simulate Consequences -> Show Preview -> Confirm
```

**Example:**
```
User: "Delete all files in /tmp/old_project"

WorldSim Simulation:
|- Action: rm -rf /tmp/old_project
|- Risk Level: CAUTION
|- Consequences:
|   - Files will be permanently deleted
|   - Cannot be undone (no trash)
|- Reversible: No
|- Safer Alternative: "Move to /tmp/trash first?"
|- Proceed? [Yes / Use Alternative / Cancel]
```

**Risk Levels:**

| Level | Icon | Behavior |
|-------|------|----------|
| `SAFE` | ✅ | No concerns, proceed |
| `CAUTION` | ⚠️ | Show warning, ask confirmation |
| `DANGEROUS` | 🚨 | Strong warning, suggest alternative |
| `BLOCKED` | 🚫 | Refuse to execute |

**Blocked Patterns (Instant Block):**

- `rm -rf /` - Delete root filesystem
- `rm -rf ~` - Delete home directory
- `:(){ :|:& };:` - Fork bomb
- `curl ... | bash` - Pipe untrusted script to shell
- `dd if=... of=/dev/sda` - Overwrite disk
- `mkfs.*` - Format filesystem

**Caution Patterns (Trigger LLM Analysis):**

- File deletion (`rm`, `del`)
- Elevated privileges (`sudo`)
- Permission changes (`chmod`, `chown`)
- Database modifications (`DROP`, `DELETE FROM`)
- Git destructive operations (`reset --hard`, `push --force`)
- Service control (`systemctl stop`)

**Python API:**

```python
from apprentice_agent.tools.worldsim import WorldSim, RiskLevel

sim = WorldSim()

# Simulate an action
result = sim.simulate("rm -rf /tmp/old_logs")

print(f"Risk: {result.risk_level.value}")       # caution
print(f"Consequences: {result.consequences}")   # ['Files will be deleted']
print(f"Reversible: {result.reversible}")       # False
print(f"Proceed: {result.should_proceed}")      # True (with caution)

# Check if blocked
if result.risk_level == RiskLevel.BLOCKED:
    print("Action blocked for safety!")

# Get formatted warning
warning = sim.format_warning(result)
print(warning)
```

**Integration with Agent:**

```python
# In agent.py - automatic safety check
warning = agent.check_action_safety("rm -rf /important")

if warning:
    print(warning)  # Shows WorldSim warning
    # Wait for user confirmation
else:
    # Safe to proceed
    execute_action()
```

**Disable:**

```bash
export WORLDSIM_ENABLED=false
```

### AURA v3.0 ALIVE (Emotionally Present AI)

AURA v3.0 (Autonomous Living Intelligence with Visible Emotion) transforms the agent from a reactive assistant into a **proactive, emotionally present thinking partner**.

**Architecture:**

```
User Input → AURA Engine → [Memory + Emotion + Patterns + Thinking] → Humanized Response
                ↓
        Proactive Heartbeat → Notifications
```

**8 Core Components:**

| Component | Description |
|-----------|-------------|
| **MarkdownStore** | Human-readable `.md` memory files (editable by user) |
| **EmotionalEngine** | Mood persistence across sessions (energy, warmth, engagement) |
| **HeartbeatMonitor** | Background proactive notifications and session awareness |
| **PatternProphet** | Cross-conversation pattern recognition (topic sequences, time patterns) |
| **VisibleThinking** | Shows internal reasoning with emoji prefixes |
| **ResponseHumanizer** | Transforms robotic text to natural speech |
| **SoulLoader** | Personality configuration via markdown soul files |
| **AURAEngine** | Main orchestrator integrating all components |

**Commands:**

| Command | Description |
|---------|-------------|
| `aura status` | Full system status (mood, energy, patterns, features) |
| `aura mood` | Current emotional state with details |
| `aura soul` | Active personality configuration |
| `aura memory` | Memory statistics by type |
| `aura insights` | Learned patterns about user behavior |
| `remember this: <fact>` | Store a fact in long-term memory |

**Example:**

```
User: "aura status"

AURA v3.0 ALIVE Status:
- Soul: AURA
- Mood: content (Balanced and comfortable)
- Energy: 85%
- Patterns: 12 learned
- Turns this session: 5
- Features: Proactive=True, Thinking=True, Humanize=True
```

**Soul Presets:**

| Soul | Personality | Use Case |
|------|-------------|----------|
| `SOUL_PERSONAL` | Warm, curious, subtly witty | Personal assistant |
| `SOUL_ENTERPRISE` | Professional, precise, efficient | Work environments |

Soul files are markdown in `aura/soul/` and can be customized:

```markdown
# My Custom Soul

## Personality
- Friendly and supportive
- Technical but accessible

## Voice
Speak naturally, like a knowledgeable colleague.

## Quirks
- Uses "Actually..." when gently correcting
- Shows enthusiasm for interesting problems
```

**Memory Files:**

AURA stores memory in human-readable markdown files at `aura/data/memory/`:

| File | Contents |
|------|----------|
| `user_profile.md` | User info, preferences, goals |
| `conversations.md` | Key conversation highlights |
| `learned_facts.md` | Things AURA has learned |
| `emotional_state.md` | Mood history and triggers |
| `patterns.md` | Recognized behavioral patterns |

**Visible Thinking:**

When enabled, AURA shows its thought process:

```
🤔 *Considering this...*

I'd be happy to help with that Python question...
```

**Proactive Notifications:**

AURA can initiate interactions:

```
---
* Good afternoon! I'm here whenever you need me.
* Remember to take breaks - you've been working for 2 hours.
```

**Configuration:**

```bash
# Enable/disable AURA
export AURA_ENABLED=true

# Choose soul preset
export AURA_SOUL=SOUL_PERSONAL  # or SOUL_ENTERPRISE

# Feature toggles
export AURA_PROACTIVE=true   # Background notifications
export AURA_THINKING=true    # Visible thinking prefixes
export AURA_HUMANIZE=true    # Response humanization
```

**Python API:**

```python
from aura.engine import create_aura

# Create AURA instance
aura = create_aura(soul="SOUL_PERSONAL")

# Process user input
context = aura.process_input("Hello! How are you?")
print(f"Detected topic: {context['topic']}")
print(f"Current mood: {context['mood']}")

# Humanize a response
response = aura.process_response("I am doing well.", context)
print(response.content)  # "Oh, I'm doing well! How can I help?"

# Remember something
aura.remember("User prefers dark mode", importance=0.8)

# Get greeting based on mood
print(aura.get_greeting())  # "Good afternoon! I'm here whenever you need me."

# Shutdown gracefully
aura.shutdown()
```

**Integration with Agent:**

AURA is automatically integrated into the agent's chat flow:

1. **Input Processing** - AURA analyzes message, updates emotional state, records patterns
2. **Context Enhancement** - Adds mood-appropriate tone to LLM prompts
3. **Response Humanization** - Transforms robotic responses to natural speech
4. **Proactive Messages** - Appends notifications when appropriate

**Disable:**

```bash
export AURA_ENABLED=false
```

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
| `MIRRORMIND_ENABLED` | `false` | Enable self-critique response improvement |
| `MIRRORMIND_THRESHOLD` | `0.75` | Minimum quality score to accept (0.0-1.0) |
| `MIRRORMIND_MAX_ITERATIONS` | `2` | Maximum improvement iterations |
| `COGNITIVE_THEATER_ENABLED` | `true` | Enable multi-perspective reasoning for decisions |
| `REFLEXION_ENABLED` | `true` | Enable learning from mistakes for code execution |
| `REFLEXION_MAX_ATTEMPTS` | `3` | Maximum retry attempts when learning |
| `SYNAPSEFORGE_ENABLED` | `true` | Enable dynamic tool creation at runtime |
| `WORLDSIM_ENABLED` | `true` | Enable consequence simulation for risky actions |

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
        ├── sesame_tts.py     # Sesame CSM 1B high-quality TTS
        ├── voice_manager.py  # Hybrid voice system (Sesame + PersonaPlex)
        ├── evoemo.py         # Emotional state detection and tracking
        ├── evoemo_prompts.py # Adaptive tone modifiers for emotions
        ├── inner_monologue.py # Real-time thought visualization and Think Aloud
        ├── knowledge_graph.py # Relationship-based memory with NetworkX
        ├── kg_extractor.py   # Knowledge extraction from text
        ├── hybrid_memory.py  # Combined vector + graph memory
        ├── metacog_guardian.py # Self-aware failure prediction system
        ├── neurodream.py     # Sleep/dream memory consolidation system
        ├── mirrormind.py     # Self-critique response improvement system
        ├── cognitive_theater.py # Multi-perspective reasoning system
        ├── reflexion.py      # Learn from mistakes system
        ├── synapseforge.py   # Dynamic tool creation system
        ├── worldsim.py       # Consequence simulation system
        ├── synthesized/      # Runtime-generated tools
        └── custom/           # Auto-generated custom tools
```

## License

MIT
