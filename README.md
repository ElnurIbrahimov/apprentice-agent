# Apprentice Agent

An AI agent with memory and reasoning capabilities, powered by local LLMs via Ollama.

## Features

- **Gradio GUI** - Modern web interface with real-time thinking process visualization
- **Observe-Plan-Act-Evaluate-Remember loop** - Structured reasoning cycle for achieving goals
- **Local LLM support** - Uses Ollama for privacy-friendly, offline inference
- **Long-term memory** - ChromaDB-powered memory system for learning from past experiences
- **Web search** - DuckDuckGo integration for gathering online information
- **Filesystem tools** - Read files and list directories on your local machine
- **Code execution** - Safely run Python code in a sandboxed environment
- **Summarization** - Automatically summarize gathered information

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally
- llama3:8b model (or another compatible model)

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

3. Pull the Ollama model:
```bash
ollama pull llama3:8b
```

4. (Optional) Copy and configure environment:
```bash
cp .env.example .env
```

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

## Configuration

Edit `apprentice_agent/config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `MODEL_NAME` | `llama3:8b` | LLM model to use |
| `CHROMADB_PATH` | `./data/chromadb` | Memory storage location |

## Available Tools

| Tool | Description | Example Action |
|------|-------------|----------------|
| `filesystem` | List/read local files | `list C:/Users/project` |
| `web_search` | Search the internet | `AI news 2024` |
| `code_executor` | Run Python code (sandboxed) | `import math; print(math.factorial(50))` |
| `summarize` | Summarize gathered info | `results` |

### Code Executor Safety

The code executor runs Python code in a sandboxed subprocess with:
- **Blocked imports**: `os`, `subprocess`, `sys`, `socket`, `requests`, etc.
- **No file access**: `open()`, `file()` are blocked
- **Timeout protection**: 30 second default limit
- **Isolated execution**: Runs in temp directory

## Architecture

```
apprentice-agent/
├── gui.py        # Gradio web interface
├── main.py       # CLI entry point
└── apprentice_agent/
    ├── agent.py      # Main agent loop (observe/plan/act/evaluate/remember)
    ├── brain.py      # OllamaBrain - LLM interface and reasoning
    ├── memory.py     # ChromaDB-powered long-term memory
    ├── config.py     # Configuration settings
    └── tools/
        ├── filesystem.py    # File operations
        ├── web_search.py    # DuckDuckGo search
        └── code_executor.py # Sandboxed Python execution
```

## License

MIT
