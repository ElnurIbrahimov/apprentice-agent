"""Ollama tool calling JSON schemas for Aura's agentic dev CLI.

Defines AGENTIC_TOOLS — the list of tool schemas passed to Ollama's
client.chat(tools=...) parameter for structured tool calling.

Each tool maps to an existing Aura tool class:
  read_file      -> CodeEditTool / FileSystemTool.read_file
  grep           -> CodeSearchTool.grep
  glob           -> CodeSearchTool.glob
  list_dir       -> FileSystemTool.list_directory
  edit_file      -> FileSystemTool.apply_search_replace
  write_file     -> FileSystemTool.write_file
  shell          -> ShellExecutorTool.run
  git            -> GitTool (status/diff/add/commit/log/push)
  search_web     -> BraveSearchTool.run / TavilyTool.search
  fetch_url      -> requests.get + HTML strip (read full web pages)
  project_structure -> CodeSearchTool.project_structure
  create_directory -> os.makedirs
  move_file      -> shutil.move
  run_tests      -> auto-detect pytest/jest/vitest/cargo/go
"""

AGENTIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents. Returns content with line numbers. Use offset/limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (absolute or relative to project root)",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Start line (1-based). Omit to start from beginning.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to read. Omit or 0 to read all.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents using regex pattern. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: project root)",
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Filter by file type: py, js, ts, rust, go, java, etc.",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern. Returns file paths sorted by modification time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts', '*.json')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: project root)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories in a directory. Shows file sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: project root)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing an exact string with a new string. The old_string must match exactly (including whitespace/indentation). Read the file first to get the exact content. Example: {\"path\": \"src/app.py\", \"old_string\": \"def hello():\", \"new_string\": \"def hello(name):\"}",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find and replace (must be unique in the file)",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or overwrite an existing file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a shell command and return stdout/stderr/exit_code. Use for running tests, installing packages, building, or any system command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (default: project root)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60, max: 300)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Run git operations: status, diff, log, add, commit, push, pull, branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Git action: status, diff, log, add, commit, push, pull, branch",
                        "enum": ["status", "diff", "log", "add", "commit", "push", "pull", "branch"],
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message (for 'commit' action)",
                    },
                    "files": {
                        "type": "string",
                        "description": "Files to add (for 'add' action, default: '.')",
                    },
                    "file": {
                        "type": "string",
                        "description": "Specific file for diff",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of log entries (for 'log' action, default: 5)",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information. Returns titles, URLs, and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 8)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_structure",
            "description": "Get a tree-like overview of the project directory structure with file counts and detected languages/frameworks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Project root directory (default: current directory)",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum directory depth to show (default: 3)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agent",
            "description": "Spawn a sub-agent for a parallel task. Use for: researching docs, reading multiple files, independent analysis. Only 1 'coder' agent can write files at a time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What the sub-agent should do",
                    },
                    "role": {
                        "type": "string",
                        "enum": ["reader", "researcher", "coder"],
                        "description": "reader=read-only (default), researcher=read+web, coder=full access (max 1 concurrent)",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page URL and return its text content (HTML stripped to readable text). Use after search_web to read full page content. Example: {\"url\": \"https://docs.python.org/3/library/json.html\"}",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory (and parent directories if needed). Example: {\"path\": \"src/components\"}",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to create"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file. Example: {\"source\": \"old_name.py\", \"destination\": \"new_name.py\"}",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Current file path"},
                    "destination": {"type": "string", "description": "New file path"},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run project tests. Auto-detects test framework (pytest, jest, vitest, cargo test, go test). Example: {\"target\": \"tests/test_auth.py\"}",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Specific test file or directory (optional — omit to run all)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": "Apply multiple edits to a single file in one atomic operation. All edits succeed or none do. Example: {\"path\": \"src/app.py\", \"edits\": [{\"old_string\": \"import os\", \"new_string\": \"import os\\nimport sys\"}, {\"old_string\": \"def main():\", \"new_string\": \"def main(args):\"}]}",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                            },
                            "required": ["old_string", "new_string"],
                        },
                        "description": "List of {old_string, new_string} pairs to apply in order",
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_observation",
            "description": "Retrieve the full text of a previously-masked tool output by its observation ID. Use this when you see a placeholder like ⟦OBS:abc123...⟧ in prior tool results and need the complete content. Example: {\"obs_id\": \"a3f9c2d014\"}",
            "parameters": {
                "type": "object",
                "properties": {
                    "obs_id": {"type": "string", "description": "Observation ID from the ⟦OBS:...⟧ placeholder"},
                },
                "required": ["obs_id"],
            },
        },
    },
]

# Quick lookup: tool name -> schema
TOOL_SCHEMA_MAP = {t["function"]["name"]: t for t in AGENTIC_TOOLS}

# Tool names that are read-only (safe to auto-approve)
READ_ONLY_TOOLS = frozenset({
    "read_file", "grep", "glob", "list_dir",
    "search_web", "project_structure", "fetch_url",
    "expand_observation",
})

# Tool names that mutate state (need approval in non-trust mode)
MUTATING_TOOLS = frozenset({
    "edit_file", "write_file", "shell",
    "create_directory", "move_file", "multi_edit",
})
