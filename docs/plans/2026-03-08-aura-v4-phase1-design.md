# AURA v4 Phase 1: Coding Agent Foundation

**Date:** 2026-03-08
**Status:** Implemented

## What Was Built

Phase 1 gives AURA the foundational tools that make Claude Code, Gemini CLI, and Codex effective as coding agents. The core insight: **coding agents are bottlenecked by search, not coding ability** — 60%+ of agent time is wasted finding code.

### New Tools

#### 1. CodeSearchTool (`aura/tools/code_search.py`)
- **grep** — Regex content search across files with context lines, file type filtering, output modes (content/files/count)
- **glob** — File pattern matching sorted by modification time
- **find_definition** — Language-aware definition finder (Python, JS/TS, Rust, Go) using regex patterns for classes, functions, interfaces, types, enums, structs, traits
- **find_references** — Word-boundary search for all references to a symbol
- **project_structure** — Tree-like directory overview with file/dir counts and language stats
- **detect_project_type** — Auto-detect project type, stack, frameworks, package manager from markers (package.json, requirements.txt, Cargo.toml, etc.)
- Performance: 330ms grep across 786 files, 440ms glob, 30ms structure

#### 2. CodeEditTool (`aura/tools/code_edit.py`)
- **read_file** — Line-numbered file reading with offset/limit (like `cat -n`)
- **edit** — Surgical find-replace with exact match primary, fuzzy fallback (85% threshold), unified diff output, automatic backup
- **create_file** — Create new files (fails if exists)
- **multi_edit** — Atomic batch edits on a single file (all-or-nothing)
- **rollback** — Undo last edit from .bak backup
- NOT sandboxed — operates on real project files for coding agent use
- Safety: blocks system directories and build/dependency directories

### Enhanced Tools

#### 3. ShellExecutorTool enhancements
- **run_streaming** — New method with real-time line-by-line output via callback
- **Expanded allowed commands** — Added python, pip, node, npm, npx, yarn, pnpm, bun, deno, tsc, eslint, prettier, vitest, jest, pytest, rg, fd, ruff, mypy, black, isort, curl, wget

#### 4. Project Context enhancements
- **detect_and_load_context** — Combines AURA.md loading with auto-detection for projects without AURA.md

### New CLI Commands

| Command | Description |
|---------|-------------|
| `/grep <pattern> [path] [--type py] [-i] [-C n]` | Search code content |
| `/search <glob-pattern>` | Find files by pattern |
| `/find def <name>` | Find class/function definitions |
| `/find ref <name>` | Find all references to a symbol |
| `/search structure [path]` | Show project tree |
| `/edit <path> [offset] [limit]` | Read file with line numbers |
| `/project info` | Auto-detect project type and stack |
| `/project init` | Create AURA.md template |
| `/project context` | Show current AURA.md content |
| `/shell <command>` | Run shell command with streaming output |
| `/bash <command>` | Alias for /shell |
| `/run <command>` | Alias for /shell |

### Agent Integration
- Both tools registered in agent's tool dict (loaded on init, not lazy)
- Tool keywords added for automatic routing via keyword detection
- Dispatch logic in `_parse_and_execute_tool_action` for code_search and code_edit
- New `_extract_symbol_name` helper for parsing symbol names from natural language

## Architecture Decisions

1. **Pure Python, no dependencies** — CodeSearchTool and CodeEditTool use only stdlib (re, os, pathlib, fnmatch, difflib, shutil). No ripgrep binary, no tree-sitter required.
2. **CodeEditTool is NOT sandboxed** — deliberate choice. A coding agent needs to edit actual project files. Safety comes from blocking system dirs and build dirs, not from sandboxing.
3. **Fuzzy edit fallback at 85%** — matches Claude Code's approach. If exact match fails, try difflib SequenceMatcher. Below 85% similarity, show the closest match to help the user.
4. **Streaming shell** — uses line-buffered Popen with threading for stderr, callback-based output. No asyncio needed.

## Files Changed
- **New:** `aura/tools/code_search.py` (693 lines)
- **New:** `aura/tools/code_edit.py` (340 lines)
- **Modified:** `aura/tools/shell_executor.py` (added streaming + dev commands)
- **Modified:** `aura/tools/project_context.py` (added detect_and_load_context)
- **Modified:** `aura/tools/__init__.py` (added imports + exports)
- **Modified:** `aura/agent.py` (tool loading, keywords, dispatch, symbol extractor)
- **Modified:** `main.py` (6 new CLI commands)
