# State-of-the-Art: General-Purpose AI Assistant CLIs & Terminal UX

**Date:** 2026-03-17
**Focus:** What makes the best CLI tools world-class for non-coding tasks (research, writing, brainstorming, analysis)

---

## Tool-by-Tool Analysis

### 1. Simon Willison's `llm` CLI

**The gold standard for Unix-philosophy AI tooling.**

GitHub: https://github.com/simonw/llm
Docs: https://llm.datasette.io

**Long-form content handling:**
- Outputs raw text to stdout by default -- no fancy rendering, which makes it perfectly composable
- Fragments (`-f/--fragment`) let you load long documents, URLs, or file paths as context
- Multiple `-f` flags concatenate fragments in order, with the prompt appended at the end
- No built-in pagination or markdown rendering -- relies on piping to `glow`, `bat`, `less`, etc.

**Research workflows:**
- No built-in web search, but tool-calling support (v0.26+) lets you define Python functions as tools the model can invoke
- Fragments can load URLs directly (`-f https://...`), pulling web content into context
- Every prompt/response is logged to SQLite automatically -- you build a searchable research database just by using the tool

**Conversation context:**
- `-c` flag continues the last conversation
- Named conversations stored in SQLite
- `llm logs` to browse history; `llm logs -q "search term"` to search
- Conversations are stored with full metadata and can be explored via Datasette (web UI for SQLite)

**Output formats:**
- Raw text to stdout (default)
- `--json` for JSON output
- Schemas (`--schema 'name, bio, age int'`) force structured JSON output conforming to a spec
- `llm logs --schema X` filters logged responses by schema, outputs newline-delimited JSON or `--data-array` for JSON arrays
- Templates with schemas can be saved: `llm --schema X --save template-name`

**Multi-step workflows:**
- Full Unix piping: `cat document.pdf | llm "summarize this"`
- Chain commands: `cat paper.txt | llm "extract key claims" | llm "rate each claim for evidence quality"`
- Fragments + piping + templates = composable multi-step pipelines
- Tool-calling (v0.26+) enables agentic loops where the model calls Python functions

**Brainstorming/creative:**
- System prompts (`-s "You are a creative strategist"`) set persona
- Templates save reusable prompt+system-prompt combos: `llm -t brainstorm "topic"`
- No special creative features -- relies on prompt engineering + model capability

**File I/O:**
- Multimodal: `-a` flag attaches images, audio, video files or URLs
- `llm 'describe this image' -a photo.jpg`
- Fragments load text files, URLs
- Output goes to stdout -- redirect to files with `>`

**Memory/personalization:**
- SQLite log IS the memory -- every interaction is permanent and searchable
- Templates serve as personalization (saved prompts, system prompts, schemas)
- Plugin system for extending with custom models, tools, behaviors
- No cross-session "memory" in the ChatGPT sense -- but the SQLite log is arguably better for power users

**Key insight:** `llm` treats AI as a Unix text filter. It doesn't try to be a chatbot -- it's a composable building block. The SQLite logging + Datasette exploration is uniquely powerful for researchers who want to mine their own interaction history.

---

### 2. Fabric (Daniel Miessler)

**The "pattern library" approach -- pre-built prompt templates for specific tasks.**

GitHub: https://github.com/danielmiessler/fabric

**Long-form content handling:**
- Patterns are Markdown-formatted prompt templates designed for readability
- `--stream` flag for streaming output
- YouTube integration: `yt --transcript "URL" | fabric -sp summarize` extracts and processes video transcripts
- Timestamps preserved with `--transcript-with-timestamps`
- Output is plain text to stdout

**Research workflows:**
- `extract_wisdom` pattern: produces structured output with Summary, Ideas, Insights, Quotes, Habits, Facts, References, One-Sentence Takeaway, Recommendations
- `analyze_claims` pattern: rates truth claims with evidence and fallacy detection
- `analyze_debate` pattern: structured debate analysis
- YouTube transcript processing is a killer feature for research
- `yt` helper tool uses yt-dlp under the hood

**Conversation context:**
- Fabric is primarily stateless and one-shot by design
- Each pattern invocation is independent
- Context comes from piping input, not from conversation history
- This is intentional -- patterns are meant to be composable, not conversational

**Output formats:**
- Plain text/Markdown to stdout
- Patterns define their own output structure (e.g., extract_wisdom has specific sections)
- `create_markmap` generates mind map visualizations (Vis.js)
- `create_conceptmap` generates visual knowledge representations
- No built-in JSON/CSV export -- but output structure is consistent per pattern

**Multi-step workflows:**
- Full Unix piping is the core design:
  ```
  wget -qO - https://example.com/article | pandoc -f html -t plain | fabric --stream --pattern summarize
  ```
- Pattern chaining: output of one pattern feeds into the next
  ```
  yt --transcript "URL" | fabric -sp extract_wisdom | fabric -sp write_essay
  ```
- This is Fabric's superpower -- it turns AI into composable Unix tools

**Brainstorming/creative:**
- `write_essay` -- writes Paul Graham-style essays
- `write_micro_essay` -- concise essay generation
- `create_conceptmap` -- visual knowledge mapping
- `improve_writing` -- editorial feedback
- Patterns encode specific creative methodologies

**File I/O:**
- Primarily text/stdin-based
- YouTube transcript extraction via `yt` tool
- No native PDF/image processing -- relies on external tools (pandoc, etc.) piped in
- Web content via wget/curl piped in

**Memory/personalization:**
- Custom patterns stored in `~/.config/fabric/patterns/`
- Users create their own patterns alongside the 200+ community patterns
- Model selection saved in config
- i18n support (10 languages as of late 2025)
- No conversation memory -- intentionally stateless

**Key insight:** Fabric's genius is the pattern library concept. Instead of remembering how to prompt for "extract key insights from a podcast," you just use `extract_wisdom`. The 200+ community-curated patterns encode collective prompt engineering knowledge. The Unix-pipe composability makes it a force multiplier.

---

### 3. Open Interpreter

**The "natural language computer control" approach -- code execution as the interface.**

GitHub: https://github.com/openinterpreter/open-interpreter

**Long-form content handling:**
- Runs actual code (Python, JS, Bash) to process content
- Can generate, edit, and manipulate PDFs, documents, spreadsheets
- No special terminal rendering -- outputs via code execution results

**Research workflows:**
- Browser control: can navigate web, scrape data, conduct research via automated browsing
- Data analysis: runs pandas, matplotlib, etc. locally for data processing
- Can search the web, download papers, extract text from PDFs -- all through code
- Computer API: controls mouse, keyboard, screen for GUI automation

**Conversation context:**
- Full conversation history maintained within a session
- Messages include code execution results
- OS mode (`interpreter --os`) maintains state of computer interactions
- No cross-session persistence by default

**Output formats:**
- Whatever code can produce: charts, files, JSON, CSV, images, PDFs
- Matplotlib visualizations rendered inline
- Code execution results displayed in terminal
- Can write output to any file format

**Multi-step workflows:**
- The model plans and executes multi-step code sequences
- Each step's output informs the next
- Can create files, modify them, run analysis, generate reports -- all in one conversation
- Human-in-the-loop confirmation before code execution (configurable)

**Brainstorming/creative:**
- Can generate visual content (images via code)
- Can create presentations, documents
- More of a "do things" tool than a "think with me" tool
- Not really optimized for back-and-forth ideation

**File I/O:**
- Strongest file handling of any CLI tool
- Vision: analyzes images via Moondream (local) or GPT-4o
- PDF generation and editing
- Any file type the underlying code can handle
- OS mode can interact with any application on the system

**Memory/personalization:**
- Custom system prompts
- Model selection
- Profile configurations
- No persistent memory across sessions

**Key insight:** Open Interpreter is the most capable tool for tasks that require *doing* things (manipulating files, running analysis, automating workflows), but it's not great for pure thinking/brainstorming. It's an execution engine, not a thinking partner.

---

### 4. Shell-GPT (sgpt)

**The "shell-integrated assistant" -- focused on terminal productivity.**

GitHub: https://github.com/TheR1D/shell_gpt

**Long-form content handling:**
- Streams responses to terminal
- No special markdown rendering in default mode
- `--code` mode outputs clean code to stdout (no markdown wrapping)

**Research workflows:**
- General query mode for information retrieval
- No web search integration
- No citation or source tracking
- Primarily a "ask questions, get answers" tool

**Conversation context:**
- `--chat` flag with named sessions: `sgpt --chat research "What is X?"`
- Chat history preserved per named session
- Chat cache persistence across sessions
- Can switch between named conversations

**Output formats:**
- Plain text to stdout (default)
- `--code` mode: clean code output
- `--shell` mode: OS-aware shell commands
- No structured output (JSON/CSV) support
- Output goes to stdout for redirection

**Multi-step workflows:**
- Piping supported: `cat file.txt | sgpt "summarize"`
- Named chat sessions allow multi-turn workflows
- Functions system: custom Python functions in `~/.config/shell_gpt/functions/`
- Limited compared to llm or Fabric for composition

**Brainstorming/creative:**
- Roles system: `--create-role "creative_writer"` defines personas
- `--list-roles` to see available roles
- Roles persist and can be reused
- No special creative workflow features

**File I/O:**
- Piping stdin for text input
- No native multimodal support
- No PDF/image handling
- Output to stdout (redirect to files)

**Memory/personalization:**
- Named chat sessions persist
- Custom roles for different interaction styles
- Custom functions for tool use
- Configuration via config file
- LiteLLM backend supports many model providers

**Key insight:** Shell-GPT is optimized for quick terminal tasks -- generating shell commands, answering quick questions, writing code snippets. It's the most lightweight and "just works" option, but the least capable for complex research or creative workflows.

---

### 5. Ollama CLI

**The local-first model runner with a clean interactive chat.**

Docs: https://docs.ollama.com/cli

**Long-form content handling:**
- Interactive chat mode with streaming
- `/set parameter num_ctx 4096` to control context window (adjustable per session)
- Command history saved to `~/.ollama/history`
- No built-in markdown rendering or pagination

**Research workflows:**
- No web search or tool-calling in base CLI
- MCP client integrations available (e.g., mcp-client-for-ollama)
- Secure Minions (with Stanford): local models work with cloud models while keeping data encrypted
- Thinking Mode: shows model reasoning steps (for reasoning models)

**Conversation context:**
- Interactive mode maintains full conversation within session
- No persistent cross-session conversations in base CLI
- Third-party GUIs (Cortex, Askimo) add conversation forking, history, favorites
- Desktop app (July 2025) adds chat history

**Output formats:**
- Plain text streaming in CLI
- API returns JSON
- No structured output in CLI mode
- Third-party tools add Markdown export

**Multi-step workflows:**
- Modelfiles define reusable configurations (model + temperature + system prompt)
- No piping/composition in interactive mode
- API mode supports integration into pipelines
- Best used as a model server that other tools (llm, Fabric) call

**File I/O:**
- Multimodal support: drag-and-drop images/PDFs in desktop app
- Vision models supported
- CLI: limited to text input/output
- API: supports image inputs for vision models

**Memory/personalization:**
- Modelfiles: package model + prompt template + parameters as reusable configs
- No built-in memory system
- Custom system prompts per Modelfile
- Third-party MCP clients add preferences, saved prompts

**Key insight:** Ollama is primarily a model server, not an interaction tool. Its CLI is functional but minimal. The real value is as infrastructure that other tools build on. The Modelfile concept (packaging model + system prompt + parameters) is a good pattern.

---

### 6. LM Studio CLI (`lms`)

**Desktop-first local model manager with CLI for automation.**

Docs: https://lmstudio.ai/docs/cli

**Long-form content handling:**
- `lms chat` opens interactive session
- Context length configurable at model load time
- Log streaming: `lms log stream --source model --filter output`

**Research workflows:**
- No built-in research features
- OpenAI-compatible API enables integration with other tools
- Structured output via JSON schemas in API calls
- Tool use support in API

**Conversation context:**
- Interactive chat in CLI
- No persistent conversation history in CLI
- Desktop app manages conversation state

**Output formats:**
- Structured Output: API enforces JSON schemas on responses
- OpenAI-compatible endpoints for chat completions
- Tool use follows OpenAI function calling format
- CLI output is plain text

**Multi-step workflows:**
- CLI designed for scripting: `lms daemon up && lms get model-name && lms server start`
- API enables pipeline integration
- Not designed for interactive composition

**File I/O:**
- Model management (download, load, unload)
- No direct file processing in CLI
- API supports what the loaded model supports

**Memory/personalization:**
- Model configs saved
- GPU offload settings
- No conversation memory or personalization

**Key insight:** LM Studio CLI is a model management tool, not an interaction tool. Use it to set up local models, then interact through other tools or the API.

---

### 7. ChatGPT CLI (kardolus/chatgpt-cli)

**The most feature-complete ChatGPT-specific CLI.**

GitHub: https://github.com/kardolus/chatgpt-cli

**Long-form content handling:**
- Streaming mode for real-time output
- Interactive mode for conversational use
- Query mode for single-shot interactions

**Research workflows:**
- MCP (Model Context Protocol) support: call MCP tools, inject results as context
- Stateful MCP sessions with auto-renewal
- Can integrate with any MCP server (web search, databases, etc.)
- Agent mode: multi-step automation with safety and budget controls

**Conversation context:**
- Thread-based: each thread has its own history
- Individualized context per thread
- Seamless conversation continuation

**Output formats:**
- Text streaming to terminal
- No structured output beyond what the model returns
- Agent mode logs to timestamped directories

**Multi-step workflows:**
- Agent mode: model plans and executes multi-step tasks
- MCP tool chaining within conversations
- Budget controls for agent runs
- Each agent run gets logged for inspection

**File I/O:**
- `--image` flag: upload images or URLs
- `--draw` + `--output`: generate images from prompts
- `--audio` flag: upload audio for analysis
- Image editing via prompts
- Multi-provider (OpenAI, Azure, Perplexity, LLaMA)

**Memory/personalization:**
- Thread-based history
- Custom system prompts
- Model/provider configuration
- Prompt files for reusable inputs

**Key insight:** This is the closest to "ChatGPT in your terminal" with full multimodal support and MCP integration. The agent mode + MCP combination makes it extensible for research workflows. Thread-based conversation management is solid.

---

### 8. OpenAI Codex CLI

**OpenAI's official coding agent, but with patterns relevant to any CLI.**

Docs: https://developers.openai.com/codex/cli

**Long-form content handling:**
- Sandboxed execution environment
- Full-resolution image inspection
- Multimodal output from custom tools (not just text)

**Research workflows:**
- AGENTS.md file for repo/project-specific instructions
- MCP server integration for third-party tools and context
- Can be run AS an MCP server for orchestration
- Spawned subagents inherit sandbox and network rules

**Conversation context:**
- Session-based with full context
- Project-profile layering
- Persisted host approvals

**Multi-step workflows:**
- Agent loop with planning, execution, verification
- Filesystem RPCs for file operations
- Python SDK for programmatic integration
- Can orchestrate via Agents SDK

**Key insight:** Codex CLI's AGENTS.md pattern (project-specific instructions file) and its approach to sandboxed execution with configurable permissions are important UX patterns for any AI CLI.

---

### 9. AI Research Tools (Elicit, Consensus)

**Not CLIs, but their patterns are instructive for CLI research workflows.**

#### Elicit (https://elicit.com)
- API launched March 2026: search 138M+ papers programmatically
- Sentence-level citations: every claim linked to exact source sentence
- Research Agents (Dec 2025): broad topic exploration, competitive landscapes
- Keyword search across Elicit, PubMed, ClinicalTrials.gov
- Strict Screening criteria for systematic reviews
- Reports up to 80 papers
- 94-99% accuracy in data extraction
- Claude Opus 4.5 integration (Dec 2025)

#### Consensus (https://consensus.app)
- Built on GPT-5 + Responses API with multi-agent workflow:
  - Planning Agent: breaks down questions
  - Search Agent: searches 200M+ papers + citation graphs
  - Reading Agent: interprets papers
  - Analysis Agent: synthesizes results
- Consensus Meter: visualizes literature agreement (yes/no/mixed/possibly)
- Auto-citations via Zotero, Mendeley, EndNote, RefWorks
- "Research context pack": papers + metadata + key findings bundle

**Key insight for CLI design:** The multi-agent decomposition (plan -> search -> read -> synthesize) and citation tracking patterns from these tools should inform how any CLI handles research workflows. The "research context pack" concept from Consensus -- bundling papers + metadata + findings -- is a pattern worth stealing.

---

## Cross-Cutting Patterns & Insights

### What Makes a CLI Feel Like a "Thinking Partner" vs Just a Chatbot

1. **Persistence of thought.** Tools that log everything (llm's SQLite) let you return to ideas. A thinking partner remembers. A chatbot forgets.

2. **Composability over conversation.** The best "thinking" happens when you can chain operations: extract -> analyze -> synthesize -> write. Fabric and llm excel here. Chatbot-style tools (Shell-GPT) feel more transactional.

3. **Structured output.** When a tool can produce consistent, parseable output (llm schemas, Fabric patterns), it enables building on previous results. Random prose is harder to build on.

4. **Context from files, not just chat.** Loading documents, URLs, transcripts as context (llm fragments, Fabric's yt tool) makes the tool a research partner. Pure chat history is limiting.

5. **Named sessions/threads.** Being able to say "continue the research-on-X conversation" (Shell-GPT named chats, chatgpt-cli threads) enables topic-focused thinking across time.

6. **Reusable prompting patterns.** Templates (llm), Patterns (Fabric), Roles (Shell-GPT) let you encode your thinking methodology, not just your questions.

### How the Best Tools Handle Context Switching

| Tool | Approach |
|------|----------|
| llm | Named conversations in SQLite. `-c` continues last, or specify by ID. Search across all. |
| Shell-GPT | Named chat sessions (`--chat topic_name`). Switch by name. |
| chatgpt-cli | Thread-based. Each thread has isolated context. |
| Fabric | Stateless. No context switching needed -- each invocation is independent. |
| Open Interpreter | Session-based only. No switching. |

**Best pattern:** llm's approach -- everything logged to SQLite, searchable, browsable, with named conversations you can continue. The database IS the context management.

### Output Formatting That Works in Terminals

1. **Streaming is mandatory.** Every good tool streams tokens as they arrive. Waiting for complete responses feels broken.

2. **Rich/Glow for markdown.** The best approach: output raw markdown, let users pipe to `glow` or `bat` for rendering. Don't bake in rendering -- it breaks piping.

3. **Structured sections.** Fabric's pattern output (## Summary, ## Key Ideas, ## Quotes) is scannable. Unstructured prose walls are hard to read in terminals.

4. **Color sparingly.** Syntax highlighting for code blocks, dim colors for metadata, bold for headers. Too much color is worse than none.

5. **Width-aware.** Wrapping at terminal width. Tables that don't overflow. This is where many tools fail.

### How Power Users Customize and Extend

| Mechanism | Tool | Power |
|-----------|------|-------|
| **Plugins** | llm | Highest. Python packages that add models, tools, behaviors. 100+ plugins. |
| **Patterns** | Fabric | High. Markdown prompt templates. 200+ community patterns. Easy to create. |
| **Roles** | Shell-GPT | Medium. Named system prompts. |
| **Functions** | Shell-GPT, llm | High. Python functions the model can call. |
| **Modelfiles** | Ollama | Medium. Package model + prompt + params. |
| **Templates** | llm | High. Saved prompt + system prompt + schema combos. |
| **MCP Servers** | chatgpt-cli, Codex | Highest. External tool integration via protocol. |
| **Custom patterns** | Fabric | High. Drop a markdown file in ~/.config/fabric/patterns/. |

### The Missing Pieces (Gaps in Current Tools)

1. **No CLI tool does real citation tracking.** Elicit and Consensus track citations, but no terminal tool does. Every CLI treats sources as text, not structured references.

2. **No tool bridges research and writing.** You can extract wisdom (Fabric), you can write essays (Fabric), but there's no workflow that maintains a bibliography, tracks which claims come from which sources, and lets you compose a cited document.

3. **Cross-session memory is primitive.** llm's SQLite log is the best, but it's search-based, not semantic. No CLI tool has ChatGPT-style "learned preferences" or Mem0-style persistent memory.

4. **Multi-document analysis is clunky.** Loading 10 papers and asking cross-cutting questions requires manually constructing huge prompts. No CLI tool has RAG built in.

5. **No brainstorming-specific UX.** All tools treat brainstorming like any other prompt. None offer: idea branching, constraint variation, forced connections, SCAMPER frameworks, or other structured ideation techniques as first-class features.

6. **Visual output in terminals is unsolved.** Fabric's create_markmap generates HTML. Matplotlib needs a display. Terminal-native visualization (beyond ASCII) is basically nonexistent.

7. **No collaborative context.** All tools are single-user. No tool helps you share research context, brainstorming sessions, or analysis results with another person.

---

## Ranking: Best-in-Class for Non-Coding Tasks

| Capability | Best Tool | Why |
|-----------|-----------|-----|
| **Research workflows** | Fabric + llm (piped) | Fabric's extract_wisdom + llm's logging + piping |
| **Writing assistance** | Fabric | write_essay, improve_writing patterns |
| **Brainstorming** | llm (with templates) | Reusable persona templates + conversation continuity |
| **Data extraction** | llm (schemas) | Structured JSON output from any content |
| **File analysis** | Open Interpreter | Can run code to process any file type |
| **Unix composability** | Fabric, then llm | Both are pipe-native; Fabric's patterns are more task-specific |
| **Conversation memory** | llm | SQLite logging + Datasette exploration |
| **Extensibility** | llm | Plugin system + tool-calling + schemas |
| **Multimodal** | chatgpt-cli (kardolus) | Image + audio + generation + MCP |
| **Ease of use** | Shell-GPT | Simplest setup and usage |
| **Local/private** | Ollama + llm | llm with ollama plugin runs everything locally |

---

## Design Principles for a World-Class Non-Coding AI CLI

Based on this research, the patterns that matter most:

1. **Be a Unix citizen first.** Stdin/stdout piping, composability with other tools, raw text output by default. (llm, Fabric)

2. **Log everything to a queryable store.** SQLite or similar. Make the interaction history a first-class, searchable, exportable artifact. (llm)

3. **Encode expertise in reusable patterns.** Named templates/patterns that capture not just prompts but entire methodologies. (Fabric, llm templates)

4. **Support structured output natively.** JSON schemas, consistent section formatting, parseable output. This is what enables multi-step workflows. (llm schemas)

5. **Make context loading effortless.** URLs, files, fragments, transcripts -- the tool should pull in context from anywhere without friction. (llm fragments, Fabric yt)

6. **Separate interaction modes.** One-shot (piped), interactive (chat), and agent (multi-step autonomous) modes serve different needs. (chatgpt-cli)

7. **Conversation persistence with search.** Named sessions, full-text search, ability to resume any past conversation. (llm, Shell-GPT)

8. **Extensible via plugins/patterns, not code.** Let non-developers create new capabilities through prompt templates, not programming. (Fabric patterns)

9. **Multimodal input is table stakes.** Images, audio, PDFs, URLs should all be first-class inputs. (llm, chatgpt-cli, Open Interpreter)

10. **The "research context pack" pattern.** Bundle sources + metadata + extracted findings + citations as a structured artifact, not just prose. (Inspired by Consensus/Elicit, not yet in any CLI tool)
