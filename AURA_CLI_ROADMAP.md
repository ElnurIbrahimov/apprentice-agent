# AURA CLI Roadmap — World-Class Terminal Agent
**Created:** 2026-03-17
**Goal:** Make Aura's CLI the best AI terminal agent across all three use cases: coding, general AI assistant, and autonomous agent orchestration.

---

## Current State

Aura CLI has a solid foundation:
- Rich terminal UI (colors, panels, status bar, banner)
- 38 slash commands, interactive model picker (Alt+M)
- Streaming response rendering with syntax highlighting
- Session persistence with full message + tool call history
- Multi-mode (chat, one-shot, voice, non-interactive, dream)
- Permission system, agentic loop with 6-model fallback chains
- Emotional modulation, memory system, proactive awareness

**But compared to Claude Code, Aider, Codex CLI, and Copilot CLI — Aura is missing critical UX patterns that have become table stakes in 2026.**

---

## Competitive Landscape (March 2026)

| Feature | Claude Code | Aider | Codex CLI | Copilot CLI | **Aura** |
|---------|------------|-------|-----------|-------------|----------|
| Streaming tokens | Token-by-token | Token-by-token | Token-by-token | Token-by-token | Word chunks |
| Inline diffs | Syntax-highlighted | Color-coded | Theme-aware | Intra-line | None |
| Checkpoint/rewind | Esc Esc | Git commits | Sandbox | Esc Esc | None |
| Plan mode | /plan (read-only) | /architect | — | Shift+Tab | /plan (basic) |
| Permission tiers | 5 modes | Auto-commit | 3 modes | 3 modes | Trust toggle |
| Context visibility | /context fuel gauge | /tokens | /clear | Auto-compress | None |
| Command palette | / + fuzzy | / prefix | / + theme | / prefix | / prefix |
| Keyboard shortcuts | 6+ customizable | Minimal | 4+ | 4+ | Alt+M only |
| Non-interactive/pipe | Yes | Yes | codex exec | & delegation | -p flag |
| Parallel agents | Background + Tasks | — | App threads | /fleet | — |
| Cross-session memory | CLAUDE.md | — | Profiles | Repository memory | Memory system |
| Git integration | Deep (commits, PRs) | Git-native | Git-aware | Native GitHub MCP | /git, /commit |
| Mid-turn steering | Interrupt + redirect | Mode switch | Enter to inject | — | — |
| Hooks/plugins | PreToolUse/PostToolUse | — | Project profiles | /plugin | /hook (basic) |
| Themes | — | — | /theme + .tmTheme | — | — |
| Unix composability | Yes | Yes | stderr/stdout split | Yes | -p flag only |

---

## The Roadmap

### Tier 1 — Table Stakes (Must-Have, High Impact)
*Features every serious AI CLI has in 2026. Without these, Aura feels a generation behind.*

#### 1.1 Inline Diff Viewer
**What:** Show syntax-highlighted diffs when Aura edits files, not just "file edited" messages.
**Why:** Every competitor shows diffs inline. Users need to see what changed before/after.
**Inspiration:** Claude Code (syntax-highlighted), Aider (color-coded unified diff), Codex CLI (theme-aware with .tmTheme support).
**Scope:**
- Unified diff format with +/- color coding (green/red)
- Syntax highlighting within diff hunks (language-aware)
- Show diff automatically after every file edit
- `/diff` command to show current git diff with highlighting
- Collapsible — show summary first ("3 files changed, +42/-18"), expand on request

#### 1.2 Checkpoint & Rewind System
**What:** Snapshot file state before edits. Let users rewind to any checkpoint.
**Why:** This is what makes autonomous editing feel safe. Claude Code and Copilot CLI both have Esc Esc rewind. Aider uses git commits. Without this, users can't let Aura run freely.
**Scope:**
- Auto-snapshot every file before edit (store in `.aura/checkpoints/`)
- `Esc Esc` or `/rewind` — interactive picker to rewind to any checkpoint
- Options: rewind files only, rewind conversation only, rewind both
- Show what will be reverted before confirming
- Limit to last 50 checkpoints, auto-prune older ones

#### 1.3 Context Window Visibility
**What:** Show token usage, context budget, and warnings in the status bar.
**Why:** Users need to know when they're running low on context. Claude Code has /context fuel gauge, Aider has /tokens. Aura shows nothing.
**Scope:**
- Token counter in status bar: `Tokens: 12.4K / 128K`
- Color-coded: green (<50%), yellow (50-80%), red (>80%)
- Warning when approaching limit: "Context 85% full — consider /compact"
- `/context` command showing detailed breakdown (system prompt, history, tools)
- `/compact` already exists — add a "compacted X messages, saved Y tokens" confirmation

#### 1.4 Keyboard Shortcuts
**What:** Standard keyboard shortcuts for common operations.
**Why:** Alt+M is the only shortcut. Every competitor has 4-6+ shortcuts. Power users expect them.
**Scope:**
- `Ctrl+L` — Clear screen (keep conversation)
- `Ctrl+N` — New session
- `Ctrl+K` — Command palette (fuzzy search all commands)
- `Ctrl+R` — Search conversation history
- `Esc Esc` — Rewind (see 1.2)
- `Shift+Tab` — Cycle permission modes
- `Shift+Enter` — Multi-line input
- `Ctrl+G` — Open system editor for long prompts
- All customizable via `~/.aura/keybindings.json`

#### 1.5 Permission Tiers (Not Just Trust Toggle)
**What:** Multiple permission levels, not just on/off trust mode.
**Why:** The trust toggle is too binary. Every competitor has 3-5 tiers.
**Scope:**
- **Plan Mode** — Read-only, no file edits or commands (research only)
- **Careful** (default) — Approve every file edit and shell command
- **Auto-Edit** — File edits auto-apply, shell commands still ask
- **Full Auto** — Everything runs, cost/iteration limits are the only guard
- `Shift+Tab` cycles through modes
- Mode indicator in status bar
- Per-project defaults in AURA.md

---

### Tier 2 — Differentiators (High Impact, Medium Effort)
*Features that elevate Aura above competitors by leveraging its unique strengths.*

#### 2.1 Editable Plan Mode
**What:** Before executing complex tasks, Aura generates a structured plan as a Markdown checklist. User can edit it. Then Aura executes the plan step-by-step.
**Why:** Cursor's Plan Mode reduces context errors by 35%. Aura already has `/plan` but it's basic. An editable plan is a "contract" between user and agent.
**Scope:**
- `/plan <task>` generates a Markdown plan with steps, files to modify, and approach
- Plan displayed inline with checkboxes
- User can edit before confirming (add/remove/reorder steps)
- During execution, current step highlighted, completed steps checked off
- If a step fails, pause and ask for guidance (not just retry blindly)

#### 2.2 Streaming Tool Execution
**What:** Show tool output in real-time, not as monolithic blocks after completion.
**Why:** Shell commands, file reads, web requests — all produce progressive output. Showing it live makes the agent feel responsive and transparent.
**Scope:**
- Shell commands: stream stdout/stderr line-by-line as they execute
- File reads: show content progressively with syntax highlighting
- Web requests: show status → headers → body progressively
- Progress bars for long operations (indexing, searching large codebases)
- Collapsible tool output — show first 5 lines by default, "show more" to expand

#### 2.3 Smart Session Picker
**What:** Interactive session browser with arrow keys, preview, and metadata.
**Why:** Current `/sessions` is text-based number selection. Every modern CLI tool uses interactive pickers.
**Scope:**
- Arrow-key navigation (like model picker)
- Per-session metadata: title, duration, message count, last model, last active
- Preview: show last 3 messages of selected session
- Filter/search by keyword
- Quick actions: resume, rename, delete, export
- Recent sessions shown on startup if no arguments

#### 2.4 Unix Composability
**What:** Make Aura a proper Unix citizen — pipe-friendly, stderr/stdout split, structured output.
**Why:** Simon Willison's `llm`, Continue CLI, Cline CLI 2.0 all support piping. Power users chain tools.
**Scope:**
- `aura -p "prompt"` outputs response to stdout, status to stderr (like Codex CLI)
- Pipe input: `git diff | aura -p "review this code"`
- `--format json` flag for structured output
- `--format markdown` for clean Markdown (no Rich formatting)
- Exit codes: 0 = success, 1 = error, 2 = budget exceeded
- `aura exec "task"` — non-interactive agent run (like `codex exec`)

#### 2.5 Mid-Turn Steering
**What:** Inject instructions while Aura is working, without interrupting the current action.
**Why:** Codex CLI lets you press Enter to inject instructions mid-turn. This is a unique real-time steering capability.
**Scope:**
- While Aura is running (thinking/executing tools), user can type and press Enter
- Message is queued and injected as a "user note" in the next iteration
- Press `Tab` while running to queue a follow-up prompt for the next turn
- Visual indicator: "Message queued — will apply on next step"

#### 2.6 Aura Themes
**What:** Customizable color themes for the terminal UI.
**Why:** Codex CLI has /theme with live preview and .tmTheme support. Long sessions need comfortable aesthetics.
**Scope:**
- Built-in themes: dark (default), light, monokai, dracula, solarized, nord
- `/theme` command with live preview
- Custom themes via `~/.aura/themes/` (JSON format)
- Theme affects: banner, status bar, response panels, diffs, code blocks
- Auto-detect dark/light terminal and suggest matching theme

---

### Tier 3 — Advanced Agent Features (High Effort, Breakthrough Impact)
*Features that push Aura beyond any competitor by leveraging its unique architecture (emotion, memory, dreams, proactive awareness).*

#### 3.1 Parallel Sub-Agents
**What:** Split complex tasks into sub-tasks and run them in parallel.
**Why:** Copilot CLI's `/fleet` is a genuine throughput multiplier. Aura already has sub-agent infrastructure (`core/sub_agent.py`).
**Scope:**
- `/fleet <task>` — Aura decomposes task, runs sub-agents in parallel
- Each sub-agent gets its own context and tool access
- Live dashboard showing all sub-agents: status, current action, progress
- Git worktree isolation — each agent works on a separate branch
- Merge results when all complete, show conflicts if any
- Budget split across sub-agents

#### 3.2 Background Agent Mode
**What:** Send tasks to background, get notified when done.
**Why:** Codex App, Cursor Automations, and Devin all support async execution. Users shouldn't have to watch.
**Scope:**
- `Ctrl+B` while agent is running — move to background
- `&` prefix on commands — start in background immediately
- `/tasks` — show all background tasks (running, completed, failed)
- Desktop notification (winotify on Windows) when background task completes
- Resume background task in foreground: `/resume <task_id>`
- Daemon integration — background tasks survive terminal close

#### 3.3 Progressive Disclosure for Tool Calls
**What:** Show tool execution at the right level of detail — summary by default, expandable.
**Why:** The emerging standard (Claude Code, Cursor) is: always show action label + time, collapse details, expand on click/key.
**Scope:**
- Default: `▸ edit_file main.py (+12/-3)` — one-line summary
- Press `Enter` or `e` to expand → full diff view
- Collapsible sections for shell output (first 5 lines shown)
- After completion: tool calls are collapsed in conversation history
- Verbose mode (`-v`) shows everything expanded

#### 3.4 Research Mode
**What:** Dedicated mode for research tasks with citation tracking and source management.
**Why:** No CLI tool does this well. Aura has web search, arXiv, memory — but no structured research workflow.
**Scope:**
- `/research <topic>` — enters research mode
- Searches web, arXiv, and memory for relevant sources
- Tracks citations with numbered references [1], [2], etc.
- Builds a research context that persists across messages
- `/sources` — show all sources with metadata
- `/export research` — export findings as Markdown with bibliography
- Leverages Aura's existing memory system for cross-session research continuity

#### 3.5 Emotional Context in CLI
**What:** Surface Aura's unique emotional system in the CLI experience.
**Why:** No other CLI has this. It's Aura's differentiator. The ALMA engine, mood, neuromodulators — make them visible and meaningful.
**Scope:**
- Subtle mood indicator in status bar (emoji or color shift)
- Aura's response style naturally adapts to mood (already happens via neuromodulators — make it visible)
- `/mood` — show current emotional state, what influenced it
- Dream insights appear as proactive suggestions: "While consolidating memories last night, I noticed a pattern..."
- After long sessions, Aura suggests breaks based on cognitive load

#### 3.6 Hooks & Automation System
**What:** Programmable hooks that fire on events, enabling workflow automation.
**Why:** Claude Code's hooks (PreToolUse, PostToolUse) are the recommended fine-grained control mechanism. Aura has `/hook` but it's basic.
**Scope:**
- Event types: pre_tool_call, post_tool_call, pre_response, post_response, session_start, session_end
- Hook definitions in AURA.md or `~/.aura/hooks.yaml`
- Shell command hooks: run a script on each event
- Built-in hooks: auto-lint after edits, auto-test after code changes, auto-commit
- `/hook list`, `/hook add <event> <command>`, `/hook remove`

---

### Tier 4 — Polish & Ecosystem (Medium Effort, Cumulative Quality)
*Features that make the day-to-day experience smoother.*

#### 4.1 Command Palette with Fuzzy Search
**What:** `Ctrl+K` opens a fuzzy-searchable palette of all commands, files, and sessions.
**Why:** Modern editors (VS Code, Sublime) all have this. Typing `/` then remembering the exact command is slower.
**Scope:**
- Fuzzy search across: slash commands, recent files, sessions, models
- Arrow keys to navigate, Enter to execute
- Preview pane for file/session selection
- Recently used commands float to top

#### 4.2 Git Power Tools
**What:** Deep git integration beyond /commit and /diff.
**Why:** Aider is git-native (auto-commits everything). Claude Code creates branches and PRs. Aura should match.
**Scope:**
- `/commit` — AI-generated commit message (already exists — improve quality)
- `/pr` — Create PR with AI-generated title and description
- `/branch <name>` — Create feature branch from current state
- `/stash` — Smart stash with AI-generated description
- Auto-commit on each Aura edit (opt-in via AURA.md `auto_commit: true`)
- `/blame <file:line>` — Explain why this line exists using git history

#### 4.3 Test Runner Integration
**What:** Run tests with formatted output — pass/fail indicators, error highlighting, auto-fix loop.
**Why:** Aider has --auto-test. Claude Code reads test output. Aura should have first-class test integration.
**Scope:**
- `/test` — runs `test_cmd` from AURA.md, parses output
- Color-coded results: green pass, red fail, yellow skip
- On failure: show failing test, offer to fix automatically
- Auto-test mode: run tests after every file edit (AURA.md `auto_test: true`)
- Test history: track pass rate over session

#### 4.4 Activity Log & Queryable History
**What:** Log all interactions to a queryable SQLite store (like Simon Willison's `llm`).
**Why:** Power users want to search past interactions, export conversations, analyze patterns.
**Scope:**
- Every prompt + response logged to `~/.aura/logs.db`
- Searchable: `aura log search "authentication bug"`
- Export: `aura log export --session <id> --format markdown`
- Stats: `aura log stats` — messages, tokens, cost, favorite models
- Feeds into Aura's memory system for cross-session learning

#### 4.5 Watch Mode
**What:** Monitor files for AI comments and auto-respond.
**Why:** Aider's Watch Mode lets users drop `AI: fix this` comments in code and Aider picks them up.
**Scope:**
- `/watch` — start monitoring current project files
- Detects `AURA:` or `AI:` comments in code
- Auto-responds to the comment, applies fix, removes the marker
- Shows a log of detected and resolved comments
- Works alongside IDE — edit in VS Code, Aura watches and acts

---

## Implementation Priority

### Phase 1: Foundation (2-3 weeks)
> Make Aura feel like a 2026 CLI tool.

1. **1.3 Context Window Visibility** — Quick win, high visibility
2. **1.4 Keyboard Shortcuts** — Framework + 5 core shortcuts
3. **1.5 Permission Tiers** — 4 modes + Shift+Tab cycling
4. **1.1 Inline Diff Viewer** — Core diff rendering engine
5. **1.2 Checkpoint & Rewind** — File snapshots + Esc Esc

### Phase 2: Power User (2-3 weeks)
> Make Aura the best tool for serious work.

6. **2.1 Editable Plan Mode** — Upgrade existing /plan
7. **2.2 Streaming Tool Execution** — Live tool output
8. **2.4 Unix Composability** — Pipe-friendly mode
9. **4.2 Git Power Tools** — /pr, /branch, auto-commit
10. **4.3 Test Runner Integration** — /test with auto-fix

### Phase 3: Differentiation (2-3 weeks)
> Features no other CLI has.

11. **2.3 Smart Session Picker** — Interactive browser
12. **2.5 Mid-Turn Steering** — Inject instructions while running
13. **3.5 Emotional Context** — Surface the ALMA engine
14. **3.4 Research Mode** — Citation tracking + source management
15. **4.1 Command Palette** — Ctrl+K fuzzy search

### Phase 4: Agent Platform (3-4 weeks)
> Make Aura the most capable autonomous agent CLI.

16. **3.1 Parallel Sub-Agents** — /fleet with dashboard
17. **3.2 Background Agent Mode** — Ctrl+B + notifications
18. **3.3 Progressive Disclosure** — Collapsible tool output
19. **3.6 Hooks & Automation** — Programmable event hooks
20. **4.5 Watch Mode** — File monitoring + auto-respond

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Keyboard shortcuts | 1 (Alt+M) | 10+ customizable |
| Permission modes | 2 (normal, trust) | 4 graduated tiers |
| Context visibility | None | Real-time token gauge |
| Diff display | None | Syntax-highlighted inline |
| Safety net | None | Checkpoint + rewind |
| Non-interactive modes | -p flag | exec, pipe, watch, background |
| Time to first useful action | ~3s (banner + init) | <1s (no banner in pipe mode) |

---

## Design Principles

1. **Progressive disclosure** — Summary by default, details on demand
2. **Earned autonomy** — Start careful, let users opt into more freedom
3. **Git is the safety net** — Every change can be reverted
4. **Unix philosophy** — Composable, pipe-friendly, exit codes meaningful
5. **Aura's uniqueness** — Emotion, memory, dreams are differentiators, not gimmicks
6. **Speed over chrome** — Fast startup, instant streaming, no unnecessary animation
7. **Customizable** — Themes, keybindings, hooks, permission levels — power users shape their experience

---

## References

- [Claude Code Docs](https://code.claude.com/docs/) — Rewind, permissions, hooks, /context
- [Aider Docs](https://aider.chat/docs/) — Architect mode, repo map, git-native workflow
- [Codex CLI](https://developers.openai.com/codex/cli/) — Sandbox, themes, mid-turn injection
- [Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/copilot-cli) — Fleet, repository memory, & delegation
- [Cursor Plan Mode](https://cursor.com/docs/agent/plan-mode) — Editable plans, 35% error reduction
- [Cline CLI 2.0](https://cline.ghost.io/introducing-cline-cli-2-0/) — Plan/Act, stdin/stdout, -y flag
- [Simon Willison's llm](https://llm.datasette.io/) — SQLite logging, fragments, schemas, plugins
- [Fabric](https://github.com/danielmiessler/fabric) — Pattern library, Unix pipes, YouTube transcripts
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/) — Time-travel debugging, checkpoints
- [Anthropic: Measuring Agent Autonomy](https://www.anthropic.com/research/measuring-agent-autonomy)
