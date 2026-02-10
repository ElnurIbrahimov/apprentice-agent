# Claude Code — IDE Integrations

**Tags:** claude-code, vscode, jetbrains, ide
**Created:** 2025
**Category:** claude-code

---

## VS Code Extension

### Installation
- Command palette: `Cmd+Shift+X` / `Ctrl+Shift+X`
- Search "Claude Code" and install
- Requires VS Code 1.98.0+

### Features
- Native graphical panel
- Inline diffs with side-by-side comparison
- @-mentions with fuzzy matching
- Plan mode review
- Multiple conversation tabs
- Resume web sessions from claude.ai

### Key Shortcuts
| Shortcut | Action |
|----------|--------|
| `Cmd+Esc` / `Ctrl+Esc` | Toggle focus editor/Claude |
| `Cmd+Shift+Esc` / `Ctrl+Shift+Esc` | Open new tab |
| `Option+K` / `Alt+K` | Insert @-mention reference |
| `Cmd+N` / `Ctrl+N` | New conversation |
| `Shift+Enter` | Multiline input |

### Extension Settings
- `selectedModel` — Model for new conversations
- `useTerminal` — Launch in terminal mode
- `initialPermissionMode` — default, plan, acceptEdits
- `preferredLocation` — sidebar or panel
- `autosave` — Auto-save before read/write
- `respectGitIgnore` — Exclude .gitignore patterns

### Diff Viewing
Hover messages to reveal rewind button with options:
1. Fork conversation from here
2. Rewind code to here
3. Fork and rewind

---

## JetBrains IDEs

### Supported
IntelliJ IDEA, PyCharm, Android Studio, WebStorm, PhpStorm, GoLand

### Features
- Quick launch: `Cmd+Esc` / `Ctrl+Esc`
- IDE diff viewer integration
- Selection context sharing
- File reference: `Cmd+Option+K` / `Alt+Ctrl+K`
- Diagnostic sharing from IDE

### Configuration
- Settings -> Tools -> Claude Code [Beta]
- WSL: Set command to `wsl -d Ubuntu -- bash -lic "claude"`
