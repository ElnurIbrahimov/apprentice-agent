# Claude Code — Keyboard Shortcuts

**Tags:** claude-code, shortcuts, keybindings
**Created:** 2025
**Category:** claude-code

---

## General Controls
| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel/interrupt |
| `Ctrl+D` | Exit |
| `Ctrl+L` | Clear screen (keeps history) |
| `Ctrl+O` | Toggle verbose output |
| `Ctrl+R` | Reverse search command history |
| `Ctrl+V` / `Cmd+V` | Paste image |
| `Ctrl+B` | Background current task |
| `Ctrl+T` | Toggle task list |
| `Ctrl+G` | Open in external editor |

## Permission & Mode
| Shortcut | Action |
|----------|--------|
| `Shift+Tab` / `Alt+M` | Cycle permission modes |
| `Option+P` / `Alt+P` | Switch model |
| `Option+T` / `Alt+T` | Toggle extended thinking |

## Text Editing
| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Delete to end of line |
| `Ctrl+U` | Delete entire line |
| `Ctrl+Y` | Paste deleted text |
| `Alt+B` / `Alt+F` | Move word back/forward |

## Multiline Input
| Method | Shortcut |
|--------|----------|
| Quick escape | `\` + `Enter` |
| macOS | `Option+Enter` |
| Standard | `Shift+Enter` |

## Navigation
| Shortcut | Action |
|----------|--------|
| `Up/Down` | Navigate history |
| `Esc Esc` | Rewind or summarize |

## Custom Keybindings (`~/.claude/keybindings.json`)
```json
{
  "$schema": "https://www.schemastore.org/claude-code-keybindings.json",
  "bindings": [
    {
      "context": "Chat",
      "bindings": {
        "ctrl+e": "chat:externalEditor",
        "shift+enter": "chat:submit",
        "ctrl+u": null
      }
    }
  ]
}
```

## Available Contexts
Global, Chat, Autocomplete, Settings, Confirmation, Tabs, Help, Transcript, HistorySearch, Task, ThemePicker, Attachments, Footer, MessageSelector, DiffDialog, ModelPicker, Select, Plugin
