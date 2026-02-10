# Claude Code — Chrome Browser Automation

**Tags:** claude-code, chrome, browser, automation
**Created:** 2025
**Category:** claude-code

---

## Setup
1. Install "Claude in Chrome" extension (v1.0.36+)
2. Launch Claude Code with `--chrome` flag or enable in settings
3. Extension connects Claude Code to your browser

## Available Tools

### Navigation & Context
| Tool | Purpose |
|------|---------|
| `tabs_context_mcp` | Get info about current browser tabs (ALWAYS call first) |
| `tabs_create_mcp` | Create new tab in MCP group |
| `navigate` | Go to URL, forward/back |
| `resize_window` | Resize browser window |
| `switch_browser` | Connect to different Chrome browser |

### Page Interaction
| Tool | Purpose |
|------|---------|
| `computer` | Mouse clicks, keyboard input, screenshots, scrolling |
| `find` | Find elements by natural language ("search bar", "login button") |
| `read_page` | Get accessibility tree of page elements |
| `form_input` | Set values in form fields |
| `javascript_tool` | Execute JavaScript in page context |
| `get_page_text` | Extract raw text content from page |

### Recording
| Tool | Purpose |
|------|---------|
| `gif_creator` | Record browser actions as animated GIF |

### Debugging
| Tool | Purpose |
|------|---------|
| `read_console_messages` | Read browser console output |
| `read_network_requests` | Monitor HTTP requests |

### Media
| Tool | Purpose |
|------|---------|
| `upload_image` | Upload screenshot/image to file input or drag target |

## Workflow Pattern

### 1. Always Start with Context
```
tabs_context_mcp(createIfEmpty=true)
```
Get tab IDs before doing anything else.

### 2. Create New Tab (Don't Reuse)
```
tabs_create_mcp()
```
Each session should create its own tab.

### 3. Navigate
```
navigate(url="https://example.com", tabId=123)
```

### 4. Interact
```
# Take screenshot to see the page
computer(action="screenshot", tabId=123)

# Find and click element
find(query="login button", tabId=123)
computer(action="left_click", coordinate=[x, y], tabId=123)

# Type text
computer(action="type", text="hello", tabId=123)
```

## GIF Recording
```
# Start recording
gif_creator(action="start_recording", tabId=123)

# Take screenshot for first frame
computer(action="screenshot", tabId=123)

# ... do actions ...

# Take final screenshot
computer(action="screenshot", tabId=123)

# Stop and export
gif_creator(action="stop_recording", tabId=123)
gif_creator(action="export", tabId=123, download=true, filename="demo.gif")
```

## Safety Rules
- Never enter sensitive financial data (bank accounts, SSN, etc.)
- Never create accounts on user's behalf
- Never authorize password-based access
- Always confirm before downloads, purchases, or sending messages
- Never bypass CAPTCHA or bot detection
- Decline cookies by default (privacy-preserving)
- Never reproduce copyrighted content from web pages
- Stop and ask user if encountering unexpected complexity

## Common Issues
- **Dialog blocking**: Avoid triggering alerts/confirms — they block all browser events
- **Tab not found**: Call `tabs_context_mcp` to refresh tab IDs
- **Element not clickable**: Take screenshot, verify coordinates, adjust click position
- **Page not loading**: Wait, then retry navigation
