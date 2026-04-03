# Phase 2: Chat UX Upgrades

**Effort:** 2 days
**Impact:** Brings chat parity with modern AI interfaces (ChatGPT, Claude.ai)

---

## 2A. Message Editing

### Problem
Users can't edit their own messages. If they made a typo or want to rephrase, they have to type a new message.

### What To Build

**In `MessageBubble.tsx`:**
- Add an "Edit" button (pencil icon) on hover for user messages
- On click: replace the message text with an editable textarea
- "Save & Resend" button: updates the message text, removes all subsequent messages, re-sends via WebSocket
- "Cancel" button: reverts to original text
- Show "(edited)" indicator on edited messages

**In `chatStore.ts`:**
```typescript
editMessage: (id: string, newText: string) => {
  set(state => ({
    messages: state.messages.map(m => 
      m.id === id ? { ...m, text: newText, edited: true } : m
    )
  }));
},
removeMessagesAfter: (id: string) => {
  set(state => {
    const idx = state.messages.findIndex(m => m.id === id);
    return { messages: state.messages.slice(0, idx + 1) };
  });
},
```

### UI
- Hover state: faint pencil icon appears top-right of user message
- Edit mode: textarea replaces text, with "Save & Resend" (purple) and "Cancel" (gray) buttons below
- "(edited)" small muted text after the message timestamp

---

## 2B. Message Retry

### Problem
If the AI gives a bad response, users must type "try again" or similar. No retry button.

### What To Build

**On assistant messages, add:**
- "Retry" button (refresh icon) on hover
- On click: calls regenerate logic (same as 1B — remove this message, re-send the preceding user message)
- Loading state while regenerating

**On failed messages (error state):**
- "Retry" button always visible (not just on hover)
- Clear the error and re-send

---

## 2C. Conversation Export

### Problem
No way to export a conversation as text, markdown, or JSON.

### What To Build

**New utility:** `web/src/utils/exportConversation.ts`

```typescript
export function exportAsMarkdown(messages: Message[]): string {
  return messages.map(m => {
    const role = m.role === 'user' ? '**You**' : '**Aura**';
    const time = new Date(m.timestamp).toLocaleString();
    return `### ${role} — ${time}\n\n${m.text}\n`;
  }).join('\n---\n\n');
}

export function exportAsJSON(messages: Message[]): string {
  return JSON.stringify(messages.map(m => ({
    role: m.role,
    text: m.text,
    timestamp: m.timestamp,
    citations: m.citations,
    attachments: m.attachments?.map(a => ({ name: a.name, type: a.type })),
  })), null, 2);
}

export function downloadExport(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

**UI:**
- Three-dot menu on conversation header → "Export as Markdown" / "Export as JSON" / "Copy to Clipboard"
- Also in the conversation right-click context menu in the sidebar

---

## 2D. Keyboard Shortcuts

### Problem
Only Ctrl+N (new chat), Ctrl+K (focus input), Ctrl+/ (settings) exist. No way to navigate tabs, toggle sidebar, etc.

### What To Build

| Shortcut | Action |
|----------|--------|
| `Ctrl+1` through `Ctrl+6` | Switch to tab 1-6 |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+Shift+N` | New conversation |
| `Ctrl+Shift+E` | Export current conversation |
| `Escape` | Close any open panel/modal |
| `Ctrl+Shift+T` | Toggle theme (dark/light) |
| `Up Arrow` (in empty input) | Edit last user message |

**Implementation:**
- Global `useEffect` with `keydown` listener in App.tsx
- Dispatch custom events or store actions
- Show shortcut hints in tooltips on tab buttons

**Shortcut help modal:**
- `Ctrl+?` or `?` when input not focused → shows shortcut cheat sheet

---

## 2E. Stop Button Per Message

### Problem
Stop button only exists in the input area. If the user scrolls up during generation, they can't stop it without scrolling back down.

### What To Build
- Add a small "Stop" button on the currently-streaming assistant message
- Appears only during active streaming for that specific message
- Calls the same stop logic as the input area stop button
- Disappears when generation completes

---

## 2F. Copy Improvements

### Problem
"Copy" on messages copies raw text. No "Copy as code" for code-only responses, no "Copy as Markdown".

### What To Build
- Single code block responses: show "Copy Code" button that copies only the code content
- Multi-section responses: "Copy as Markdown" preserves formatting
- Success feedback: button text changes to "Copied!" for 1.5s with checkmark

---

## Definition of Done — Phase 2
- [ ] User messages have edit button → textarea → save & resend flow
- [ ] "(edited)" indicator shows on edited messages
- [ ] Assistant messages have retry button on hover
- [ ] Failed messages always show retry button
- [ ] Conversation export as Markdown and JSON works
- [ ] Export accessible from conversation header menu and sidebar context menu
- [ ] Keyboard shortcuts for tabs, sidebar, theme, new chat, export
- [ ] Shortcut cheat sheet modal (Ctrl+?)
- [ ] Stop button visible on the streaming message itself
- [ ] Code blocks have "Copy Code" button
- [ ] All copy buttons show "Copied!" feedback
