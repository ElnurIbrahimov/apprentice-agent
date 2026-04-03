# Phase 1: Fix Broken Things

**Effort:** 1 day
**Impact:** Fixes every embarrassing bug users would hit immediately

---

## 1A. Real Light Theme

### Problem
Settings has Dark/Light/System toggle. Selecting "Light" adds class `light` to `<html>` but there are zero `.light` CSS overrides in `index.css`. The entire UI stays dark.

### What To Build
Add a complete `.light` override block in `web/src/index.css`:

```css
html.light {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #f1f5f9;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #94a3b8;
  --border: #e2e8f0;
  --surface-0: #ffffff;
  --surface-1: #f8fafc;
  --surface-2: #f1f5f9;
  --surface-3: #e2e8f0;
  --surface-4: #cbd5e1;
  --accent: #7c3aed;
  --accent-glow: rgba(124, 58, 237, 0.15);
  --chat-bg: #ffffff;
  --chat-sidebar: #f8fafc;
  --chat-assistant: #f1f5f9;
  --chat-border: #e2e8f0;
  --chat-text: #0f172a;
  /* ... override ALL dark-mode CSS vars */
}
```

Also:
- Mesh gradient background blobs: reduce opacity or switch to light pastels
- Grain overlay: reduce or remove
- MessageBubble: user bubbles need light-aware border/shadow
- CodeBlock: switch to a light code theme
- Sidebar panels: ensure readability on light backgrounds
- Scrollbars: light track color

### System Theme
Wire `prefers-color-scheme: dark` media query for "System" option:
```css
@media (prefers-color-scheme: light) {
  html.system { /* same as html.light */ }
}
```

---

## 1B. Fix Regenerate Button

### Problem
Long-press context menu on messages shows "Regenerate" but it just calls `console.log('regenerate', message.id)`.

### What To Build
In `MessageBubble.tsx` (or wherever the context menu lives):

```typescript
const handleRegenerate = useCallback(async (messageId: string) => {
  // 1. Find the user message that preceded this assistant message
  const messages = chatStore.getState().messages;
  const idx = messages.findIndex(m => m.id === messageId);
  if (idx < 0) return;
  
  // Find the preceding user message
  let userMsg = null;
  for (let i = idx - 1; i >= 0; i--) {
    if (messages[i].role === 'user') { userMsg = messages[i]; break; }
  }
  if (!userMsg) return;
  
  // 2. Remove the assistant message and everything after it
  chatStore.getState().removeMessagesFrom(messageId);
  
  // 3. Re-send the user message via WebSocket
  sendMessage(userMsg.text, userMsg.attachments);
}, []);
```

Also add `removeMessagesFrom(messageId)` to chatStore if it doesn't exist.

---

## 1C. Wire showThinking Setting

### Problem
`settingsStore` has `showThinking: true` toggle in Behavior settings, but no component reads it.

### What To Build
In `MessageBubble.tsx`:
- Import `useSettingsStore`
- When `showThinking === false`, hide `ThinkingShimmer` and the "Thought for Ns" pill
- When `showThinking === true` (default), show them as normal

In `ChatContainer.tsx`:
- When `showThinking === false`, skip rendering `ThinkingShimmer` during streaming

---

## 1D. Fix Mobile Swipe Drawer

### Problem
Swipe-from-left-edge gesture in ChatContainer works mechanically (20px edge zone, physics, spring animation) but the drawer body contains only placeholder text: "Conversation history will appear here."

### What To Build
Replace the placeholder with the actual `ConversationList` component:

```tsx
// In the swipe drawer body:
<ConversationList
  conversations={conversations}
  currentId={currentConversationId}
  onSelect={(id) => { switchConversation(id); closeDrawer(); }}
  onNew={() => { newConversation(); closeDrawer(); }}
  compact  // Smaller variant for drawer context
/>
```

Also:
- Add a "New Chat" button at the top of the drawer
- Add a close button (X) in the drawer header
- Dim the main content when drawer is open (backdrop)

---

## 1E. Fix CitationsPanel Toggle

### Problem
The toggle button for CitationsPanel has `display: citationsPanelOpen ? 'none' : 'flex'` — the button disappears when the panel opens. No close button is consistently visible on the panel itself.

### What To Build
- Keep the toggle button always visible (change to a toggle icon that rotates or changes)
- Add a clear "X" close button in the CitationsPanel header
- When open: button shows "Close Citations" state
- When closed: button shows "N Citations" badge

---

## 1F. Fix Duplicate Code

### Problem
`EMOTION_COLORS`, `NEURO_INFO`, `PERSONALITY_INFO` objects are duplicated verbatim between `EmotionPanel.tsx` and `MiniApp.tsx`.

### What To Build
Extract to `web/src/utils/emotionConstants.ts`:
```typescript
export const EMOTION_COLORS = { ... };
export const NEURO_INFO = { ... };
export const PERSONALITY_INFO = { ... };
```
Import in both files.

---

## Definition of Done — Phase 1
- [ ] Light theme renders correctly across all components (chat, sidebar, settings, all tabs)
- [ ] System theme follows OS preference
- [ ] Regenerate button works: removes assistant message, re-sends user message
- [ ] showThinking toggle hides/shows thinking indicators
- [ ] Mobile swipe drawer shows real ConversationList with new chat button
- [ ] CitationsPanel toggle always visible, panel has close button
- [ ] Emotion constants extracted to shared module
- [ ] No visual regressions in dark theme
