# Phase 8: Mobile-First Overhaul + PWA

**Effort:** 3 days
**Impact:** Makes the web UI a genuine mobile app alternative — installable, fast, works offline for basics

---

## The Vision

The web dashboard becomes a Progressive Web App that can be installed on any phone, loads instantly, and feels native. Combined with the Telegram MiniApp (which already exists), this gives users 3 ways to access Aura on mobile: browser, PWA, and Telegram.

---

## 8A. PWA Setup

### manifest.json
**New file:** `web/public/manifest.json`
```json
{
  "name": "Aura AI",
  "short_name": "Aura",
  "description": "Your private AI assistant",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#030303",
  "theme_color": "#7c3aed",
  "orientation": "any",
  "icons": [
    { "src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/icons/icon-maskable.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ]
}
```

### Service Worker Enhancement
`web/public/sw.js` already exists. Enhance it:
- **Cache-first** for static assets (JS, CSS, fonts, icons)
- **Network-first** for API calls (fall back to cached response if offline)
- **Offline page**: show a "You're offline — reconnecting..." page when network is down
- **Background sync**: queue failed messages and retry when back online

### Registration
Already registered in main.tsx. Ensure it's production-only:
```typescript
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register('/sw.js');
}
```

### Install Prompt
Detect `beforeinstallprompt` event and show a custom install banner:
```
┌────────────────────────────────────────────┐
│ 📱 Install Aura for quick access  [Install] [Dismiss] │
└────────────────────────────────────────────┘
```
Show once, remember dismissal in localStorage.

---

## 8B. Mobile Navigation Overhaul

### Problem
Current: 6 tab buttons in a horizontal header. On small screens, tab labels hide (icons only). No bottom nav. Sidebar requires hamburger menu.

### Solution: Bottom Tab Bar on Mobile

```
Desktop (>1024px):        Mobile (<1024px):
┌──────────────────┐     ┌──────────────────┐
│ [Sidebar] [Tabs] │     │ [Content area]   │
│ [Content]        │     │                  │
│                  │     │                  │
│                  │     ├──────────────────┤
└──────────────────┘     │ [🗨️][🔧][📊][⚙️] │
                         └──────────────────┘
```

### Bottom Tab Bar Component
```typescript
// web/src/components/BottomTabBar.tsx
const MOBILE_TABS = [
  { id: 'chat', icon: MessageSquare, label: 'Chat' },
  { id: 'tools', icon: Wrench, label: 'Tools' },
  { id: 'monitoring', icon: Activity, label: 'Monitor' },
  { id: 'settings', icon: Settings, label: 'Settings' },
];
```

- Fixed at bottom, 56px height
- Safe area padding (`env(safe-area-inset-bottom)`)
- Active tab: filled icon + label + accent color
- Inactive: outline icon only
- Haptic feedback on tap
- Swipe left/right between tabs (gesture handler)
- Hide when keyboard is open (detect `visualViewport.resize`)

### Sidebar → Drawer on Mobile
- Full-screen overlay drawer (not 256px sidebar)
- Swipe-from-left to open (already exists but broken)
- Backdrop tap to close
- Conversation list + quick actions

---

## 8C. Touch Interactions

### Pull-to-Refresh
At the top of the chat scroll area:
- Pull down 60px+ → refresh messages / reconnect WebSocket
- Spring animation with loading indicator
- `overscroll-behavior: contain` to prevent browser default

### Swipe Gestures
- **Swipe left on message** → Show action buttons (copy, share, delete)
- **Swipe between tabs** → Navigate bottom tabs
- **Long-press message** → Context menu (already exists, but improve)
  - Fix: regenerate actually works (Phase 1)
  - Add: share, bookmark, translate

### Haptic Patterns
```typescript
const haptics = {
  light: () => navigator.vibrate?.(10),     // tab switch, button tap
  medium: () => navigator.vibrate?.(25),    // send message, action confirm
  heavy: () => navigator.vibrate?.(50),     // error, long-press trigger
  success: () => navigator.vibrate?.([10, 50, 10]),  // generation complete
};
```

---

## 8D. Mobile Chat Optimizations

### Input Bar
- Sticky at bottom with safe-area padding
- Auto-resize textarea (1-5 lines)
- Attachment button opens native file picker (camera option on mobile)
- Voice input with animated waveform
- "Send" button transforms to "Stop" during generation

### Message List
- Virtualized rendering for long conversations (react-window or similar)
  - Current: renders all 500 messages → laggy on phones
  - After: only renders ~20 visible messages + buffer
- Image messages: lazy load with blur placeholder
- Code blocks: horizontal scroll instead of wrap (fits narrow screens better)
- Reduce font size slightly on mobile (14px → 13px body)

### Connection Status
- Subtle top bar: green dot when connected, yellow during reconnect, red when offline
- Auto-reconnect with exponential backoff (already exists)
- Show "Reconnecting..." toast instead of full-width banner on mobile

---

## 8E. Offline Support

### What Works Offline
- Browsing conversation history (cached in localStorage/IndexedDB)
- Settings changes
- Reading previous AI responses
- Viewing saved artifacts/images

### What Shows "Offline" State
- Chat input: disabled with "Offline — messages will be sent when reconnected"
- New chat: queued and sent on reconnect
- Polling: paused entirely, resumes on reconnect
- Generate/Execute: disabled with clear messaging

### Conversation Cache
Move conversation storage from in-memory `chatStore` to IndexedDB for persistence:
```typescript
// web/src/utils/conversationDB.ts
import { openDB } from 'idb';

const db = openDB('aura-conversations', 1, {
  upgrade(db) {
    db.createObjectStore('conversations', { keyPath: 'id' });
    db.createObjectStore('messages', { keyPath: 'id' }).createIndex('conversationId', 'conversationId');
  },
});
```

---

## 8F. Performance Budget

Target metrics for mobile (Lighthouse):
- **FCP** < 1.5s
- **LCP** < 2.5s
- **CLS** < 0.1
- **TTI** < 3s
- **Bundle size** < 200KB gzipped (excluding lazy chunks)

To achieve:
- Lazy load everything except Chat tab
- Code-split Shiki grammars per language
- Preload critical fonts (Plus Jakarta Sans)
- Inline critical CSS
- Compress images (icons) to WebP

---

## 8G. Telegram MiniApp Sync

### Problem
`MiniApp.tsx` is a separate app sharing no code with the main web app. Duplicate emotion constants, separate WS hook, no shared components.

### What To Build
- Extract shared utilities: emotion constants, format utils, haptics, sounds
- Share store between main app and MiniApp (same Zustand stores)
- MiniApp becomes a "lite" layout using shared components
- Same conversation state — switching between MiniApp and web app shows same chat

---

## Definition of Done — Phase 8
- [ ] PWA manifest with icons, standalone display, theme color
- [ ] Service worker caches static assets, network-first for API
- [ ] "Install Aura" prompt shown to mobile users
- [ ] Bottom tab bar on mobile with 4 key tabs
- [ ] Safe area padding everywhere (notch, home indicator)
- [ ] Sidebar becomes full-screen drawer on mobile
- [ ] Pull-to-refresh in chat
- [ ] Swipe gestures on messages (actions) and between tabs
- [ ] Haptic feedback on key interactions
- [ ] Input bar: auto-resize, camera attachment, voice waveform
- [ ] Virtualized message list (handles 500+ messages smoothly)
- [ ] Offline mode: cached conversations, queued messages, clear status
- [ ] Conversation storage in IndexedDB (persists across sessions)
- [ ] Lighthouse mobile score > 90
- [ ] Telegram MiniApp shares code with main app
- [ ] No regressions on desktop
