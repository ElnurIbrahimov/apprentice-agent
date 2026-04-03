# Phase 7: Performance + Polish

**Effort:** 2 days
**Impact:** Faster load, smoother UX, reduced server load

---

## 7A. Reduce Polling (8+ intervals → 2-3)

### Problem
At any time, 8+ concurrent `setInterval` timers are firing:
- `/api/status` (30s)
- `/api/thoughts` (10s)
- `/api/aura` (15s)
- `/api/alma/state` (30s)
- `/api/introspection/recent` (30s)
- `/api/activity/events` (3s)
- `/api/amem/stats` (10s)
- `/api/neurodream` (15s)
- `/api/proactive/messages` (15s)

This creates unnecessary network chatter and server load, especially when most tabs aren't visible.

### Solution: Tab-Aware Polling

Only poll endpoints relevant to the **active tab**:

| Tab | Needed Endpoints | Interval |
|-----|-----------------|----------|
| Chat | status, proactive | 30s, 15s |
| Monitoring | thoughts, aura, alma, introspection | 10s, 15s, 30s, 30s |
| Tools | status (once) | none |
| Advanced | amem, neurodream | 10s, 15s |
| Activity | activity/events | 3s |
| Settings | none | none |

**Implementation:**
```typescript
// In usePolling, accept a list of active endpoints
function useSmartPolling(activeTab: string) {
  const endpointsForTab = POLLING_CONFIG[activeTab] || [];
  // Only start intervals for endpoints in the list
  // Cleanup intervals when tab changes
}
```

Also:
- Pause ALL polling when browser tab is hidden (`document.hidden`)
- Resume only the active-tab endpoints when visible again
- Already partially done in `usePolling` — extend to tab awareness

### Combine Endpoints (Backend Change)
Create a single `/api/dashboard` endpoint that returns status + mood + thoughts + aura in one response:
```python
@router.get("/api/dashboard")
async def dashboard_state():
    return {
        "status": get_status(),
        "mood": get_mood(),
        "thoughts": get_recent_thoughts(5),
        "energy": get_energy_state(),
    }
```
One 30s poll replaces four separate ones for the monitoring tab.

---

## 7B. Drop framer-motion

### Problem
`framer-motion` is ~12MB unpacked, 150KB+ gzipped in the bundle. Usage is minimal — a few `AnimatePresence`/`motion.div` entrance animations that can be replaced with CSS.

### What To Replace
Search for all `import.*framer-motion` and `motion\.` usage:
- `AnimatePresence` → CSS `@keyframes` with conditional rendering
- `motion.div` with `initial`/`animate`/`exit` → CSS classes with `animate-*` (already using Tailwind animations elsewhere)
- `motion.button` → regular button with hover/active CSS

### Example Replacement
```tsx
// Before (framer-motion)
<motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
  {content}
</motion.div>

// After (CSS)
<div className="animate-fade-in-up">
  {content}
</div>
```

Add to Tailwind config:
```js
animation: {
  'fade-in-up': 'fadeInUp 0.2s ease-out',
  'fade-out': 'fadeOut 0.15s ease-in',
}
```

### After removing
```bash
npm uninstall framer-motion
```
Bundle size reduction: ~150KB gzipped.

---

## 7C. Lazy Load Heavy Components

### Problem
All tabs load eagerly even if user only uses Chat.

### What To Build
```typescript
// In App.tsx
const ReasoningTreePanel = React.lazy(() => import('./components/ReasoningTreePanel'));
const NeuroDreamPanel = React.lazy(() => import('./components/NeuroDreamPanel'));
const AMEMPanel = React.lazy(() => import('./components/AMEMPanel'));
const ActivityTimeline = React.lazy(() => import('./components/ActivityTimeline'));
const SettingsPage = React.lazy(() => import('./components/SettingsPage'));
const ToolsPanel = React.lazy(() => import('./components/ToolsPanel'));
// New panels from Phase 3-6:
const CodeInterpreter = React.lazy(() => import('./components/CodeInterpreter'));
const ArtifactsPanel = React.lazy(() => import('./components/ArtifactsPanel'));
const ImageGenPanel = React.lazy(() => import('./components/ImageGenPanel'));
```

Wrap in `<Suspense fallback={<TabSkeleton />}>`.

### Tab Skeleton
Simple loading placeholder matching the tab's expected layout:
```tsx
function TabSkeleton() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="animate-pulse flex flex-col gap-3 w-64">
        <div className="h-4 bg-surface-2 rounded w-3/4" />
        <div className="h-4 bg-surface-2 rounded w-1/2" />
        <div className="h-4 bg-surface-2 rounded w-5/6" />
      </div>
    </div>
  );
}
```

---

## 7D. Responsive Settings Page

### Problem
Settings two-column layout (w-48 nav + content) breaks on narrow screens.

### What To Build
- Below 640px: stack nav and content vertically
- Nav becomes horizontal scrollable pill bar
- Or: nav becomes a dropdown selector
- Content area gets full width

```tsx
// Settings layout
<div className="flex flex-col sm:flex-row h-full">
  {/* Nav — horizontal on mobile, vertical on desktop */}
  <nav className="flex sm:flex-col gap-1 overflow-x-auto sm:overflow-visible sm:w-48 border-b sm:border-b-0 sm:border-r border-chat-border p-2">
    {sections.map(s => (
      <button key={s.id} className="whitespace-nowrap sm:whitespace-normal ...">{s.label}</button>
    ))}
  </nav>
  <div className="flex-1 overflow-auto p-4">{/* content */}</div>
</div>
```

---

## 7E. Remove Custom Event Anti-Pattern

### Problem
Cross-component communication uses `window.dispatchEvent(new CustomEvent('aura:new-chat'))` instead of store actions. This is brittle and untraceable.

### What To Build
Replace all custom events with store actions:
- `aura:new-chat` → `chatStore.getState().newConversation()`
- `aura:focus-input` → `chatStore.getState().setFocusInput(true)` + `useEffect` in MessageInput
- `aura:switch-tab` → `chatStore.getState().setActiveTab(tab)` (add to store)

Search for all `dispatchEvent` and `addEventListener` with `aura:` prefix and replace.

---

## 7F. Deduplicate Model Picker

### Problem
Model selection dropdown is copy-pasted between Sidebar and MessageInput with separate implementations.

### What To Build
Extract to `web/src/components/ModelPicker.tsx`:
```typescript
interface ModelPickerProps {
  compact?: boolean;  // MessageInput uses compact, sidebar uses full
  position?: 'above' | 'below';
}
```
Use in both Sidebar and MessageInput. Single source of truth.

---

## Definition of Done — Phase 7
- [ ] Polling is tab-aware — only polls endpoints needed by active tab
- [ ] All polling pauses when browser tab is hidden
- [ ] `/api/dashboard` combined endpoint (optional backend change)
- [ ] framer-motion removed, animations replaced with CSS
- [ ] Bundle size reduced by ~150KB gzipped
- [ ] Heavy components lazy-loaded with Suspense
- [ ] Tab skeleton loading state
- [ ] Settings page responsive on narrow screens
- [ ] Custom events replaced with store actions
- [ ] Model picker extracted to shared component
- [ ] No polling regressions (all data still updates correctly)
