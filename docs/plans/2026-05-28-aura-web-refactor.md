# Aura Web Refactor & Polish Implementation Plan

This document provides a detailed, step-by-step implementation plan for resolving identified performance risks, code decomposition issues, and architectural hotspots in the Aura Web React application (`D:/Aura/web`).

---

## Phase 1: Zustand Store Selectorization (Performance Fix)

### Overview
**Goal:** Eliminate unnecessary re-renders caused by non-selective Zustand store destructuring.
**Impact:** Components like `ChatContainer` and `MessageInput` currently re-render when unrelated store properties (e.g., `messages`) change.

### Steps & Commands

#### Task 1.1: `App.tsx` Selectorization

**Command:**
```bash
cd D:/Aura/web
npx tsc --noEmit -p tsconfig.json
```
**Expected Output:** `Found 0 errors.` (before changes)

**Implementation Detail:**
Replace the following destructured state access in `App.tsx`:

```typescript
const { sidebarOpen, setSidebarOpen, toggleSidebar, conversations, connectionStatus, isLoading } = useChatStore();
```

With atomic selectors:

```typescript
const sidebarOpen = useChatStore(s => s.sidebarOpen);
const setSidebarOpen = useChatStore(s => s.setSidebarOpen);
const toggleSidebar = useChatStore(s => s.toggleSidebar);
const conversations = useChatStore(s => s.conversations);
const connectionStatus = useChatStore(s => s.connectionStatus);
const isLoading = useChatStore(s => s.isLoading);
```

**Verification:**
```bash
npx tsc --noEmit -p tsconfig.json
```
**Expected Output:** `Found 0 errors.`

---

#### Task 1.2: `ChatContainer.tsx` Selectorization

**Command:**
```bash
grep -n "useChatStore()" src/components/ChatContainer.tsx
```
**Expected Output:** Lines showing generic hook usage.

**Implementation Detail:**
Replace generic `useChatStore()` calls with specific property selectors:

```typescript
const messages = useChatStore(s => s.messages);
const isLoading = useChatStore(s => s.isLoading);
```

**Verification:**
```bash
npx tsc --noEmit -p tsconfig.json
```

---

#### Task 1.3: `MessageInput.tsx` Selectorization

**Command:**
```bash
grep -n "useChatStore()" src/components/MessageInput.tsx
```

**Implementation Detail:**
Replace destructured state with atomic selectors for `messages`, `selectedModel`, `availableModels`, etc.

**Verification:**
```bash
npx tsc --noEmit -p tsconfig.json
```

### Success Criteria
- TypeScript compilation passes (`tsconfig.json`).
- No visual regressions in application routing or tab state.

---

## Phase 2: `App.tsx` Decomposition (Architecture Reorganization)

### Overview
**Goal:** Reduce `App.tsx` cyclomatic complexity and render surface by extracting authentication logic and tab routing.

### Steps

#### Task 2.1: Create `AuthGate.tsx`

**Command:**
```bash
touch src/components/AuthGate.tsx
```

**Content:**
Extract the authentication state (`auth`, `LoginPage`, `OnboardingFlow`) from `App.tsx`.
Move `AuthState`, session probing, and onboarding logic to this component.

**Implementation Detail:**
`AuthGate.tsx` receives children and renders them once authenticated.

**Verification:**
```bash
grep -n "AuthGate" src/App.tsx
```
**Expected Output:** `import { AuthGate } from './components/AuthGate';`

---

#### Task 2.2: Create `TabRouter.tsx`

**Command:**
```bash
touch src/components/TabRouter.tsx
```

**Content:**
Extract `renderTabContent`, `TABS`, `SubTabBar`, and all lazy imports.
This component manages active tab state and sub-navigation.

**Verification:**
```bash
npx tsc --noEmit -p tsconfig.json
```

---

#### Task 2.3: Refactor `App.tsx`

**Implementation Detail:**
Remove `AuthState` and `renderTabContent`.
Wrap `MainApp` with `<AuthGate>`.

**Success Criteria:**
- `App.tsx` under 150 lines.
- `npx tsc --noEmit` passes.

---

## Phase 3: `ChatContainer.tsx` Decomposition (Component Extraction)

### Overview
**Goal:** Extract distinct UI surfaces from `ChatContainer.tsx` to improve maintainability.

### Steps

#### Task 3.1: Extract `ChatToolbar.tsx` (Export/Share Menu)

**Command:**
```bash
touch src/components/ChatToolbar.tsx
```

**Content:**
`exportMenuOpen`, `ExportMenu`, `handleExportMarkdown`, `handleExportJSON`, `handleExportHTML`, `handleCopyConversation`, `handleShareLink`, `ShareModal` logic.

**Expected Lines in `ChatContainer.tsx`:** -120 lines

---

#### Task 3.2: Extract `InConversationSearch.tsx`

**Command:**
```bash
touch src/components/InConversationSearch.tsx
```

**Content:**
`searchOpen`, `searchQuery`, `searchIndex`, search navigation, and `filteredMessages` logic.

**Expected Lines in `ChatContainer.tsx`:** -100 lines

---

#### Task 3.3: Extract `ShareModal.tsx`

**Command:**
```bash
touch src/components/ShareModal.tsx
```

**Content:**
`shareModalOpen`, `shareUrl`, `shareLoading`, `shareCopied`, modal markup.

**Expected Lines in `ChatContainer.tsx`:** -60 lines

---

#### Task 3.4: Refactor `ChatContainer.tsx`

**Command:**
```bash
npx tsc --noEmit -p tsconfig.json
```
**Implementation Detail:**
Remove extracted logic; import and render child components.

---

## Phase 4: `apiFetch` Resilience & Timeout

### Overview
**Goal:** Add timeout and retry logic to `src/utils/apiFetch.ts` without changing its public signature.

### Steps

#### Task 4.1: Update `apiFetch.ts`

**Implementation Detail:**

```typescript
const DEFAULT_TIMEOUT = 15000; // 15 seconds

export function apiFetch(
  input: RequestInfo | URL,
  init?: RequestInit & { retry?: boolean }
): Promise<Response> {
  const controller = new AbortController();
  const signal = init?.signal;

  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);

  if (signal) {
    signal.addEventListener('abort', () => controller.abort());
  }

  return fetch(input, {
    credentials: 'include',
    ...(init ?? {}),
    signal: controller.signal,
  }).catch(async (error) => {
    clearTimeout(timeoutId);
    if (
      init?.retry !== false &&
      (error instanceof TypeError || (error as any)?.status === 503)
    ) {
      console.warn('[apiFetch] Retrying request...');
      return fetch(input, {
        credentials: 'include',
        ...(init ?? {}),
        signal: controller.signal,
      });
    }
    throw error;
  }).finally(() => {
    clearTimeout(timeoutId);
  });
}
```

**Verification:**
```bash
npx tsc --noEmit -p tsconfig.json
```

---

## Phase 5: `useIdleBehaviors` Mousemove Throttle

### Overview
**Goal:** Prevent `mousemove` from firing excessively (hundreds of times/sec) by throttling to 500ms using `requestAnimationFrame`.

### Steps

#### Task 5.1: Update `useIdleBehaviors.ts`

**Implementation Detail:**
In `recordActivity`, replace:

```typescript
recordActivity(); // Called on every mousemove
```

With a throttled version:

```typescript
let throttleTimer: number | null = null;
const throttledRecordActivity = () => {
  if (throttleTimer) return;
  throttleTimer = window.setTimeout(() => {
    throttleTimer = null;
    recordActivity();
  }, 500);
};
```

Update `window.addEventListener('mousemove', throttledRecordActivity)`.

**Verification:**
```bash
npx tsc --noEmit -p tsconfig.json
```

---

## Phase 6: Remove Dead Service Worker Stub

### Overview
**Goal:** Remove unused `navigator.serviceWorker.register` and `if ('serviceWorker'...` block from `main.tsx` if `sw.js` is not present in `public/`.

### Steps

#### Task 6.1: Verify `public/sw.js`

**Command:**
```bash
test -f D:/Aura/web/public/sw.js && echo "EXISTS" || echo "MISSING"
```
**Expected Output:** `MISSING`

#### Task 6.2: Edit `main.tsx`

**Implementation Detail:**
Remove:
```typescript
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}
```

**Verification:**
```bash
npx tsc --noEmit -p tsconfig.json
```

---

## Phase 7: Consolidation & Polish

### Steps

#### Task 7.1: Remove Unused Imports

**Command:**
```bash
npx eslint . --ext .ts,.tsx --fix
```
**Expected Output:** No unused variable/import errors after refactoring.

#### Task 7.2: Inline Style Audit

**Command:**
```bash
grep -n "style={{" src/components/ChatContainer.tsx
```
**Expected Output:** Significantly fewer `style={{` instances after Phase 3 decomposition.

**Detail:** Convert remaining inline style objects to Tailwind utility classes in `ThinkingHistory` and UI borders.

#### Task 7.3: Final TypeScript Check

**Command:**
```bash
npx tsc --noEmit -p tsconfig.json
```
**Expected Output:** `Found 0 errors.`

---

## Commit History (Recommended)

- `refactor: selectorize Zustand stores in App and ChatContainer`
- `refactor: extract AuthGate and TabRouter from App.tsx`
- `refactor: decompose ChatContainer into sub-components`
- `feat(apiFetch): add 15s timeout and retry logic`
- `perf(idle): throttle mousemove activity tracking to 500ms`
- `fix: remove dead service worker registration`
- `refactor: consolidate inline styles and remove dead code`

---

**Plan complete and saved to `D:/Aura/docs/plans/2026-05-28-aura-web-refactor.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration.
**2. Parallel Session (separate)** — Open new session with `executing-plans` skill, batch execution with checkpoints.

**Which approach?**
