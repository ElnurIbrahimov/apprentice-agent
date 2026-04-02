# Phase 1: Quick Wins — Error Recovery, Persistence, Gallery

**Effort:** 1-2 days
**Impact:** Fixes the most frustrating UX problems immediately

---

## 1A. AI Error Recovery Loop

### Problem
When generated HTML/React/JS code throws an error in the iframe, the user sees it in the console panel but the AI has no idea. The user has to manually describe the error. v0.dev and Bolt.new auto-detect and auto-fix.

### What Already Works
- Iframe errors are captured via `window.onerror` and `unhandledrejection` in `buildSrcdoc()` (ArtifactsPanel)
- Errors are relayed to parent via `postMessage({ type: 'artifact-error', ... })`
- Console logs are already captured (up to 100 messages)
- CodePanel already has a "Fix Error" button for Python errors

### What To Build

**ArtifactsPanel** — `extension-src/src/panels/ArtifactsPanel.tsx`:
1. Add state: `autoFixAttempts: number` (max 3), `isAutoFixing: boolean`
2. When `artifact-error` message is received AND `autoFixAttempts < 3`:
   - Set `isAutoFixing = true`
   - Build a repair prompt:
     ```
     The code you generated has an error:
     
     ERROR: {error.message}
     LINE: {error.lineno}
     
     Current code:
     ```{type}
     {currentCode}
     ```
     
     Fix the error and return the complete corrected code.
     ```
   - Send to `/api/generate/raw` with the same system prompt + conversation history
   - On response: replace code, re-render, increment `autoFixAttempts`
   - Show a subtle toast: "Auto-fixing error (attempt 1/3)..."
3. If still broken after 3 attempts: show the error normally + "Fix manually" option
4. Reset `autoFixAttempts` on any new user prompt

**WebCreatorPanel** — `extension-src/src/panels/WebCreatorPanel.tsx`:
- Same pattern but simpler — inject error interception into the iframe (it may not have it yet)
- On error: append error context to conversation history, auto-regenerate

### UI
- Small animated indicator in the preview toolbar: "Fixing..." with a spinning icon
- Error count badge on the console button
- Toggle in settings: "Auto-fix errors" (default ON)

---

## 1B. Artifact Persistence

### Problem
ArtifactsPanel loses all content on panel switch or extension restart. WebCreatorPanel already persists to `chrome.storage.local` — Artifacts doesn't.

### What To Build

**File:** `extension-src/src/panels/ArtifactsPanel.tsx`

1. On every code change, save to `chrome.storage.local`:
   ```typescript
   const STORAGE_KEY = 'aura_artifacts_state';
   
   // Save (debounced, 1s)
   chrome.storage.local.set({
     [STORAGE_KEY]: {
       code: currentCode,
       type: currentType,
       messages: conversationHistory,  // if any
       activeFile: currentFilename,    // for live mode
       timestamp: Date.now()
     }
   });
   ```

2. On mount, restore from storage:
   ```typescript
   useEffect(() => {
     chrome.storage.local.get(STORAGE_KEY, (data) => {
       if (data[STORAGE_KEY]) {
         setCode(data[STORAGE_KEY].code);
         setType(data[STORAGE_KEY].type);
         // etc.
       }
     });
   }, []);
   ```

3. Add a "Clear" button to reset state

---

## 1C. Artifact Gallery

### Problem
Users create artifacts but can't save, name, browse, or revisit them. Each generation replaces the previous one. Version history helps but is limited to 20 unnamed versions.

### What To Build

**New file:** `extension-src/src/utils/artifactGallery.ts`

```typescript
interface SavedArtifact {
  id: string;           // nanoid
  name: string;         // user-provided or auto-generated
  type: ArtifactType;
  code: string;
  thumbnail?: string;   // base64 screenshot of the iframe (optional)
  createdAt: number;
  updatedAt: number;
  tags?: string[];
}

const GALLERY_KEY = 'aura_artifact_gallery';
const MAX_ARTIFACTS = 50;
const MAX_CODE_SIZE = 100_000; // 100KB per artifact

export const galleryStore = {
  async list(): Promise<SavedArtifact[]> { ... },
  async save(artifact: Omit<SavedArtifact, 'id' | 'createdAt' | 'updatedAt'>): Promise<string> { ... },
  async get(id: string): Promise<SavedArtifact | null> { ... },
  async delete(id: string): Promise<void> { ... },
  async update(id: string, updates: Partial<SavedArtifact>): Promise<void> { ... },
};
```

**UI in ArtifactsPanel:**
- "Save" button in toolbar → modal to name the artifact
- "Gallery" button → grid view of saved artifacts with thumbnails
- Click to load, long-press/right-click to delete or rename
- "Fork" option: load a saved artifact as a new starting point
- Search/filter by type and tags

**Thumbnail generation:**
- After render completes, use `html2canvas(iframeDoc.body)` or capture via `iframe.contentWindow` to get a small preview image
- Store as compressed base64 (max 20KB per thumbnail)
- Fallback: colored placeholder with artifact type icon

---

## 1D. Console Panel Improvements

### Problem
Console panel exists but is basic — just a scrollable list. No filtering, no clearing, no log levels.

### What To Build
- Filter buttons: All | Errors | Warnings | Info
- Clear button
- Click-to-copy on any log entry
- Error entries are expandable (show full stack trace)
- Badge count on console button showing unread error count
- Auto-open console on first error (configurable)

---

## Definition of Done — Phase 1
- [ ] Artifact JS errors auto-trigger up to 3 fix attempts before showing to user
- [ ] WebCreator JS errors auto-trigger fix attempts
- [ ] ArtifactsPanel state persists across panel switches and extension restarts
- [ ] Users can save, name, browse, load, and delete artifacts in a gallery
- [ ] Console panel has filtering, clearing, and error count badge
- [ ] All new features have a settings toggle to disable
