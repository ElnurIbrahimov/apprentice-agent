> SUPERSEDED 2026-04-13. Current source of truth: D:/Aura/CURRENT_STATE.md

# AURA Extension Panels — Dev Roadmap
> Making Artifacts, WebCreator, and Code panels state-of-the-art
> Created: 2026-03-21 | Status: Planning

---

## Current State

| Panel | Status | Key Gap |
|-------|--------|---------|
| **ArtifactsPanel** | Functional — 7 types, sandboxed iframe, live WS mode, quick starts | No streaming preview, no version history, full srcdoc replacement, no console |
| **WebCreatorPanel** | Functional — chat iteration, 7 templates, device preview, conversation history | No streaming preview, no visual element selection, no version timeline, no theme panel |
| **CodePanel** | Broken UX — execution is **faked** (LLM pretends to run code) | No real execution. Backend has real `code_executor.py` but panel doesn't use it |

---

## Phase 1: Real Code Execution (CodePanel)
*The biggest UX gap — code panel is currently useless for real work*

### 1.1 Pyodide Web Worker (Client-Side Python)
**What:** Run real Python in the browser via WebAssembly (Pyodide). Zero server cost, works offline.

**Architecture:**
```
CodePanel (React)
    │
    ├── User writes/generates code
    │
    ├── Import Router: parse imports
    │   ├── All Pyodide-compatible? → Web Worker
    │   └── Needs unsupported package? → Backend API
    │
    ├── Web Worker (Pyodide)
    │   ├── CPython 3.12 compiled to WASM (~25MB, cached after first load)
    │   ├── Monkey-patched sys.stdout → postMessage (streaming output)
    │   ├── Supports: numpy, pandas, matplotlib, scipy, scikit-learn, sympy
    │   ├── Stateful: variables persist between runs (like Jupyter)
    │   └── matplotlib → PNG/SVG via io.BytesIO → base64 data URL
    │
    └── Backend Fallback (/api/code/execute)
        ├── Uses existing code_executor.py sandbox
        ├── For packages Pyodide can't handle
        └── File I/O, networking, large datasets
```

**Pyodide-compatible packages (confirmed):**
- numpy, pandas, matplotlib, scipy, scikit-learn
- sympy, sqlite3, json, csv, re, math, statistics
- PIL/Pillow, seaborn (partial)

**NOT supported by Pyodide (route to backend):**
- torch, tensorflow, transformers
- requests (use pyodide.http.pyfetch instead)
- subprocess, socket, os.system

**Files to create/modify:**
- `extension-src/src/workers/pyodide-worker.ts` — NEW: Web Worker that loads and runs Pyodide
- `extension-src/src/panels/CodePanel.tsx` — MODIFY: Wire to real execution
- `api/routes/code.py` — CHECK: Ensure execution endpoint exists and returns structured output

**Output protocol (unified across client and server):**
```json
{"type": "stdout", "text": "Hello world\n"}
{"type": "stderr", "text": "Warning: ...\n"}
{"type": "image", "mime": "image/png", "data": "base64..."}
{"type": "html", "content": "<table>...</table>"}
{"type": "error", "traceback": "...", "ename": "ValueError", "evalue": "..."}
{"type": "result", "repr": "42", "type_name": "int"}
```

### 1.2 Rich Output Rendering
- **Matplotlib/charts:** Intercept `plt.show()`, render to PNG bytes → base64 data URL → `<img>` in output panel
- **DataFrames:** `df.to_html()` → sanitized HTML table with sort/filter. Or `df.to_json(orient='records')` → interactive table component
- **Images (PIL):** Convert to PNG bytes → base64 → inline `<img>`
- **Errors:** Syntax-highlighted tracebacks, clickable line references, "Fix Error" button sends code + error to LLM
- **MIME-type dispatch (Jupyter protocol):** Check `_repr_html_()` → HTML, `_repr_png_()` → image, `_repr_latex_()` → KaTeX, fallback `repr()` → preformatted text

### 1.3 Stateful Sessions (Variable Persistence)
- Keep Pyodide Web Worker alive between runs — variables from run 1 available in run 2
- After each execution, run introspection:
  ```python
  {name: {"type": type(val).__name__, "repr": repr(val)[:200], "shape": getattr(val, 'shape', None)}
   for name, val in globals().items() if not name.startswith('_')}
  ```
- Render collapsible **Variable Inspector** panel showing live namespace
- "Reset" button terminates and restarts the Web Worker
- Optional: persist key variables to IndexedDB via pickle for cross-session continuity

### 1.4 Auto-Install Missing Packages
- On `ModuleNotFoundError`, auto-detect package name
- Try `micropip.install('package_name')` in Pyodide
- If Pyodide can't install it → offer to re-run on backend
- Show progress: "Installing numpy... done (2.1s)"

### 1.5 Code Editor Upgrade
- Replace raw `<textarea>` edit mode with **CodeMirror 6** (lazy-loaded, ~75KB gzipped)
- Python syntax highlighting, bracket matching, auto-indent
- Only instantiate CM6 when user clicks "Edit" — code display stays as Prism.js highlighted `<pre>`

---

## Phase 2: Streaming Preview Infrastructure (All 3 Panels)
*The biggest perceived quality jump — users see content materialize in real-time*

### 2.1 Streaming Response Handler
**Current:** All 3 panels use `fetch` → `resp.json()` (waits for complete response)
**Target:** Stream tokens via SSE/fetch ReadableStream, update preview incrementally

```typescript
// Shared streaming utility
async function* streamChat(message: string, model: string, signal?: AbortSignal) {
  const resp = await fetch(`${HTTP}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({ message, model }),
    signal,
  });
  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield decoder.decode(value, { stream: true });
  }
}
```

**Backend requirement:** Ensure `/api/chat/stream` endpoint exists that streams SSE/chunked text. Check `api/routes/` for existing streaming support.

### 2.2 Smart Preview Controller (250ms Debounced)
```typescript
class StreamingPreviewController {
  private buffer = '';
  private timer: number | null = null;
  private lastGoodHTML = '';

  appendChunk(chunk: string) {
    this.buffer += chunk;
    if (!this.timer) {
      this.timer = setTimeout(() => {
        this.timer = null;
        if (this.isSafeToRender()) this.render();
      }, 250);
    }
  }

  private isSafeToRender(): boolean {
    // Not inside unclosed <script> or <style> tag
    // Not mid-HTML-tag (inside < without closing >)
    const html = this.buffer;
    const lastScriptOpen = html.lastIndexOf('<script');
    const lastScriptClose = html.lastIndexOf('</script>');
    if (lastScriptOpen > lastScriptClose) return false;
    const lastStyleOpen = html.lastIndexOf('<style');
    const lastStyleClose = html.lastIndexOf('</style>');
    if (lastStyleOpen > lastStyleClose) return false;
    return true;
  }

  private render() {
    // DOM morph or srcdoc update
    // Save scroll position, form state
    // Update iframe
    // Restore state
    this.lastGoodHTML = this.buffer;
  }

  onComplete() {
    clearTimeout(this.timer!);
    this.render(); // Final render with complete content
  }
}
```

### 2.3 DOM Morphing with Idiomorph (~4KB)
- Replace full `srcdoc` replacement with `Idiomorph.morph()` for incremental DOM updates
- Preserves: scroll position, form values, focus state, CSS transitions, running animations
- Falls back to full srcdoc replacement for complete type changes (e.g., HTML → Mermaid)

### 2.4 Preview Skeleton
- Show shimmer/skeleton placeholder immediately when generation starts
- Replace with real content on first successful render
- Stop button: `AbortController.abort()` → render whatever we have

### 2.5 Streaming Code Display
- Code pane shows tokens appearing in real-time with a blinking cursor
- Prism.js re-highlights on each chunk (fast enough for <10KB)
- Auto-scroll to bottom of code display

---

## Phase 3: Version History (Artifacts + WebCreator)
*Enables fearless iteration — try anything, restore anytime*

### 3.1 Version Model
```typescript
interface ArtifactVersion {
  id: string;
  timestamp: number;
  prompt: string;           // What the user asked for
  code: string;             // Generated code
  type: ArtifactType;       // html, react, svg, mermaid, chart, markdown
  thumbnail?: string;       // Base64 screenshot of preview (optional)
  parentId?: string;        // For branching/forking
}
```

### 3.2 Version Timeline UI
- Horizontal strip at bottom of panel showing version thumbnails
- Click any version to restore it
- Current version highlighted
- "Fork from here" creates a branch point
- Stored in `chrome.storage.local` keyed by session

### 3.3 Visual Diff
- Side-by-side code diff (green/red lines) between any two versions
- Uses a lightweight diff algorithm (~3KB) for line-based comparison
- Optional: pixel-diff of preview screenshots (stretch goal)

---

## Phase 4: Console Overlay (Artifacts)
*Essential for debugging generated code*

### 4.1 Console Interceptor (in iframe)
```javascript
// Injected into every artifact iframe
['log', 'warn', 'error', 'info'].forEach(method => {
  const original = console[method];
  console[method] = (...args) => {
    original(...args);
    parent.postMessage({
      type: 'console',
      level: method,
      args: args.map(a => {
        try { return JSON.stringify(a); }
        catch { return String(a); }
      }),
      timestamp: Date.now(),
    }, '*');
  };
});
```

### 4.2 Console Drawer UI
- Collapsible drawer at bottom of preview area
- Color-coded: log (white), warn (yellow), error (red), info (blue)
- Badge showing unread count when collapsed
- Clear button
- Filter by level

---

## Phase 5: Visual Element Selection (WebCreator)
*Click any element in the preview → AI edits just that element*

### 5.1 Element Highlighter (in iframe)
```javascript
// Injected into WebCreator preview iframe
document.addEventListener('mousemove', (e) => {
  // Remove previous highlight
  document.querySelector('.aura-highlight')?.classList.remove('aura-highlight');
  // Add highlight to hovered element
  e.target.classList.add('aura-highlight');
});

document.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  const el = e.target;
  parent.postMessage({
    type: 'element-selected',
    tagName: el.tagName,
    classes: el.className,
    id: el.id,
    text: el.textContent?.slice(0, 100),
    outerHTML: el.outerHTML.slice(0, 500),
    path: getElementPath(el), // e.g., "body > div.hero > h1"
  }, '*');
});
```

### 5.2 Selection UI
- Blue outline on hover (via injected CSS: `.aura-highlight { outline: 2px solid #3b82f6; outline-offset: 2px; }`)
- On click: floating toolbar appears with "Edit with AI", "Delete", "Duplicate"
- "Edit with AI" pre-fills chat input with: `Modify the ${tagName} element "${text}": `
- AI receives only that element's outerHTML + surrounding context, not the full page

---

## Phase 6: Syntax Highlighting Upgrade (All 3 Panels)
*Replace custom regex highlighting with proper multi-language support*

### 6.1 Prism.js Integration (~5KB gzipped)
- Languages: JavaScript, TypeScript, CSS, HTML, Python, JSON, Bash
- Line numbers via CSS counters
- Copy button with checkmark feedback (already exists, keep it)
- Word wrap toggle

### 6.2 CodeMirror 6 for Edit Mode (lazy-loaded, ~75KB)
- Only instantiate when user clicks "Edit" button
- Python mode for CodePanel
- HTML/CSS/JS mode for Artifacts/WebCreator
- Bracket matching, auto-indent, search

---

## Phase 7: Dynamic npm via esm.sh (Artifacts)
*Import any npm package on the fly*

### 7.1 Import Map in Iframe Shell
```html
<script type="importmap">
{
  "imports": {
    "react": "https://esm.sh/react@19",
    "react-dom/client": "https://esm.sh/react-dom@19/client",
    "recharts": "https://esm.sh/recharts",
    "lucide-react": "https://esm.sh/lucide-react",
    "framer-motion": "https://esm.sh/framer-motion",
    "three": "https://esm.sh/three",
    "d3": "https://esm.sh/d3"
  }
}
</script>
```

### 7.2 Auto-Detection of Unknown Imports
- Parse generated code for `import` statements
- If import not in pre-bundled map → dynamically add to import map via `esm.sh/{package}`
- Show brief "Loading framer-motion..." indicator
- Cache resolved packages in extension storage

### 7.3 Pre-Bundled Runtime (Artifact Iframe Shell)
- React 19 + ReactDOM
- Tailwind CSS (CDN)
- Chart.js 4
- Mermaid 11 (CDN, lazy)
- Error boundary wrapper
- Console interceptor
- postMessage bridge

---

## Phase 8: WebCreator Polish
*Theme panel, SEO preview, multi-page, detachable preview*

### 8.1 Theme Panel
- Color swatches: Primary, Secondary, Accent, Background, Text
- Click swatch → color picker → changes cascade via CSS variables
- "AI Suggest Palette" button generates harmonious alternatives
- Light/dark mode toggle

### 8.2 SEO Preview
- Google search result mockup (title, URL, description)
- Open Graph card preview (image, title, description)
- Character count indicators (title < 60, description < 155)
- "AI optimize" button generates SEO-friendly meta tags

### 8.3 Multi-Page Support
- Page list in sidebar header
- "Add Page" creates new page with shared header/footer/nav
- AI auto-updates nav links when pages added/removed
- Pages stored as separate HTML strings in state

### 8.4 Detachable Preview
- "Pop out" icon opens preview in new `window.open()` window
- Synced via `BroadcastChannel` — changes in sidebar reflect instantly
- When detached window closes, preview returns to inline
- Essential for the narrow sidebar constraint

### 8.5 Export Upgrades
- Download as clean HTML zip
- Open in CodeSandbox (POST to CodeSandbox API)
- Open in StackBlitz
- Copy shareable data URL (for small pages)

---

## Bundle Impact Analysis

| Addition | Size (gzipped) | Load Strategy |
|----------|----------------|---------------|
| Prism.js (6 languages) | ~5 KB | Initial load |
| Idiomorph (DOM morph) | ~4 KB | Initial load |
| Streaming utilities | ~2 KB | Initial load |
| Console interceptor | ~1 KB | Initial load |
| Version history UI | ~3 KB | Initial load |
| **Initial total** | **~15 KB** | |
| CodeMirror 6 (edit mode) | ~75 KB | Lazy on "Edit" click |
| KaTeX (math rendering) | ~95 KB + fonts | Lazy on first math block |
| Pyodide (Python WASM) | ~25 MB | Lazy on first code execution, cached |
| Mermaid (diagrams) | ~800 KB | CDN in iframe only |

**Core bundle increase: ~15KB.** Everything else is lazy-loaded.

---

## Implementation Order

| # | Phase | Effort | Impact | Why This Order |
|---|-------|--------|--------|----------------|
| 1 | Real Code Execution | Large | Critical | CodePanel is currently broken/fake |
| 2 | Streaming Preview | Medium | High | Biggest perceived quality jump, benefits all 3 panels |
| 3 | Version History | Medium | High | Enables fearless iteration |
| 4 | Console Overlay | Small | Medium | Essential for Artifacts debugging |
| 5 | Visual Element Selection | Medium | High | WebCreator's killer feature |
| 6 | Prism.js Upgrade | Small | Medium | Better DX across all panels |
| 7 | Dynamic npm (esm.sh) | Small | Medium | Unlocks any library in Artifacts |
| 8 | WebCreator Polish | Large | Medium | Theme, SEO, multi-page, detach |

---

## Research Sources

- **Pyodide** — CPython 3.12 compiled to WASM, supports numpy/pandas/matplotlib, runs in Web Worker
- **Claude Artifacts** — Sandboxed iframe, pre-bundled React, error boundaries, no streaming preview
- **v0 by Vercel** — Streaming preview with 200-300ms debounce, shadcn/ui pre-bundled, esm.sh for dynamic imports
- **Bolt.new** — WebContainers (full Node.js in browser), real HMR, multi-file projects
- **ChatGPT Code Interpreter** — Server-side sandboxed execution, stateful sessions, rich output (images, tables, files)
- **Lovable** — Click-to-edit visual selection, Supabase integration, GitHub sync
- **Idiomorph** — ~4KB DOM morphing library, preserves state during updates
- **Prism.js** — ~5KB syntax highlighter, 200+ languages, sufficient quality for sidebar
- **CodeMirror 6** — ~75KB modular editor, best option for embedded editing (Monaco is 4MB+, too heavy)
- **esm.sh** — CDN that serves any npm package as ESM, works with import maps
- **JupyterLite** — Pyodide + Web Worker architecture reference for browser-based Python
- **Streamdown (Vercel)** — Streaming markdown renderer with lazy Shiki highlighting

---

## What NOT To Do

1. **Don't use Monaco Editor** — 4MB+ bundle, designed for full IDEs, overkill for sidebar
2. **Don't use WebContainers** — Requires COOP/COEP headers, Service Workers, conflicts with extension architecture
3. **Don't render Mermaid client-side in the main bundle** — 800KB+. Load via CDN inside iframe only
4. **Don't stream preview on every token** — Causes jank and error flashes. 250ms debounce minimum
5. **Don't try to support all npm packages in Pyodide** — Route unsupported packages to backend
6. **Don't build collaborative editing yet** — Turn-based (lock during generation) is sufficient for v1
7. **Don't add Shiki** — 280KB+ for slightly better highlighting. Prism.js is 95% as good at 1/50th the size

---

*This roadmap turns three functional panels into three extraordinary ones. Phase 1 (real code execution) alone would make AURA's Code panel competitive with ChatGPT's Code Interpreter. Phases 1-4 together would put the extension ahead of every browser AI sidebar on the market.*