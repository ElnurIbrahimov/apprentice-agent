# Agent 8 — UX Consistency Audit Report

**Scope:** Full UX consistency audit across all 20 panels of the AURA Chrome extension sidebar.
**Files audited:** `extension/sidebar.html`, `extension/sidebar.js`
**Status:** All identified issues fixed directly in source files.

---

## Audit Categories & Findings

### 1. Loading States

**Chat / Search / Write / Ask / Wisebase / PDF / Summary / YouTube / Research / Math / Artifacts / Compare / Agent**
All panels that stream output use the `activeStream` pattern. The `submitBtn` field is set on the stream object and `finalizeStream()` re-enables it on completion or error. Loading is visually indicated by the button entering `:disabled` state during requests.

**Issues found:**
- **PDF panel** — `$('pdf-send').disabled = true` was missing before the `activeStream` assignment. The button was only disabled via the `submitBtn` field inside `activeStream`, meaning the button remained clickable until the first chunk arrived. **Fixed:** Added explicit `$('pdf-send').disabled = true` before the `activeStream` block in `sendPdfQuestion()`.
- **Voice/Rec summarize** — `submitBtn` was set to `null` in the stream object, preventing re-enable after completion. **Fixed (by concurrent agent):** Set to `$('rec-summarize')`.

All other panels with streaming correctly set `submitBtn` to the appropriate element before the stream begins.

---

### 2. Empty States

**Panels with results areas:**
- Chat: shows welcome message on load — OK
- Wisebase: shows empty prompt when no notes exist — OK
- PDF: shows "No PDF loaded" state — OK
- YouTube: shows idle state before URL entry — OK
- Compare: shows prompt to add items — OK
- Research: result area hidden until content arrives (`#res-result` display toggle) — OK
- Math: result area hidden until content arrives (`#math-result` display toggle) — OK
- Artifacts: editor starts empty, which is appropriate — OK

No critical empty state gaps found.

---

### 3. Error States

All streaming panels bubble errors through `finalizeStream()` which re-enables the submit button and can display error text in the result element. Error display is consistent across panels.

**Note:** Research panel error display relies on the result element being visible. The `#res-result.on` CSS class was missing its `display:block` rule. **Fixed (by concurrent agent):** Added `#res-result.on{display:block;animation:up .18s ease-out}`.

---

### 4. Disabled States (Buttons During Requests)

Buttons must visually indicate disabled state via CSS. Missing `:disabled` CSS rules found and fixed:

| Panel | Button | Fix Applied |
|-------|--------|-------------|
| Translate | `#tr-btn` | Added `:disabled{background:var(--b1);cursor:not-allowed;opacity:.55}` |
| Write | `#write-submit` | Added `:disabled{background:var(--b1);cursor:not-allowed;opacity:.55}` |
| Grammar | `#gr-btn` | Added `:disabled{background:var(--b1);cursor:not-allowed;opacity:.55}` |
| Artifacts | `#art-go` | Added `:disabled{background:var(--b1);cursor:not-allowed;opacity:.55}` |
| YouTube Summarize | `#yt-summarize-btn` | Added `:disabled{background:var(--b1);cursor:not-allowed;opacity:.55}`, removed `!important` from `:hover` |
| YouTube Auto | `#yt-auto-btn` | Added `:disabled{background:var(--b1);cursor:not-allowed;opacity:.55}`, removed `!important` from `:hover` |
| Rec/Voice Summarize | `#rec-summarize` | Added `:disabled{opacity:.45;cursor:not-allowed}` |

Panels already covered before this audit: Chat send, Search send, Write send, Ask send, Wisebase send, Agent submit.

**Note on `!important`:** The original YouTube button hover rules used `background:var(--p2) !important`. The `!important` flag overrides disabled state background styles. Removed to allow `:disabled` rules to take effect.

---

### 5. CSS Consistency — Design System Variables

Surveyed all panel CSS for hardcoded hex/rgb colors vs design system variables (`--bg`, `--s1`, `--s2`, `--s3`, `--b1`, `--b2`, `--p`, `--p2`, `--pl`, `--tx`, `--mu`, `--di`, `--gr`, `--rd`, etc.).

**Issues found:**
- `#yt-url-inp::placeholder` had no color rule — browser defaulted to system color. **Fixed:** Added `#yt-url-inp::placeholder{color:var(--di)}` to match all other textarea/input placeholder rules.
- Research result bottom margin was `margin:0 14px 0` (no bottom spacing). **Fixed:** Changed to `margin:0 14px 14px`.

All other surveyed panel elements use design system variables correctly. No hardcoded hex colors found in newly audited panels.

---

### 6. Missing `flex-shrink:0`

Icon/badge elements that could collapse under flex pressure. Surveyed all panel headers and inline icon usages. No critical `flex-shrink:0` omissions found beyond what concurrent agents had already addressed on toolbar nav items.

---

### 7. Overflow / Clip

Result areas in all panels that show streaming text use `overflow-y:auto` or `overflow:hidden` at the panel scroll container level. No clipping issues found that would hide content unexpectedly.

---

### 8. Panel Scroll

All panels use `.panel { overflow-y: auto }` inherited from the base panel class or define their own scroll container. Verified for: research, math, artifacts, compare, youtube. No panels found where content would overflow without scroll.

---

### 9. Missing Model Pill Slots

The `PILL_SLOTS` object in `sidebar.js` maps feature keys to `<span id="mdl-*">` elements in `sidebar.html`. `initModelPills()` inserts pill widgets into these slots at startup.

**Missing slots found and fixed:**

| Panel | HTML Slot Added | JS PILL_SLOTS Entry | JS FEATURE_DEFS Entry |
|-------|----------------|--------------------|-----------------------|
| OCR | `<span id="mdl-ocr">` in OCR panel header | `ocr: 'mdl-ocr'` | `{ key:'ocr', label:'OCR', icon:'🔲', desc:'Extract text from screen regions' }` |
| Image Generator | `<span id="mdl-image">` in Image panel header | `image: 'mdl-image'` | `{ key:'image', label:'Image Generator', icon:'🖼️', desc:'Text-to-image generation' }` |
| Artifacts | `<span id="mdl-artifacts">` in Artifacts panel header | `artifacts: 'mdl-artifacts'` | `{ key:'artifacts', label:'Artifacts', icon:'⌨️', desc:'Generate runnable code/HTML/SVG' }` |

All three panels now have their model pill headers structured as:
```html
<div style="display:flex;align-items:center;justify-content:space-between;...">
  <h2 style="margin:0">Panel Name</h2>
  <span id="mdl-panelname"></span>
</div>
```

---

### 10. Input Focus on Panel Switch

`switchPanel(name)` in `sidebar.js` was missing focus calls for most panels. Users switching panels would have no focused input, requiring a click before typing.

**Fixed — full focus map added to `switchPanel()`:**

| Panel | Focus Action |
|-------|-------------|
| `chat` | `inp.focus()` (was already present) |
| `search` | `searchInp.focus()` (was already present) |
| `translate` | `trInp.focus()` — **added** |
| `write` | `writeInp.focus()` — **added** |
| `grammar` | `setTimeout(() => $('gr-inp')?.focus(), 0)` — **added** |
| `ask` | `setTimeout(() => $('ask-inp')?.focus(), 0)` — **added** |
| `wisebase` | `loadWisebase(); setTimeout(() => $('wb-inp')?.focus(), 0)` — **added** |
| `models` | `loadModelPanel()` (no text input to focus) |
| `summary` | No auto-focus (user clicks Summarize to trigger) |
| `youtube` | `setTimeout(() => $('yt-url-inp')?.focus(), 0)` — **added** |
| `compare` | `initComparePanel()` (no single primary input) |
| `research` | `setTimeout(() => $('res-inp')?.focus(), 0)` — **added** |
| `math` | `setTimeout(() => $('math-inp')?.focus(), 0)` — **added** |
| `artifacts` | `setTimeout(() => $('art-inp')?.focus(), 0)` — **added** |
| `agent` | `setTimeout(() => $('agent-task')?.focus(), 0)` — **added** |

`setTimeout(..., 0)` is used for panels that may have conditional rendering to ensure the DOM is ready before focusing.

---

### 11. Keyboard Shortcuts (Enter / Ctrl+Enter)

Textareas that span multiple lines should use `Ctrl+Enter` to submit. Single-line inputs should use `Enter`.

**Missing keyboard handlers found and fixed:**

| Panel | Input | Shortcut Added |
|-------|-------|----------------|
| Translate | `#tr-inp` (textarea) | `Ctrl+Enter` → click `#tr-btn` |
| Grammar | `#gr-inp` (textarea) | `Ctrl+Enter` → click `#gr-btn` |

Both handlers follow the existing pattern used in Chat (`Ctrl+Enter` for multiline textarea submit):
```js
element.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); btn.click(); }
});
```

All other panels with text input were already covered before this audit.

---

### 12. Mobile / Narrow Layout

The sidebar is constrained to a fixed width Chrome extension panel (typically 380-420px). Panels were reviewed for overflow at narrow widths:

- Compare panel: dual-column layout collapses correctly
- YouTube panel: URL input box stretches to full width — OK
- Research panel: single-column layout — OK
- Math panel: single-column layout — OK
- Artifacts panel: CodeMirror/textarea fills width — OK

No critical narrow-layout breakage found.

---

### 13. Animation / Transition Consistency

Design system targets `.13s` transitions for interactive elements.

Surveyed button and element transitions across all panels. New panels (YouTube, Compare, Research, Math, Artifacts) use `transition:.13s` on buttons and inputs, consistent with existing panels.

**Typography — Panel Headers:**
Design system spec: panel `<h2>` headers should be `font-size:18px; font-weight:700`.

**Issues found and fixed:**
- Compare panel `<h2>` was `font-size:14px` — **Fixed (by concurrent agent):** Changed to `18px`.
- Artifacts panel title was `font-size:13px` — **Fixed (by concurrent agent):** Changed to `18px`.

All 20 panel headers now consistently use `18px/700`.

**Typography — Labels:**
10px uppercase labels verified in: Translate (language selector labels), Write (tone/style labels), Grammar (mode labels). Consistent.

---

## Summary Table

| Issue | Panels Affected | Status |
|-------|----------------|--------|
| Missing `:disabled` CSS for buttons | Translate, Write, Grammar, Artifacts, YouTube x2, Voice | Fixed |
| YouTube button `!important` on hover blocking disabled style | YouTube | Fixed |
| Missing `::placeholder` color for YouTube URL input | YouTube | Fixed |
| Missing `flex-shrink:0` on critical elements | None critical found | N/A |
| Missing model pill slots (HTML + JS) | OCR, Image, Artifacts | Fixed |
| Missing input focus on panel switch | Translate, Write, Grammar, Ask, Wisebase, YouTube, Research, Math, Artifacts, Agent | Fixed |
| Missing `Ctrl+Enter` keyboard shortcut | Translate, Grammar | Fixed |
| PDF send button not disabled before stream starts | PDF | Fixed |
| Voice summarize `submitBtn: null` | Voice/Rec | Fixed (concurrent agent) |
| Compare panel h2 wrong size (14px) | Compare | Fixed (concurrent agent) |
| Artifacts panel title wrong size (13px) | Artifacts | Fixed (concurrent agent) |
| `#res-result.on` missing `display:block` | Research | Fixed (concurrent agent) |
| `#math-result.on` missing `display:flex` | Math | Fixed (concurrent agent) |
| Research result missing bottom margin | Research | Fixed |

---

## Known Limitations / Out of Scope

- **YouTube panel inline styles:** The YouTube panel uses extensive inline styles on its result child elements. These are functional but not aligned with the design system CSS class approach. Refactoring these to CSS classes would improve maintainability but is a larger task beyond this audit's scope.
- **Compare panel keyboard navigation:** Tab order through comparison slots is not explicitly managed. Acceptable for current complexity.
- **Agent panel streaming:** The agent panel uses a different streaming mechanism than `activeStream`. Its button disable/enable logic was not modified in this audit; it appeared correct in the existing implementation.
- **OCR and Image panels:** These panels do not have streaming text output — they show results differently (image display, text block). The model pills added will allow model selection but the result display patterns were not changed.
