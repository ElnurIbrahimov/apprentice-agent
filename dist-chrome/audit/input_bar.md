# Input Bar Redesign — Audit Report

**Date:** 2026-03-07
**Files modified:** `extension/sidebar.html`

## What changed

### HTML (sidebar.html ~line 1515)

Replaced the old `#inp-area > #inp-box > #inp-top + #modes` structure with a new unified `#inp-bar` component:

```
#inp-bar
  #inp-meta          ← top row
    #inp-toggles
      #m-think.inp-tog
      #m-research.inp-tog
      #m-page.inp-tog
    #mdl-chat          ← model pill (was inside #modes)
  #inp-row           ← bottom row
    #inp (textarea)
    #send (button)
```

Old layout had: textarea + send on top, model + mode pills below.
New layout has: mode toggles + model on top, textarea + send below — matching Sider's unified bar pattern.

### CSS changes

**Base styles (replaced ~lines 262-283):**
- Removed: `#inp-area`, `#inp-box`, `#inp-box:focus-within`, `#inp-top`, `#modes`, `.mpill` and variants
- Added: `#inp-bar`, `#inp-bar:focus-within`, `#inp-meta`, `#inp-toggles`, `.inp-tog` and variants, `#inp-row`

**Dark theme overrides (replaced ~lines 968-1003):**
- Removed: dark overrides for `#inp-area`, `#inp-box`, `#inp-top`, `#modes`, `.mpill`
- Added: dark overrides for `#inp-bar`, `#inp-bar:focus-within`, `.inp-tog` hover/on states, `#send` gradient

**Media query (max-width:320px):**
- Replaced `#inp-area{padding-left/right:8px}` and `#inp-top{padding:8px}` with `#inp-bar{margin-left/right:6px}` and `#inp-row{padding:5px 6px 7px}`

## ID compatibility — no JS changes needed

All IDs referenced in sidebar.js are preserved:
- `$('inp')` → `<textarea id="inp">` — unchanged
- `$('send')` (as `sendBtn`) → `<button id="send">` — unchanged
- `$('m-think')`, `$('m-research')`, `$('m-page')` — unchanged IDs, now carry class `inp-tog` instead of `mpill`
- `$('mdl-chat')` — unchanged, moved from inside `#modes` to inside `#inp-meta`

## What was NOT changed

- sidebar.js — zero modifications required
- `#mpill` (connection status pill in header) — separate element, unaffected
- PDF panel `#pdf-inp-area` — separate element, unaffected
- All other panels — unaffected
- Mode toggle logic (`thinkingMode`, `deepResearch`, `.classList.toggle('on')`) — works unchanged since it targets element IDs
