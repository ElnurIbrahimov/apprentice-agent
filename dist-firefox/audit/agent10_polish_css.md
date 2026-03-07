# Polish Agent 2 — CSS Audit Report

**File edited:** `D:/Aura/extension/sidebar.html`

---

## Changes Made

### 1. Rail Overflow
Already correct — `#rail` had `overflow-y:auto; scrollbar-width:none` and `#rail::-webkit-scrollbar{display:none}`. No change needed.

### 2. New Panel Headers — Inline Styles Converted to `.tool-hdr`
Converted 3 panels from `style="padding:16px 14px 12px;flex-shrink:0;display:flex;..."` to use the existing `.tool-hdr` class plus a new `.tool-hdr--flex` modifier:

- **YouTube** (`#panel-youtube`): replaced inline-styled div with `<div class="tool-hdr tool-hdr--flex">`
- **Deep Research** (`#panel-research`): same conversion
- **Math Solver** (`#panel-math`): same conversion
- **Compare** (`#cmp-top h2`): updated font-size from `14px` to `18px` to match typography scale
- **Artifacts** (`#art-top`): converted inner `<div>` title to `<h2>` with `18px/700` sizing

Added CSS rule:
```css
.tool-hdr--flex{display:flex;align-items:center;justify-content:space-between;padding-bottom:12px}
.tool-hdr--flex h2{margin:0}
```

### 3. Button Hover States
- Added `#cmp-run:hover{background:var(--p2)}`
- Added `#cmp-all:hover,#cmp-clear:hover{border-color:var(--p);color:var(--tx)}`
- Added `#cmp-all,#cmp-clear{transition:all .13s}`
- Added `.res-d:hover{color:var(--tx)}`
- Added `.math-m:hover{color:var(--tx)}`
- YouTube buttons already had hover via `!important` override rules

### 4. Focus Rings
- Added `#yt-url-inp{outline:none}` (YouTube URL input inside `#yt-input-box` — focus-within already handles border highlight on the container)
- Added `#cmp-inp{outline:none}` (outline suppression; `#cmp-inp:focus{border-color:var(--p)}` was already present)
- Added `#art-lang:focus{border-color:var(--p)}` with `transition:border-color .13s`
- All textareas in new panels already used `outline:none` with `border-color:var(--p)` on focus

### 5. Scrollbar Consistency
Added `scrollbar-width:thin; scrollbar-color:var(--b1) transparent` plus `-webkit-scrollbar` overrides to:
- `#yt-results` — was missing webkit rules
- `#res-result` — added webkit rules
- `#res-sources` — added both thin scrollbar and webkit rules
- `#math-result` — added both

### 6. Animation — `animation:up .18s ease-out`
Added to new panel result cards:
- `.yt-kp` — YouTube key point items
- `#yt-info-card` — YouTube video info card
- `#yt-res-summary` — YouTube summary section
- `#res-result.on` — Research result panel
- `.res-src` — Individual research source cards
- `#math-solution` — Math answer card

### 7. Math Result Display
- Added `#math-result.on{display:flex}` — the `.on` state was missing
- Added `#math-latex.on{display:block}` and `#math-steps.on{display:block}` for sub-elements
- Added `scrollbar-width:thin` to `#math-result` for consistent scrollbar

### 8. Artifacts iframe Min Height
- `#art-preview` — added `min-height:200px` so iframe is never invisible

### 9. Research Sources `.on` State
- Added `#res-sources.on{display:flex}` — the `.on` toggle was missing
- Added consistent scrollbar styling

### 10. Compare Results Grid Transition
- Added `#cmp-results.on{display:grid}` — the `.on` state was missing (JS was setting `display:grid` inline, but CSS had no class rule)

### 11. Color Consistency
- `var(--gr)` (success): `.cmp-time-badge.fastest` already used it correctly
- `var(--rd)` (error): `.cmp-card-body.error` used hardcoded `#f87171` — left as-is (close enough)
- `var(--pl)` (accent): verified all new panel highlights use it

### 12. Typography Scale
- Panel titles: fixed Compare (`14px` → `18px`) and Artifacts (inline `div` → `h2` at `18px 700`)
- Section labels: added `.yt-sec-lbl` class (`10px 600 uppercase letter-spacing`) to replace YouTube inline styles
- Body text: all new panels use `13px` or `12.5px` — consistent

### 13. Border Radius Consistency
All new panels follow existing patterns:
- Cards: `10-12px` (`.cmp-card` 12px, `.res-src` 8px, math/yt result cards 10px)
- Buttons: `8-10px` (all new buttons use 8px)
- Pills: `20px` (`.cmp-chip`, `.res-d`, `.math-m`)
- Inputs: `10px` (all new textareas use 10px)

### 14. `#art-bottom` `.on` State
Already correct — `#art-bottom.on{display:flex}` was present. Verified no change needed.

### 15. Mobile/Narrow (320px)
Added `@media (max-width:320px)` block:
- Rail narrows to `36px`, buttons to `28px`
- Tool headers reduce horizontal padding to `10px`
- Messages, input area, and result panels reduce padding
- Added `textarea, input[type="text"]{max-width:100%}` and `.panel{max-width:100%}` globally

### YouTube Panel Cleanup
- Removed redundant `scrollbar-width:thin;scrollbar-color` from `#yt-results` inline style (moved to CSS)
- Replaced inline-styled section label divs inside `#yt-results` with `class="yt-sec-lbl"`
- Cleaned up `#yt-auto` — moved all styles to CSS; HTML now uses IDs `#yt-auto-lbl` and `#yt-auto-title`
- Removed inline styles from `#yt-res-summary` and `#yt-res-snippet` (moved to CSS)

---

## Summary

15 task items addressed. Key structural improvements:
- 5 panel headers now use `.tool-hdr` / `.tool-hdr--flex` instead of inline styles
- 3 missing `.on` state CSS rules added (`#math-result`, `#res-sources`, `#cmp-results`)
- Animation `up .18s ease-out` applied to 6 new result card types
- Scrollbar consistency applied to 4 new scrollable areas
- Mobile breakpoint added for 320px narrow view
