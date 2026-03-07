# AURA Extension UI Polish Audit
**Date:** 2026-03-07
**Files modified:** `sidebar.html`, `sidebar.js`

---

## Approach

All improvements were added as a "Polish Layer v2" CSS block appended at the end of the existing `<style>` tag in `sidebar.html`. This means every override uses CSS specificity to overrule original declarations without deleting them — safe to diff and revert.

`sidebar.js` received one functional patch to the `switchPanel` function.

---

## Changes Made

### 1. CSS Variable Additions
Added to `:root`:
- `--r-sm/md/lg/pill` — radius scale tokens for consistent border-radius across all components
- `--b3` — extra-subtle border tint
- `--p3`, `--gr2`, `--rd2` — darker shades of primary, green, red for hover states
- `--sh-sm/md/lg` — elevation shadow scale

### 2. Panel Transition (NEW)
**Before:** Panels switched instantly with `display:none` / `display:flex`.
**After:** `@keyframes panelIn` with `opacity: 0 → 1` + `translateY(6px → 0)` using a `cubic-bezier(.22,.68,0,1.2)` spring curve (0.2s). Feels like Sider / Linear.

`sidebar.js` `switchPanel()` updated to use `requestAnimationFrame` so the animation replays on every switch rather than only on initial load.

### 3. Header Refinements
- Height bumped from 46px → 48px for better vertical breathing room
- Added iridescent gradient underline (`#hdr::after`) — purple to lavender, fades at edges
- Logo letter-spacing tightened, gap reduced to 6px for cleaner proportions
- Pulse dot animation replaced with `pulseGlow` — includes box-shadow breathing, not just opacity
- Model pill: hover state darkens background to `--s3`, transitions smoothly
- Online dot `.on` gains a double-ring glow using `box-shadow`
- `.ihbtn` gets `transform:scale(.9)` on `:active` for tactile feedback

### 4. Context Bar
- Background changed to proper CSS gradient using CSS vars, removing hardcoded `#13082c`/`#3a1d60`
- Added `ctxSlideIn` animation — slides down from above when shown
- Label font-size reduced to 9px, letter-spacing increased for sharper uppercase label
- Close button (ctx-x) gains a rounded hover background

### 5. Chat Empty State
- Avatar `#eav`: added outer ring glow (`box-shadow: 0 0 0 6px rgba(...)`) + hover scale animation
- Chips: added `transform:translateX(2px)` on hover (slide-in feel), box-shadow, active scale-down
- Chip SVG color transitions independently on hover

### 6. Message Bubbles
- `mrow` uses new `msgIn` keyframe — 6px translateY spring instead of 4px linear
- AI avatar gains double-ring glow matching logo pulse
- User bubble gets additional `0 0 0 1px` inner border shadow for depth
- AI bubble border lightens on `.mrow.ai:hover` — subtle reveal
- Timestamp font reduced to 9.5px

### 7. Reaction Bar
- `.rx` buttons get `transform:scale(1.08)` on hover, `scale(.95)` on active

### 8. Code Blocks
- Pre block gets `border-radius:9px` and `box-shadow:var(--sh-sm)`
- Code header darkened, font-size 9.5px, dimmer default color
- Copy button redesigned — transparent until hover, then purple tint
- Inline code gets purple tinted background `rgba(124,58,237,.12)` instead of near-black

### 9. Input Area
- Background updated to `rgba(6,6,18,.85)` — slightly lighter than before, better contrast
- `#inp-box` uses `var(--s1)` background instead of `--bg` (warmer, more depth)
- Focus state adds `box-shadow: 0 0 0 3px rgba(124,58,237,.1)` + larger outer shadow — standard ring pattern matching all other inputs
- Send button: gradient from `--p` to `--ub2`, drop shadow, `:active` scale, disabled uses `--s3` (dark, not white)
- Mode pills: font reduced to 10.5px, hover uses purple tint background

### 10. Rail Navigation (major)
**Before:** Active rail button was `background:var(--pg)` with no other indicator.
**After:**
- Active button (`.rbtn.on`) gets `::before` pseudo-element — a 2.5px × 18px vertical pill on the left edge, gradient `--pl → --p` with purple glow. Matches Linear/Arc style.
- Hover: `transform:scale(1.05)` — slight lift
- Active click: `transform:scale(.93)` — tactile press
- Rail background slightly lighter (`rgba(8,6,20,.78)`) with stronger blur
- Separators reduced to `rgba(255,255,255,.07)` — more subtle

### 11. Tooltips (improved)
- Tooltip `::after` adds `box-shadow:0 4px 16px rgba(0,0,0,.5)` for elevation
- Added `tipFadeIn` animation — fades in + slides from right by 4px
- Border uses `rgba(124,58,237,.25)` purple tint instead of `--b2`

### 12. Primary Action Buttons (ALL panels)
All primary buttons (`#sum-go`, `#res-go`, `#math-go`, `#write-submit`, `#send`, `#gr-btn`, `#ask-send`, `#wb-sbtn`, `#art-go`, `#img-gen`, `#ocr-capture`, `#pdf-send`, `#rec-whisper-btn`, `#mdl-save`, `#search-btn`, `#tr-btn`, `#pdf-auto-btn`):
- Now use `linear-gradient(135deg, var(--p), var(--ub2))` — consistent purple gradient
- All have `box-shadow:0 2px 10-14px rgba(124,58,237,.35-.4)` — lift effect
- Hover: `opacity:.88-.9` + `translateY(-1px/-2px)` — subtle float
- Active: `transform:none` (springs back)
- Disabled: `background:var(--s3)`, `box-shadow:none`, `opacity:.45`, `cursor:not-allowed`

### 13. Secondary Action Buttons
All secondary/ghost buttons (`#sum-to-chat`, `#sum-copy`, `#ocr-to-chat`, `#ocr-translate`, `#ocr-copy`, `#art-copy-code`, `#art-send-chat`, `#img-dl`, `#pdf-upload-btn`, `#wb-all`, `#mdl-reload`, `#rec-summarize`, `#rec-save-wb`, etc.):
- Consistent `border-radius:9px`
- Hover: `border-color:rgba(124,58,237,.4)`, `color:var(--pl)`, `background:rgba(124,58,237,.08)`, `translateY(-1px)`

### 14. Toggle Groups (mode pills, type buttons)
`.wtype`, `.sum-fmt`, `.gr-mode`, `.img-style`, `.res-d`, `.math-m`, `.ask-btn`:
- All unified to `border-radius:var(--r-pill)` and same transition timing
- `.on` state: `border-color:rgba(124,58,237,.55)`, `background:rgba(124,58,237,.15)` — more visible than before
- Hover: subtle background tint `rgba(255,255,255,.04)`

### 15. Input Fields (all panels)
All text inputs and textareas (`#ask-inp`, `#wb-inp`, `#gr-inp`, `#res-inp`, `#math-inp`, `#art-inp`, `#agent-task`, `#img-prompt`, `#img-neg`, `#pdf-inp`, `#cmp-inp`, `#yt-url-inp`):
- `border-radius:9-11px` (consistent)
- `transition:border-color .2s, box-shadow .2s`
- Focus: `border-color:rgba(124,58,237,.55)` + `box-shadow:0 0 0 3px rgba(124,58,237,.1)` ring

### 16. Search Input Box
`#search-box` and `#yt-input-box`:
- Background set to `var(--s1)` — warm dark instead of pure `--bg`
- Focus ring matches all other inputs

### 17. Tab Systems
`.wtab`, `.art-tab`:
- Active state font-weight bumped to 700 (was 600)
- Transitions added

### 18. Cards (result cards, source cards, WB cards)
`.src-card`, `.wb-card`, `.cmp-card`:
- `border-radius:9-12px`
- Hover: `translateX(2px)` for source/wb cards (directional hint)
- `cmp-card` gets elevation shadow

### 19. Tool Cards
- Hover lifts `translateY(-2px)` with increased shadow and a `0 0 0 1px rgba(124,58,237,.15)` inner border
- SVG stroke color transitions to full `--pl` on hover
- Active snaps back with fast transition override

### 20. Wisebase Empty State
- Color changed to `--di` (dimmer), more helpful padding, slightly larger line-height

### 21. Compare Panel Skeletons
- `shimmer` keyframe rewritten for smoother gradient scan
- Skeleton bar background uses `rgba(255,255,255,.05)` — CSS-variable safe

### 22. Scrollbars (ALL scroll containers)
All 16 scrollable containers now use:
- `scrollbar-width:thin; scrollbar-color:rgba(124,58,237,.18) transparent`
- `::-webkit-scrollbar{width:3px}` + thumb `rgba(124,58,237,.2)` with `border-radius:3px`
- This replaces the inconsistent mix of `var(--b1)` and hardcoded values

### 23. Focus Rings (global)
Added global `:focus-visible` rule for `button`, `input`, `textarea`, `select`:
- `outline:2px solid rgba(124,58,237,.55); outline-offset:2px`
- Uses `:focus-visible` not `:focus` — only shows on keyboard navigation, not mouse click

### 24. Status Text Lines
All small status/step-count elements unified to `font-size:11px; color:rgba(122,122,157,.65); letter-spacing:.01em`.

### 25. Context Bar Animation
`@keyframes ctxSlideIn` — slides down 5px + fades in when context bar appears.

### 26. Content Result Areas
All result areas (`.write-result`, `#gr-result`, `#sum-result`, etc.) that `.on` show via JS get `animation:contentIn .2s ease` — fade + 4px slide in.

### 27. Voice Recording
- Improved `recpulse` animation — extends glow ring to 8px (more dramatic)
- Button hover uses `--rd2` (darker red)

### 28. Agent Step Log
`.agent-step` border-bottom changed to `rgba(255,255,255,.05)` (CSS var compatible), font-size 11.5px.

### 29. PDF and YouTube Auto-detect Cards
Both use `border-radius:11px`, proper gradient background, animation on show.

### 30. Header Glow Pulse (logo dot)
Replaced `@keyframes pulse` (opacity only) with `pulseGlow` that also animates box-shadow — creates a breathing glow effect.

---

## What Was NOT Changed
- HTML structure — no elements added or removed
- JavaScript logic — only `switchPanel` function patched
- Inline styles on YouTube panel HTML (added `!important` overrides via CSS for `#yt-input-box`)
- Any color that was already correct in the original

---

## Audit Issues Found

| Issue | Severity | Fix Applied |
|---|---|---|
| Panel switch had no transition | High | panelIn animation + JS rAF patch |
| Hardcoded colors in `#ctx` background (`#13082c`, `#3a1d60`) | Medium | Replaced with CSS gradient |
| Rail active indicator was just a background tint (no accent bar) | Medium | Added `::before` pill indicator |
| Tooltip `::after` had no animation | Low | tipFadeIn keyframe added |
| Disabled buttons used `var(--b1)` (white transparent) — appeared broken | Medium | Changed to `var(--s3)` (dark) |
| `recpulse` animation had 0px initial shadow — jumpy appearance | Low | Rewrote animation |
| Focus rings were absent on most elements | High | Global `:focus-visible` rule |
| `.panel.active` set `display:flex` with no transition (snap) | High | animation override |
| Scrollbar colors inconsistent (mix of `--b1` and `--di`) | Low | Unified to purple tint |
| Code copy button was hard to see (styled like primary) | Low | Redesigned as ghost |
| Primary buttons had no elevation shadow | Medium | Added gradient + shadow |
| Send button disabled state used `var(--b1)` (invisible) | Medium | Fixed to `--s3` + reduced opacity |
