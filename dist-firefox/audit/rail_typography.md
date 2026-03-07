# Rail + Typography Polish Audit
**Date:** 2026-03-07
**Files modified:** `extension/sidebar.html`, `extension/sidebar.js`

---

## Task 1: Rail button text labels

All 18 rail buttons now have `<span class="rail-lbl">Label</span>` beneath their SVG icons.

**Always-visible buttons (top 8 + More):**
Chat, Ask, Search, Wisebase, Translate, Write, Voice, More

**Hidden behind More (13 extra):**
Grammar, OCR, PDF, Summary, YouTube, Research, Math, Image, Artifacts, Compare, Agent, Models, Tools

**Bottom utility:**
Clear (rail-clear)

The `rbtn` class was updated to `flex-direction: column` with `gap: 3px` and `padding: 7px 4px` so the icon + label stack cleanly. Rail width increased from 44px to 58px.

---

## Task 2: More toggle

A `#rail-more` button is placed at position 9 (after the separator following Voice). All extra panels are wrapped in `<div class="rail-extra">` which starts collapsed (`max-height: 0; opacity: 0`).

Clicking More toggles `.open` on both `#rail-more` and `.rail-extra`, triggering the CSS transition (`max-height: 600px; opacity: 1` at 0.25s).

JS added to `sidebar.js` at line ~344 (after `$('rail-clear')` listener):
```js
$('rail-more')?.addEventListener('click', () => {
  const open = $('rail-more').classList.toggle('open');
  document.querySelectorAll('.rail-extra').forEach(el => el.classList.toggle('open', open));
});
```

The existing `switchPanel()` function and `.rbtn[data-panel]` event delegation still work unchanged because all `data-panel` attributes are preserved. The extra buttons inside `.rail-extra` are fully functional when expanded.

---

## Task 3: Typography polish

Added to the CSS block:

| Rule | Purpose |
|---|---|
| `.panel h2` | 17px/700/−0.3px tracking, overrides individual panel h2s consistently |
| `.sect-lbl` | 10px/600/0.06em uppercase — reusable section label class |
| `.panel p, .panel .result-text` | 13px/1.6 body text baseline |
| `.panel .meta` | 11px/1.4 secondary/meta text |
| `.panel code, .panel pre` | JetBrains Mono > Fira Code > Cascadia Code stack, 12px |

These are additive — they don't override the many specific rules already in the file; the panel h2 rule does unify existing per-panel h2 declarations.

---

## Task 4: Page summarize bar

**HTML** added inside `#panel-chat` between the context bar and `#msgs`:
```html
<div id="page-sum-bar" style="display:none">
  <span id="page-sum-title"></span>
  <button id="page-sum-btn">Summarize</button>
</div>
```

**CSS:** Purple-tinted pill bar (`rgba(124,58,237,.07)` background, `rgba(124,58,237,.15)` border), 10px radius. Title truncates with ellipsis, Summarize button is a ghost button in `--p` color.

**JS logic (`sidebar.js`):**
- `showPageSumBar(title)` — called after `loadPage()` succeeds (both full text and tab-only fallback). Shows bar with page title.
- `hidePageSumBar()` — called from `clearCtx()` (user dismisses context) and from the Summarize button click handler.
- `$('page-sum-btn')` click fires `sendMessage('Summarize this page for me')` then hides the bar.

---

## Notes / caveats

- The `rail-extra` div uses CSS `overflow:hidden` + `max-height` transition (not `display:none` toggle) so the animation works smoothly. The `.rail-extra` is always in the DOM flow but collapsed.
- `Wisebase` label is 8 chars and fits at 52px max-width at 9px font — verified fits without truncation.
- `data-tip` tooltips on rail buttons still work (the `::after` pseudo-element is positioned relative to `.rbtn` which keeps `position:relative`).
- The narrow breakpoint at `max-width:340px` drops rail to 48px and labels to 8px as a fallback.
- `btn-new` in the header was not given a label per spec (it lives in `#hdr`, not the rail).
