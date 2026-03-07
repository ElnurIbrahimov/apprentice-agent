# Page Summary Feature — Complete Implementation Spec

All code below is paste-ready. Insert each section into the indicated file.

---

## 1. sidebar.html — CSS (paste inside `<style>` after the OCR PANEL section)

```css
/* ═══════════════════════════════════════
   PAGE SUMMARY PANEL
═══════════════════════════════════════ */
#panel-summary{gap:0}
#sum-hdr{padding:16px 14px 12px;flex-shrink:0;display:flex;align-items:center;justify-content:space-between}
#sum-hdr h2{font-size:18px;font-weight:700;margin:0}
#sum-formats{display:flex;gap:5px;padding:0 14px 10px;flex-shrink:0}
.sum-fmt{background:var(--s2);border:1px solid var(--b1);border-radius:20px;color:var(--mu);
  font-size:11px;padding:4px 12px;cursor:pointer;font-family:inherit;transition:all .13s}
.sum-fmt:hover{color:var(--tx)}
.sum-fmt.on{border-color:var(--p);color:var(--pl);background:var(--pg)}
#sum-go{margin:0 14px 10px;background:var(--p);border:none;border-radius:10px;color:#fff;
  font-size:13px;font-weight:600;font-family:inherit;padding:11px 16px;cursor:pointer;
  transition:background .13s;display:flex;align-items:center;justify-content:center;gap:8px;
  flex-shrink:0;width:calc(100% - 28px)}
#sum-go:hover{background:var(--p2)}
#sum-go:disabled{background:var(--b1);cursor:not-allowed;opacity:.55}
#sum-go svg{width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex-shrink:0}
#sum-status{padding:4px 14px 8px;font-size:11.5px;color:var(--mu);flex-shrink:0;min-height:22px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#sum-result{flex:1;overflow-y:auto;margin:0 14px;background:var(--s1);border:1px solid var(--b1);
  border-radius:10px;padding:12px 13px;font-size:13px;line-height:1.65;color:var(--tx);
  display:none;scrollbar-width:thin;scrollbar-color:var(--b1) transparent}
#sum-result.on{display:block}
#sum-badges{display:flex;gap:6px;padding:8px 14px;flex-shrink:0;flex-wrap:wrap;display:none}
#sum-badges.on{display:flex}
.sum-badge{background:var(--s2);border:1px solid var(--b1);border-radius:20px;
  font-size:10.5px;color:var(--mu);padding:3px 10px;white-space:nowrap}
.sum-badge.hi{border-color:var(--p);color:var(--pl);background:var(--pg)}
#sum-actions{padding:0 14px 12px;display:flex;gap:6px;flex-shrink:0;display:none}
#sum-actions.on{display:flex}
#sum-to-chat,#sum-copy{background:var(--s2);border:1px solid var(--b1);border-radius:8px;
  color:var(--mu);font-size:12px;font-family:inherit;padding:7px 14px;cursor:pointer;transition:all .13s}
#sum-to-chat:hover,#sum-copy:hover{border-color:var(--p);color:var(--pl)}
#sum-empty{display:flex;flex-direction:column;align-items:center;gap:10px;
  padding:40px 14px;color:var(--mu);text-align:center}
#sum-empty svg{width:38px;height:38px;stroke:var(--di);fill:none;stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round}
#sum-empty p{font-size:12px;line-height:1.5;color:var(--di)}
```

---

## 2. sidebar.html — Panel HTML (paste before `<!-- RIGHT RAIL -->`)

```html
<!-- PAGE SUMMARY PANEL -->
<div class="panel" id="panel-summary">
  <div id="sum-hdr">
    <h2>Page Summary</h2>
    <span id="mdl-summary"></span>
  </div>
  <div id="sum-formats">
    <button class="sum-fmt on" data-fmt="bullets">Bullets</button>
    <button class="sum-fmt" data-fmt="paragraph">Paragraph</button>
    <button class="sum-fmt" data-fmt="tldr">TL;DR</button>
  </div>
  <button id="sum-go">
    <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>
    Summarize This Page
  </button>
  <div id="sum-status"></div>
  <div id="sum-empty">
    <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>
    <p>Open any webpage and click<br><strong>Summarize This Page</strong></p>
  </div>
  <div id="sum-result"></div>
  <div id="sum-badges">
    <span class="sum-badge" id="sum-wc"></span>
    <span class="sum-badge hi" id="sum-rt"></span>
  </div>
  <div id="sum-actions">
    <button id="sum-to-chat">Send to Chat</button>
    <button id="sum-copy">Copy</button>
  </div>
</div><!-- /panel-summary -->
```

---

## 3. sidebar.html — Rail Button (paste inside `#rail` div, after the OCR button and before the `<div class="rsep">` that precedes Image Gen)

```html
<button class="rbtn" data-panel="summary" data-tip="Summary">
  <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>
</button>
```

---

## 4. sidebar.js — State variable (paste near the top with other panel state variables, after `let agentRunning = false;`)

```js
// Summary panel state
let summaryFormat = 'bullets';
let summaryText = '';
```

---

## 5. sidebar.js — PILL_SLOTS addition (add `summary: 'mdl-summary'` to the existing object)

Change:
```js
const PILL_SLOTS = {
  chat: 'mdl-chat', search: 'mdl-search', translate: 'mdl-translate',
  write: 'mdl-write', grammar: 'mdl-grammar', ask: 'mdl-ask',
  pdf: 'mdl-pdf', voice: 'mdl-voice',
};
```

To:
```js
const PILL_SLOTS = {
  chat: 'mdl-chat', search: 'mdl-search', translate: 'mdl-translate',
  write: 'mdl-write', grammar: 'mdl-grammar', ask: 'mdl-ask',
  pdf: 'mdl-pdf', voice: 'mdl-voice', summary: 'mdl-summary',
};
```

---

## 6. sidebar.js — FEATURE_DEFS addition (add summary entry to the array)

Add to `FEATURE_DEFS`:
```js
{ key: 'summary', label: 'Page Summary', icon: '📄', desc: 'One-click page summarization' },
```

---

## 7. sidebar.js — switchPanel hook addition

In the existing `switchPanel` function, add a `summary` branch. Change:
```js
  else if (name === 'models') loadModelPanel();
```
To:
```js
  else if (name === 'models') loadModelPanel();
  else if (name === 'summary') { /* panel is ready, user clicks Summarize */ }
```

---

## 8. sidebar.js — Page Summary feature code (paste as a new section, after the OCR section)

```js
// ══════════════════════════════════════════════════════════════════════════
// PAGE SUMMARY PANEL
// ══════════════════════════════════════════════════════════════════════════

const sumGo      = $('sum-go');
const sumStatus  = $('sum-status');
const sumResult  = $('sum-result');
const sumBadges  = $('sum-badges');
const sumActions = $('sum-actions');
const sumEmpty   = $('sum-empty');
const sumWc      = $('sum-wc');
const sumRt      = $('sum-rt');

// Format toggle buttons
document.querySelectorAll('.sum-fmt').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.sum-fmt').forEach(b => b.classList.remove('on'));
    this.classList.add('on');
    summaryFormat = this.dataset.fmt;
  });
});

async function summarizeCurrentPage() {
  // Get page content via background → content script
  sumGo.disabled = true;
  sumStatus.textContent = 'Reading page…';
  sumResult.classList.remove('on');
  sumBadges.classList.remove('on');
  sumActions.classList.remove('on');
  sumEmpty.style.display = 'none';
  summaryText = '';

  let pageData;
  try {
    pageData = await new Promise((resolve) => {
      ext.runtime.sendMessage({ type: 'GET_PAGE_CONTENT' }, (resp) => {
        resolve(resp);
      });
    });
  } catch (e) {
    sumStatus.textContent = 'Could not read page.';
    sumGo.disabled = false;
    sumEmpty.style.display = '';
    return;
  }

  if (!pageData?.ok || !pageData.text) {
    sumStatus.textContent = pageData?.error || 'Could not read page content.';
    sumGo.disabled = false;
    sumEmpty.style.display = '';
    return;
  }

  const title = pageData.title || 'Untitled Page';
  sumStatus.textContent = 'Summarizing: ' + title.slice(0, 55) + (title.length > 55 ? '…' : '');

  try {
    const res = await fetch(`${HTTP}/api/summarize/page`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: pageData.text,
        url: pageData.url || '',
        title: title,
        format: summaryFormat,
        model: getModel('summary'),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      sumStatus.textContent = 'Error: ' + (err.detail || res.statusText);
      sumGo.disabled = false;
      sumEmpty.style.display = '';
      return;
    }

    const data = await res.json();
    summaryText = data.summary || '';

    // Render markdown summary
    sumResult.innerHTML = md(summaryText);
    sumResult.classList.add('on');

    // Badges
    const wordCount = data.word_count || 0;
    sumWc.textContent = wordCount.toLocaleString() + ' words on page';
    sumRt.textContent = data.reading_time_saved || '';
    sumBadges.classList.add('on');

    // Action buttons
    sumActions.classList.add('on');
    sumStatus.textContent = '';

  } catch (e) {
    sumStatus.textContent = 'Request failed: ' + e.message;
    sumEmpty.style.display = '';
  } finally {
    sumGo.disabled = false;
  }
}

sumGo.addEventListener('click', summarizeCurrentPage);

// Send to Chat: set pendingCtx with summary, switch panel
$('sum-to-chat').addEventListener('click', () => {
  if (!summaryText) return;
  pendingCtx = {
    text: summaryText,
    title: 'Page Summary',
    url: '',
    action: 'ask',
  };
  showCtx(summaryText, 'Page Summary');
  switchPanel('chat');
});

// Copy summary to clipboard
$('sum-copy').addEventListener('click', function() {
  if (!summaryText) return;
  navigator.clipboard.writeText(summaryText).then(() => {
    const orig = this.textContent;
    this.textContent = 'Copied!';
    setTimeout(() => { this.textContent = orig; }, 1500);
  });
});
```

---

## 9. background.js — GET_PAGE_CONTENT handler

The handler is **already implemented** in background.js (lines 111–131). No changes needed.

It handles `GET_PAGE_CONTENT` by querying the active tab and forwarding `{ type: 'EXTRACT_PAGE' }` to the content script, then relays the response back to the sidebar. The sidebar's `summarizeCurrentPage()` function uses this exact pattern.

---

## Notes

- The `EXTRACT_PAGE` message is handled by the existing content script (`content.js`), which extracts `document.body.innerText`, `document.title`, and `window.location.href`.
- The model pill for `summary` uses `id="mdl-summary"` in the panel header. The `PILL_SLOTS` addition ensures `initModelPills()` injects the picker automatically.
- The `summaryText` variable persists the last summary so "Send to Chat" and "Copy" always have data even after re-renders.
- Backend truncates input at 50,000 chars and sets `truncated: true` in response (the frontend doesn't surface this flag but the model is told in the prompt).
- The rail button SVG is a document-with-lines icon matching the page document theme, consistent with other rail icons.
