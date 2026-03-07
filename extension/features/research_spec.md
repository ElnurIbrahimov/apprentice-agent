# Deep Research Panel — Complete Implementation Spec

Paste each section into the corresponding file. All code is complete and ready to use.

---

## 1. main.py — Register the router

In `D:\Aura\api\main.py`, add `research` to the import line and add `include_router`:

```python
# Line 30 — add "research" to the import
from api.routes import ..., summarize, research

# After line 380 (app.include_router(summarize.router))
app.include_router(research.router)
```

---

## 2. sidebar.html — CSS (paste inside `<style>` before `</style>`)

```css
/* ═══════════════════════════════════════
   RESEARCH PANEL
═══════════════════════════════════════ */
#panel-research{gap:0}

/* Depth selector */
#res-depth{display:flex;gap:6px;padding:12px 14px 0;flex-shrink:0}
.rdepth-btn{
  flex:1;padding:6px 0;border-radius:8px;border:1px solid var(--b1);
  background:var(--s1);color:var(--mu);font-size:11.5px;font-family:inherit;
  cursor:pointer;transition:all .13s;text-align:center;line-height:1.3;
}
.rdepth-btn:hover{border-color:var(--p);color:var(--tx)}
.rdepth-btn.on{background:var(--pg);border-color:var(--p);color:var(--pl);font-weight:600}
.rdepth-btn small{display:block;font-size:9.5px;color:var(--di);font-weight:400;margin-top:1px}
.rdepth-btn.on small{color:var(--pl);opacity:.7}

/* Query input */
#res-inp{
  margin:10px 14px 0;background:var(--bg);border:1px solid var(--b1);
  border-radius:10px;color:var(--tx);font-family:inherit;font-size:13px;
  line-height:1.5;padding:10px 13px;outline:none;resize:none;
  min-height:72px;transition:border-color .18s;width:calc(100% - 28px);flex-shrink:0;
}
#res-inp:focus{border-color:var(--p)}
#res-inp::placeholder{color:var(--di)}

/* Research button */
#res-btn{
  margin:8px 14px 0;width:calc(100% - 28px);flex-shrink:0;
  background:linear-gradient(135deg,var(--p),var(--p2));
  border:none;border-radius:9px;color:#fff;font-size:13px;font-weight:600;
  font-family:inherit;padding:9px 0;cursor:pointer;transition:opacity .13s;
}
#res-btn:hover{opacity:.9}
#res-btn:disabled{opacity:.5;cursor:not-allowed}

/* Progress area */
#res-progress{
  display:none;margin:10px 14px 0;padding:12px 14px;
  background:var(--s1);border:1px solid var(--b1);border-radius:10px;
  flex-shrink:0;
}
#res-progress.on{display:flex;align-items:center;gap:10px}
.res-dots{display:flex;gap:4px;flex-shrink:0}
.res-dot{
  width:6px;height:6px;border-radius:50%;background:var(--pl);
  animation:resdot 1.2s ease-in-out infinite;
}
.res-dot:nth-child(2){animation-delay:.2s}
.res-dot:nth-child(3){animation-delay:.4s}
@keyframes resdot{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
#res-status{font-size:12px;color:var(--mu);flex:1}

/* Results area */
#res-results{flex:1;overflow-y:auto;padding:10px 14px 14px;display:flex;flex-direction:column;gap:10px;
  scrollbar-width:thin;scrollbar-color:var(--b1) transparent}
#res-results::-webkit-scrollbar{width:3px}
#res-results::-webkit-scrollbar-thumb{background:var(--b1);border-radius:2px}

/* Sources */
#res-sources-wrap{display:none}
#res-sources-wrap.on{display:block}
.res-sources-lbl{font-size:10px;color:var(--mu);font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:6px}
#res-sources{display:flex;flex-direction:column;gap:4px}
.res-src{
  display:flex;align-items:center;gap:8px;padding:6px 10px;
  background:var(--s1);border:1px solid var(--b1);border-radius:8px;
  text-decoration:none;transition:border-color .13s;overflow:hidden;
}
.res-src:hover{border-color:var(--p)}
.res-src-fav{
  width:14px;height:14px;border-radius:3px;flex-shrink:0;
  background:var(--s2);display:flex;align-items:center;justify-content:center;overflow:hidden;
}
.res-src-fav img{width:14px;height:14px;object-fit:cover}
.res-src-num{font-size:10px;color:var(--di);flex-shrink:0;min-width:16px}
.res-src-title{font-size:11.5px;color:var(--pl2);flex:1;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.res-src-domain{font-size:10px;color:var(--di);flex-shrink:0;max-width:90px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* Report */
#res-report-wrap{display:none}
#res-report-wrap.on{display:block}
.res-report-lbl{font-size:10px;color:var(--mu);font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:8px}
#res-report{
  background:var(--s1);border:1px solid var(--b1);border-radius:10px;
  padding:14px 16px;font-size:13px;line-height:1.7;color:var(--tx);
}
#res-report h2{font-size:14px;font-weight:700;color:var(--pl2);margin:14px 0 6px;padding-bottom:4px;
  border-bottom:1px solid var(--b1)}
#res-report h2:first-child{margin-top:0}
#res-report h3{font-size:13px;font-weight:600;color:var(--pl);margin:10px 0 4px}
#res-report p{margin-bottom:8px}
#res-report ul,#res-report ol{padding-left:18px;margin-bottom:8px}
#res-report li{margin-bottom:3px}
#res-report strong{color:var(--pl2)}
#res-report a.cite{
  display:inline-flex;align-items:center;justify-content:center;
  width:16px;height:16px;border-radius:4px;font-size:9.5px;font-weight:700;
  background:var(--pg);border:1px solid rgba(124,58,237,.3);color:var(--pl);
  text-decoration:none;vertical-align:super;line-height:1;margin:0 1px;
  transition:background .13s;cursor:pointer;
}
#res-report a.cite:hover{background:var(--p);color:#fff}

/* Action buttons */
#res-actions{display:none;gap:6px;flex-shrink:0;padding-top:2px}
#res-actions.on{display:flex}
.res-act{
  flex:1;padding:7px 0;border-radius:8px;border:1px solid var(--b1);
  background:var(--s1);color:var(--mu);font-size:11.5px;font-family:inherit;
  cursor:pointer;transition:all .13s;
}
.res-act:hover{border-color:var(--p);color:var(--pl)}
.res-act.primary{background:var(--pg);border-color:var(--p);color:var(--pl)}
```

---

## 3. sidebar.html — Panel HTML (paste after `</div><!-- /panel-agent -->`)

```html
  <div class="panel" id="panel-research">
    <!-- Header -->
    <div class="tool-hdr" style="display:flex;align-items:center;justify-content:space-between;padding:12px 14px 0;flex-shrink:0">
      <h2 style="font-size:14px;font-weight:700;color:var(--tx)">Deep Research</h2>
      <span id="mdl-research"></span>
    </div>

    <!-- Depth selector -->
    <div id="res-depth">
      <button class="rdepth-btn on" data-depth="quick">Quick<small>~30s</small></button>
      <button class="rdepth-btn" data-depth="standard">Standard<small>1-2 min</small></button>
      <button class="rdepth-btn" data-depth="deep">Deep<small>3-5 min</small></button>
    </div>

    <!-- Query input -->
    <textarea id="res-inp" placeholder="What do you want to research?"></textarea>

    <!-- Research button -->
    <button id="res-btn">Research</button>

    <!-- Progress -->
    <div id="res-progress">
      <div class="res-dots">
        <div class="res-dot"></div>
        <div class="res-dot"></div>
        <div class="res-dot"></div>
      </div>
      <div id="res-status">Starting...</div>
    </div>

    <!-- Results -->
    <div id="res-results">
      <!-- Sources -->
      <div id="res-sources-wrap">
        <div class="res-sources-lbl">Sources</div>
        <div id="res-sources"></div>
      </div>

      <!-- Report -->
      <div id="res-report-wrap">
        <div class="res-report-lbl">Research Report</div>
        <div id="res-report"></div>
      </div>

      <!-- Actions -->
      <div id="res-actions">
        <button class="res-act primary" id="res-copy">Copy Report</button>
        <button class="res-act" id="res-save">Save to Wisebase</button>
        <button class="res-act" id="res-chat">Send to Chat</button>
      </div>
    </div>
  </div><!-- /panel-research -->
```

---

## 4. sidebar.html — Rail Button (paste before the `<div class="rsp"></div>` near the bottom of `#rail`)

```html
  <button class="rbtn" data-panel="research" data-tip="Research">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="10" cy="10" r="7"/>
      <line x1="15.5" y1="15.5" x2="21" y2="21"/>
      <path d="M7 10h6M10 7v6"/>
      <rect x="14" y="2" width="8" height="6" rx="1" fill="none"/>
      <line x1="15.5" y1="5" x2="20.5" y2="5"/>
    </svg>
  </button>
```

---

## 5. sidebar.js — PILL_SLOTS addition

Find the existing `PILL_SLOTS` object and add the research entry:

**Find:**
```javascript
const PILL_SLOTS = {
  chat: 'mdl-chat', search: 'mdl-search', translate: 'mdl-translate',
  write: 'mdl-write', grammar: 'mdl-grammar', ask: 'mdl-ask',
  pdf: 'mdl-pdf', voice: 'mdl-voice',
};
```

**Replace with:**
```javascript
const PILL_SLOTS = {
  chat: 'mdl-chat', search: 'mdl-search', translate: 'mdl-translate',
  write: 'mdl-write', grammar: 'mdl-grammar', ask: 'mdl-ask',
  pdf: 'mdl-pdf', voice: 'mdl-voice', research: 'mdl-research',
};
```

---

## 6. sidebar.js — Research feature code (paste at end of file, before the closing)

```javascript
// ═══════════════════════════════════════════════════════════
//  DEEP RESEARCH PANEL
// ═══════════════════════════════════════════════════════════

let researchDepth = 'quick';
let lastResearchReport = '';

// ── DOM refs ──────────────────────────────────────────────
const resInp         = $('res-inp');
const resBtn         = $('res-btn');
const resProgress    = $('res-progress');
const resStatus      = $('res-status');
const resSources     = $('res-sources');
const resSourcesWrap = $('res-sources-wrap');
const resReport      = $('res-report');
const resReportWrap  = $('res-report-wrap');
const resActions     = $('res-actions');

// ── Depth selector ────────────────────────────────────────
document.querySelectorAll('.rdepth-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    researchDepth = btn.dataset.depth;
    document.querySelectorAll('.rdepth-btn').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
  });
});

// ── Citation link renderer ────────────────────────────────
function renderCitations(html, sources) {
  return html.replace(/\[(\d+)\]/g, (match, num) => {
    const src = sources.find(s => s.index === parseInt(num));
    if (!src) return match;
    return `<a class="cite" href="${src.url}" target="_blank" rel="noopener" title="${esc(src.title)}">${num}</a>`;
  });
}

// ── Main research function ────────────────────────────────
async function runDeepResearch() {
  const query = resInp.value.trim();
  if (!query) { resInp.focus(); return; }

  // Reset UI
  resBtn.disabled = true;
  resSourcesWrap.classList.remove('on');
  resReportWrap.classList.remove('on');
  resActions.classList.remove('on');
  resSources.innerHTML = '';
  resReport.innerHTML = '';
  resProgress.classList.add('on');
  resStatus.textContent = 'Starting...';
  lastResearchReport = '';

  try {
    const resp = await fetch(`${HTTP}/api/research`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        depth: researchDepth,
        model: getModel('research'),
      }),
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${err}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\n');
      buf = lines.pop(); // keep incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let msg;
        try { msg = JSON.parse(trimmed); } catch { continue; }

        if (msg.status === 'searching' || msg.status === 'analyzing' || msg.status === 'writing') {
          resStatus.textContent = msg.message || msg.status;

        } else if (msg.status === 'done') {
          resProgress.classList.remove('on');

          // Render sources
          if (msg.sources && msg.sources.length) {
            resSourcesWrap.classList.add('on');
            resSources.innerHTML = msg.sources.map(s => {
              const faviconUrl = `https://www.google.com/s2/favicons?domain=${encodeURIComponent(s.domain)}&sz=32`;
              return `<a class="res-src" href="${s.url}" target="_blank" rel="noopener" title="${esc(s.snippet)}">
                <div class="res-src-fav"><img src="${faviconUrl}" alt="" loading="lazy" onerror="this.style.display='none'"/></div>
                <span class="res-src-num">[${s.index}]</span>
                <span class="res-src-title">${esc(s.title)}</span>
                <span class="res-src-domain">${esc(s.domain)}</span>
              </a>`;
            }).join('');
          }

          // Render report
          if (msg.report) {
            lastResearchReport = msg.report;
            resReportWrap.classList.add('on');
            const rendered = md(msg.report);
            const withCites = renderCitations(rendered, msg.sources || []);
            resReport.innerHTML = withCites;
          }

          resActions.classList.add('on');
        }
      }
    }
  } catch (err) {
    resProgress.classList.remove('on');
    resReport.innerHTML = `<em style="color:#f87171">Research failed: ${esc(err.message)}</em>`;
    resReportWrap.classList.add('on');
  } finally {
    resBtn.disabled = false;
  }
}

// ── Button handlers ───────────────────────────────────────
resBtn.addEventListener('click', runDeepResearch);

resInp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runDeepResearch();
});

// Copy report
$('res-copy').addEventListener('click', () => {
  if (!lastResearchReport) return;
  navigator.clipboard.writeText(lastResearchReport).then(() => {
    const btn = $('res-copy');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
});

// Save to Wisebase
$('res-save').addEventListener('click', async () => {
  if (!lastResearchReport) return;
  const btn = $('res-save');
  btn.disabled = true;
  btn.textContent = 'Saving...';
  try {
    const query = resInp.value.trim();
    await fetch(`${HTTP}/api/knowledge/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: lastResearchReport,
        title: `Research: ${query}`,
        tags: ['research', 'deep-research'],
      }),
    });
    btn.textContent = 'Saved!';
    setTimeout(() => { btn.textContent = 'Save to Wisebase'; btn.disabled = false; }, 2000);
  } catch (e) {
    btn.textContent = 'Failed';
    setTimeout(() => { btn.textContent = 'Save to Wisebase'; btn.disabled = false; }, 2000);
  }
});

// Send to Chat
$('res-chat').addEventListener('click', () => {
  if (!lastResearchReport) return;
  const query = resInp.value.trim();
  pendingCtx = lastResearchReport;
  showCtx(`Research: ${query}`, 'Research Report');
  switchPanel('chat');
});
```

---

## Notes

- The rail button uses a magnifying glass + document SVG. It stacks on the rail with other tool buttons.
- `researchDepth` defaults to `'quick'` in the JS state but the HTML default button marked `.on` is also "Quick" — consistent.
- `renderCitations` walks the rendered markdown HTML and turns `[1]` tokens into clickable `.cite` anchor links that jump to the source URL.
- The streaming reader uses a newline-delimited JSON (NDJSON) protocol matching the backend's `StreamingResponse`.
- Ctrl+Enter submits the research query (same pattern as the write panel).
- `getModel('research')` reads from the model routing system — null means backend decides (defaults to `qwen3.5:397b-cloud`).
- `Save to Wisebase` posts to `/api/knowledge/save` with `text`, `title`, and `tags` fields matching the `SaveRequest` schema in `knowledge.py`.
