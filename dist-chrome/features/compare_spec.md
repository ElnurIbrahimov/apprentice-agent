# Compare Panel — Group Chat / Model Compare Feature Spec

Lets users send one prompt to multiple AI models simultaneously and view responses side-by-side. No PILL_SLOTS needed — the panel IS its own model selector.

---

## 1. Rail Button

Insert after the `agent` rail button (before `models`), inside `#rail` in sidebar.html:

```html
<button class="rbtn" data-panel="compare" data-tip="Compare">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
    <path d="M8 10h8M8 14h5" opacity=".5"/>
    <circle cx="20" cy="6" r="3" fill="currentColor" stroke="none" opacity=".7"/>
    <circle cx="20" cy="6" r="1.2" fill="white" stroke="none"/>
  </svg>
</button>
```

---

## 2. HTML Panel

Insert as a sibling of the other `.panel` divs inside `#main` in sidebar.html:

```html
<!-- COMPARE PANEL -->
<div class="panel" id="panel-compare">

  <!-- Scrollable body -->
  <div id="cmp-body" style="flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:0;scrollbar-width:thin;scrollbar-color:var(--b1) transparent">

    <!-- Header + model selector -->
    <div id="cmp-top" style="padding:14px 14px 10px;border-bottom:1px solid var(--b1);flex-shrink:0">
      <h2 style="font-size:14px;font-weight:700;color:var(--tx);margin-bottom:10px;display:flex;align-items:center;gap:7px">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--pl)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
        </svg>
        Compare Models
      </h2>

      <!-- Model chip row -->
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:10px;color:var(--mu);font-weight:600;letter-spacing:.05em;text-transform:uppercase">Select models</span>
        <div style="display:flex;gap:4px">
          <button id="cmp-all" style="font-size:10px;padding:2px 7px;border-radius:5px;background:var(--s2);border:1px solid var(--b1);color:var(--mu);cursor:pointer">All</button>
          <button id="cmp-clear" style="font-size:10px;padding:2px 7px;border-radius:5px;background:var(--s2);border:1px solid var(--b1);color:var(--mu);cursor:pointer">Clear</button>
        </div>
      </div>
      <div id="cmp-chips" style="display:flex;flex-wrap:wrap;gap:5px;min-height:28px"></div>
    </div>

    <!-- Prompt input -->
    <div id="cmp-input-area" style="padding:10px 14px;border-bottom:1px solid var(--b1);flex-shrink:0;display:flex;gap:8px;align-items:flex-end">
      <textarea
        id="cmp-inp"
        rows="2"
        placeholder="Ask all selected models…"
        style="flex:1;resize:none;background:var(--s1);border:1px solid var(--b1);border-radius:10px;padding:8px 10px;font-size:12.5px;color:var(--tx);font-family:inherit;line-height:1.5;outline:none;transition:border-color .13s"
      ></textarea>
      <button id="cmp-run"
        style="height:36px;padding:0 14px;border-radius:9px;background:var(--p);border:none;color:#fff;font-size:12px;font-weight:600;cursor:pointer;flex-shrink:0;transition:opacity .13s"
      >Ask All</button>
    </div>

    <!-- Empty state -->
    <div id="cmp-empty" style="display:flex;flex-direction:column;align-items:center;gap:10px;padding:36px 20px;color:var(--mu);text-align:center">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="var(--di)" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
      </svg>
      <div style="font-size:12px;line-height:1.55">Select models above, type a prompt,<br>and hit <strong style="color:var(--pl)">Ask All</strong> to compare responses.</div>
    </div>

    <!-- Results grid -->
    <div id="cmp-results" style="display:none;padding:12px 14px;display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px"></div>

  </div>
</div>
<!-- END COMPARE PANEL -->
```

---

## 3. CSS

Add inside the `<style>` block in sidebar.html (or append to it):

```css
/* ═══════════════════════════════════════
   COMPARE PANEL
═══════════════════════════════════════ */

/* Model selector chips */
.cmp-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 9px;
  border-radius: 20px;
  font-size: 11px;
  cursor: pointer;
  border: 1px solid var(--b1);
  background: var(--s2);
  color: var(--mu);
  transition: all .13s;
  white-space: nowrap;
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}
.cmp-chip:hover { border-color: var(--p); color: var(--tx); }
.cmp-chip.on {
  background: var(--pg);
  border-color: var(--p);
  color: var(--pl2);
}
.cmp-chip .cmp-chip-icon { font-size: 10px; flex-shrink: 0; }

/* Response card */
.cmp-card {
  background: var(--s1);
  border: 1px solid var(--b1);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  animation: up .18s ease-out;
  min-height: 120px;
}
.cmp-card-hdr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px 7px;
  border-bottom: 1px solid var(--b1);
  gap: 6px;
  flex-shrink: 0;
}
.cmp-model-badge {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 10.5px;
  font-weight: 700;
  color: var(--pl2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}
.cmp-model-badge .cmp-badge-icon { flex-shrink: 0; font-size: 10px; }
.cmp-time-badge {
  font-size: 9.5px;
  padding: 1px 6px;
  border-radius: 10px;
  background: var(--s2);
  border: 1px solid var(--b1);
  color: var(--mu);
  white-space: nowrap;
  flex-shrink: 0;
}
.cmp-time-badge.fastest {
  background: rgba(16,185,129,.15);
  border-color: rgba(16,185,129,.35);
  color: var(--gr);
}
.cmp-card-body {
  flex: 1;
  padding: 10px 10px 8px;
  font-size: 12px;
  color: var(--tx);
  line-height: 1.6;
  overflow-y: auto;
  max-height: 320px;
  scrollbar-width: thin;
  scrollbar-color: var(--b1) transparent;
}
.cmp-card-body.error { color: #f87171; font-style: italic; }
.cmp-card-footer {
  padding: 6px 10px;
  border-top: 1px solid var(--b1);
  flex-shrink: 0;
}
.cmp-send-btn {
  font-size: 10.5px;
  padding: 3px 10px;
  border-radius: 6px;
  background: var(--s2);
  border: 1px solid var(--b1);
  color: var(--mu);
  cursor: pointer;
  transition: all .13s;
}
.cmp-send-btn:hover { background: var(--pg); border-color: var(--p); color: var(--pl); }

/* Loading skeleton */
.cmp-skeleton {
  background: var(--s1);
  border: 1px solid var(--b1);
  border-radius: 12px;
  overflow: hidden;
  min-height: 120px;
  position: relative;
}
.cmp-skeleton::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(167,139,250,.07) 50%,
    transparent 100%
  );
  background-size: 200% 100%;
  animation: cmp-shimmer 1.4s ease-in-out infinite;
}
@keyframes cmp-shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position:  200% 0; }
}
.cmp-skeleton-hdr {
  height: 34px;
  border-bottom: 1px solid var(--b1);
  padding: 8px 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.cmp-skeleton-bar {
  height: 9px;
  border-radius: 5px;
  background: var(--s3);
}
.cmp-skeleton-body { padding: 10px; display: flex; flex-direction: column; gap: 7px; }

/* Narrow panel: 1 column */
@media (max-width: 340px) {
  #cmp-results { grid-template-columns: 1fr !important; }
}

#cmp-inp:focus { border-color: var(--p); }
#cmp-run:disabled { opacity: .45; cursor: default; }
```

---

## 4. JavaScript

Add at the bottom of `sidebar.js` (before the final closing lines / after `initModelPills()`):

```javascript
// ══════════════════════════════════════════════════════════════════════════
// COMPARE PANEL
// ══════════════════════════════════════════════════════════════════════════

let compareSelectedModels = new Set();
let compareInitialized = false;

// Default pre-selections (first 3 cloud models by preference order)
const COMPARE_DEFAULT_MODELS = [
  'minimax-m2.7:cloud',
  'qwen3.5:397b-cloud',
  'kimi-k2.5:cloud',
];

function initComparePanel() {
  if (compareInitialized) return;

  const chipsEl = $('cmp-chips');
  if (!chipsEl) return;

  function buildChips(cloudList, localList) {
    chipsEl.innerHTML = '';

    function addChip(modelName, isCloud) {
      const chip = document.createElement('button');
      chip.className = 'cmp-chip';
      chip.dataset.model = modelName;
      chip.title = modelName;

      const icon = document.createElement('span');
      icon.className = 'cmp-chip-icon';
      icon.textContent = isCloud ? '\u2601' : '\uD83D\uDDA5';  // ☁ or 🖥

      const label = document.createElement('span');
      label.style.cssText = 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
      label.textContent = modelName.replace(/:cloud$/, '');

      chip.appendChild(icon);
      chip.appendChild(label);

      if (
        compareSelectedModels.size < 3 &&
        (COMPARE_DEFAULT_MODELS.includes(modelName) || compareSelectedModels.size === 0)
      ) {
        compareSelectedModels.add(modelName);
      }

      if (compareSelectedModels.has(modelName)) chip.classList.add('on');

      chip.addEventListener('click', () => {
        if (compareSelectedModels.has(modelName)) {
          compareSelectedModels.delete(modelName);
          chip.classList.remove('on');
        } else {
          compareSelectedModels.add(modelName);
          chip.classList.add('on');
        }
      });

      chipsEl.appendChild(chip);
    }

    cloudList.forEach(m => addChip(m, true));
    localList.forEach(m => addChip(m, false));

    // Re-apply selection state to any defaults that ARE in the lists
    COMPARE_DEFAULT_MODELS.forEach(m => {
      if (cloudList.includes(m) || localList.includes(m)) {
        compareSelectedModels.add(m);
        const el = chipsEl.querySelector(`[data-model="${CSS.escape(m)}"]`);
        if (el) el.classList.add('on');
      }
    });

    compareInitialized = true;
  }

  if (mdlCloudList.length || mdlLocalList.length) {
    buildChips(mdlCloudList, mdlLocalList);
  } else {
    // Lazy-load from Ollama directly
    fetch('http://localhost:11434/api/tags')
      .then(r => r.json())
      .then(d => {
        const all = (d.models || []).map(m => m.name);
        mdlCloudList = all.filter(n => n.includes(':cloud'));
        mdlLocalList = all.filter(n => !n.includes(':cloud'));
        buildChips(mdlCloudList, mdlLocalList);
      })
      .catch(() => {
        fetch(`${HTTP}/api/models/available`)
          .then(r => r.json())
          .then(d => {
            mdlCloudList = (d.cloud || []).map(m => m.name);
            mdlLocalList = (d.local || []).map(m => m.name);
            buildChips(mdlCloudList, mdlLocalList);
          })
          .catch(() => {
            chipsEl.innerHTML = '<span style="font-size:11px;color:var(--mu)">No models found — is Ollama running?</span>';
          });
      });
  }
}

// "All" / "Clear" buttons
$('cmp-all') && $('cmp-all').addEventListener('click', () => {
  const chipsEl = $('cmp-chips');
  if (!chipsEl) return;
  chipsEl.querySelectorAll('.cmp-chip').forEach(c => {
    compareSelectedModels.add(c.dataset.model);
    c.classList.add('on');
  });
});

$('cmp-clear') && $('cmp-clear').addEventListener('click', () => {
  compareSelectedModels.clear();
  const chipsEl = $('cmp-chips');
  if (chipsEl) chipsEl.querySelectorAll('.cmp-chip').forEach(c => c.classList.remove('on'));
});

// Textarea: Enter submits (Shift+Enter = newline)
$('cmp-inp') && $('cmp-inp').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runCompare(); }
});

$('cmp-run') && $('cmp-run').addEventListener('click', runCompare);

async function runCompare() {
  const inp = $('cmp-inp');
  const resultsEl = $('cmp-results');
  const emptyEl = $('cmp-empty');
  const runBtn = $('cmp-run');

  if (!inp || !resultsEl) return;

  const prompt = inp.value.trim();
  if (!prompt) {
    inp.style.borderColor = 'var(--rd)';
    setTimeout(() => { inp.style.borderColor = ''; }, 800);
    return;
  }
  if (compareSelectedModels.size === 0) {
    const chipsEl = $('cmp-chips');
    if (chipsEl) { chipsEl.style.outline = '1px solid var(--rd)'; setTimeout(() => { chipsEl.style.outline = ''; }, 800); }
    return;
  }

  runBtn.disabled = true;
  if (emptyEl) emptyEl.style.display = 'none';

  const selectedModels = [...compareSelectedModels];

  // Show loading skeletons
  resultsEl.style.display = 'grid';
  resultsEl.innerHTML = '';
  selectedModels.forEach(model => {
    const skel = document.createElement('div');
    skel.className = 'cmp-skeleton';
    skel.dataset.skelModel = model;
    skel.innerHTML = `
      <div class="cmp-skeleton-hdr">
        <div class="cmp-skeleton-bar" style="width:55%;"></div>
        <div class="cmp-skeleton-bar" style="width:18%;margin-left:auto"></div>
      </div>
      <div class="cmp-skeleton-body">
        <div class="cmp-skeleton-bar" style="width:92%"></div>
        <div class="cmp-skeleton-bar" style="width:78%"></div>
        <div class="cmp-skeleton-bar" style="width:85%"></div>
        <div class="cmp-skeleton-bar" style="width:60%"></div>
      </div>`;
    resultsEl.appendChild(skel);
  });

  try {
    const res = await fetch(`${HTTP}/api/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: prompt, models: selectedModels }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();  // { results: [...], fastest: str, query: str }

    renderCompareResults(data.results, data.fastest);
  } catch (err) {
    resultsEl.innerHTML = `<div style="grid-column:1/-1;color:var(--rd);font-size:12px;padding:12px 0">Failed to compare: ${esc(String(err))}</div>`;
  } finally {
    runBtn.disabled = false;
  }
}

function renderCompareResults(results, fastest) {
  const resultsEl = $('cmp-results');
  if (!resultsEl) return;

  resultsEl.innerHTML = '';
  resultsEl.style.display = 'grid';

  results.forEach(result => {
    const isCloud = result.model.includes(':cloud');
    const isFastest = result.model === fastest && !result.error;
    const displayName = result.model.replace(/:cloud$/, '');

    const card = document.createElement('div');
    card.className = 'cmp-card';

    // Header
    const hdr = document.createElement('div');
    hdr.className = 'cmp-card-hdr';

    const badge = document.createElement('div');
    badge.className = 'cmp-model-badge';
    badge.title = result.model;
    badge.innerHTML = `
      <span class="cmp-badge-icon">${isCloud ? '\u2601' : '\uD83D\uDDA5'}</span>
      <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(displayName)}</span>`;

    const timeBadge = document.createElement('span');
    timeBadge.className = 'cmp-time-badge' + (isFastest ? ' fastest' : '');
    const timeLabel = result.elapsed_ms >= 1000
      ? (result.elapsed_ms / 1000).toFixed(1) + 's'
      : result.elapsed_ms + 'ms';
    timeBadge.textContent = isFastest ? '\u26A1 ' + timeLabel : timeLabel;  // ⚡

    hdr.appendChild(badge);
    hdr.appendChild(timeBadge);

    // Body
    const body = document.createElement('div');
    body.className = 'cmp-card-body' + (result.error ? ' error' : '');
    if (result.error) {
      body.textContent = 'Error: ' + result.error;
    } else {
      body.innerHTML = md(result.response || '');
    }

    // Footer
    const footer = document.createElement('div');
    footer.className = 'cmp-card-footer';

    const sendBtn = document.createElement('button');
    sendBtn.className = 'cmp-send-btn';
    sendBtn.textContent = 'Send to Chat';
    sendBtn.addEventListener('click', () => {
      if (!result.response) return;
      const prefix = `[${displayName} says]: `;
      switchPanel('chat');
      const inp2 = $('inp');
      if (inp2) {
        inp2.value = prefix + result.response.slice(0, 500) + (result.response.length > 500 ? '…' : '');
        inp2.focus();
      }
    });

    footer.appendChild(sendBtn);
    card.appendChild(hdr);
    card.appendChild(body);
    if (!result.error) card.appendChild(footer);

    resultsEl.appendChild(card);
  });
}

// Hook into switchPanel — init chips on first visit to compare
const _origSwitchPanel = switchPanel;
// We patch by overriding after declaration — wrap via re-assignment
// (switchPanel is a function declaration so we store the compare init hook separately)
document.querySelectorAll('.rbtn[data-panel="compare"]').forEach(btn => {
  btn.addEventListener('click', () => {
    initComparePanel();
  });
});
```

---

## 5. Integration Checklist

- [ ] Add rail button HTML inside `#rail` in `sidebar.html`
- [ ] Add panel HTML (`#panel-compare`) inside `#main` in `sidebar.html`
- [ ] Add CSS block to `<style>` in `sidebar.html`
- [ ] Append JS block to end of `sidebar.js`
- [ ] Verify `httpx` is in `requirements.txt` (used by new `multi_model.py`)
- [ ] Backend: `multi_model.py` now directly calls `http://localhost:11434/api/generate` — no agent_service lock contention
- [ ] Test: select 3 models, type prompt, click "Ask All" — results grid appears with timing badges, fastest highlighted in green

---

## 6. Notes

- The `/api/compare` endpoint caps at 6 models to avoid hammering Ollama bridge.
- Models with `:cloud` suffix get the cloud icon; all others get the desktop icon.
- "Send to Chat" pre-fills the chat input — the user still clicks Send to actually send, avoiding accidental submissions.
- `compareInitialized` flag prevents redundant chip rebuilds on repeated panel visits. To force a refresh (e.g. after loading new models), set `compareInitialized = false` then call `initComparePanel()`.
- The skeleton shimmer animation uses the existing `--s3` CSS variable for card background and `--pl` purple tint for the sweep, matching AURA's visual language.
