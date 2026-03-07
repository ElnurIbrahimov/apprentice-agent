# Math Solver Panel — Complete Implementation Spec

Paste-ready code for integrating the Math Solver feature into AURA's sidebar.

---

## 1. HTML PANEL — add inside `<div id="main">` before `</div><!-- /main -->`

```html
<!-- MATH SOLVER PANEL -->
<div class="panel" id="panel-math">
  <div class="tool-hdr" style="display:flex;align-items:center;justify-content:space-between;padding-bottom:12px">
    <h2 style="margin:0">Math Solver</h2>
    <span id="mdl-math"></span>
  </div>
  <div id="math-modes">
    <button class="math-mode on" data-mode="solve">Solve</button>
    <button class="math-mode" data-mode="explain">Explain</button>
    <button class="math-mode" data-mode="graph_data">Graph</button>
  </div>
  <textarea id="math-inp" placeholder="Enter a math problem... e.g. 'Solve x² + 5x + 6 = 0' or 'Integrate x²dx'"></textarea>
  <button id="math-btn">Solve →</button>
  <div id="math-result">
    <div id="math-answer"></div>
    <ol id="math-steps"></ol>
    <div id="math-latex"></div>
    <div id="math-graph-wrap" style="display:none">
      <canvas id="math-graph" width="280" height="180"></canvas>
    </div>
    <div id="math-actions" style="display:none">
      <button id="math-copy">Copy Solution</button>
      <button id="math-to-chat">Send to Chat</button>
    </div>
  </div>
</div><!-- /panel-math -->
```

---

## 2. RAIL BUTTON — add inside `<div id="rail">` after the last `<div class="rsep"></div>` block

```html
<button class="rbtn" data-panel="math" data-tip="Math">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M4 7h16M4 12h10M4 17h6"/>
    <path d="M19 14l-3 6M16 14l3 6"/>
  </svg>
</button>
```

---

## 3. CSS — add inside the `<style>` block (after grammar panel styles is a good spot)

```css
/* ═══════════════════════════════════════
   MATH SOLVER PANEL
═══════════════════════════════════════ */

#panel-math { gap: 0; }

#math-modes {
  display: flex;
  gap: 5px;
  padding: 12px 14px 0;
  flex-shrink: 0;
}

.math-mode {
  background: var(--s2);
  border: 1px solid var(--b1);
  border-radius: 20px;
  color: var(--mu);
  font-size: 11px;
  padding: 4px 12px;
  cursor: pointer;
  font-family: inherit;
  transition: all .13s;
}

.math-mode:hover { border-color: var(--p); color: var(--pl); }
.math-mode.on { background: var(--pg); border-color: var(--p); color: var(--pl); font-weight: 600; }

#math-inp {
  margin: 10px 14px 0;
  background: var(--bg);
  border: 1px solid var(--b1);
  border-radius: 10px;
  color: var(--tx);
  font-family: inherit;
  font-size: 13px;
  line-height: 1.5;
  padding: 10px 13px;
  outline: none;
  resize: none;
  transition: border-color .18s;
  min-height: 70px;
  width: calc(100% - 28px);
  flex-shrink: 0;
}
#math-inp:focus { border-color: var(--p); }

#math-btn {
  margin: 8px 14px 0;
  align-self: flex-start;
  background: var(--p);
  border: none;
  border-radius: 8px;
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  font-family: inherit;
  padding: 9px 20px;
  cursor: pointer;
  transition: background .13s;
  flex-shrink: 0;
}
#math-btn:hover { background: var(--p2); }
#math-btn:disabled { opacity: .5; cursor: not-allowed; }

#math-result {
  flex: 1;
  overflow-y: auto;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  scrollbar-width: thin;
  scrollbar-color: var(--b1) transparent;
}

#math-answer {
  display: none;
  background: linear-gradient(135deg, #13082c, #1e0d40);
  border: 1.5px solid var(--p);
  border-radius: 10px;
  padding: 14px 16px;
  font-size: 15px;
  font-weight: 600;
  color: var(--pl2);
  line-height: 1.5;
  word-break: break-word;
}
#math-answer.has-text { display: block; }

#math-steps {
  display: none;
  list-style: decimal;
  padding-left: 20px;
  margin: 0;
  color: var(--tx);
  font-size: 12.5px;
  line-height: 1.65;
}
#math-steps.has-items { display: block; }
#math-steps li {
  padding: 4px 0;
  border-bottom: 1px solid var(--b1);
}
#math-steps li:last-child { border-bottom: none; }

#math-latex {
  display: none;
  background: var(--s1);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 10px 14px;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  color: var(--pl);
  text-align: center;
  overflow-x: auto;
  word-break: break-all;
}
#math-latex.has-content { display: block; }

/* KaTeX overrides to match dark theme */
#math-latex .katex { color: var(--pl2); }

#math-graph-wrap {
  background: var(--s1);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

#math-graph {
  border-radius: 4px;
  max-width: 100%;
}

#math-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}

#math-copy, #math-to-chat {
  flex: 1;
  background: var(--s2);
  border: 1px solid var(--b1);
  border-radius: 8px;
  color: var(--mu);
  font-size: 12px;
  font-family: inherit;
  padding: 8px 12px;
  cursor: pointer;
  transition: all .13s;
}
#math-copy:hover, #math-to-chat:hover {
  border-color: var(--p);
  color: var(--pl);
}
```

---

## 4. JAVASCRIPT — add to sidebar.js (after the grammar section is a natural fit)

```javascript
// ══════════════════════════════════════════════════════════════════════════
//  MATH SOLVER
// ══════════════════════════════════════════════════════════════════════════

let mathMode = 'solve';
let _katexLoaded = false;
let _katexLoading = false;

// Mode toggle
document.querySelectorAll('.math-mode').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.math-mode').forEach(b => b.classList.remove('on'));
    this.classList.add('on');
    mathMode = this.dataset.mode;
  });
});

// KaTeX lazy loader
function loadKaTeX() {
  return new Promise((resolve, reject) => {
    if (_katexLoaded) { resolve(); return; }
    if (_katexLoading) {
      // Poll until loaded
      const poll = setInterval(() => {
        if (_katexLoaded) { clearInterval(poll); resolve(); }
      }, 100);
      return;
    }
    _katexLoading = true;

    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
    document.head.appendChild(css);

    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
    script.onload = () => { _katexLoaded = true; _katexLoading = false; resolve(); };
    script.onerror = () => { _katexLoading = false; reject(new Error('KaTeX load failed')); };
    document.head.appendChild(script);
  });
}

async function renderLatex(container, latex) {
  if (!latex) { container.classList.remove('has-content'); return; }
  container.classList.add('has-content');
  try {
    await loadKaTeX();
    container.innerHTML = window.katex.renderToString(latex, {
      throwOnError: false,
      displayMode: true,
    });
  } catch (e) {
    // Fallback: monospace raw LaTeX
    container.textContent = latex;
  }
}

async function solveMath() {
  const problem = $('math-inp').value.trim();
  if (!problem) return;

  const btn = $('math-btn');
  const answerEl = $('math-answer');
  const stepsEl = $('math-steps');
  const latexEl = $('math-latex');
  const graphWrap = $('math-graph-wrap');
  const actionsEl = $('math-actions');

  // Loading state
  btn.disabled = true;
  btn.textContent = 'Solving…';
  answerEl.textContent = '…';
  answerEl.classList.add('has-text');
  stepsEl.innerHTML = '';
  stepsEl.classList.remove('has-items');
  latexEl.classList.remove('has-content');
  graphWrap.style.display = 'none';
  actionsEl.style.display = 'none';

  try {
    const res = await fetch(`${HTTP}/api/math/solve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        problem,
        mode: mathMode,
        model: getModel('math'),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Render solution answer
    answerEl.textContent = data.solution || '(no solution)';
    answerEl.classList.add('has-text');

    // Render steps
    if (data.steps && data.steps.length > 0) {
      stepsEl.innerHTML = data.steps
        .map(s => `<li>${esc(String(s))}</li>`)
        .join('');
      stepsEl.classList.add('has-items');
    }

    // Render LaTeX
    await renderLatex(latexEl, data.latex);

    // Render graph if present
    if (data.graph_data && data.graph_data.points && data.graph_data.points.length > 0) {
      graphWrap.style.display = 'flex';
      drawMathGraph(data.graph_data);
    }

    // Show action buttons
    actionsEl.style.display = 'flex';

  } catch (e) {
    answerEl.textContent = 'Error: ' + e.message;
    answerEl.classList.add('has-text');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Solve →';
  }
}

function drawMathGraph(graphData) {
  const canvas = $('math-graph');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const PAD = 30;

  ctx.clearRect(0, 0, W, H);

  const points = graphData.points;
  if (!points || points.length === 0) return;

  const xMin = graphData.x_min !== undefined ? graphData.x_min : Math.min(...points.map(p => p[0]));
  const xMax = graphData.x_max !== undefined ? graphData.x_max : Math.max(...points.map(p => p[0]));
  const yVals = points.map(p => p[1]);
  const yMin = Math.min(...yVals);
  const yMax = Math.max(...yVals);
  const yRange = yMax - yMin || 1;
  const xRange = xMax - xMin || 1;

  // Map data coords to canvas coords
  const cx = x => PAD + ((x - xMin) / xRange) * (W - PAD * 2);
  const cy = y => H - PAD - ((y - yMin) / yRange) * (H - PAD * 2);

  // Background
  ctx.fillStyle = '#0e0c1e';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  const gridX = 5, gridY = 4;
  for (let i = 0; i <= gridX; i++) {
    const x = PAD + (i / gridX) * (W - PAD * 2);
    ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, H - PAD); ctx.stroke();
  }
  for (let i = 0; i <= gridY; i++) {
    const y = PAD + (i / gridY) * (H - PAD * 2);
    ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
  }

  // Axes (only draw if zero is in range)
  ctx.strokeStyle = 'rgba(167,139,250,0.35)';
  ctx.lineWidth = 1;
  if (xMin <= 0 && xMax >= 0) {
    const zeroX = cx(0);
    ctx.beginPath(); ctx.moveTo(zeroX, PAD); ctx.lineTo(zeroX, H - PAD); ctx.stroke();
  }
  if (yMin <= 0 && yMax >= 0) {
    const zeroY = cy(0);
    ctx.beginPath(); ctx.moveTo(PAD, zeroY); ctx.lineTo(W - PAD, zeroY); ctx.stroke();
  }

  // X axis labels
  ctx.fillStyle = 'rgba(122,122,157,0.8)';
  ctx.font = '9px system-ui, sans-serif';
  ctx.textAlign = 'center';
  [xMin, 0, xMax].forEach(val => {
    if (val >= xMin && val <= xMax) {
      ctx.fillText(String(Math.round(val)), cx(val), H - 4);
    }
  });

  // Y axis labels
  ctx.textAlign = 'right';
  [yMin, (yMin + yMax) / 2, yMax].forEach(val => {
    ctx.fillText(String(Math.round(val)), PAD - 4, cy(val) + 3);
  });

  // Draw curve
  ctx.beginPath();
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  points.forEach(([x, y], i) => {
    const px = cx(x), py = cy(y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();

  // Curve label
  if (graphData.label) {
    ctx.fillStyle = '#a78bfa';
    ctx.font = 'bold 10px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(graphData.label, PAD + 4, PAD + 12);
  }
}

// Action buttons
$('math-btn').addEventListener('click', solveMath);

$('math-inp').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    solveMath();
  }
});

$('math-copy').addEventListener('click', function() {
  const answerEl = $('math-answer');
  const stepsEl = $('math-steps');
  const stepsText = Array.from(stepsEl.querySelectorAll('li'))
    .map((li, i) => `${i + 1}. ${li.textContent}`)
    .join('\n');
  const full = answerEl.textContent + (stepsText ? '\n\n' + stepsText : '');
  navigator.clipboard.writeText(full).then(() => {
    const btn = $('math-copy');
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy Solution', 1500);
  });
});

$('math-to-chat').addEventListener('click', function() {
  const problem = $('math-inp').value.trim();
  const solution = $('math-answer').textContent.trim();
  if (!problem || !solution) return;
  pendingCtx = `Math problem: ${problem}\n\nSolution: ${solution}`;
  showCtx(pendingCtx, 'Math Result');
  switchPanel('chat');
});
```

---

## 5. PILL_SLOTS ADDITION — update the existing PILL_SLOTS object in sidebar.js

Change:
```javascript
const PILL_SLOTS = {
  chat: 'mdl-chat', search: 'mdl-search', translate: 'mdl-translate',
  write: 'mdl-write', grammar: 'mdl-grammar', ask: 'mdl-ask',
  pdf: 'mdl-pdf', voice: 'mdl-voice',
};
```

To:
```javascript
const PILL_SLOTS = {
  chat: 'mdl-chat', search: 'mdl-search', translate: 'mdl-translate',
  write: 'mdl-write', grammar: 'mdl-grammar', ask: 'mdl-ask',
  pdf: 'mdl-pdf', voice: 'mdl-voice', math: 'mdl-math',
};
```

---

## 6. REGISTER ROUTE IN main.py

In `D:\Aura\api\main.py`, update the import line (line 30) to add `math as math_route`:

```python
from api.routes import chat, status, upload, features, multi_agent, reasoning_tree, introspection, proactive, memory, context, conversation_starters, thinking, idle_behaviors, self_improvement, thinking_mode, state_machine, tools_new, activity, multi_model, knowledge, search, pdf, transcribe, ocr, image_gen, agent_action, models as models_route, summarize, math as math_route
```

Then add after `app.include_router(summarize.router)`:

```python
app.include_router(math_route.router)
```

---

## 7. NOTES

- KaTeX is loaded lazily on first Solve click — no upfront bandwidth cost.
- If KaTeX CDN fails, raw LaTeX string is shown in monospace (no crash).
- Graph canvas is only shown when the backend returns `graph_data` with points.
- Ctrl+Enter in the textarea triggers solve (power user shortcut).
- "Send to Chat" populates the context bar and switches to the chat panel, so the user can ask follow-up questions about the solution.
- The backend JSON extraction handles: raw JSON, markdown-fenced JSON, JSON buried in prose.
- Default model is `gemini-3-flash-preview:cloud` (same as agent_action.py). Users can override per-feature via the model pill.
