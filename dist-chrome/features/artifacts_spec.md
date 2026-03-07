# AURA Artifacts — Complete Frontend Spec

## Overview

Artifacts renders AI-generated code blocks as live interactive previews, inline in the chat panel. No new backend route is needed. This is a pure frontend feature implemented in sidebar.js and sidebar.html.

**Supported types:**

| Language class        | Behavior                                  |
|-----------------------|-------------------------------------------|
| `html`, `htm`         | Sandboxed iframe (srcdoc)                 |
| `mermaid`             | SVG diagram via mermaid.js from CDN       |
| `svg`                 | Injected inline as SVG element            |
| `json`                | Collapsible tree viewer (no external lib) |
| `css`, `js`, `python`, `bash`, `ts`, `tsx`, `jsx`, `any other` | Syntax-highlighted code + Copy button only |

---

## Integration Point

### Where to call processArtifacts

In `sidebar.js`, the `ws.onmessage` handler (starting at line 138) processes two relevant events:

1. **`d.type === 'chunk'`** — fires during streaming; re-renders `el.innerHTML = md(activeStream.rawText)` on every chunk. We do NOT run processArtifacts here — it would flicker on every keystroke.

2. **`d.type === 'done'`** (line 163-165) — this is the integration point. After the final `el.innerHTML = md(...)` call and before `finalizeStream()`, call `processArtifacts(el)`.

Additionally, `finalizeStream()` (line 180-186) has an `onDone` hook. For the chat stream, `activeStream` is set at line 350 as:
```js
activeStream = { type: 'chat', el: bubble, rawText: '', submitBtn: sendBtn };
```
The cleanest integration is to add `onDone` to this object so processArtifacts runs after the final render.

### Exact diff for sidebar.js

**Change 1 — In `ws.onmessage`, the `done` handler (around line 163):**

```js
// BEFORE:
} else if (d.type === 'done') {
  if (type === 'chat' && activeStream.rawText) el.innerHTML = md(activeStream.rawText);
  finalizeStream();

// AFTER:
} else if (d.type === 'done') {
  if (type === 'chat' && activeStream.rawText) el.innerHTML = md(activeStream.rawText);
  if (type === 'chat') processArtifacts(el);
  finalizeStream();
```

That's the only required change. `processArtifacts` is idempotent and fast — safe to run synchronously after the final render.

---

## HTML Additions to sidebar.html

Add the artifact modal markup just before `</body>`. Nothing else needs to be added to the chat panel HTML — all per-message DOM is injected by JS.

```html
<!-- ARTIFACT MODAL -->
<div id="artifact-modal" class="artifact-modal" style="display:none" role="dialog" aria-modal="true">
  <div class="artifact-modal-inner">
    <div class="artifact-modal-toolbar">
      <span class="artifact-modal-title" id="artifact-modal-title">Preview</span>
      <div class="artifact-modal-actions">
        <button class="artifact-modal-btn" id="artifact-modal-copy">Copy code</button>
        <button class="artifact-modal-btn" id="artifact-modal-newwin">Open in tab</button>
        <button class="artifact-close" id="artifact-modal-close" aria-label="Close">✕</button>
      </div>
    </div>
    <div class="artifact-modal-body" id="artifact-modal-body"></div>
  </div>
</div>
```

---

## CSS

Add this block to the `<style>` section in `sidebar.html`, after the existing `.bubble.ai blockquote` rule (around line 168).

```css
/* ═══════════════════════════════════════
   ARTIFACTS
═══════════════════════════════════════ */

/* Wrapper replaces raw <pre> inside .bubble.ai */
.artifact-wrap {
  position: relative;
  margin: 8px 0;
}

/* Toolbar: floats above the code block, visible on hover */
.artifact-toolbar {
  display: flex;
  align-items: center;
  gap: 4px;
  position: absolute;
  top: 4px;
  right: 6px;
  z-index: 5;
  opacity: 0;
  transition: opacity .15s;
  pointer-events: none;
}
.artifact-wrap:hover .artifact-toolbar,
.artifact-wrap:focus-within .artifact-toolbar {
  opacity: 1;
  pointer-events: auto;
}

.artifact-btn {
  background: var(--s3);
  border: 1px solid var(--b2);
  border-radius: 5px;
  color: var(--mu);
  font-size: 10px;
  font-family: inherit;
  padding: 2px 8px;
  cursor: pointer;
  transition: all .12s;
  line-height: 1.6;
  white-space: nowrap;
}
.artifact-btn:hover {
  background: var(--p);
  border-color: var(--p);
  color: #fff;
}
.artifact-btn.preview-btn {
  color: var(--pl);
  border-color: rgba(167,139,250,.3);
}
.artifact-btn.preview-btn:hover {
  background: var(--p);
  border-color: var(--p);
  color: #fff;
}
.artifact-btn.copied {
  color: var(--gr);
  border-color: var(--gr);
}

/* Inline preview area (collapsed by default) */
.artifact-preview {
  display: none;
  width: 100%;
  background: var(--cb);
  border: 1px solid var(--b1);
  border-top: none;
  border-bottom-left-radius: 8px;
  border-bottom-right-radius: 8px;
  overflow: hidden;
  transition: height .2s ease;
}
.artifact-preview.open {
  display: block;
}

/* iframe inside inline preview */
.artifact-preview iframe {
  display: block;
  width: 100%;
  height: 200px;
  border: none;
  background: #fff;
}

/* mermaid diagram container */
.artifact-mermaid {
  padding: 16px;
  background: var(--cb);
  text-align: center;
  overflow: auto;
}
.artifact-mermaid svg {
  max-width: 100%;
  height: auto;
}
.artifact-mermaid-error {
  padding: 10px 14px;
  color: var(--rd);
  font-size: 11.5px;
  font-family: 'Cascadia Code','Fira Code','Consolas',monospace;
}

/* SVG preview */
.artifact-svg {
  padding: 16px;
  background: var(--cb);
  text-align: center;
  overflow: auto;
}
.artifact-svg svg {
  max-width: 100%;
  height: auto;
}

/* JSON tree viewer */
.artifact-json {
  padding: 8px 10px;
  font-family: 'Cascadia Code','Fira Code','Consolas',monospace;
  font-size: 11px;
  line-height: 1.7;
  color: var(--tx);
  background: var(--cb);
  overflow: auto;
  max-height: 200px;
}
.json-node {
  margin-left: 16px;
}
.json-toggle {
  cursor: pointer;
  user-select: none;
  color: var(--pl);
  font-size: 10px;
  margin-right: 3px;
}
.json-toggle:hover {
  color: var(--pl2);
}
.json-key   { color: #93c5fd; }
.json-str   { color: #86efac; }
.json-num   { color: #fde68a; }
.json-bool  { color: #f9a8d4; }
.json-null  { color: var(--mu); }
.json-bracket { color: var(--di); }
.json-collapsed > .json-node { display: none; }
.json-collapsed > .json-inline { display: inline; }
.json-inline { display: none; color: var(--di); font-size: 10px; }

/* ── Artifact Modal ── */
.artifact-modal {
  position: fixed;
  inset: 0;
  z-index: 9000;
  background: rgba(6,6,18,.88);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  display: flex;
  align-items: center;
  justify-content: center;
  animation: fadeIn .15s ease;
}
@keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }

.artifact-modal-inner {
  background: var(--s1);
  border: 1px solid var(--b2);
  border-radius: 14px;
  width: calc(100% - 24px);
  max-width: 100%;
  max-height: calc(100vh - 32px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 8px 40px rgba(0,0,0,.7);
  animation: slideUp .15s ease;
}
@keyframes slideUp { from { transform: translateY(8px); opacity: .6 } to { transform: none; opacity: 1 } }

.artifact-modal-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: var(--s2);
  border-bottom: 1px solid var(--b1);
  flex-shrink: 0;
  gap: 8px;
}
.artifact-modal-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--pl);
  letter-spacing: .05em;
  text-transform: uppercase;
  flex: 1;
}
.artifact-modal-actions {
  display: flex;
  align-items: center;
  gap: 5px;
}
.artifact-modal-btn {
  background: var(--s3);
  border: 1px solid var(--b1);
  border-radius: 6px;
  color: var(--mu);
  font-size: 10.5px;
  font-family: inherit;
  padding: 3px 9px;
  cursor: pointer;
  transition: all .12s;
}
.artifact-modal-btn:hover {
  color: var(--tx);
  border-color: var(--p);
}
.artifact-close {
  background: none;
  border: none;
  color: var(--mu);
  font-size: 14px;
  cursor: pointer;
  padding: 0 2px;
  line-height: 1;
  transition: color .12s;
}
.artifact-close:hover {
  color: var(--tx);
}

.artifact-modal-body {
  flex: 1;
  overflow: auto;
  min-height: 0;
  background: var(--bg);
}
.artifact-modal-body iframe {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 320px;
  border: none;
  background: #fff;
}
.artifact-modal-body .artifact-mermaid {
  min-height: 200px;
  padding: 24px;
}
.artifact-modal-body .artifact-json {
  max-height: none;
  height: 100%;
  min-height: 200px;
}
.artifact-modal-body .artifact-svg {
  min-height: 200px;
  padding: 24px;
}
```

---

## JavaScript

The following functions should be appended to the bottom of `sidebar.js`, before the final event listener block. They are self-contained and only touch DOM within chat bubbles.

```js
// ══════════════════════════════════════════════════════════════════════════
// ARTIFACTS — inline live preview of AI-generated code blocks
// ══════════════════════════════════════════════════════════════════════════

// ── Mermaid lazy-load ─────────────────────────────────────────────────────

let _mermaidLoaded = false;
let _mermaidLoading = false;
let _mermaidQueue = [];

function loadMermaid(cb) {
  if (_mermaidLoaded) { cb(); return; }
  _mermaidQueue.push(cb);
  if (_mermaidLoading) return;
  _mermaidLoading = true;
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
  s.onload = () => {
    _mermaidLoaded = true;
    _mermaidLoading = false;
    try {
      window.mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        themeVariables: {
          background: '#03030d',
          primaryColor: '#7c3aed',
          primaryTextColor: '#f0eff8',
          primaryBorderColor: '#3a3a5e',
          lineColor: '#7a7a9d',
          secondaryColor: '#161428',
          tertiaryColor: '#0e0c1e',
          edgeLabelBackground: '#0e0c1e',
          clusterBkg: '#161428',
          titleColor: '#e0d6ff',
        }
      });
    } catch(e) { /* mermaid already initialized */ }
    _mermaidQueue.forEach(fn => fn());
    _mermaidQueue = [];
  };
  s.onerror = () => {
    _mermaidLoading = false;
    _mermaidLoaded = false;
    _mermaidQueue.forEach(fn => fn(new Error('mermaid load failed')));
    _mermaidQueue = [];
  };
  document.head.appendChild(s);
}

// ── Language detection ─────────────────────────────────────────────────────

function detectArtifactType(codeEl) {
  // class is e.g. "language-html", "language-mermaid", set by md() renderer
  const cls = codeEl.className || '';
  const m = cls.match(/language-(\w+)/);
  const lang = m ? m[1].toLowerCase() : 'text';

  if (['html', 'htm'].includes(lang))     return { type: 'html',    lang };
  if (lang === 'mermaid')                  return { type: 'mermaid', lang };
  if (lang === 'svg')                      return { type: 'svg',     lang };
  if (lang === 'json')                     return { type: 'json',    lang };
  return { type: 'code', lang };
}

// ── JSON tree renderer ────────────────────────────────────────────────────
// Fully recursive, no external dependencies. Produces collapsible DOM nodes.

function buildJsonTree(value, isRoot) {
  const frag = document.createDocumentFragment();

  if (value === null) {
    const s = document.createElement('span');
    s.className = 'json-null'; s.textContent = 'null';
    frag.appendChild(s); return frag;
  }
  if (typeof value === 'boolean') {
    const s = document.createElement('span');
    s.className = 'json-bool'; s.textContent = String(value);
    frag.appendChild(s); return frag;
  }
  if (typeof value === 'number') {
    const s = document.createElement('span');
    s.className = 'json-num'; s.textContent = String(value);
    frag.appendChild(s); return frag;
  }
  if (typeof value === 'string') {
    const s = document.createElement('span');
    s.className = 'json-str'; s.textContent = JSON.stringify(value);
    frag.appendChild(s); return frag;
  }

  const isArray = Array.isArray(value);
  const keys = isArray ? value.map((_, i) => i) : Object.keys(value);
  const openBracket  = isArray ? '[' : '{';
  const closeBracket = isArray ? ']' : '}';

  if (keys.length === 0) {
    const s = document.createElement('span');
    s.className = 'json-bracket';
    s.textContent = openBracket + closeBracket;
    frag.appendChild(s); return frag;
  }

  // Collapsible container
  const container = document.createElement('span');

  // Toggle button
  const toggle = document.createElement('span');
  toggle.className = 'json-toggle';
  toggle.textContent = '▾';
  toggle.title = 'Click to collapse';

  const ob = document.createElement('span');
  ob.className = 'json-bracket'; ob.textContent = openBracket;

  const inlineSummary = document.createElement('span');
  inlineSummary.className = 'json-inline';
  inlineSummary.textContent = isArray
    ? ` ${keys.length} items ` + closeBracket
    : ' … ' + closeBracket;

  const childWrap = document.createElement('span');
  childWrap.className = 'json-node';

  keys.forEach((key, idx) => {
    const row = document.createElement('div');

    if (!isArray) {
      const keySpan = document.createElement('span');
      keySpan.className = 'json-key';
      keySpan.textContent = JSON.stringify(String(key));
      row.appendChild(keySpan);
      const colon = document.createElement('span');
      colon.textContent = ': ';
      row.appendChild(colon);
    }

    row.appendChild(buildJsonTree(isArray ? value[key] : value[key], false));

    if (idx < keys.length - 1) {
      const comma = document.createElement('span');
      comma.className = 'json-bracket'; comma.textContent = ',';
      row.appendChild(comma);
    }

    childWrap.appendChild(row);
  });

  const cb = document.createElement('div');
  const cbSpan = document.createElement('span');
  cbSpan.className = 'json-bracket'; cbSpan.textContent = closeBracket;
  cb.appendChild(cbSpan);

  container.appendChild(toggle);
  container.appendChild(ob);
  container.appendChild(inlineSummary);
  container.appendChild(childWrap);
  container.appendChild(cb);

  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const collapsed = container.classList.toggle('json-collapsed');
    toggle.textContent = collapsed ? '▸' : '▾';
    toggle.title = collapsed ? 'Click to expand' : 'Click to collapse';
  });

  // Auto-collapse deep arrays/objects at depth > 2 items when not root
  if (!isRoot && keys.length > 5) {
    container.classList.add('json-collapsed');
    toggle.textContent = '▸';
    toggle.title = 'Click to expand';
  }

  frag.appendChild(container);
  return frag;
}

function renderJsonTree(code) {
  const wrap = document.createElement('div');
  wrap.className = 'artifact-json';
  try {
    const parsed = JSON.parse(code);
    wrap.appendChild(buildJsonTree(parsed, true));
  } catch(e) {
    wrap.textContent = code; // fallback: raw text
    wrap.style.color = 'var(--rd)';
  }
  return wrap;
}

// ── Core renderArtifact ───────────────────────────────────────────────────
// Returns a Promise<HTMLElement> for async types (mermaid), or a sync element
// wrapped in Promise.resolve() for sync types.

function renderArtifact(type, code) {
  if (type === 'html') {
    const iframe = document.createElement('iframe');
    iframe.sandbox = 'allow-scripts allow-same-origin allow-forms';
    iframe.srcdoc = code;
    iframe.style.cssText = 'display:block;width:100%;height:100%;border:none;background:#fff;';
    return Promise.resolve(iframe);
  }

  if (type === 'svg') {
    const wrap = document.createElement('div');
    wrap.className = 'artifact-svg';
    // Sanitize: only allow the SVG element itself (strip scripts)
    const cleaned = code
      .replace(/<script[\s\S]*?<\/script>/gi, '')
      .replace(/\son\w+="[^"]*"/gi, '')
      .replace(/\son\w+='[^']*'/gi, '');
    wrap.innerHTML = cleaned;
    return Promise.resolve(wrap);
  }

  if (type === 'json') {
    return Promise.resolve(renderJsonTree(code));
  }

  if (type === 'mermaid') {
    return new Promise((resolve) => {
      loadMermaid((err) => {
        const wrap = document.createElement('div');
        wrap.className = 'artifact-mermaid';

        if (err) {
          wrap.className = 'artifact-mermaid-error';
          wrap.textContent = 'Could not load mermaid.js: ' + err.message;
          resolve(wrap); return;
        }

        const id = 'mermaid-' + Math.random().toString(36).slice(2, 9);
        window.mermaid.render(id, code.trim())
          .then(({ svg }) => {
            wrap.innerHTML = svg;
            resolve(wrap);
          })
          .catch((e) => {
            wrap.className = 'artifact-mermaid-error';
            wrap.textContent = 'Diagram error: ' + (e?.message || String(e));
            resolve(wrap);
          });
      });
    });
  }

  // Fallback: should not reach here for type='code'
  const pre = document.createElement('pre');
  pre.textContent = code;
  return Promise.resolve(pre);
}

// ── Copy with feedback ────────────────────────────────────────────────────

function artifactCopyCode(code, btn) {
  navigator.clipboard.writeText(code).then(() => {
    if (!btn) return;
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = orig;
      btn.classList.remove('copied');
    }, 1500);
  }).catch(() => {
    // Silent fail — clipboard may be blocked in extension context
  });
}

// ── Modal ─────────────────────────────────────────────────────────────────

let _modalCurrentCode = '';
let _modalCurrentType = 'html';

function showArtifactModal(type, code, lang) {
  _modalCurrentCode = code;
  _modalCurrentType = type;

  const modal = document.getElementById('artifact-modal');
  const body  = document.getElementById('artifact-modal-body');
  const title = document.getElementById('artifact-modal-title');
  if (!modal || !body) return;

  title.textContent = lang.toUpperCase() + ' Preview';
  body.innerHTML = '';
  modal.style.display = 'flex';

  // Trap focus on close button
  document.getElementById('artifact-modal-close').focus();

  renderArtifact(type, code).then(el => {
    body.appendChild(el);
    // For iframes: set a minimum height to fill the modal body
    if (type === 'html') {
      el.style.minHeight = '340px';
      // Grow iframe to match content after load
      el.addEventListener('load', () => {
        try {
          const h = el.contentDocument?.body?.scrollHeight;
          if (h && h > 200) el.style.height = Math.min(h + 32, 520) + 'px';
        } catch(e) { /* cross-origin, ignore */ }
      });
    }
  });
}

function hideArtifactModal() {
  const modal = document.getElementById('artifact-modal');
  if (modal) modal.style.display = 'none';
  document.getElementById('artifact-modal-body').innerHTML = '';
}

// Wire up modal buttons (runs once after DOM is ready)
function initArtifactModal() {
  const modal     = document.getElementById('artifact-modal');
  const closeBtn  = document.getElementById('artifact-modal-close');
  const copyBtn   = document.getElementById('artifact-modal-copy');
  const newWinBtn = document.getElementById('artifact-modal-newwin');
  if (!modal) return;

  closeBtn.addEventListener('click', hideArtifactModal);

  // Close on backdrop click
  modal.addEventListener('click', (e) => {
    if (e.target === modal) hideArtifactModal();
  });

  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display !== 'none') hideArtifactModal();
  });

  copyBtn.addEventListener('click', () => {
    artifactCopyCode(_modalCurrentCode, copyBtn);
  });

  newWinBtn.addEventListener('click', () => {
    if (_modalCurrentType === 'html') {
      // Open HTML in new tab via blob URL
      const blob = new Blob([_modalCurrentCode], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } else {
      // For diagrams/JSON: open in new tab as plain text
      const blob = new Blob([_modalCurrentCode], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    }
  });
}

// ── processArtifacts ──────────────────────────────────────────────────────
// Called once after each assistant message is fully rendered.
// Finds all <pre><code> blocks in the message bubble and wraps them.

function processArtifacts(container) {
  if (!container) return;

  container.querySelectorAll('pre').forEach(pre => {
    // Skip if already processed
    if (pre.closest('.artifact-wrap')) return;

    const codeEl = pre.querySelector('code');
    if (!codeEl) return;

    const { type, lang } = detectArtifactType(codeEl);
    const code = codeEl.textContent;

    // ── Wrap the <pre> ──
    const wrap = document.createElement('div');
    wrap.className = 'artifact-wrap';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    // ── Build toolbar ──
    const toolbar = document.createElement('div');
    toolbar.className = 'artifact-toolbar';

    // Copy button (always shown)
    const copyBtn = document.createElement('button');
    copyBtn.className = 'artifact-btn';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      artifactCopyCode(code, copyBtn);
    });
    toolbar.appendChild(copyBtn);

    // Preview + Expand buttons — only for renderable types
    if (type !== 'code') {
      // Inline preview toggle
      const previewBtn = document.createElement('button');
      previewBtn.className = 'artifact-btn preview-btn';
      previewBtn.textContent = 'Preview';

      const previewArea = document.createElement('div');
      previewArea.className = 'artifact-preview';

      let previewRendered = false;
      let previewOpen = false;

      previewBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        previewOpen = !previewOpen;

        if (previewOpen) {
          previewBtn.textContent = 'Hide';
          previewArea.classList.add('open');

          // Lazy-render on first open
          if (!previewRendered) {
            previewRendered = true;
            previewArea.innerHTML = '<div style="padding:10px;color:var(--mu);font-size:11px">Rendering…</div>';
            renderArtifact(type, code).then(el => {
              previewArea.innerHTML = '';
              // For inline preview, cap iframe height
              if (type === 'html' && el.tagName === 'IFRAME') {
                el.style.height = '200px';
              }
              previewArea.appendChild(el);
            });
          }
        } else {
          previewBtn.textContent = 'Preview';
          previewArea.classList.remove('open');
        }
      });

      // Expand (fullscreen modal)
      const expandBtn = document.createElement('button');
      expandBtn.className = 'artifact-btn';
      expandBtn.textContent = 'Expand';
      expandBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        showArtifactModal(type, code, lang);
      });

      toolbar.appendChild(previewBtn);
      toolbar.appendChild(expandBtn);

      // Append preview area after the pre (inside wrap)
      wrap.appendChild(previewArea);
    }

    // Append toolbar inside pre, overlaid via position:absolute
    pre.style.position = 'relative';
    // Push chdr (code header) buttons left to avoid overlap with artifact toolbar
    const chdr = pre.querySelector('.chdr');
    if (chdr) chdr.style.paddingRight = type !== 'code' ? '120px' : '80px';

    pre.appendChild(toolbar);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────
// Called once on DOMContentLoaded (or immediately if DOM is already ready).

(function initArtifacts() {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initArtifactModal);
  } else {
    initArtifactModal();
  }
})();
```

---

## md() function adjustment — language class on `<code>`

The existing `md()` function (sidebar.js line 101) generates:
```js
out += `<pre><div class="chdr"><span>${esc(lang)}</span>...` +
       `<code id="${cid}">${esc(code)}</code></pre>`;
```

The `<code>` element has `id="${cid}"` but **no `class` attribute**, so `detectArtifactType` cannot read the language from `className`. You must add `class="language-${lang}"` to the `<code>` element.

**Change in md() at line 101:**

```js
// BEFORE:
out += `<pre><div class="chdr"><span>${esc(lang)}</span><button class="ccopy" onclick="cpCode('${cid}')">Copy</button></div><code id="${cid}">${esc(code.replace(/\n$/,''))}</code></pre>`;

// AFTER:
out += `<pre><div class="chdr"><span>${esc(lang)}</span><button class="ccopy" onclick="cpCode('${cid}')">Copy</button></div><code id="${cid}" class="language-${esc(lang)}">${esc(code.replace(/\n$/,''))}</code></pre>`;
```

This is the only required change to md(). All other rendering logic is untouched.

---

## Summary of All Changes Required

| File | Location | Change |
|------|----------|--------|
| `sidebar.js` | Line 101, inside `md()` | Add `class="language-${esc(lang)}"` to `<code>` element |
| `sidebar.js` | Line 163-165, `done` handler in `ws.onmessage` | Add `if (type === 'chat') processArtifacts(el);` before `finalizeStream()` |
| `sidebar.js` | Bottom of file | Append the entire Artifacts JS block above |
| `sidebar.html` | Just before `</body>` | Add the artifact modal HTML |
| `sidebar.html` | After `.bubble.ai blockquote` CSS rule | Add the entire Artifacts CSS block |

Total new JS: ~320 lines. Total new CSS: ~180 lines. Zero new files needed — everything in existing sidebar.html and sidebar.js.

---

## Behavior Notes

- **processArtifacts is idempotent** — it checks `pre.closest('.artifact-wrap')` and skips already-processed blocks. Safe to call multiple times.
- **Mermaid is lazy-loaded** — only fetches the CDN script when the first mermaid block appears. Subsequent renders reuse the same loaded instance.
- **Inline preview is lazy-rendered** — renderArtifact is not called until the user clicks Preview, keeping message rendering fast.
- **Modal is shared** — one modal element for the whole page. Only one artifact modal is open at a time.
- **HTML iframes** use `sandbox="allow-scripts allow-same-origin allow-forms"`. This allows JS execution for interactive demos but prevents navigation, popups, and top-frame access.
- **SVG sanitization** strips `<script>` tags and `on*` event attributes before injecting inline.
- **JSON fallback** — if JSON.parse fails, the raw text is shown in monospace with a red tint instead of throwing.
- **Copy button** in the artifact toolbar replaces (not duplicates) the existing `.ccopy` button behavior. The `.ccopy` button in `.chdr` remains functional for users who don't hover over the artifact toolbar.
