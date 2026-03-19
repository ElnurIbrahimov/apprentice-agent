# YouTube Summarizer — Complete Integration Spec

All code below is exact and ready to paste. No placeholders.

---

## 1. HTML Panel — add inside `#main` alongside other `.panel` divs

Paste this block after the last existing `<div class="panel" id="panel-...">` and before the closing `</div>` of `#main`:

```html
<!-- ═══════════════════════════════════════
     YOUTUBE PANEL
═══════════════════════════════════════ -->
<div class="panel" id="panel-youtube">
  <!-- header row -->
  <div class="tool-hdr" style="padding:16px 14px 12px;flex-shrink:0;display:flex;align-items:center;justify-content:space-between;">
    <h2 style="font-size:18px;font-weight:700;">YouTube Summary</h2>
    <span id="mdl-youtube"></span>
  </div>

  <!-- auto-detect banner (hidden until YouTube tab detected) -->
  <div id="yt-auto" style="display:none;margin:0 14px 12px;padding:10px 12px;
      background:linear-gradient(135deg,#1a0533,#2d0d5e);
      border:1px solid var(--p);border-radius:10px;flex-shrink:0;">
    <div style="font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
        color:var(--pl);margin-bottom:4px;">YouTube detected</div>
    <div id="yt-auto-title" style="font-size:12px;color:var(--pl2);
        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:6px;"></div>
    <button id="yt-auto-btn" style="background:var(--p);border:none;border-radius:7px;
        color:#fff;font-size:12px;font-weight:600;font-family:inherit;
        padding:6px 14px;cursor:pointer;transition:background .13s;">
      Summarize this video
    </button>
  </div>

  <!-- manual URL input -->
  <div style="padding:0 14px 10px;flex-shrink:0;">
    <div style="display:flex;gap:6px;align-items:center;
        background:var(--bg);border:1px solid var(--b1);border-radius:12px;
        padding:8px 8px 8px 13px;transition:border-color .18s;" id="yt-input-box">
      <input id="yt-url-inp" type="text" placeholder="Paste YouTube URL…"
        style="flex:1;background:transparent;border:none;outline:none;
          color:var(--tx);font-family:inherit;font-size:13px;" />
      <button id="yt-summarize-btn"
        style="background:var(--p);border:none;border-radius:8px;color:#fff;
          cursor:pointer;display:flex;align-items:center;justify-content:center;
          width:30px;height:30px;transition:background .13s;flex-shrink:0;">
        <!-- play icon -->
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none"
            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
      </button>
    </div>
  </div>

  <!-- status / loading -->
  <div id="yt-status" style="padding:0 14px 6px;font-size:11.5px;color:var(--mu);
      flex-shrink:0;min-height:18px;display:flex;align-items:center;gap:8px;"></div>

  <!-- results area (scrollable) -->
  <div id="yt-results" style="flex:1;overflow-y:auto;padding:0 14px 16px;
      scrollbar-width:thin;scrollbar-color:var(--b1) transparent;display:none;">

    <!-- video info card -->
    <div id="yt-info-card" style="background:var(--s2);border:1px solid var(--b1);
        border-radius:10px;padding:12px 13px;margin-bottom:10px;">
      <div id="yt-res-title" style="font-size:14px;font-weight:700;color:var(--tx);
          margin-bottom:4px;line-height:1.35;"></div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <span id="yt-res-channel" style="font-size:11.5px;color:var(--pl);font-weight:600;"></span>
        <span id="yt-res-duration" style="font-size:11.5px;color:var(--mu);"></span>
      </div>
    </div>

    <!-- summary section -->
    <div style="margin-bottom:10px;">
      <div style="font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
          color:var(--mu);margin-bottom:6px;">Summary</div>
      <div id="yt-res-summary" style="font-size:13px;color:var(--tx);line-height:1.65;
          background:var(--s1);border:1px solid var(--b1);border-radius:10px;
          padding:11px 13px;"></div>
    </div>

    <!-- key points section -->
    <div style="margin-bottom:10px;">
      <div style="font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
          color:var(--mu);margin-bottom:6px;">Key Points</div>
      <ul id="yt-res-points" style="list-style:none;padding:0;margin:0;
          display:flex;flex-direction:column;gap:5px;"></ul>
    </div>

    <!-- transcript snippet (collapsible) -->
    <div>
      <button id="yt-snippet-toggle"
        style="background:none;border:none;color:var(--mu);font-size:11px;
          font-family:inherit;cursor:pointer;padding:0;margin-bottom:6px;
          display:flex;align-items:center;gap:4px;transition:color .13s;">
        <svg viewBox="0 0 24 24" width="11" height="11" fill="none"
            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
            id="yt-snippet-chevron">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
        Transcript snippet
      </button>
      <div id="yt-res-snippet" style="display:none;font-size:12px;color:var(--mu);
          line-height:1.6;background:var(--s1);border:1px solid var(--b1);
          border-radius:8px;padding:10px 12px;font-style:italic;"></div>
    </div>

  </div>
</div>
```

---

## 2. Rail Button — add to `#rail` in sidebar.html

Paste this after the existing `<button class="rbtn" data-panel="pdf" ...>` block and its separator,
or group it with media-related buttons. Recommended position: after the `pdf` button separator:

```html
<button class="rbtn" data-panel="youtube" data-tip="YouTube">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <rect x="2" y="5" width="20" height="14" rx="2" ry="2"/>
    <polygon points="10 9 15 12 10 15 10 9" fill="currentColor" stroke="none"/>
  </svg>
</button>
```

---

## 3. CSS — add inside the `<style>` block in sidebar.html

Add this after the `/* CHATPDF PANEL */` section (or at the end of the style block):

```css
/* ═══════════════════════════════════════
   YOUTUBE PANEL
═══════════════════════════════════════ */
#panel-youtube { gap: 0; }

#yt-input-box:focus-within { border-color: var(--p); }

#yt-summarize-btn:hover { background: var(--p2) !important; }

#yt-auto-btn:hover { background: var(--p2) !important; }

#yt-snippet-toggle:hover { color: var(--tx); }

#yt-snippet-toggle:hover #yt-snippet-chevron { stroke: var(--tx); }

.yt-kp {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  background: var(--s1);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 8px 11px;
  font-size: 12.5px;
  color: var(--tx);
  line-height: 1.5;
}

.yt-kp::before {
  content: '';
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--pl);
  flex-shrink: 0;
  margin-top: 5px;
}

#yt-status .dots { display: inline-flex; gap: 3px; }
#yt-status .dots span {
  width: 4px; height: 4px; border-radius: 50%;
  background: var(--pl); opacity: .35;
  animation: db 1.1s ease-in-out infinite;
}
#yt-status .dots span:nth-child(2) { animation-delay: .18s; }
#yt-status .dots span:nth-child(3) { animation-delay: .36s; }
```

---

## 4. sidebar.js additions

### 4a. State variable — add near the other panel state variables (around line 31, after `let agentRunning = false;`)

```js
// YouTube panel state
let ytAutoUrl = '';   // URL detected from active YouTube tab
```

### 4b. PILL_SLOTS addition — update the existing PILL_SLOTS object (around line 1367–1371)

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
  pdf: 'mdl-pdf', voice: 'mdl-voice', youtube: 'mdl-youtube',
};
```

### 4c. switchPanel hook — update the `switchPanel` function

Add this `else if` branch inside `switchPanel` after the `else if (name === 'models')` line:

```js
else if (name === 'youtube') {
  if (ytAutoUrl) {
    $('yt-url-inp').value = ytAutoUrl;
  }
  $('yt-url-inp').focus();
}
```

### 4d. Message listener — add inside the `ext.runtime.onMessage.addListener` callback in sidebar.js

Find the block that handles `PDF_TAB_DETECTED` (or the general message handler at the bottom of sidebar.js — search for `ext.runtime.onMessage`). Add this handler alongside it:

```js
ext.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'YT_TAB_DETECTED') {
    ytAutoUrl = msg.url;
    // Show auto-detect banner
    const banner = $('yt-auto');
    const titleEl = $('yt-auto-title');
    if (banner && titleEl) {
      titleEl.textContent = msg.title || msg.url;
      banner.style.display = 'block';
    }
    // If already on the youtube panel, populate input
    if (activePanel === 'youtube') {
      $('yt-url-inp').value = msg.url;
    }
  }
  if (msg.type === 'PDF_TAB_DETECTED') {
    // existing PDF handler — leave as-is, this is just context
  }
});
```

NOTE: If sidebar.js already has a single `ext.runtime.onMessage.addListener`, add the `YT_TAB_DETECTED` case inside it rather than registering a second listener. Look for the existing listener and add:

```js
case 'YT_TAB_DETECTED': {
  ytAutoUrl = msg.url;
  const banner = $('yt-auto');
  const titleEl = $('yt-auto-title');
  if (banner && titleEl) {
    titleEl.textContent = msg.title || msg.url;
    banner.style.display = 'block';
  }
  if (activePanel === 'youtube') {
    $('yt-url-inp').value = msg.url;
  }
  break;
}
```

### 4e. Core YouTube JS — paste this entire block at the end of sidebar.js (before the final closing if any)

```js
// ══════════════════════════════════════════════════════════════════════════
// YOUTUBE PANEL
// ══════════════════════════════════════════════════════════════════════════

async function loadYoutubeSummary(url) {
  url = (url || $('yt-url-inp').value || '').trim();
  if (!url) return;

  const statusEl   = $('yt-status');
  const resultsEl  = $('yt-results');
  const summarizeBtn = $('yt-summarize-btn');
  const autoBtn    = $('yt-auto-btn');

  // Loading state
  resultsEl.style.display = 'none';
  statusEl.innerHTML = '<div class="dots"><span></span><span></span><span></span></div> Fetching transcript…';
  summarizeBtn.disabled = true;
  if (autoBtn) autoBtn.disabled = true;

  try {
    const resp = await fetch(`${HTTP}/api/youtube/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
      signal: AbortSignal.timeout(90000),
    });

    if (!resp.ok) {
      let errMsg = `Error ${resp.status}`;
      try { const d = await resp.json(); errMsg = d.detail || errMsg; } catch {}
      statusEl.textContent = '⚠ ' + errMsg;
      return;
    }

    const data = await resp.json();
    renderYoutubeResults(data);
    statusEl.textContent = '';
    resultsEl.style.display = 'block';

  } catch (err) {
    if (err.name === 'TimeoutError') {
      statusEl.textContent = '⚠ Request timed out. The video may be very long.';
    } else if (err.name === 'TypeError' && err.message.includes('fetch')) {
      statusEl.textContent = '⚠ Cannot reach backend. Is the server running?';
    } else {
      statusEl.textContent = '⚠ ' + (err.message || 'Unknown error');
    }
  } finally {
    summarizeBtn.disabled = false;
    if (autoBtn) autoBtn.disabled = false;
  }
}

function renderYoutubeResults(data) {
  // Video info
  $('yt-res-title').textContent    = data.title    || 'Untitled Video';
  $('yt-res-channel').textContent  = data.channel  ? '▶ ' + data.channel : '';
  $('yt-res-duration').textContent = data.duration ? '⏱ ' + data.duration : '';

  // Summary
  $('yt-res-summary').textContent = data.summary || 'No summary available.';

  // Key points
  const ul = $('yt-res-points');
  ul.innerHTML = '';
  const points = Array.isArray(data.key_points) ? data.key_points : [];
  if (points.length === 0) {
    const li = document.createElement('li');
    li.className = 'yt-kp';
    li.textContent = 'No key points extracted.';
    ul.appendChild(li);
  } else {
    points.forEach(pt => {
      const li = document.createElement('li');
      li.className = 'yt-kp';
      li.textContent = pt;
      ul.appendChild(li);
    });
  }

  // Transcript snippet
  const snippet = data.transcript_snippet || '';
  $('yt-res-snippet').textContent = snippet;
  // Hide snippet area initially — user clicks toggle
  $('yt-res-snippet').style.display = 'none';
  // Reset chevron
  const chevron = $('yt-snippet-chevron');
  if (chevron) chevron.style.transform = '';
}

// Summarize button click
$('yt-summarize-btn').addEventListener('click', () => {
  loadYoutubeSummary($('yt-url-inp').value);
});

// Enter key in URL input
$('yt-url-inp').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); loadYoutubeSummary($('yt-url-inp').value); }
});

// Auto-detect "Summarize this video" button
$('yt-auto-btn').addEventListener('click', () => {
  if (ytAutoUrl) {
    $('yt-url-inp').value = ytAutoUrl;
    loadYoutubeSummary(ytAutoUrl);
  }
});

// Transcript snippet toggle
$('yt-snippet-toggle').addEventListener('click', () => {
  const snip    = $('yt-res-snippet');
  const chevron = $('yt-snippet-chevron');
  const open    = snip.style.display !== 'none';
  snip.style.display    = open ? 'none' : 'block';
  if (chevron) chevron.style.transform = open ? '' : 'rotate(90deg)';
});
```

---

## 5. background.js addition — YouTube tab detection

Add this block inside the existing `ext.tabs.onUpdated.addListener` callback, right after the PDF detection block (around line 53–61). Or add a second `tabs.onUpdated.addListener` call right after the first:

```js
// ── YouTube Tab Detection ─────────────────────────────────────────────────────

ext.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (
    changeInfo.status === 'complete' &&
    (tab.url || '').includes('youtube.com/watch')
  ) {
    ext.runtime.sendMessage({
      type: 'YT_TAB_DETECTED',
      url: tab.url,
      title: tab.title || tab.url,
    }).catch(() => {
      // Sidebar may not be open — that's fine
    });
  }
});
```

If you prefer to merge it into the existing `tabs.onUpdated` listener, replace the existing one with:

```js
ext.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete') return;
  const url = tab.url || '';

  // PDF detection
  if (/\.pdf($|\?)/i.test(url)) {
    ext.runtime.sendMessage({
      type: 'PDF_TAB_DETECTED',
      url: url,
      title: tab.title || url,
    });
  }

  // YouTube detection
  if (url.includes('youtube.com/watch')) {
    ext.runtime.sendMessage({
      type: 'YT_TAB_DETECTED',
      url: url,
      title: tab.title || url,
    }).catch(() => {});
  }
});
```

---

## 6. main.py addition — register the youtube router

In `D:\Aura\api\main.py`, add the import and router registration:

### Import line (around line 30, add `youtube` to the import):

```python
from api.routes import (
    chat, status, upload, features, multi_agent, reasoning_tree,
    introspection, proactive, memory, context, conversation_starters,
    thinking, idle_behaviors, self_improvement, thinking_mode,
    state_machine, tools_new, activity, multi_model, knowledge,
    search, pdf, transcribe, ocr, image_gen, agent_action,
    models as models_route,
    youtube,   # <-- ADD THIS
)
```

Or as a separate import line after the existing import:

```python
from api.routes import youtube
```

### Router registration (after `app.include_router(models_route.router)`, around line 379):

```python
app.include_router(youtube.router)
```

---

## 7. Dependencies

Install via pip (add to requirements.txt if you have one):

```
youtube-transcript-api>=0.6.2
httpx>=0.24.0   # already used by pdf.py, likely installed
```

Install command:
```
pip install youtube-transcript-api
```

`httpx` is already a dependency of the project (used in `pdf.py`).

---

## 8. Summary of what the feature does

- User navigates to a YouTube video in any tab → background.js detects the URL change and sends `YT_TAB_DETECTED` to sidebar
- Sidebar shows a purple auto-detect banner: "YouTube detected — [video title] — [Summarize this video]"
- User can also manually paste any YouTube URL (youtu.be, /watch, /shorts, /embed formats all supported)
- On click: POSTs to `POST /api/youtube/summarize`
- Backend extracts the 11-char video ID, fetches the transcript via `youtube-transcript-api` (supports auto-generated captions), truncates to 12,000 chars, fetches og:title/channel/duration from the YouTube page, then sends everything to `nemotron-3-super:cloud` via Ollama with a structured prompt
- LLM response is parsed into a summary paragraph + 5 key bullet points
- Results panel shows: video title + channel + duration card, summary block, key points list, collapsible transcript snippet
- Error states handled: no transcript (private/disabled), Ollama offline, invalid URL, timeout
- The model pill (id="mdl-youtube") is wired into PILL_SLOTS so users can override the LLM model per the existing model picker system
