/**
 * AURA Sidebar — sidebar.js
 * Multi-panel: Chat | Search | Translate | Write | Tools
 */

const HTTP = 'http://localhost:8000';
const WS   = 'ws://localhost:8000/api/chat/stream';

// Firefox compatibility shim
const ext = typeof browser !== 'undefined' ? browser : chrome;

// ── State ──────────────────────────────────────────────────────────────────
let ws = null, wsReady = false;
let conversationId = null;
let pendingCtx = null;
let activePanel = 'chat';
let thinkingMode = false;
let deepResearch = false;

// Active streaming context: { type, el, rawText, submitBtn, onFirstChunk? }
// type = 'chat' | 'search' | 'translate' | 'write'
let activeStream = null;

// Write panel state
let writeTab  = 'write';
let writeType = 'Essay';
let writeTone = 'Formal';
let writeLen  = 'Medium';

// New panel state
let pdfCtx = null;          // { text, page_count, word_count, filename? }
let lastOcrText = '';       // last OCR result
let imgStyle = '';          // selected image style suffix
let agentRunning = false;
let agentStep = 0;

// Summary panel state
let summaryFormat = 'bullets';
let summaryText = '';

// YouTube panel state
let ytAutoUrl = '';

// Compare panel state
let compareSelectedModels = new Set();
let compareInitialized = false;
const COMPARE_DEFAULT_MODELS = ['gemini-3-flash-preview:cloud','qwen3.5:397b-cloud','kimi-k2-thinking:cloud'];

// Research panel state
let resDepth = 'standard';

// Math panel state
let mathMode = 'solve';

// Artifacts panel state
let artCode = '';
let artLang = 'html';

// ── DOM ────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const msgs    = $('msgs');
const empty   = $('empty');
const inp     = $('inp');
const sendBtn = $('send');
const mdot    = $('mdot');
const mname   = $('mname');
const moodEl  = $('mood');
const ctxEl   = $('ctx');
const ctxLbl  = $('ctx-lbl');
const ctxTxt  = $('ctx-txt');

const searchInp     = $('search-inp');
const searchBtn     = $('search-btn');
const searchResults = $('search-results');
const searchEmpty   = $('search-empty');
const searchLoading = $('search-loading');

const trFrom = $('tr-from');
const trTo   = $('tr-to');
const trInp  = $('tr-inp');
const trBtn  = $('tr-btn');
const trOut  = $('tr-out');

const writeInp    = $('write-inp');
const writeSubmit = $('write-submit');
const writeResult = $('write-result');

// ── Markdown ───────────────────────────────────────────────────────────────

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
           .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function inline(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/__(.+?)__/g,'<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g,'<em>$1</em>')
    .replace(/_([^_\n]+)_/g,'<em>$1</em>')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\[(.+?)\]\((https?:\/\/[^\)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function md(raw) {
  let out = '', listBuf = '', listType = null;
  const flush = () => {
    if (!listType) return;
    out += `<${listType}>${listBuf}</${listType}>`;
    listBuf = ''; listType = null;
  };
  const lines = raw.split('\n');
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) {
      flush();
      const lang = l.slice(3).trim() || 'text';
      let code = ''; i++;
      while (i < lines.length && !/^```/.test(lines[i])) { code += lines[i] + '\n'; i++; }
      const cid = 'c' + Math.random().toString(36).slice(2,8);
      out += `<pre><div class="chdr"><span>${esc(lang)}</span><button class="ccopy" onclick="cpCode('${cid}')">Copy</button></div><code id="${cid}">${esc(code.replace(/\n$/,''))}</code></pre>`;
      i++; continue;
    }
    const hm = l.match(/^(#{1,3})\s+(.+)/);
    if (hm) { flush(); out += `<h${hm[1].length}>${inline(hm[2])}</h${hm[1].length}>`; i++; continue; }
    if (/^---+$/.test(l.trim())) { flush(); out += '<hr>'; i++; continue; }
    const bq = l.match(/^>\s*(.*)/);
    if (bq) { flush(); out += `<blockquote>${inline(bq[1])}</blockquote>`; i++; continue; }
    const ul = l.match(/^[-*]\s+(.+)/);
    if (ul) { if (listType !== 'ul') { flush(); listType='ul'; } listBuf += `<li>${inline(ul[1])}</li>`; i++; continue; }
    const ol = l.match(/^\d+\.\s+(.+)/);
    if (ol) { if (listType !== 'ol') { flush(); listType='ol'; } listBuf += `<li>${inline(ol[1])}</li>`; i++; continue; }
    if (l.trim() === '') { flush(); out += '<p></p>'; i++; continue; }
    flush();
    out += `<p>${inline(l)}</p>`; i++;
  }
  flush();
  return out.replace(/(<p><\/p>){2,}/g,'<p></p>');
}

window.cpCode = function(id) {
  const el = document.getElementById(id);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent).then(() => {
    const btn = el.closest('pre').querySelector('.ccopy');
    if (btn) { btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 1500); }
  });
};

// ── WebSocket ──────────────────────────────────────────────────────────────

function connectWS() {
  if (ws && ws.readyState <= 1) return;
  ws = new WebSocket(WS);
  ws.onopen  = () => { wsReady = true; setOnline(true); };
  ws.onclose = () => { wsReady = false; setOnline(false); setTimeout(connectWS, 5000); };
  ws.onerror = () => { wsReady = false; setOnline(false); };
  ws.onmessage = (ev) => {
    let d; try { d = JSON.parse(ev.data); } catch { return; }
    if (!activeStream) return;

    const { type, el } = activeStream;

    if (d.type === 'chunk') {
      // First chunk: fire onFirstChunk hook if present
      if (activeStream.onFirstChunk) {
        activeStream.onFirstChunk();
        activeStream.onFirstChunk = null;
      }
      activeStream.rawText += d.content || '';

      if (type === 'chat') {
        el.querySelector('.dots')?.remove();
        el.innerHTML = md(activeStream.rawText);
        msgs.scrollTop = msgs.scrollHeight;
      } else if (type === 'translate') {
        el.textContent = activeStream.rawText;
        el.classList.add('has-text');
      } else {
        // search or write — render markdown
        el.innerHTML = md(activeStream.rawText);
      }
    } else if (d.type === 'done') {
      if (type === 'chat' && activeStream.rawText) el.innerHTML = md(activeStream.rawText);
      finalizeStream();
    } else if (d.type === 'error') {
      const errMsg = d.content || d.error || 'Error';
      if (activeStream.onFirstChunk) { activeStream.onFirstChunk(); activeStream.onFirstChunk = null; }
      if (type === 'chat') {
        el.innerHTML = `<em style="color:#f87171">⚠ ${esc(errMsg)}</em>`;
      } else {
        el.textContent = '⚠ ' + errMsg;
        el.classList.add('has-text');
      }
      finalizeStream();
    }
  };
}

function finalizeStream() {
  if (!activeStream) return;
  const stream = activeStream; // capture before nulling
  if (stream.submitBtn) stream.submitBtn.disabled = false;
  activeStream = null;
  if (stream.onDone) stream.onDone(stream.rawText); // pass rawText for post-processing
}

// ── Status ─────────────────────────────────────────────────────────────────

async function fetchStatus() {
  try {
    const r = await fetch(`${HTTP}/api/status`, { signal: AbortSignal.timeout(4000) });
    if (!r.ok) { setOnline(false); return; }
    const d = await r.json();
    setOnline(true);
    if (!wsReady) connectWS();
    const m = (d.last_model_used || d.model || '').replace(/:cloud$/, '');
    mname.textContent = m.length > 22 ? m.slice(-22) : m;
    mname.title = d.last_model_used || '';
    if (d.mood?.emoji) moodEl.textContent = d.mood.emoji;
  } catch { setOnline(false); }
}

function setOnline(on) {
  mdot.classList.toggle('on', on);
  if (!on) { mname.textContent = 'offline'; moodEl.textContent = ''; }
}

// ── Panel switching ────────────────────────────────────────────────────────

function switchPanel(name) {
  activePanel = name;
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  const panel = $('panel-' + name);
  if (panel) panel.classList.add('active');
  document.querySelectorAll('.rbtn[data-panel]').forEach(b => {
    b.classList.toggle('on', b.dataset.panel === name);
  });
  if (name === 'chat') inp.focus();
  else if (name === 'search') searchInp.focus();
  else if (name === 'translate') trInp.focus();
  else if (name === 'write') writeInp.focus();
  else if (name === 'wisebase') loadWisebase();
  else if (name === 'models') loadModelPanel();
  else if (name === 'summary') { /* panel is ready; user clicks Summarize to trigger */ }
  else if (name === 'youtube') { if (ytAutoUrl) $('yt-url-inp').value = ytAutoUrl; }
  else if (name === 'compare') initComparePanel();
  else if (name === 'research') { /* ready */ }
  else if (name === 'math') { /* ready */ }
  else if (name === 'artifacts') { /* ready */ }
}

document.querySelectorAll('.rbtn[data-panel]').forEach(btn => {
  btn.addEventListener('click', () => switchPanel(btn.dataset.panel));
});

$('rail-clear').addEventListener('click', clearAll);

// ── Context bar ────────────────────────────────────────────────────────────

function showCtx(text, label) {
  ctxLbl.textContent = label || 'Context';
  ctxTxt.textContent = text.slice(0, 200) + (text.length > 200 ? '…' : '');
  ctxEl.classList.add('on');
}

function clearCtx() {
  pendingCtx = null;
  ctxEl.classList.remove('on');
}

$('ctx-x').addEventListener('click', clearCtx);

// ── Chat messages ──────────────────────────────────────────────────────────

function hideEmpty() { if (empty) empty.style.display = 'none'; }
function scrollBot()  { msgs.scrollTop = msgs.scrollHeight; }

function ts() {
  const n = new Date();
  return n.getHours().toString().padStart(2,'0') + ':' + n.getMinutes().toString().padStart(2,'0');
}

function addUserMsg(text) {
  hideEmpty();
  const row = document.createElement('div');
  row.className = 'mrow user';
  row.innerHTML = `
    <div class="mwrap">
      <div class="bubble user">${esc(text)}</div>
      <div class="ts">${ts()}</div>
    </div>
    <div class="av u-av">E</div>`;
  msgs.appendChild(row);
  scrollBot();
}

function addAIMsg() {
  hideEmpty();
  const row = document.createElement('div');
  row.className = 'mrow ai';

  const bubble = document.createElement('div');
  bubble.className = 'bubble ai';
  bubble.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';

  const rxbar = document.createElement('div');
  rxbar.className = 'rxbar';
  rxbar.innerHTML = `
    <button class="rx" data-rx="copy">
      <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
      Copy
    </button>
    <button class="rx" data-rx="good">
      <svg viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z"/><path d="M7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3"/></svg>
    </button>
    <button class="rx" data-rx="bad">
      <svg viewBox="0 0 24 24"><path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17"/></svg>
    </button>`;

  rxbar.querySelector('[data-rx=copy]').addEventListener('click', function() {
    navigator.clipboard.writeText(bubble.innerText).then(() => {
      this.textContent = '✓ Copied';
      this.classList.add('ok');
      setTimeout(() => {
        this.innerHTML = '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg> Copy';
        this.classList.remove('ok');
      }, 1500);
    });
  });

  const wrap = document.createElement('div');
  wrap.className = 'mwrap';
  const tsEl = document.createElement('div');
  tsEl.className = 'ts';
  tsEl.textContent = ts();
  wrap.appendChild(bubble);
  wrap.appendChild(rxbar);
  wrap.appendChild(tsEl);

  row.innerHTML = '<div class="av ai-av">A</div>';
  row.appendChild(wrap);
  msgs.appendChild(row);
  scrollBot();
  return bubble;
}

function sysmsg(text) {
  const el = document.createElement('div');
  el.style.cssText = 'text-align:center;color:var(--mu);font-size:11px;padding:6px 0;';
  el.textContent = text;
  msgs.appendChild(el);
  scrollBot();
}

// ── Send (chat) ────────────────────────────────────────────────────────────

// Page-context keywords — if any match, inject full page text automatically
const PAGE_KEYWORDS = /\b(this page|this site|this article|this post|this video|current page|what('s| is) (this|the) (page|site|article)|summarize this|explain this|what does this (say|mean)|translate this|tldr|tl;dr)\b/i;

async function sendMessage(override, modelKey) {
  const text = (override !== undefined ? override : inp.value).trim();
  if (!text) return;
  if (!wsReady) { sysmsg('AURA is offline — start the backend server.'); return; }
  if (activeStream) return;

  let full = text;

  // ── Auto page context (like Sider) ────────────────────────────────────────
  if (!pendingCtx) {
    const wantsFullPage = PAGE_KEYWORDS.test(text);
    if (wantsFullPage) {
      // Full page text for page-specific questions
      const pageResp = await new Promise(r => ext.runtime.sendMessage({ type: 'GET_PAGE_CONTENT' }, r));
      if (pageResp?.ok && pageResp.text) {
        pendingCtx = { text: pageResp.text.slice(0, 20000), title: pageResp.title, url: pageResp.url, action: 'ask' };
        showCtx(pageResp.text, pageResp.title || 'Current page');
      }
    } else {
      // Always inject URL + title so the model knows what page you're on
      const tabResp = await new Promise(r => ext.runtime.sendMessage({ type: 'GET_CURRENT_TAB' }, r));
      if (tabResp?.ok && tabResp.url && !tabResp.url.startsWith('chrome://') && !tabResp.url.startsWith('about:')) {
        full = `[Current page: ${tabResp.title || tabResp.url} — ${tabResp.url}]\n\n${text}`;
      }
    }
  }

  if (pendingCtx) {
    full = `[Context: ${pendingCtx.title || pendingCtx.url || 'selection'}]\n${pendingCtx.text}\n\n---\n${text}`;
    clearCtx();
  }
  if (thinkingMode) full = '[Think step by step before answering]\n' + full;
  if (deepResearch)  full = '[Do deep research on this, search the web if needed]\n' + full;

  addUserMsg(text);
  if (override === undefined) { inp.value = ''; autoH(); }
  sendBtn.disabled = true;
  const bubble = addAIMsg();
  activeStream = { type: 'chat', el: bubble, rawText: '', submitBtn: sendBtn };
  ws.send(JSON.stringify({ type: 'chat', message: full, conversation_id: conversationId, model: getModel(modelKey || 'chat') }));
}

// ── Mode pills ─────────────────────────────────────────────────────────────

$('m-think').addEventListener('click', function() {
  thinkingMode = !thinkingMode;
  this.classList.toggle('on', thinkingMode);
});

$('m-research').addEventListener('click', function() {
  deepResearch = !deepResearch;
  this.classList.toggle('on', deepResearch);
});

$('m-page').addEventListener('click', function() {
  loadPage();
});

// ── Page loading ───────────────────────────────────────────────────────────

function loadPage() {
  return new Promise(resolve => {
    ext.runtime.sendMessage({ type: 'GET_PAGE_CONTENT' }, resp => {
      if (resp?.ok) {
        pendingCtx = { text: resp.text, title: resp.title, url: resp.url, action: 'ask' };
        showCtx(resp.text, resp.title || 'Current page');
        if (activePanel === 'chat') sysmsg(`Page loaded: "${(resp.title || 'page').slice(0,40)}"`);
      } else {
        if (activePanel === 'chat') sysmsg('Could not read page content.');
      }
      resolve(resp);
    });
  });
}

// ── Suggestion chips ───────────────────────────────────────────────────────

document.querySelectorAll('.chip').forEach(c =>
  c.addEventListener('click', () => sendMessage(c.dataset.q))
);

// ── Input handlers ─────────────────────────────────────────────────────────

sendBtn.addEventListener('click', () => sendMessage());
inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
inp.addEventListener('input', autoH);

function autoH() {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 140) + 'px';
}

// ── Clear ──────────────────────────────────────────────────────────────────

function clearAll() {
  while (msgs.firstChild) msgs.removeChild(msgs.firstChild);
  msgs.appendChild(empty);
  empty.style.display = '';
  conversationId = null;
  clearCtx();
  thinkingMode = false;
  deepResearch = false;
  $('m-think').classList.remove('on');
  $('m-research').classList.remove('on');
}

$('btn-new').addEventListener('click', clearAll);

// ══════════════════════════════════════════════════════════════════════════
// SEARCH PANEL
// ══════════════════════════════════════════════════════════════════════════

async function doSearch(q) {
  q = q.trim();
  if (!q) return;

  // Reset results area
  [...searchResults.querySelectorAll('.src-section,.ans-section,.ans-label')].forEach(el => el.remove());
  searchEmpty.style.display = 'none';
  searchLoading.classList.add('on');
  searchBtn.disabled = true;

  try {
    const searchModel = getModel('search');
    const searchUrl = `${HTTP}/api/search?q=${encodeURIComponent(q)}&limit=5` + (searchModel ? `&model=${encodeURIComponent(searchModel)}` : '');
    const res = await fetch(searchUrl);
    const data = await res.json();

    searchLoading.classList.remove('on');

    // Render source cards
    if (data.sources && data.sources.length > 0) {
      const srcSection = document.createElement('div');
      srcSection.className = 'src-section';
      const srcLabel = document.createElement('div');
      srcLabel.className = 'src-label';
      srcLabel.textContent = 'Sources';
      const srcCards = document.createElement('div');
      srcCards.className = 'src-cards';
      data.sources.forEach((src, i) => {
        const card = document.createElement('a');
        card.className = 'src-card';
        card.href = src.url;
        card.target = '_blank';
        card.rel = 'noopener';
        card.innerHTML = `<div class="src-card-n">${i + 1}</div>` +
          `<div class="src-card-t">${esc(src.title)}</div>` +
          `<div class="src-card-u">${esc(src.url)}</div>`;
        srcCards.appendChild(card);
      });
      srcSection.appendChild(srcLabel);
      srcSection.appendChild(srcCards);
      searchResults.appendChild(srcSection);
    }

    // Render answer
    if (data.answer) {
      const lbl = document.createElement('div');
      lbl.className = 'src-label ans-label';
      lbl.style.margin = '14px 0 8px';
      lbl.textContent = 'Answer';
      searchResults.appendChild(lbl);
      const ansDiv = document.createElement('div');
      ansDiv.className = 'ans-section';
      ansDiv.innerHTML = md(data.answer);
      searchResults.appendChild(ansDiv);
    }

    if (!data.sources?.length && !data.answer) {
      searchEmpty.style.display = '';
    }
  } catch (err) {
    searchLoading.classList.remove('on');
    searchEmpty.style.display = '';
    searchEmpty.querySelector('p').textContent = 'Search error: ' + err.message;
  } finally {
    searchBtn.disabled = false;
  }
}

searchBtn.addEventListener('click', () => doSearch(searchInp.value));
searchInp.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(searchInp.value); });

// ══════════════════════════════════════════════════════════════════════════
// TRANSLATE PANEL
// ══════════════════════════════════════════════════════════════════════════

trBtn.addEventListener('click', () => {
  const text = trInp.value.trim();
  if (!text) return;
  if (!wsReady) { trOut.textContent = 'AURA is offline.'; trOut.classList.add('has-text'); return; }
  if (activeStream) return;

  const from = trFrom.value === 'auto' ? 'the detected language' : trFrom.value;
  const to   = trTo.value;

  trOut.textContent = 'Translating…';
  trOut.classList.remove('has-text');
  trBtn.disabled = true;

  activeStream = {
    type: 'translate',
    el: trOut,
    rawText: '',
    submitBtn: trBtn,
    onFirstChunk: () => { trOut.textContent = ''; }
  };

  ws.send(JSON.stringify({
    type: 'chat',
    message: `Translate the following text from ${from} to ${to}. Output only the translation, no explanation:\n\n${text}`,
    model: getModel('translate'),
    conversation_id: null
  }));
});

// ══════════════════════════════════════════════════════════════════════════
// WRITE PANEL
// ══════════════════════════════════════════════════════════════════════════

// Tab switching
document.querySelectorAll('.wtab').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.wtab').forEach(b => b.classList.remove('on'));
    this.classList.add('on');
    writeTab = this.dataset.tab;
    writeInp.placeholder = writeTab === 'improve'
      ? 'Paste text to improve…'
      : 'Enter your topic or prompt…';
    writeResult.classList.remove('on');
    writeResult.innerHTML = '';
  });
});

// Content type selection
document.querySelectorAll('.wtype').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.wtype').forEach(b => b.classList.remove('on'));
    this.classList.add('on');
    writeType = this.dataset.type;
  });
});

// Tone / length option groups
document.querySelectorAll('.wopt[data-opt="tone"]').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.wopt[data-opt="tone"]').forEach(b => b.classList.remove('on'));
    this.classList.add('on');
    writeTone = this.dataset.val;
  });
});

document.querySelectorAll('.wopt[data-opt="len"]').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.wopt[data-opt="len"]').forEach(b => b.classList.remove('on'));
    this.classList.add('on');
    writeLen = this.dataset.val;
  });
});

writeSubmit.addEventListener('click', () => {
  const text = writeInp.value.trim();
  if (!text) return;
  if (!wsReady) { alert('AURA is offline.'); return; }
  if (activeStream) return;

  writeResult.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';
  writeResult.classList.add('on');
  writeSubmit.disabled = true;

  let prompt;
  if (writeTab === 'improve') {
    prompt = `Improve the following text. Make it ${writeTone.toLowerCase()} in tone and ${writeLen.toLowerCase()} in length. Output only the improved text:\n\n${text}`;
  } else {
    prompt = `Write a ${writeTone.toLowerCase()}, ${writeLen.toLowerCase()}-length ${writeType} about: ${text}`;
  }

  activeStream = {
    type: 'write',
    el: writeResult,
    rawText: '',
    submitBtn: writeSubmit,
    onFirstChunk: () => { writeResult.innerHTML = ''; }
  };

  ws.send(JSON.stringify({ type: 'chat', message: prompt, model: getModel('write'), conversation_id: null }));
});

// ══════════════════════════════════════════════════════════════════════════
// TOOLS PANEL
// ══════════════════════════════════════════════════════════════════════════

document.querySelectorAll('.tool-card').forEach(card => {
  card.addEventListener('click', () => handleToolAction(card.dataset.action));
});

function handleToolAction(action) {
  switch (action) {
    case 'summarize-page':
      switchPanel('chat');
      loadPage().then(resp => {
        if (resp?.ok) sendMessage('Please summarize this page concisely.');
      });
      break;
    case 'key-points':
      switchPanel('chat');
      loadPage().then(resp => {
        if (resp?.ok) sendMessage('Extract the 5 most important key points from this page.');
      });
      break;
    case 'explain-page':
      switchPanel('chat');
      loadPage().then(resp => {
        if (resp?.ok) sendMessage('Explain this page content in simple terms for a general audience.');
      });
      break;
    case 'questions':
      switchPanel('chat');
      loadPage().then(resp => {
        if (resp?.ok) sendMessage('Generate 5 insightful questions based on this page content.');
      });
      break;
    case 'deep-research':
      switchPanel('search');
      searchInp.focus();
      searchInp.placeholder = 'Enter topic for deep research…';
      break;
    case 'fact-check':
      switchPanel('search');
      searchInp.value = 'Fact check: ';
      searchInp.focus();
      searchInp.selectionStart = searchInp.selectionEnd = searchInp.value.length;
      break;
    case 'write-essay':
      switchPanel('write');
      document.querySelectorAll('.wtab').forEach(b => b.classList.remove('on'));
      document.querySelector('.wtab[data-tab="write"]').classList.add('on');
      writeTab = 'write';
      document.querySelectorAll('.wtype').forEach(b => b.classList.remove('on'));
      document.querySelector('.wtype[data-type="Essay"]').classList.add('on');
      writeType = 'Essay';
      writeInp.focus();
      break;
    case 'improve-writing':
      switchPanel('write');
      document.querySelectorAll('.wtab').forEach(b => b.classList.remove('on'));
      document.querySelector('.wtab[data-tab="improve"]').classList.add('on');
      writeTab = 'improve';
      writeInp.placeholder = 'Paste text to improve…';
      writeInp.focus();
      break;
  }
}

// ── Background messages ────────────────────────────────────────────────────

ext.runtime.onMessage.addListener(msg => {
  if (msg.type === 'PREFILL_TEXT') {
    pendingCtx = { text: msg.text, action: msg.action || 'ask', url: msg.url || '', title: msg.title || '' };
    // Populate Ask panel fields
    const selTxt = $('ask-sel-txt');
    const selSrc = $('ask-src');
    if (selTxt) selTxt.textContent = msg.text.slice(0, 300) + (msg.text.length > 300 ? '…' : '');
    if (selSrc) selSrc.textContent = msg.title || msg.url || '';
    switchPanel('ask');
  }
  if (msg.type === 'YT_TAB_DETECTED') {
    ytAutoUrl = msg.url;
    const banner = $('yt-auto');
    const titleEl = $('yt-auto-title');
    if (banner && titleEl) { titleEl.textContent = msg.title || msg.url; banner.style.display = 'block'; }
    if (activePanel === 'youtube' && $('yt-url-inp')) $('yt-url-inp').value = msg.url;
  }
  if (msg.type === 'PDF_TAB_DETECTED') {
    const autoDiv = $('pdf-auto');
    const titleDiv = $('pdf-auto-title');
    const btn = $('pdf-auto-btn');
    if (autoDiv) {
      autoDiv.classList.add('on');
      if (titleDiv) titleDiv.textContent = msg.title || msg.url;
      if (btn) btn.onclick = () => loadPdfUrl(msg.url);
    }
  }
  if (msg.type === 'OCR_RESULT') {
    if (msg.error && msg.error !== '' && msg.error !== 'Cancelled') {
      $('ocr-result').textContent = '⚠ OCR error: ' + msg.error;
    } else if (msg.text) {
      lastOcrText = msg.text;
      $('ocr-result').textContent = msg.text;
      $('ocr-actions').style.display = '';
    }
  }
  if (msg.type === 'SWITCH_PANEL' && msg.panel) {
    switchPanel(msg.panel);
  }
});

// ── Init ───────────────────────────────────────────────────────────────────

ext.runtime.sendMessage({ type: 'SIDEBAR_READY' });
fetchStatus();
setInterval(fetchStatus, 30_000);
connectWS();

// ══════════════════════════════════════════════════════════════════════════
// ASK PANEL (Step 3)
// ══════════════════════════════════════════════════════════════════════════

document.querySelectorAll('.ask-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (!pendingCtx?.text) return;
    const msg = btn.dataset.prompt + pendingCtx.text;
    showCtx(pendingCtx.text, pendingCtx.title || pendingCtx.url || 'Selection');
    switchPanel('chat');
    sendMessage(msg, 'ask');
  });
});

$('ask-send').addEventListener('click', () => {
  const q = $('ask-inp').value.trim();
  if (!q) return;
  const msg = pendingCtx?.text ? q + '\n\n[Context]\n' + pendingCtx.text : q;
  showCtx(pendingCtx?.text || q, pendingCtx?.title || 'Question');
  switchPanel('chat');
  sendMessage(msg, 'ask');
});

$('ask-inp').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); $('ask-send').click(); }
});

// ══════════════════════════════════════════════════════════════════════════
// WISEBASE PANEL (Step 4)
// ══════════════════════════════════════════════════════════════════════════

async function loadWisebase(query = '') {
  const list = $('wb-list');
  list.innerHTML = '<div id="wb-empty" style="text-align:center;padding:30px;color:var(--mu);font-size:12px">Loading…</div>';

  try {
    const url = query
      ? `${HTTP}/api/knowledge/search?q=${encodeURIComponent(query)}&limit=20`
      : `${HTTP}/api/knowledge/list?limit=20`;
    const res = await fetch(url);
    const data = await res.json();

    const items = query
      ? (data.results || []).map(r => ({
          episode_id: r.episode_id, content: r.content,
          title: r.title, url: r.url, saved_at: r.saved_at,
        }))
      : (data.items || []);

    renderWbCards(items);
  } catch (err) {
    list.innerHTML = `<div style="text-align:center;padding:30px;color:var(--mu);font-size:12px">Error: ${esc(err.message)}</div>`;
  }
}

function renderWbCards(items) {
  const list = $('wb-list');
  list.innerHTML = '';
  if (!items.length) {
    list.innerHTML = '<div id="wb-empty" style="text-align:center;padding:30px;color:var(--mu);font-size:12px">No saved clips found.</div>';
    return;
  }
  items.forEach(item => {
    const card = document.createElement('div');
    card.className = 'wb-card';
    const dateStr = item.saved_at ? new Date(item.saved_at).toLocaleDateString() : '';
    const title = item.title || item.url || item.episode_id.slice(0, 8);
    const snippet = item.content.slice(0, 120);
    card.innerHTML = `
      <div class="wb-card-title">${esc(title)}</div>
      <div class="wb-card-snippet">${esc(snippet)}</div>
      <div class="wb-card-date">${esc(dateStr)}</div>
      <button class="wb-card-del" title="Delete">×</button>`;

    // Click card → load into context + switch to chat
    card.addEventListener('click', e => {
      if (e.target.classList.contains('wb-card-del')) return;
      pendingCtx = { text: item.content, title: title, url: item.url || '' };
      showCtx(item.content, title);
      switchPanel('chat');
    });

    // Delete
    card.querySelector('.wb-card-del').addEventListener('click', async e => {
      e.stopPropagation();
      if (!confirm('Delete this clip?')) return;
      try {
        await fetch(`${HTTP}/api/knowledge/${encodeURIComponent(item.episode_id)}`, { method: 'DELETE' });
        card.remove();
      } catch (err) {
        alert('Delete failed: ' + err.message);
      }
    });

    list.appendChild(card);
  });
}

$('wb-sbtn').addEventListener('click', () => loadWisebase($('wb-inp').value.trim()));
$('wb-all').addEventListener('click', () => { $('wb-inp').value = ''; loadWisebase(); });
$('wb-inp').addEventListener('keydown', e => { if (e.key === 'Enter') $('wb-sbtn').click(); });

// ══════════════════════════════════════════════════════════════════════════
// GRAMMAR PANEL (Step 5)
// ══════════════════════════════════════════════════════════════════════════

let grMode = 'grammar';

document.querySelectorAll('.gr-mode').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.gr-mode').forEach(b => b.classList.remove('on'));
    this.classList.add('on');
    grMode = this.dataset.mode;
  });
});

$('gr-btn').addEventListener('click', () => {
  const text = $('gr-inp').value.trim();
  if (!text) return;
  if (!wsReady) { alert('AURA is offline.'); return; }
  if (activeStream) return;

  const result = $('gr-result');
  result.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';
  result.classList.add('on');
  $('gr-btn').disabled = true;

  const prompts = {
    grammar: `Fix grammar and spelling. Return the corrected text, then the separator line "---CHANGES---", then each change as "original → corrected" on its own line.\n\nText:\n${text}`,
    style: `Fix grammar, spelling, and improve clarity and style. Return the corrected text, then "---CHANGES---", then each change as "original → corrected".\n\nText:\n${text}`,
    rewrite: `Completely rewrite for maximum clarity and flow. Return the rewritten text, then "---CHANGES---", then a brief summary of what changed.\n\nText:\n${text}`,
  };

  const grOrigText = text;
  activeStream = {
    type: 'write',
    el: result,
    rawText: '',
    submitBtn: $('gr-btn'),
    onFirstChunk: () => { result.innerHTML = ''; },
    onDone: (rawText) => renderGrammarResult(result, grOrigText, rawText),
  };

  ws.send(JSON.stringify({ type: 'chat', message: prompts[grMode], model: getModel('grammar'), conversation_id: null }));
});

function renderGrammarResult(el, original, rawText) {
  const fullText = rawText || el.textContent;
  const sep = fullText.indexOf('---CHANGES---');

  if (sep === -1) {
    el.innerHTML = md(fullText);
    return;
  }

  const corrected = fullText.slice(0, sep).trim();
  const changesRaw = fullText.slice(sep + 14).trim();

  // Word-diff the original vs corrected
  const diffHtml = renderWordDiff(original.trim(), corrected);

  const changesHtml = changesRaw
    ? `<div class="gr-changes"><div class="gr-change-lbl">Changes</div>` +
      changesRaw.split('\n').filter(l => l.trim()).map(l =>
        `<div class="gr-change">${esc(l)}</div>`).join('') + `</div>`
    : '';

  el.innerHTML = `<div class="gr-corrected">${diffHtml}</div>${changesHtml}`;
}

// LCS word diff (~50 lines)
function wordDiff(a, b) {
  const wa = a.split(/(\s+)/), wb = b.split(/(\s+)/);
  const m = wa.length, n = wb.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = wa[i-1] === wb[j-1] ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);
  const ops = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && wa[i-1] === wb[j-1]) { ops.unshift({ t: '=', w: wa[i-1] }); i--; j--; }
    else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) { ops.unshift({ t: '+', w: wb[j-1] }); j--; }
    else { ops.unshift({ t: '-', w: wa[i-1] }); i--; }
  }
  return ops;
}

function renderWordDiff(a, b) {
  return wordDiff(a, b).map(op => {
    if (op.t === '=') return esc(op.w);
    if (op.t === '+') return `<span class="gr-ins">${esc(op.w)}</span>`;
    return `<span class="gr-del">${esc(op.w)}</span>`;
  }).join('');
}


// ══════════════════════════════════════════════════════════════════════════
// CHATPDF PANEL (Step 6)
// ══════════════════════════════════════════════════════════════════════════

$('pdf-upload-btn').addEventListener('click', () => $('pdf-file').click());

$('pdf-file').addEventListener('change', async () => {
  const file = $('pdf-file').files[0];
  if (!file) return;
  await uploadPdf(file);
});

async function uploadPdf(file) {
  const status = $('pdf-status');
  status.textContent = 'Extracting text…';
  $('pdf-inp-area').style.display = 'none';

  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch(`${HTTP}/api/pdf/extract`, { method: 'POST', body: form });
    const data = await res.json();
    if (data.detail) throw new Error(data.detail);
    pdfCtx = { text: data.text, page_count: data.page_count, word_count: data.word_count };
    status.textContent = `✓ ${data.filename || file.name} — ${data.page_count} pages, ${data.word_count} words`;
    $('pdf-inp-area').style.display = '';
  } catch (err) {
    status.textContent = '⚠ Error: ' + err.message;
  }
}

async function loadPdfUrl(url) {
  const status = $('pdf-status');
  status.textContent = 'Fetching PDF…';
  $('pdf-inp-area').style.display = 'none';
  $('pdf-auto').classList.remove('on');

  try {
    const res = await fetch(`${HTTP}/api/pdf/extract-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (data.detail) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    pdfCtx = { text: data.text, page_count: data.page_count, word_count: data.word_count };
    status.textContent = `✓ PDF loaded — ${data.page_count} pages, ${data.word_count} words`;
    $('pdf-inp-area').style.display = '';
  } catch (err) {
    status.textContent = '⚠ Error: ' + err.message;
  }
}

function sendPdfQuestion() {
  const q = $('pdf-inp').value.trim();
  if (!q || !pdfCtx) return;
  if (!wsReady) { alert('AURA is offline.'); return; }
  if (activeStream) return;

  // Show question bubble
  const qDiv = document.createElement('div');
  qDiv.className = 'pdf-q';
  qDiv.textContent = q;
  $('pdf-msgs').appendChild(qDiv);
  $('pdf-msgs').scrollTop = $('pdf-msgs').scrollHeight;
  $('pdf-inp').value = '';

  // Create answer bubble
  const aDiv = document.createElement('div');
  aDiv.className = 'pdf-a';
  aDiv.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';
  $('pdf-msgs').appendChild(aDiv);
  $('pdf-msgs').scrollTop = $('pdf-msgs').scrollHeight;

  const contextPrefix = `[PDF Context — ${pdfCtx.page_count} pages]\n${pdfCtx.text.slice(0, 35000)}\n\n---\nQuestion: `;
  activeStream = {
    type: 'write',
    el: aDiv,
    rawText: '',
    submitBtn: $('pdf-send'),
    onFirstChunk: () => { aDiv.innerHTML = ''; },
    onDone: () => { $('pdf-msgs').scrollTop = $('pdf-msgs').scrollHeight; },
  };

  ws.send(JSON.stringify({ type: 'chat', message: contextPrefix + q, model: getModel('pdf'), conversation_id: null }));
}


$('pdf-send').addEventListener('click', sendPdfQuestion);
$('pdf-inp').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendPdfQuestion(); }
});

// ══════════════════════════════════════════════════════════════════════════
// REC NOTE PANEL (Step 7)
// ══════════════════════════════════════════════════════════════════════════

let recRecognition = null;
let fullTranscript = '';
let recTimerInterval = null;
let recSeconds = 0;

// Detect Speech Recognition support
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRecognitionAPI) {
  $('rec-ff-note').style.display = '';
  $('rec-controls').style.opacity = '0.3';
  $('rec-controls').style.pointerEvents = 'none';
} else {
  $('rec-ff-note').style.display = 'none';
}

function startRec() {
  if (!SpeechRecognitionAPI) return;
  fullTranscript = '';
  $('rec-transcript').textContent = '';
  $('rec-notes').classList.remove('on');
  $('rec-notes').innerHTML = '';
  $('rec-foot').style.display = 'none';

  recRecognition = new SpeechRecognitionAPI();
  recRecognition.continuous = true;
  recRecognition.interimResults = true;
  recRecognition.lang = 'en-US';

  let interim = '';
  recRecognition.onresult = (e) => {
    interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) fullTranscript += e.results[i][0].transcript + ' ';
      else interim += e.results[i][0].transcript;
    }
    $('rec-transcript').textContent = fullTranscript + interim;
    $('rec-transcript').scrollTop = $('rec-transcript').scrollHeight;
  };

  recRecognition.onerror = (e) => { $('rec-status').textContent = 'Error: ' + e.error; };
  recRecognition.start();

  $('rec-btn').textContent = '■ Stop';
  $('rec-btn').classList.add('recording');
  $('rec-status').textContent = 'Listening…';
  recSeconds = 0;
  recTimerInterval = setInterval(() => {
    recSeconds++;
    const m = Math.floor(recSeconds / 60);
    const s = (recSeconds % 60).toString().padStart(2, '0');
    $('rec-timer').textContent = `${m}:${s}`;
  }, 1000);
}

function stopRec() {
  if (recRecognition) { recRecognition.stop(); recRecognition = null; }
  clearInterval(recTimerInterval);
  $('rec-btn').textContent = '● Record';
  $('rec-btn').classList.remove('recording');
  $('rec-status').textContent = 'Stopped';
  if (fullTranscript.trim()) $('rec-foot').style.display = '';
}

$('rec-btn').addEventListener('click', () => {
  if (recRecognition) stopRec();
  else startRec();
});

$('rec-summarize').addEventListener('click', () => {
  if (!fullTranscript.trim()) return;
  if (!wsReady) { alert('AURA is offline.'); return; }
  if (activeStream) return;

  const notes = $('rec-notes');
  notes.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';
  notes.classList.add('on');

  activeStream = {
    type: 'write',
    el: notes,
    rawText: '',
    submitBtn: null,
    onFirstChunk: () => { notes.innerHTML = ''; },
  };

  ws.send(JSON.stringify({
    type: 'chat',
    message: `Turn this voice transcript into structured, well-organized notes with headings and bullet points:\n\n${fullTranscript}`,
    model: getModel('voice'),
    conversation_id: null,
  }));
});

$('rec-save-wb').addEventListener('click', async () => {
  if (!fullTranscript.trim()) return;
  try {
    await fetch(`${HTTP}/api/knowledge/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: fullTranscript.trim(),
        title: `Voice note ${new Date().toLocaleDateString()}`,
        source_type: 'voice_note',
      }),
    });
    $('rec-status').textContent = '✓ Saved to Wisebase';
  } catch (err) {
    $('rec-status').textContent = '⚠ Save failed: ' + err.message;
  }
});

// Whisper upload (Firefox fallback)
$('rec-whisper-btn').addEventListener('click', async () => {
  const file = $('rec-audio-file').files[0];
  if (!file) return;
  $('rec-status').textContent = 'Transcribing…';
  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch(`${HTTP}/api/transcribe`, { method: 'POST', body: form });
    const data = await res.json();
    if (data.detail) throw new Error(data.detail);
    fullTranscript = data.text || '';
    $('rec-transcript').textContent = fullTranscript;
    $('rec-status').textContent = '✓ Transcribed';
    if (fullTranscript.trim()) $('rec-foot').style.display = '';
  } catch (err) {
    $('rec-status').textContent = '⚠ ' + err.message;
  }
});

// ══════════════════════════════════════════════════════════════════════════
// OCR PANEL (Step 8)
// ══════════════════════════════════════════════════════════════════════════

$('ocr-capture').addEventListener('click', () => {
  ext.runtime.sendMessage({ type: 'OCR_START' });
  $('ocr-result').textContent = 'Draw a region on the page…';
  $('ocr-actions').style.display = 'none';
});

$('ocr-to-chat').addEventListener('click', () => {
  if (!lastOcrText) return;
  pendingCtx = { text: lastOcrText, title: 'OCR extract', url: '' };
  showCtx(lastOcrText, 'OCR extract');
  switchPanel('chat');
});

$('ocr-translate').addEventListener('click', () => {
  if (!lastOcrText) return;
  $('tr-inp').value = lastOcrText;
  switchPanel('translate');
});

$('ocr-copy').addEventListener('click', () => {
  if (!lastOcrText) return;
  navigator.clipboard.writeText(lastOcrText).then(() => {
    $('ocr-copy').textContent = '✓ Copied';
    setTimeout(() => { $('ocr-copy').textContent = 'Copy'; }, 1500);
  });
});

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
  sumGo.disabled = true;
  sumStatus.textContent = 'Reading page\u2026';
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
  sumStatus.textContent = 'Summarizing: ' + title.slice(0, 55) + (title.length > 55 ? '\u2026' : '');

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

    sumResult.innerHTML = md(summaryText);
    sumResult.classList.add('on');

    const wordCount = data.word_count || 0;
    sumWc.textContent = wordCount.toLocaleString() + ' words on page';
    sumRt.textContent = data.reading_time_saved || '';
    sumBadges.classList.add('on');

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

$('sum-to-chat').addEventListener('click', () => {
  if (!summaryText) return;
  pendingCtx = { text: summaryText, title: 'Page Summary', url: '', action: 'ask' };
  showCtx(summaryText, 'Page Summary');
  switchPanel('chat');
});

$('sum-copy').addEventListener('click', function() {
  if (!summaryText) return;
  navigator.clipboard.writeText(summaryText).then(() => {
    const orig = this.textContent;
    this.textContent = 'Copied!';
    setTimeout(() => { this.textContent = orig; }, 1500);
  });
});

// ══════════════════════════════════════════════════════════════════════════
// IMAGE GENERATOR PANEL (Step 9)
// ══════════════════════════════════════════════════════════════════════════

document.querySelectorAll('.img-style').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.img-style').forEach(b => b.classList.remove('on'));
    this.classList.add('on');
    imgStyle = this.dataset.style;
  });
});

$('img-gen').addEventListener('click', async () => {
  const prompt = $('img-prompt').value.trim();
  if (!prompt) return;
  const neg = $('img-neg').value.trim();
  const fullPrompt = imgStyle ? `${prompt}, ${imgStyle}` : prompt;

  $('img-status').textContent = 'Generating… (15–60s)';
  $('img-out').style.display = 'none';
  $('img-acts').style.display = 'none';
  $('img-comfy-note').style.display = 'none';
  $('img-gen').disabled = true;

  try {
    const res = await fetch(`${HTTP}/api/image/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: fullPrompt, negative_prompt: neg, steps: 20 }),
    });
    const data = await res.json();

    if (res.status === 503) {
      $('img-status').textContent = '';
      $('img-comfy-note').style.display = '';
      return;
    }

    if (data.image_b64) {
      const img = $('img-out');
      img.src = 'data:image/png;base64,' + data.image_b64;
      img.style.display = 'block';
      $('img-acts').style.display = '';
      $('img-status').textContent = '✓ Generated';
    } else {
      $('img-status').textContent = '⚠ ' + (data.detail || 'Generation failed');
    }
  } catch (err) {
    $('img-status').textContent = '⚠ ' + err.message;
  } finally {
    $('img-gen').disabled = false;
  }
});

$('img-dl').addEventListener('click', () => {
  const img = $('img-out');
  if (!img.src) return;
  const a = document.createElement('a');
  a.href = img.src;
  a.download = 'aura-image.png';
  a.click();
});

// ══════════════════════════════════════════════════════════════════════════
// BROWSER AGENT PANEL (Step 10)
// ══════════════════════════════════════════════════════════════════════════

const sleep = ms => new Promise(r => setTimeout(r, ms));
const msgBg = msg => new Promise(r => ext.runtime.sendMessage(msg, r));

function logAgent(text) {
  const log = $('agent-log');
  const el = document.createElement('div');
  el.className = 'agent-step';
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

async function runAgentLoop(task, history) {
  $('agent-step-count').textContent = `Step ${agentStep} of 15`;

  if (!agentRunning || agentStep >= 15) {
    logAgent(agentStep >= 15 ? '⚠ Max steps (15) reached.' : '■ Stopped.');
    $('agent-start').style.display = '';
    $('agent-stop').style.display = 'none';
    agentRunning = false;
    return;
  }

  agentStep++;

  const dom = await msgBg({ type: 'AGENT_DOM' });
  if (!dom?.ok) {
    logAgent('⚠ Could not read page DOM.');
    agentRunning = false;
    $('agent-start').style.display = '';
    $('agent-stop').style.display = 'none';
    return;
  }

  const domStr = dom.dom
    .map(e => `[${e.index}] ${e.type} "${e.text}" → ${e.selector}`)
    .join('\n');

  const prompt =
    `Task: "${task}"\nURL: ${dom.url}\nTitle: ${dom.title}\n` +
    `History: ${JSON.stringify(history.slice(-5))}\n\n` +
    `Interactive elements on page:\n${domStr.slice(0, 3000)}\n\n` +
    `Respond ONLY with valid JSON (no markdown, no explanation):\n` +
    `{"action":"click"|"type"|"scroll"|"navigate"|"done","selector":"","text":"","url":"","amount":300,"description":""}`;

  try {
    const action = await fetch(`${HTTP}/api/agent/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model: getModel('agent') }),
    }).then(r => r.json());

    logAgent(`Step ${agentStep}: [${action.action}] ${action.description || ''}`);

    if (action.action === 'done') {
      logAgent('✓ Task complete.');
      agentRunning = false;
      $('agent-start').style.display = '';
      $('agent-stop').style.display = 'none';
      return;
    }

    if (action.action === 'navigate') {
      await msgBg({ type: 'AGENT_NAV', url: action.url });
      await sleep(2500);
    } else {
      const result = await msgBg({ type: 'AGENT_EXEC', action });
      if (!result?.ok) logAgent(`  ⚠ ${result?.error || 'Action failed'}`);
      await sleep(600);
    }

    history.push({ step: agentStep, ...action });
    await runAgentLoop(task, history);

  } catch (err) {
    logAgent('⚠ Error: ' + err.message);
    agentRunning = false;
    $('agent-start').style.display = '';
    $('agent-stop').style.display = 'none';
  }
}

$('agent-start').addEventListener('click', () => {
  const task = $('agent-task').value.trim();
  if (!task) return;
  $('agent-log').innerHTML = '';
  agentRunning = true;
  agentStep = 0;
  $('agent-start').style.display = 'none';
  $('agent-stop').style.display = '';
  logAgent(`▶ Starting: "${task.slice(0, 60)}"`);
  runAgentLoop(task, []);
});

$('agent-stop').addEventListener('click', () => {
  agentRunning = false;
  $('agent-stop').style.display = 'none';
  $('agent-start').style.display = '';
  logAgent('■ Stopped by user.');
});

// ── Model Panel ─────────────────────────────────────────────────────────────
// Per-feature model selection — stored in chrome.storage.local

const FEATURE_DEFS = [
  { key: 'chat',    label: 'Chat',           icon: '💬', desc: 'Main conversation' },
  { key: 'search',  label: 'Search',         icon: '🔍', desc: 'Web search answer' },
  { key: 'translate',label:'Translate',      icon: '🌐', desc: 'Language translation' },
  { key: 'write',   label: 'Write',          icon: '✍️', desc: 'Writing assistant' },
  { key: 'grammar', label: 'Grammar',        icon: '✅', desc: 'Grammar & style check' },
  { key: 'ask',     label: 'Ask / Explain',  icon: '⚡', desc: 'Quick-action context prompts' },
  { key: 'pdf',     label: 'PDF Chat',       icon: '📄', desc: 'Chat with PDF content' },
  { key: 'voice',   label: 'Voice Notes',    icon: '🎤', desc: 'Transcript summarization' },
  { key: 'agent',   label: 'Browser Agent',  icon: '🤖', desc: 'Page action planning' },
  { key: 'summary',   label: 'Page Summary',   icon: '📄', desc: 'One-click page summarization' },
  { key: 'youtube',   label: 'YouTube',        icon: '▶️',  desc: 'Summarize YouTube videos' },
  { key: 'research',  label: 'Deep Research',  icon: '🔬', desc: 'Multi-source web research' },
  { key: 'math',      label: 'Math Solver',    icon: '➗', desc: 'Step-by-step math solving' },
];

// Loaded from chrome.storage.local, applied to every request
let featureModels = {};   // { chat: 'model-name', ... }
let mdlCloudList = [];    // cloud model names from Ollama
let mdlLocalList = [];    // local model names from Ollama

// Load saved model prefs on startup
ext.storage.local.get(['featureModels'], d => {
  featureModels = d.featureModels || {};
});

// Get model for a feature (returns null = let backend decide)
function getModel(feature) {
  return featureModels[feature] || null;
}

// ── Inline model pill (Sider-style) ─────────────────────────────────────────

const PILL_SLOTS = {
  chat: 'mdl-chat', search: 'mdl-search', translate: 'mdl-translate',
  write: 'mdl-write', grammar: 'mdl-grammar', ask: 'mdl-ask',
  pdf: 'mdl-pdf', voice: 'mdl-voice', summary: 'mdl-summary',
  youtube: 'mdl-youtube', research: 'mdl-research', math: 'mdl-math',
};

function buildPill(featureKey) {
  const wrapper = document.createElement('span');
  wrapper.style.cssText = 'position:relative;display:inline-block;vertical-align:middle';

  const btn = document.createElement('button');
  btn.className = 'mdl-pill';

  const dot = document.createElement('span');
  dot.className = 'mdl-pill-dot';

  const name = document.createElement('span');
  name.className = 'mdl-pill-name';

  const arrow = document.createElement('span');
  arrow.className = 'mdl-pill';
  arrow.style.cssText = 'background:none;border:none;padding:0;margin-left:1px';
  arrow.innerHTML = `<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>`;

  btn.appendChild(dot);
  btn.appendChild(name);
  btn.appendChild(arrow);

  const drop = document.createElement('div');
  drop.className = 'mdl-drop';
  wrapper.appendChild(btn);
  wrapper.appendChild(drop);

  function refresh() {
    const m = featureModels[featureKey];
    name.textContent = m ? m.replace(/:cloud$/, '') : 'Auto';
    dot.className = 'mdl-pill-dot' + (m ? ' set' : '');
  }

  function pick(model) {
    if (model) featureModels[featureKey] = model;
    else delete featureModels[featureKey];
    ext.storage.local.set({ featureModels });
    refresh();
    drop.classList.remove('open');
    // Refresh all pills for this feature (in case multiple exist)
    document.querySelectorAll(`[data-pill="${featureKey}"]`).forEach(p => {
      if (p._refresh) p._refresh();
    });
  }

  function openDrop() {
    drop.innerHTML = '';

    // Auto option
    const autoEl = document.createElement('div');
    autoEl.className = 'mdl-drop-item' + (!featureModels[featureKey] ? ' on' : '');
    autoEl.textContent = '⚡  Auto';
    autoEl.addEventListener('click', () => pick(null));
    drop.appendChild(autoEl);

    const allModels = [...mdlCloudList, ...mdlLocalList];

    if (!allModels.length) {
      const hint = document.createElement('div');
      hint.style.cssText = 'padding:10px 10px 6px;font-size:11px;color:rgba(160,148,210,.6);text-align:center';
      hint.textContent = 'No models found — is Ollama running?';
      drop.appendChild(hint);
    } else {
      if (mdlCloudList.length) {
        const sec = document.createElement('div');
        sec.className = 'mdl-drop-sec';
        sec.textContent = `☁  Cloud  (${mdlCloudList.length})`;
        drop.appendChild(sec);
        mdlCloudList.forEach(m => {
          const el = document.createElement('div');
          el.className = 'mdl-drop-item' + (featureModels[featureKey] === m ? ' on' : '');
          el.textContent = m.replace(/:cloud$/, '');
          el.title = m;
          el.addEventListener('click', () => pick(m));
          drop.appendChild(el);
        });
      }
      if (mdlLocalList.length) {
        const sec = document.createElement('div');
        sec.className = 'mdl-drop-sec';
        sec.textContent = `🖥  Local  (${mdlLocalList.length})`;
        drop.appendChild(sec);
        mdlLocalList.forEach(m => {
          const el = document.createElement('div');
          el.className = 'mdl-drop-item' + (featureModels[featureKey] === m ? ' on' : '');
          el.textContent = m;
          el.title = m;
          el.addEventListener('click', () => pick(m));
          drop.appendChild(el);
        });
      }
    }

    // Show first so getBoundingClientRect() has real dimensions
    drop.classList.remove('up');
    drop.classList.add('open');

    // Fixed positioning — escapes panel overflow:hidden
    const pRect = btn.getBoundingClientRect();
    drop.style.right = Math.max(4, window.innerWidth - pRect.right) + 'px';
    drop.style.left = 'auto';

    // Flip upward if not enough space below
    const dropRect = drop.getBoundingClientRect();
    if (window.innerHeight - pRect.bottom < dropRect.height + 16) {
      drop.style.top = 'auto';
      drop.style.bottom = (window.innerHeight - pRect.top + 5) + 'px';
      drop.classList.add('up');
    } else {
      drop.style.top = (pRect.bottom + 5) + 'px';
      drop.style.bottom = 'auto';
    }
  }

  btn.addEventListener('click', e => {
    e.stopPropagation();
    if (drop.classList.contains('open')) { drop.classList.remove('open'); return; }
    document.querySelectorAll('.mdl-drop.open').forEach(d => d.classList.remove('open'));

    // Lazy-load models — try Ollama directly first (works even when backend is offline)
    if (!mdlCloudList.length) {
      fetch('http://localhost:11434/api/tags')
        .then(r => r.json())
        .then(d => {
          const all = (d.models || []).map(m => m.name);
          mdlCloudList = all.filter(n => n.includes(':cloud'));
          mdlLocalList = all.filter(n => !n.includes(':cloud'));
          openDrop();
        })
        .catch(() => {
          // Ollama unreachable — try backend
          fetch(`${HTTP}/api/models/available`)
            .then(r => r.json())
            .then(d => {
              mdlCloudList = (d.cloud || []).map(m => m.name);
              mdlLocalList = (d.local || []).map(m => m.name);
              openDrop();
            })
            .catch(() => openDrop());
        });
    } else {
      openDrop();
    }
  });

  document.addEventListener('click', () => drop.classList.remove('open'));

  btn.dataset.pill = featureKey;
  btn._refresh = refresh;
  refresh();
  return wrapper;
}

function initModelPills() {
  for (const [key, slotId] of Object.entries(PILL_SLOTS)) {
    const slot = $(slotId);
    if (slot) slot.appendChild(buildPill(key));
  }
}

initModelPills();

async function loadModelPanel() {
  const body = $('mdl-body');
  body.innerHTML = '<div style="color:var(--mu);font-size:12px;padding:24px 0;text-align:center">Loading models…</div>';

  try {
    const r = await fetch(`${HTTP}/api/models/available`);
    const d = await r.json();
    mdlCloudList = (d.cloud || []).map(m => m.name);
    mdlLocalList = (d.local || []).map(m => m.name);
    renderModelPanel();
  } catch {
    body.innerHTML = '<div style="color:var(--rd);font-size:12px;padding:24px 0;text-align:center">Backend offline — start the server first.</div>';
  }
}

function buildModelSelect(featureKey, currentModel) {
  const sel = document.createElement('select');
  sel.dataset.feature = featureKey;
  sel.style.cssText = `
    width:100%;background:var(--s2);border:1px solid var(--b1);
    border-radius:8px;color:var(--tx);font-size:12px;
    padding:7px 10px;cursor:pointer;outline:none;
    transition:border-color .15s;
  `;

  // Auto option
  const autoOpt = document.createElement('option');
  autoOpt.value = '';
  autoOpt.textContent = '⚡ Auto (backend decides)';
  if (!currentModel) autoOpt.selected = true;
  sel.appendChild(autoOpt);

  // Cloud models group
  if (mdlCloudList.length) {
    const g = document.createElement('optgroup');
    g.label = `☁ Cloud models (${mdlCloudList.length})`;
    mdlCloudList.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      // Clean display: strip :cloud suffix for readability
      opt.textContent = name.replace(/:cloud$/, '') + ' ☁';
      if (name === currentModel) opt.selected = true;
      g.appendChild(opt);
    });
    sel.appendChild(g);
  }

  // Local models group
  if (mdlLocalList.length) {
    const g = document.createElement('optgroup');
    g.label = `🖥 Local models (${mdlLocalList.length})`;
    mdlLocalList.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (name === currentModel) opt.selected = true;
      g.appendChild(opt);
    });
    sel.appendChild(g);
  }

  sel.addEventListener('change', () => {
    if (sel.value) {
      featureModels[featureKey] = sel.value;
      sel.style.borderColor = 'var(--p)';
    } else {
      delete featureModels[featureKey];
      sel.style.borderColor = '';
    }
    ext.storage.local.set({ featureModels });
  });

  return sel;
}

function renderModelPanel() {
  const body = $('mdl-body');
  body.innerHTML = '';

  // Info banner
  const info = document.createElement('div');
  info.style.cssText = 'background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.2);border-radius:8px;padding:8px 10px;font-size:11px;color:var(--mu);margin-bottom:4px';
  info.textContent = `${mdlCloudList.length} cloud models available. Changes apply instantly — no restart needed.`;
  body.appendChild(info);

  FEATURE_DEFS.forEach(feat => {
    const cur = featureModels[feat.key] || '';

    const card = document.createElement('div');
    card.style.cssText = 'background:var(--s1);border:1px solid var(--b1);border-radius:10px;padding:10px 12px;display:flex;flex-direction:column;gap:7px';

    const hdr = document.createElement('div');
    hdr.style.cssText = 'display:flex;align-items:center;gap:6px';
    hdr.innerHTML = `
      <span style="font-size:14px">${feat.icon}</span>
      <div>
        <div style="font-size:12px;font-weight:600;color:var(--pl2)">${feat.label}</div>
        <div style="font-size:10.5px;color:var(--mu)">${feat.desc}</div>
      </div>
    `;
    card.appendChild(hdr);
    card.appendChild(buildModelSelect(feat.key, cur));

    if (cur) {
      const active = document.createElement('div');
      active.style.cssText = 'font-size:10px;color:var(--p);font-weight:500';
      active.textContent = `Using: ${cur}`;
      card.appendChild(active);
    }

    body.appendChild(card);
  });
}

$('mdl-save').addEventListener('click', () => {
  // Models are saved instantly on change — this just gives visual feedback
  $('mdl-save').textContent = 'Saved ✓';
  document.querySelectorAll('#mdl-body select').forEach(s => s.style.borderColor = '');
  setTimeout(() => { $('mdl-save').textContent = 'Apply Changes'; }, 1500);
});

$('mdl-reload').addEventListener('click', loadModelPanel);

// ══════════════════════════════════════════════════════════════════════════
// YOUTUBE PANEL
// ══════════════════════════════════════════════════════════════════════════

async function loadYoutubeSummary(url) {
  url = (url || $('yt-url-inp').value || '').trim();
  if (!url) return;
  const statusEl = $('yt-status');
  const resultsEl = $('yt-results');
  const btn = $('yt-summarize-btn');
  const autoBtn = $('yt-auto-btn');
  resultsEl.style.display = 'none';
  statusEl.innerHTML = '<div class="dots"><span></span><span></span><span></span></div> Fetching transcript…';
  btn.disabled = true;
  if (autoBtn) autoBtn.disabled = true;
  try {
    const resp = await fetch(`${HTTP}/api/youtube/summarize`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url}), signal: AbortSignal.timeout(90000),
    });
    if (!resp.ok) {
      let e = `Error ${resp.status}`;
      try { const d = await resp.json(); e = d.detail || e; } catch {}
      statusEl.textContent = '⚠ ' + e; return;
    }
    const data = await resp.json();
    $('yt-res-title').textContent = data.title || 'Untitled';
    $('yt-res-channel').textContent = data.channel ? '▶ ' + data.channel : '';
    $('yt-res-duration').textContent = data.duration ? '⏱ ' + data.duration : '';
    $('yt-res-summary').textContent = data.summary || 'No summary available.';
    const ul = $('yt-res-points');
    ul.innerHTML = '';
    (Array.isArray(data.key_points) ? data.key_points : []).forEach(pt => {
      const li = document.createElement('li'); li.className = 'yt-kp'; li.textContent = pt; ul.appendChild(li);
    });
    $('yt-res-snippet').textContent = data.transcript_snippet || '';
    $('yt-res-snippet').style.display = 'none';
    const ch = $('yt-snippet-chevron'); if (ch) ch.style.transform = '';
    statusEl.textContent = '';
    resultsEl.style.display = 'block';
  } catch (err) {
    statusEl.textContent = '⚠ ' + (err.name === 'TimeoutError' ? 'Request timed out.' : err.message || 'Unknown error');
  } finally {
    btn.disabled = false;
    if (autoBtn) autoBtn.disabled = false;
  }
}

$('yt-summarize-btn').addEventListener('click', () => loadYoutubeSummary($('yt-url-inp').value));
$('yt-url-inp').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); loadYoutubeSummary($('yt-url-inp').value); } });
$('yt-auto-btn').addEventListener('click', () => { if (ytAutoUrl) { $('yt-url-inp').value = ytAutoUrl; loadYoutubeSummary(ytAutoUrl); } });
$('yt-snippet-toggle').addEventListener('click', () => {
  const s = $('yt-res-snippet'), ch = $('yt-snippet-chevron');
  const open = s.style.display !== 'none';
  s.style.display = open ? 'none' : 'block';
  if (ch) ch.style.transform = open ? '' : 'rotate(90deg)';
});

// ══════════════════════════════════════════════════════════════════════════
// COMPARE PANEL
// ══════════════════════════════════════════════════════════════════════════

function initComparePanel() {
  if (compareInitialized) return;
  const chipsEl = $('cmp-chips');
  if (!chipsEl) return;
  function buildChips(cloudList, localList) {
    chipsEl.innerHTML = '';
    function addChip(m, isCloud) {
      const chip = document.createElement('button');
      chip.className = 'cmp-chip'; chip.dataset.model = m; chip.title = m;
      chip.innerHTML = `<span style="flex-shrink:0">${isCloud ? '☁' : '🖥'}</span><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(m.replace(/:cloud$/,''))}</span>`;
      if (COMPARE_DEFAULT_MODELS.includes(m)) { compareSelectedModels.add(m); chip.classList.add('on'); }
      chip.addEventListener('click', () => {
        if (compareSelectedModels.has(m)) { compareSelectedModels.delete(m); chip.classList.remove('on'); }
        else { compareSelectedModels.add(m); chip.classList.add('on'); }
      });
      chipsEl.appendChild(chip);
    }
    cloudList.forEach(m => addChip(m, true));
    localList.forEach(m => addChip(m, false));
    compareInitialized = true;
  }
  if (mdlCloudList.length || mdlLocalList.length) { buildChips(mdlCloudList, mdlLocalList); return; }
  fetch('http://localhost:11434/api/tags').then(r=>r.json()).then(d=>{
    const all = (d.models||[]).map(m=>m.name);
    mdlCloudList = all.filter(n=>n.includes(':cloud'));
    mdlLocalList = all.filter(n=>!n.includes(':cloud'));
    buildChips(mdlCloudList, mdlLocalList);
  }).catch(()=>{
    fetch(`${HTTP}/api/models/available`).then(r=>r.json()).then(d=>{
      mdlCloudList = (d.cloud||[]).map(m=>m.name);
      mdlLocalList = (d.local||[]).map(m=>m.name);
      buildChips(mdlCloudList, mdlLocalList);
    }).catch(()=>{ chipsEl.innerHTML='<span style="font-size:11px;color:var(--mu)">No models — is Ollama running?</span>'; });
  });
}

$('cmp-all') && $('cmp-all').addEventListener('click', () => {
  $('cmp-chips').querySelectorAll('.cmp-chip').forEach(c => { compareSelectedModels.add(c.dataset.model); c.classList.add('on'); });
});
$('cmp-clear') && $('cmp-clear').addEventListener('click', () => {
  compareSelectedModels.clear();
  $('cmp-chips').querySelectorAll('.cmp-chip').forEach(c => c.classList.remove('on'));
});
$('cmp-inp') && $('cmp-inp').addEventListener('keydown', e => { if (e.key==='Enter'&&!e.shiftKey){e.preventDefault();runCompare();} });
$('cmp-run') && $('cmp-run').addEventListener('click', runCompare);

async function runCompare() {
  const inp = $('cmp-inp'), resultsEl = $('cmp-results'), emptyEl = $('cmp-empty'), runBtn = $('cmp-run');
  if (!inp||!resultsEl) return;
  const prompt = inp.value.trim();
  if (!prompt) { inp.style.borderColor='var(--rd)'; setTimeout(()=>{inp.style.borderColor='';},800); return; }
  if (!compareSelectedModels.size) return;
  runBtn.disabled = true;
  if (emptyEl) emptyEl.style.display = 'none';
  const models = [...compareSelectedModels];
  resultsEl.style.display = 'grid'; resultsEl.innerHTML = '';
  models.forEach(m => {
    const sk = document.createElement('div'); sk.className='cmp-skeleton'; sk.dataset.skelModel=m;
    sk.innerHTML=`<div class="cmp-skeleton-hdr"><div class="cmp-skeleton-bar" style="width:55%"></div><div class="cmp-skeleton-bar" style="width:18%;margin-left:auto"></div></div><div class="cmp-skeleton-body"><div class="cmp-skeleton-bar" style="width:92%"></div><div class="cmp-skeleton-bar" style="width:78%"></div><div class="cmp-skeleton-bar" style="width:64%"></div></div>`;
    resultsEl.appendChild(sk);
  });
  try {
    const res = await fetch(`${HTTP}/api/compare`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:prompt,models})});
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    resultsEl.innerHTML = '';
    (data.results||[]).forEach(r => {
      const isCloud = r.model.includes(':cloud');
      const isFastest = r.model===data.fastest && !r.error;
      const displayName = r.model.replace(/:cloud$/,'');
      const card = document.createElement('div'); card.className='cmp-card';
      const timeLabel = r.elapsed_ms>=1000 ? (r.elapsed_ms/1000).toFixed(1)+'s' : r.elapsed_ms+'ms';
      card.innerHTML=`<div class="cmp-card-hdr"><div class="cmp-model-badge"><span>${isCloud?'☁':'🖥'}</span><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(displayName)}</span></div><span class="cmp-time-badge${isFastest?' fastest':''}">${isFastest?'⚡ ':''}${timeLabel}</span></div><div class="cmp-card-body${r.error?' error':''}">${r.error?esc('Error: '+r.error):md(r.response||'')}</div>`;
      if (!r.error) {
        const foot = document.createElement('div'); foot.className='cmp-card-footer';
        const btn2 = document.createElement('button'); btn2.className='cmp-send-btn'; btn2.textContent='Send to Chat';
        btn2.addEventListener('click',()=>{ const i=$('inp'); if(i){i.value=`[${displayName}]: ${(r.response||'').slice(0,500)}`; switchPanel('chat'); i.focus();} });
        foot.appendChild(btn2); card.appendChild(foot);
      }
      resultsEl.appendChild(card);
    });
  } catch (err) {
    resultsEl.innerHTML=`<div style="grid-column:1/-1;color:var(--rd);font-size:12px;padding:12px 0">Failed: ${esc(String(err))}</div>`;
  } finally { runBtn.disabled=false; }
}

// ══════════════════════════════════════════════════════════════════════════
// DEEP RESEARCH PANEL
// ══════════════════════════════════════════════════════════════════════════

document.querySelectorAll('.res-d').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.res-d').forEach(b=>b.classList.remove('on'));
    this.classList.add('on'); resDepth = this.dataset.depth;
  });
});

$('res-go').addEventListener('click', async () => {
  const query = $('res-inp').value.trim();
  if (!query) return;
  const statusEl = $('res-status'), resultEl = $('res-result'), sourcesEl = $('res-sources'), goBtn = $('res-go');
  goBtn.disabled = true; resultEl.style.display='none'; sourcesEl.style.display='none';
  statusEl.textContent = 'Searching the web…';
  try {
    const resp = await fetch(`${HTTP}/api/research`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query, depth: resDepth, model: getModel('research')}),
    });
    if (!resp.ok) { const d=await resp.json().catch(()=>({})); statusEl.textContent='⚠ '+(d.detail||resp.statusText); return; }
    const reader = resp.body.getReader(); const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const ev = JSON.parse(line);
          if (ev.status && ev.status !== 'done') { statusEl.textContent = ev.message || ev.status; }
          if (ev.status === 'done') {
            statusEl.textContent = `Done — ${ev.sources?.length||0} sources`;
            resultEl.innerHTML = md(ev.report||''); resultEl.style.display='block';
            if (ev.sources?.length) {
              sourcesEl.innerHTML = '<div style="font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--mu);margin-bottom:6px;">Sources</div>';
              sourcesEl.style.display = 'flex';
              ev.sources.slice(0,6).forEach(s => {
                const d2 = document.createElement('div'); d2.className='res-src';
                d2.innerHTML=`<a href="${esc(s.url)}" target="_blank" rel="noopener">[${s.index}] ${esc(s.title||s.domain)}</a><div class="res-src-snip">${esc(s.snippet||'')}</div>`;
                sourcesEl.appendChild(d2);
              });
            }
          }
        } catch {}
      }
    }
  } catch (err) { statusEl.textContent = '⚠ ' + (err.message||'Request failed'); }
  finally { goBtn.disabled=false; }
});

// ══════════════════════════════════════════════════════════════════════════
// MATH SOLVER PANEL
// ══════════════════════════════════════════════════════════════════════════

document.querySelectorAll('.math-m').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.math-m').forEach(b=>b.classList.remove('on'));
    this.classList.add('on'); mathMode = this.dataset.mode;
  });
});

$('math-go').addEventListener('click', async () => {
  const problem = $('math-inp').value.trim();
  if (!problem) return;
  const statusEl=$('math-status'), resultEl=$('math-result'), goBtn=$('math-go');
  goBtn.disabled=true; resultEl.style.display='none'; statusEl.textContent='Solving…';
  try {
    const resp = await fetch(`${HTTP}/api/math/solve`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({problem, mode: mathMode, model: getModel('math')}),
    });
    if (!resp.ok) { const d=await resp.json().catch(()=>({})); statusEl.textContent='⚠ '+(d.detail||resp.statusText); return; }
    const data = await resp.json();
    statusEl.textContent = '';
    const solEl=$('math-solution'), latEl=$('math-latex'), stepsEl=$('math-steps');
    solEl.textContent = data.solution || 'No solution returned.';
    if (data.latex) { latEl.textContent = data.latex; latEl.style.display='block'; } else { latEl.style.display='none'; }
    stepsEl.innerHTML = '';
    if (data.steps?.length) {
      stepsEl.style.display='block';
      data.steps.forEach((s,i) => {
        const d2=document.createElement('div'); d2.className='math-step';
        d2.innerHTML=`<span class="math-step-num">${i+1}.</span>${esc(s)}`;
        stepsEl.appendChild(d2);
      });
    } else { stepsEl.style.display='none'; }
    resultEl.style.display='flex';
  } catch (err) { statusEl.textContent='⚠ '+(err.message||'Request failed'); }
  finally { goBtn.disabled=false; }
});

$('math-inp').addEventListener('keydown', e => {
  if (e.key==='Enter' && (e.ctrlKey||e.metaKey)) { e.preventDefault(); $('math-go').click(); }
});

// ══════════════════════════════════════════════════════════════════════════
// ARTIFACTS PANEL
// ══════════════════════════════════════════════════════════════════════════

document.querySelectorAll('.art-tab').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.art-tab').forEach(b=>b.classList.remove('on'));
    this.classList.add('on');
    const tab = this.dataset.tab;
    $('art-preview').classList.toggle('on', tab==='preview');
    $('art-code').classList.toggle('on', tab==='code');
  });
});

$('art-lang').addEventListener('change', function() { artLang = this.value; });

$('art-go').addEventListener('click', async () => {
  const prompt = $('art-inp').value.trim();
  if (!prompt) return;
  const statusEl=$('art-status'), goBtn=$('art-go'), preview=$('art-preview'), codeEl=$('art-code'), bottomEl=$('art-bottom');
  goBtn.disabled=true; statusEl.textContent='Generating…';
  preview.srcdoc=''; codeEl.textContent=''; artCode='';
  bottomEl.classList.remove('on');
  const lang = $('art-lang').value;
  const systemNote = lang==='svg' ? 'Respond with only the SVG code, no explanation.' :
                     lang==='markdown' ? 'Respond with only Markdown, no explanation.' :
                     'Respond with only a complete HTML file including CSS and JS. No explanation, no markdown fences.';
  const fullPrompt = `${systemNote}\n\nTask: ${prompt}`;
  try {
    const resp = await fetch(`${HTTP}/api/chat`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: fullPrompt, stream: false}),
    });
    if (!resp.ok) { const d=await resp.json().catch(()=>({})); statusEl.textContent='⚠ '+(d.detail||resp.statusText); return; }
    const data = await resp.json();
    let code = (data.response||data.message||'').trim();
    // Strip markdown code fences
    code = code.replace(/^```[\w]*\n?/,'').replace(/\n?```$/,'').trim();
    artCode = code;
    codeEl.textContent = code;
    if (lang==='svg') { preview.srcdoc=`<html><body style="margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh;background:#fff">${code}</body></html>`; }
    else if (lang==='markdown') { preview.srcdoc=`<html><head><style>body{font-family:system-ui,sans-serif;padding:16px;line-height:1.6;max-width:700px;margin:0 auto}</style></head><body>${md(code)}</body></html>`; }
    else { preview.srcdoc=code; }
    statusEl.textContent='';
    bottomEl.classList.add('on');
  } catch (err) { statusEl.textContent='⚠ '+(err.message||'Request failed'); }
  finally { goBtn.disabled=false; }
});

$('art-copy-code').addEventListener('click', function() {
  if (!artCode) return;
  navigator.clipboard.writeText(artCode).then(()=>{ const o=this.textContent; this.textContent='Copied!'; setTimeout(()=>{this.textContent=o;},1500); });
});

$('art-send-chat').addEventListener('click', () => {
  if (!artCode) return;
  pendingCtx = { text: artCode, title: 'Artifact', url: '', action: 'ask' };
  showCtx(artCode.slice(0,80)+'…', 'Artifact');
  switchPanel('chat');
});
