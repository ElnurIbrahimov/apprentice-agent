/**
 * AURA Chrome Extension — Content Script (content.js)
 * Injects a Shadow DOM floating toolbar on text selection.
 * Runs on every page via manifest content_scripts.
 */

// Firefox compatibility shim (file-level so all code below can use ext.*)
const ext = typeof browser !== 'undefined' ? browser : chrome;

(function () {
  // Remove stale elements from any previous injection (handles extension reload)
  const _prevDock = document.getElementById('aura-dock-host');
  if (_prevDock) _prevDock.remove();
  const _prevHost = document.getElementById('aura-host');
  if (_prevHost) _prevHost.remove();
  window.__auraToolbarMounted = true;

  // ── Shadow DOM Host ────────────────────────────────────────────────────────────

  const host = document.createElement('div');
  host.id = 'aura-host';
  Object.assign(host.style, {
    position: 'fixed',
    top: '0',
    left: '0',
    zIndex: '2147483647',
    pointerEvents: 'none',
  });
  document.documentElement.appendChild(host);

  const shadow = host.attachShadow({ mode: 'open' });

  // ── Styles (inside Shadow DOM — isolated from page) ────────────────────────

  const style = document.createElement('style');
  style.textContent = `
    #toolbar {
      display: none;
      position: fixed;
      background: #1a1a2e;
      border: 1px solid #7c3aed;
      border-radius: 8px;
      padding: 4px 6px;
      gap: 4px;
      box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
      pointer-events: auto;
      z-index: 2147483647;
      align-items: center;
    }
    #toolbar.visible {
      display: flex;
    }
    .aura-btn {
      background: transparent;
      border: none;
      color: #e2e8f0;
      font-size: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      padding: 4px 8px;
      border-radius: 5px;
      cursor: pointer;
      white-space: nowrap;
      transition: background 0.15s;
    }
    .aura-btn:hover {
      background: #7c3aed;
      color: #fff;
    }
    .aura-divider {
      width: 1px;
      height: 16px;
      background: #3d3d5c;
    }
    #toast {
      display: none;
      position: fixed;
      background: #059669;
      color: #fff;
      font-size: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      padding: 6px 12px;
      border-radius: 6px;
      pointer-events: none;
      z-index: 2147483647;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    #toast.visible {
      display: block;
    }
  `;
  shadow.appendChild(style);

  // ── Toolbar DOM ────────────────────────────────────────────────────────────────

  const toolbar = document.createElement('div');
  toolbar.id = 'toolbar';

  const buttons = [
    { label: '✦ Explain', action: 'explain' },
    { label: '◈ Summarize', action: 'summarize' },
    { label: '◉ Ask AURA', action: 'ask' },
  ];

  buttons.forEach((btn, i) => {
    const el = document.createElement('button');
    el.className = 'aura-btn';
    el.textContent = btn.label;
    el.dataset.action = btn.action;
    toolbar.appendChild(el);
    if (i < buttons.length - 1) {
      const div = document.createElement('div');
      div.className = 'aura-divider';
      toolbar.appendChild(div);
    }
  });

  // Save button with divider
  const divSave = document.createElement('div');
  divSave.className = 'aura-divider';
  toolbar.appendChild(divSave);

  const saveBtn = document.createElement('button');
  saveBtn.className = 'aura-btn';
  saveBtn.textContent = '⊕ Save';
  saveBtn.dataset.action = 'save';
  toolbar.appendChild(saveBtn);

  shadow.appendChild(toolbar);

  // ── Toast ─────────────────────────────────────────────────────────────────

  const toast = document.createElement('div');
  toast.id = 'toast';
  shadow.appendChild(toast);

  // ── Floating Action Dock (direct DOM injection — no shadow DOM) ───────────

  const dockHost = document.createElement('div');
  dockHost.id = 'aura-dock-host';
  Object.assign(dockHost.style, {
    position: 'fixed',
    right: '0',
    top: '50%',
    transform: 'translateY(-50%)',
    zIndex: '2147483647',
    pointerEvents: 'auto',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '0',
    padding: '7px 4px',
    background: 'rgba(7, 5, 18, 0.92)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    border: '1px solid rgba(124, 58, 237, 0.3)',
    borderRight: 'none',
    borderRadius: '12px 0 0 12px',
    boxShadow: '-3px 0 20px rgba(0,0,0,0.5)',
    transition: 'border-color 0.2s',
    boxSizing: 'border-box',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  });
  document.body.appendChild(dockHost);

  // Logo — always visible
  const dockLogo = document.createElement('div');
  Object.assign(dockLogo.style, {
    width: '32px',
    height: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'rgba(160, 148, 210, 0.9)',
    cursor: 'default',
    flexShrink: '0',
  });
  dockLogo.innerHTML = `<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;
  dockHost.appendChild(dockLogo);

  // Action buttons container — hidden until hover
  const dockActions = document.createElement('div');
  Object.assign(dockActions.style, {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '3px',
    overflow: 'hidden',
    maxHeight: '0',
    opacity: '0',
    transition: 'max-height 0.25s ease, opacity 0.2s ease, padding-top 0.2s ease',
    paddingTop: '0',
  });
  dockHost.appendChild(dockActions);

  const _dockItems = [
    { svg: '<path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>', action: 'dock-chat', tip: 'Chat with AURA' },
    { svg: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>', action: 'dock-search', tip: 'Search' },
    null,
    { svg: '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><path d="M9 21V9"/>', action: 'dock-thispage', tip: 'This Page' },
    { svg: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>', action: 'dock-translate', tip: 'Translate' },
    null,
    { svg: '<path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>', action: 'dock-save', tip: 'Save to Memory' },
  ];

  _dockItems.forEach(item => {
    if (!item) {
      const sep = document.createElement('div');
      Object.assign(sep.style, {
        width: '18px',
        height: '1px',
        background: 'rgba(255,255,255,0.08)',
        margin: '2px 0',
        flexShrink: '0',
      });
      dockActions.appendChild(sep);
      return;
    }
    const btn = document.createElement('button');
    btn.dataset.action = item.action;
    btn.title = item.tip;
    Object.assign(btn.style, {
      width: '32px',
      height: '32px',
      minWidth: '32px',
      minHeight: '32px',
      borderRadius: '8px',
      background: 'transparent',
      border: 'none',
      padding: '0',
      margin: '0',
      color: 'rgba(160, 148, 210, 0.6)',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: '0',
      boxSizing: 'border-box',
      outline: 'none',
    });
    btn.innerHTML = `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${item.svg}</svg>`;
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(124, 58, 237, 0.2)';
      btn.style.color = 'rgba(224, 214, 255, 1)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'transparent';
      btn.style.color = 'rgba(160, 148, 210, 0.6)';
    });
    dockActions.appendChild(btn);
  });

  // Expand/collapse on hover via JS (more reliable than CSS :hover on injected elements)
  dockHost.addEventListener('mouseenter', () => {
    dockActions.style.maxHeight = '320px';
    dockActions.style.opacity = '1';
    dockActions.style.paddingTop = '5px';
    dockHost.style.borderColor = 'rgba(124, 58, 237, 0.5)';
    dockHost.style.boxShadow = '-4px 0 28px rgba(0,0,0,0.55)';
  });
  dockHost.addEventListener('mouseleave', () => {
    dockActions.style.maxHeight = '0';
    dockActions.style.opacity = '0';
    dockActions.style.paddingTop = '0';
    dockHost.style.borderColor = 'rgba(124, 58, 237, 0.3)';
    dockHost.style.boxShadow = '-3px 0 20px rgba(0,0,0,0.5)';
  });

  // Click handler
  dockHost.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    const url = window.location.href;
    const title = document.title;

    if (action === 'dock-chat') {
      ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'chat' });
    } else if (action === 'dock-search') {
      ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'search' });
    } else if (action === 'dock-translate') {
      ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'translate' });
    } else if (action === 'dock-thispage') {
      const pageText = document.body?.innerText?.slice(0, 25000) || '';
      ext.runtime.sendMessage({ type: 'OPEN_WITH_TEXT', action: 'ask', text: pageText, url, title });
    } else if (action === 'dock-save') {
      const selText = getSelectionText();
      const textToSave = selText || `${title}\n${url}`;
      ext.runtime.sendMessage(
        { type: 'SAVE_KNOWLEDGE', text: textToSave, url, title },
        (response) => {
          if (response && response.ok) showToast('Saved to AURA memory ✓');
          else showToast('Save failed — is backend running?', 3000);
        }
      );
    }
  });

  function showToast(message, durationMs = 2000) {
    toast.textContent = message;
    toast.classList.add('visible');
    toast.style.top = (parseInt(toolbar.style.top || '0') + 40) + 'px';
    toast.style.left = toolbar.style.left;
    setTimeout(() => toast.classList.remove('visible'), durationMs);
  }

  // ── Selection Handling ────────────────────────────────────────────────────

  function getSelectionText() {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return '';
    return sel.toString().trim();
  }

  function positionToolbar() {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const TOOLBAR_HEIGHT = 38;
    const MARGIN = 8;
    let left = rect.left + (rect.width / 2) - 100;
    if (left < 4) left = 4;
    if (left + 200 > window.innerWidth) left = window.innerWidth - 204;
    toolbar.style.top = `${Math.round(rect.top - TOOLBAR_HEIGHT - MARGIN)}px`;
    toolbar.style.left = `${Math.round(left)}px`;
  }

  function showToolbar() {
    toolbar.classList.add('visible');
    positionToolbar();
    host.style.pointerEvents = 'auto';
  }

  function hideToolbar() {
    toolbar.classList.remove('visible');
    host.style.pointerEvents = 'none';
  }

  document.addEventListener('mouseup', () => {
    setTimeout(() => {
      const text = getSelectionText();
      if (text.length > 0) showToolbar();
      else hideToolbar();
    }, 50);
  });

  document.addEventListener('mousedown', (e) => {
    if (!host.contains(e.target)) hideToolbar();
  });

  document.addEventListener('selectionchange', () => {
    if (getSelectionText().length === 0) hideToolbar();
  });

  // ── Button Handlers ───────────────────────────────────────────────────────

  toolbar.addEventListener('click', (e) => {
    const btn = e.target.closest('.aura-btn');
    if (!btn) return;
    const action = btn.dataset.action;
    const text = getSelectionText();
    if (!text) return;
    const url = window.location.href;
    const title = document.title;

    if (action === 'save') {
      ext.runtime.sendMessage(
        { type: 'SAVE_KNOWLEDGE', text, url, title },
        (response) => {
          if (response && response.ok) showToast('Saved to AURA memory ✓');
          else showToast('Save failed — is backend running?', 3000);
        }
      );
    } else {
      ext.runtime.sendMessage({ type: 'OPEN_WITH_TEXT', action, text, url, title });
    }
    hideToolbar();
  });

  // ── DOM Serializer (for Browser Agent) ───────────────────────────────────

  function bestSelector(el) {
    if (el.id) return '#' + CSS.escape(el.id);
    const al = el.getAttribute('aria-label');
    if (al) return `[aria-label="${al}"]`;
    const path = [];
    let cur = el;
    for (let i = 0; i < 4 && cur && cur !== document.body; i++, cur = cur.parentElement) {
      let s = cur.tagName.toLowerCase();
      if (cur.id) { path.unshift('#' + CSS.escape(cur.id)); break; }
      const idx = [...(cur.parentElement?.children || [])].indexOf(cur) + 1;
      path.unshift(s + ':nth-child(' + idx + ')');
    }
    return path.join('>');
  }

  function serializeDOM() {
    const els = [];
    document.querySelectorAll('a,button,input,textarea,select,[role="button"],[onclick]')
      .forEach((el, i) => {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return;
        els.push({
          index: i,
          type: el.tagName.toLowerCase(),
          text: (el.innerText || el.value || el.placeholder || el.title || '').slice(0, 80).trim(),
          selector: bestSelector(el),
          href: el.href || '',
        });
      });
    return els.slice(0, 80);
  }

  function execAction(action) {
    if (action.action === 'scroll') {
      window.scrollBy(0, action.amount || 300);
      return { ok: true };
    }
    const el = document.querySelector(action.selector);
    if (!el) return { ok: false, error: 'Element not found: ' + action.selector };
    if (action.action === 'click') { el.click(); return { ok: true }; }
    if (action.action === 'type') {
      el.focus();
      el.value = action.text;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return { ok: true };
    }
    return { ok: false, error: 'Unknown action: ' + action.action };
  }

  // ── OCR Overlay ──────────────────────────────────────────────────────────

  function showOcrOverlay(dataUrl, sendResponse) {
    // Create fullscreen overlay
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
      position: 'fixed', top: '0', left: '0', width: '100vw', height: '100vh',
      zIndex: '2147483646', cursor: 'crosshair', background: 'rgba(0,0,0,0.4)',
    });

    // Show the screenshot as background for reference
    const img = new Image();
    img.src = dataUrl;
    img.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;opacity:0.7;pointer-events:none;';
    overlay.appendChild(img);

    // Canvas for drawing selection rect
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    Object.assign(canvas.style, {
      position: 'absolute', top: '0', left: '0', width: '100%', height: '100%',
    });
    overlay.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const hint = document.createElement('div');
    Object.assign(hint.style, {
      position: 'fixed', top: '12px', left: '50%', transform: 'translateX(-50%)',
      background: 'rgba(0,0,0,0.75)', color: '#fff', padding: '6px 14px',
      borderRadius: '6px', fontSize: '13px', pointerEvents: 'none',
    });
    hint.textContent = 'Drag to select region • Press Esc to cancel';
    overlay.appendChild(hint);

    document.body.appendChild(overlay);

    let startX, startY, dragging = false;
    const dpr = window.devicePixelRatio || 1;

    function drawRect(x, y, w, h) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#7c3aed';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = 'rgba(124,58,237,0.12)';
      ctx.fillRect(x, y, w, h);
    }

    overlay.addEventListener('mousedown', (e) => {
      startX = e.clientX; startY = e.clientY; dragging = true;
    });

    overlay.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      drawRect(startX, startY, e.clientX - startX, e.clientY - startY);
    });

    overlay.addEventListener('mouseup', (e) => {
      dragging = false;
      const x = Math.min(startX, e.clientX);
      const y = Math.min(startY, e.clientY);
      const w = Math.abs(e.clientX - startX);
      const h = Math.abs(e.clientY - startY);
      document.body.removeChild(overlay);
      if (w < 5 || h < 5) { sendResponse({ ok: false }); return; }
      sendResponse({ ok: true, x, y, w, h, dpr });
    });

    document.addEventListener('keydown', function onEsc(e) {
      if (e.key === 'Escape') {
        document.body.removeChild(overlay);
        document.removeEventListener('keydown', onEsc);
        sendResponse({ ok: false });
      }
    });
  }

  // ── Message Listener ───────────────────────────────────────────────────────

  ext.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.type === 'EXTRACT_PAGE') {
      sendResponse({
        text: document.body?.innerText?.slice(0, 50000) || '',
        url: window.location.href,
        title: document.title,
      });
      return false;
    }

    if (msg.type === 'GET_DOM') {
      sendResponse({ ok: true, dom: serializeDOM(), url: location.href, title: document.title });
      return false;
    }

    if (msg.type === 'EXEC_ACTION') {
      sendResponse(execAction(msg.action));
      return false;
    }

    if (msg.type === 'SHOW_OCR_OVERLAY') {
      showOcrOverlay(msg.dataUrl, sendResponse);
      return true; // async
    }
  });

})();
