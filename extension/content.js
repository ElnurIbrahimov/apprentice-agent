(function() {
  "use strict";
  const ext = typeof browser !== "undefined" ? browser : chrome;
  function safeSend(msg, cb) {
    var _a, _b, _c, _d, _e;
    try {
      if (cb) {
        ext.runtime.sendMessage(msg, cb);
      } else {
        ext.runtime.sendMessage(msg);
      }
    } catch (e) {
      const err = e;
      if (((_a = err == null ? void 0 : err.message) == null ? void 0 : _a.includes("Extension context invalidated")) || ((_b = err == null ? void 0 : err.message) == null ? void 0 : _b.includes("context invalidated"))) {
        (_c = document.getElementById("aura-dock-host")) == null ? void 0 : _c.remove();
        (_d = document.getElementById("aura-host")) == null ? void 0 : _d.remove();
        (_e = document.getElementById("aura-quick-action-host")) == null ? void 0 : _e.remove();
        window.__auraToolbarMounted = false;
      }
    }
  }
  (function() {
    if (window.__auraToolbarMounted) return;
    window.__auraToolbarMounted = true;
    const _prevDock = document.getElementById("aura-dock-host");
    if (_prevDock) _prevDock.remove();
    const _prevHost = document.getElementById("aura-host");
    if (_prevHost) _prevHost.remove();
    const host = document.createElement("div");
    host.id = "aura-host";
    Object.assign(host.style, {
      position: "fixed",
      top: "0",
      left: "0",
      zIndex: "2147483647",
      pointerEvents: "none"
    });
    document.documentElement.appendChild(host);
    const shadow = host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = `
    /* ── Toolbar appear/disappear animations ── */
    @keyframes aura-toolbar-in {
      from {
        opacity: 0;
        transform: scale(0.92) translateY(-4px);
      }
      to {
        opacity: 1;
        transform: scale(1) translateY(0);
      }
    }
    @keyframes aura-toolbar-out {
      from {
        opacity: 1;
        transform: scale(1) translateY(0);
      }
      to {
        opacity: 0;
        transform: scale(0.92) translateY(-4px);
      }
    }

    #toolbar {
      display: none;
      position: fixed;
      background: rgba(10, 8, 24, 0.88);
      backdrop-filter: blur(20px) saturate(1.5);
      -webkit-backdrop-filter: blur(20px) saturate(1.5);
      border: 1px solid rgba(124, 58, 237, 0.25);
      border-radius: 12px;
      padding: 5px 8px;
      gap: 3px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255,255,255,0.05) inset;
      pointer-events: auto;
      z-index: 2147483647;
      align-items: center;
      transform-origin: center bottom;
    }
    #toolbar.visible {
      display: flex;
      animation: aura-toolbar-in 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    #toolbar.hiding {
      display: flex;
      animation: aura-toolbar-out 0.15s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    #toolbar.below {
      transform-origin: center top;
    }

    .aura-btn {
      background: transparent;
      border: none;
      color: rgba(226, 232, 240, 0.9);
      font-size: 12.5px;
      font-weight: 500;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      padding: 5px 10px;
      border-radius: 8px;
      cursor: pointer;
      white-space: nowrap;
      display: flex;
      align-items: center;
      gap: 5px;
      transition: background 0.15s ease, transform 0.15s ease, color 0.15s ease;
      line-height: 1;
    }
    .aura-btn:hover {
      background: rgba(124, 58, 237, 0.3);
      color: #fff;
      transform: scale(1.02);
    }
    .aura-btn:active {
      transform: scale(0.98);
    }
    .aura-btn svg {
      flex-shrink: 0;
    }

    .aura-divider {
      width: 1px;
      height: 14px;
      background: rgba(255, 255, 255, 0.08);
      flex-shrink: 0;
    }

    #toast {
      display: none;
      position: fixed;
      background: rgba(5, 150, 105, 0.92);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      color: #fff;
      font-size: 12px;
      font-weight: 500;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      padding: 6px 14px;
      border-radius: 8px;
      pointer-events: none;
      z-index: 2147483647;
      box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    #toast.visible {
      display: block;
    }
  `;
    shadow.appendChild(style);
    const toolbar = document.createElement("div");
    toolbar.id = "toolbar";
    const ICON_SPARKLES = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/><path d="M19 1l.5 1.5L21 3l-1.5.5L19 5l-.5-1.5L17 3l1.5-.5L19 1z"/><path d="M5 19l.5 1.5L7 21l-1.5.5L5 23l-.5-1.5L3 21l1.5-.5L5 19z"/></svg>`;
    const ICON_LAYERS = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>`;
    const ICON_MESSAGE = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/></svg>`;
    const ICON_BOOKMARK = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/></svg>`;
    const buttons = [
      { label: "Explain", icon: ICON_SPARKLES, action: "explain" },
      { label: "Summarize", icon: ICON_LAYERS, action: "summarize" },
      { label: "Ask AURA", icon: ICON_MESSAGE, action: "ask" },
      { label: "Save", icon: ICON_BOOKMARK, action: "save" }
    ];
    buttons.forEach((btn, i) => {
      if (i > 0) {
        const div = document.createElement("div");
        div.className = "aura-divider";
        toolbar.appendChild(div);
      }
      const el = document.createElement("button");
      el.className = "aura-btn";
      el.innerHTML = btn.icon + `<span>${btn.label}</span>`;
      el.dataset.action = btn.action;
      toolbar.appendChild(el);
    });
    shadow.appendChild(toolbar);
    const toast = document.createElement("div");
    toast.id = "toast";
    shadow.appendChild(toast);
    const dockShadowHost = document.createElement("div");
    dockShadowHost.id = "aura-dock-shadow";
    Object.assign(dockShadowHost.style, { position: "fixed", right: "0", top: "0", zIndex: "2147483647", pointerEvents: "none" });
    document.body.appendChild(dockShadowHost);
    const dockShadow = dockShadowHost.attachShadow({ mode: "closed" });
    const dockStyle = document.createElement("style");
    dockStyle.textContent = `
    @keyframes aura-breathe {
      0%, 100% { box-shadow: 0 0 4px rgba(124, 58, 237, 0.2), 0 0 8px rgba(124, 58, 237, 0.1); }
      50% { box-shadow: 0 0 8px rgba(124, 58, 237, 0.45), 0 0 16px rgba(124, 58, 237, 0.2); }
    }

    #aura-dock-host {
      position: fixed;
      right: 0;
      top: 50%;
      transform: translateY(-50%);
      z-index: 2147483647;
      pointer-events: auto;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0;
      padding: 7px 4px;
      background: rgba(10, 8, 24, 0.88);
      backdrop-filter: blur(20px) saturate(1.5);
      -webkit-backdrop-filter: blur(20px) saturate(1.5);
      border: 1px solid rgba(124, 58, 237, 0.25);
      border-right: none;
      border-radius: 12px 0 0 12px;
      box-shadow: -3px 0 20px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05) inset;
      transition: border-color 0.25s ease, box-shadow 0.25s ease;
      box-sizing: border-box;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    }
    #aura-dock-host:hover {
      border-color: rgba(124, 58, 237, 0.5);
      box-shadow: -4px 0 28px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.08) inset;
    }

    .dock-logo-wrap {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
      cursor: default;
      flex-shrink: 0;
    }
    .dock-logo-icon {
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: rgba(160, 148, 210, 0.9);
      border-radius: 8px;
      animation: aura-breathe 3s ease-in-out infinite;
    }
    .dock-logo-label {
      font-size: 8px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: rgba(160, 148, 210, 0.5);
      text-transform: uppercase;
      line-height: 1;
      opacity: 1;
      transition: opacity 0.2s ease;
    }
    #aura-dock-host:hover .dock-logo-label {
      opacity: 0;
    }

    .dock-actions {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 3px;
      overflow: hidden;
      max-height: 0;
      opacity: 0;
      padding-top: 0;
      transition: max-height 0.3s cubic-bezier(0.16, 1, 0.3, 1),
                  opacity 0.2s ease,
                  padding-top 0.25s ease;
    }
    #aura-dock-host:hover .dock-actions {
      max-height: 320px;
      opacity: 1;
      padding-top: 5px;
    }

    .dock-sep {
      width: 18px;
      height: 1px;
      background: rgba(255, 255, 255, 0.08);
      margin: 2px 0;
      flex-shrink: 0;
    }

    .dock-btn {
      width: 32px;
      height: 32px;
      min-width: 32px;
      min-height: 32px;
      border-radius: 8px;
      background: transparent;
      border: none;
      padding: 0;
      margin: 0;
      color: rgba(160, 148, 210, 0.6);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      box-sizing: border-box;
      outline: none;
      transition: background 0.15s ease, color 0.15s ease, transform 0.15s ease;
    }
    .dock-btn:hover {
      background: rgba(124, 58, 237, 0.25);
      color: rgba(224, 214, 255, 1);
      transform: scale(1.05);
    }
    .dock-btn:active {
      transform: scale(0.95);
    }
  `;
    dockShadow.appendChild(dockStyle);
    const dockHost = document.createElement("div");
    dockHost.id = "aura-dock-host";
    dockShadow.appendChild(dockHost);
    const dockLogoWrap = document.createElement("div");
    dockLogoWrap.className = "dock-logo-wrap";
    const dockLogoIcon = document.createElement("div");
    dockLogoIcon.className = "dock-logo-icon";
    dockLogoIcon.innerHTML = `<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;
    dockLogoWrap.appendChild(dockLogoIcon);
    const dockLogoLabel = document.createElement("div");
    dockLogoLabel.className = "dock-logo-label";
    dockLogoLabel.textContent = "A";
    dockLogoWrap.appendChild(dockLogoLabel);
    dockHost.appendChild(dockLogoWrap);
    const dockActions = document.createElement("div");
    dockActions.className = "dock-actions";
    dockHost.appendChild(dockActions);
    const _dockItems = [
      { svg: '<path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>', action: "dock-chat", tip: "Chat with AURA" },
      { svg: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>', action: "dock-search", tip: "Search" },
      null,
      { svg: '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><path d="M9 21V9"/>', action: "dock-thispage", tip: "This Page" },
      { svg: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>', action: "dock-translate", tip: "Translate" },
      null,
      { svg: '<path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>', action: "dock-save", tip: "Save to Memory" }
    ];
    _dockItems.forEach((item) => {
      if (!item) {
        const sep = document.createElement("div");
        sep.className = "dock-sep";
        dockActions.appendChild(sep);
        return;
      }
      const btn = document.createElement("button");
      btn.className = "dock-btn";
      btn.dataset.action = item.action;
      btn.title = item.tip;
      btn.innerHTML = `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${item.svg}</svg>`;
      dockActions.appendChild(btn);
    });
    dockHost.addEventListener("click", (e) => {
      const target = e.target;
      const btn = target.closest("[data-action]");
      if (!btn) return;
      const action = btn.dataset.action;
      const url = window.location.href;
      const title = document.title;
      if (action === "dock-chat") {
        safeSend({ type: "OPEN_PANEL", panel: "chat" });
      } else if (action === "dock-search") {
        safeSend({ type: "OPEN_PANEL", panel: "search" });
      } else if (action === "dock-translate") {
        safeSend({ type: "OPEN_PANEL", panel: "translate" });
      } else if (action === "dock-thispage") {
        const extracted = extractMainContent();
        safeSend({ type: "OPEN_WITH_TEXT", action: "ask", text: extracted.text, url, title });
      } else if (action === "dock-save") {
        const selText = getSelectionText();
        const textToSave = selText || `${title}
${url}`;
        safeSend(
          { type: "SAVE_KNOWLEDGE", text: textToSave, url, title },
          (response) => {
            if (response && response.ok) showToast("Saved to AURA memory ✓");
            else showToast("Save failed — is backend running?", 3e3);
          }
        );
      }
    });
    let _toastTimer = null;
    function showToast(message, durationMs = 2e3) {
      if (_toastTimer) {
        clearTimeout(_toastTimer);
        _toastTimer = null;
      }
      toast.textContent = message;
      toast.classList.add("visible");
      if (toolbar.classList.contains("visible") && toolbar.style.top) {
        toast.style.top = parseInt(toolbar.style.top) + 40 + "px";
        toast.style.left = toolbar.style.left;
      } else {
        toast.style.top = "20px";
        toast.style.left = Math.round(window.innerWidth / 2 - 100) + "px";
      }
      _toastTimer = setTimeout(() => {
        toast.classList.remove("visible");
        _toastTimer = null;
      }, durationMs);
    }
    function getSelectionText() {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return "";
      return sel.toString().trim();
    }
    let _hideAnimTimer = null;
    function positionToolbar() {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      const range = sel.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      const TOOLBAR_HEIGHT = 42;
      const GAP = 10;
      const vw = window.innerWidth;
      toolbar.style.visibility = "hidden";
      toolbar.style.display = "flex";
      const tbRect = toolbar.getBoundingClientRect();
      const tbWidth = tbRect.width || 320;
      toolbar.style.visibility = "";
      toolbar.style.display = "";
      let left = rect.left + rect.width / 2 - tbWidth / 2;
      if (left < 8) left = 8;
      if (left + tbWidth > vw - 8) left = vw - tbWidth - 8;
      const showBelow = rect.top < TOOLBAR_HEIGHT + GAP + 10;
      toolbar.classList.toggle("below", showBelow);
      if (showBelow) {
        toolbar.style.top = `${Math.round(rect.bottom + GAP)}px`;
      } else {
        toolbar.style.top = `${Math.round(rect.top - TOOLBAR_HEIGHT - GAP)}px`;
      }
      toolbar.style.left = `${Math.round(left)}px`;
    }
    function showToolbar() {
      if (_hideAnimTimer) {
        clearTimeout(_hideAnimTimer);
        _hideAnimTimer = null;
      }
      toolbar.classList.remove("hiding");
      toolbar.classList.add("visible");
      positionToolbar();
      host.style.pointerEvents = "auto";
    }
    function hideToolbar() {
      if (!toolbar.classList.contains("visible")) return;
      toolbar.classList.remove("visible");
      toolbar.classList.add("hiding");
      if (_hideAnimTimer) clearTimeout(_hideAnimTimer);
      _hideAnimTimer = setTimeout(() => {
        toolbar.classList.remove("hiding");
        host.style.pointerEvents = "none";
        _hideAnimTimer = null;
      }, 150);
    }
    document.addEventListener("mouseup", () => {
      setTimeout(() => {
        const text = getSelectionText();
        if (text.length > 0) showToolbar();
        else hideToolbar();
      }, 50);
    });
    document.addEventListener("mousedown", (e) => {
      if (!host.contains(e.target)) hideToolbar();
    });
    document.addEventListener("selectionchange", () => {
      if (getSelectionText().length === 0) hideToolbar();
    });
    toolbar.addEventListener("click", (e) => {
      const target = e.target;
      const btn = target.closest(".aura-btn");
      if (!btn) return;
      const action = btn.dataset.action;
      const text = getSelectionText();
      if (!text) return;
      const url = window.location.href;
      const title = document.title;
      if (action === "save") {
        safeSend(
          { type: "SAVE_KNOWLEDGE", text, url, title },
          (response) => {
            if (response && response.ok) showToast("Saved to AURA memory ✓");
            else showToast("Save failed — is backend running?", 3e3);
          }
        );
      } else {
        safeSend({ type: "OPEN_WITH_TEXT", action, text, url, title });
      }
      hideToolbar();
    });
    function bestSelector(el) {
      var _a;
      if (el.id) return "#" + CSS.escape(el.id);
      const al = el.getAttribute("aria-label");
      if (al) return `[aria-label="${al}"]`;
      const path = [];
      let cur = el;
      for (let i = 0; i < 4 && cur && cur !== document.body; i++, cur = cur.parentElement) {
        const s = cur.tagName.toLowerCase();
        if (cur.id) {
          path.unshift("#" + CSS.escape(cur.id));
          break;
        }
        const siblings = [...((_a = cur.parentElement) == null ? void 0 : _a.children) || []];
        const idx = siblings.indexOf(cur) + 1;
        path.unshift(s + ":nth-child(" + idx + ")");
      }
      return path.join(">");
    }
    function serializeDOM() {
      const els = [];
      const nodes = document.querySelectorAll(
        'a,button,input,textarea,select,[role="button"],[onclick]'
      );
      let idx = 0;
      for (const el of nodes) {
        if (els.length >= 80) break;
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        const htmlEl = el;
        const inputEl = el;
        els.push({
          index: idx++,
          type: el.tagName.toLowerCase(),
          text: (htmlEl.innerText || inputEl.value || inputEl.placeholder || htmlEl.title || "").slice(0, 80).trim(),
          selector: bestSelector(htmlEl),
          href: el.href || ""
        });
      }
      return els;
    }
    function execAction(action) {
      if (action.action === "scroll") {
        window.scrollBy(0, action.amount || 300);
        return { ok: true };
      }
      let el;
      try {
        el = document.querySelector(action.selector);
      } catch (e) {
        return { ok: false, error: "Invalid selector: " + action.selector };
      }
      if (!el) return { ok: false, error: "Element not found: " + action.selector };
      if (action.action === "click") {
        el.click();
        return { ok: true };
      }
      if (action.action === "type") {
        el.focus();
        el.value = action.text || "";
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return { ok: true };
      }
      if (action.action === "selectOption") {
        if (el.tagName.toLowerCase() !== "select") return { ok: false, error: "Element is not a <select>" };
        const selectEl = el;
        const opt = [...selectEl.options].find(
          (o) => o.value === action.value || o.text === action.value
        );
        if (!opt) return { ok: false, error: "Option not found: " + action.value };
        selectEl.value = opt.value;
        selectEl.dispatchEvent(new Event("change", { bubbles: true }));
        return { ok: true };
      }
      return { ok: false, error: "Unknown action: " + action.action };
    }
    function showOcrOverlay(dataUrl, sendResponse) {
      const overlay = document.createElement("div");
      Object.assign(overlay.style, {
        position: "fixed",
        top: "0",
        left: "0",
        width: "100vw",
        height: "100vh",
        zIndex: "2147483646",
        cursor: "crosshair",
        background: "rgba(0,0,0,0.4)"
      });
      const img = new Image();
      img.src = dataUrl;
      img.style.cssText = "position:fixed;top:0;left:0;width:100%;height:100%;opacity:0.7;pointer-events:none;";
      overlay.appendChild(img);
      const canvas = document.createElement("canvas");
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      Object.assign(canvas.style, {
        position: "absolute",
        top: "0",
        left: "0",
        width: "100%",
        height: "100%"
      });
      overlay.appendChild(canvas);
      const ctx = canvas.getContext("2d");
      const hint = document.createElement("div");
      Object.assign(hint.style, {
        position: "fixed",
        top: "12px",
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(0,0,0,0.75)",
        color: "#fff",
        padding: "6px 14px",
        borderRadius: "6px",
        fontSize: "13px",
        pointerEvents: "none"
      });
      hint.textContent = "Drag to select region • Press Esc to cancel";
      overlay.appendChild(hint);
      document.body.appendChild(overlay);
      let startX = 0;
      let startY = 0;
      let dragging = false;
      const dpr = window.devicePixelRatio || 1;
      function drawRect(x, y, w, h) {
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#7c3aed";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "rgba(124,58,237,0.12)";
        ctx.fillRect(x, y, w, h);
      }
      overlay.addEventListener("mousedown", (e) => {
        startX = e.clientX;
        startY = e.clientY;
        dragging = true;
      });
      overlay.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        drawRect(startX, startY, e.clientX - startX, e.clientY - startY);
      });
      function onEsc(e) {
        if (e.key === "Escape") {
          if (document.body.contains(overlay)) document.body.removeChild(overlay);
          document.removeEventListener("keydown", onEsc);
          sendResponse({ ok: false });
        }
      }
      overlay.addEventListener("mouseup", (e) => {
        dragging = false;
        const x = Math.min(startX, e.clientX);
        const y = Math.min(startY, e.clientY);
        const w = Math.abs(e.clientX - startX);
        const h = Math.abs(e.clientY - startY);
        document.removeEventListener("keydown", onEsc);
        if (document.body.contains(overlay)) document.body.removeChild(overlay);
        if (w < 5 || h < 5) {
          sendResponse({ ok: false });
          return;
        }
        sendResponse({ ok: true, x, y, w, h, dpr });
      });
      document.addEventListener("keydown", onEsc);
      const ocrCleanupObserver = new MutationObserver(() => {
        if (!document.body.contains(overlay)) {
          document.removeEventListener("keydown", onEsc);
          ocrCleanupObserver.disconnect();
        }
      });
      ocrCleanupObserver.observe(document.body, { childList: true });
    }
    const MAX_TEXT_BYTES = 5e4;
    const CONTENT_SELECTORS = [
      "article",
      "main",
      '[role="main"]',
      ".post-content",
      ".article-body",
      ".entry-content",
      ".post-body",
      ".article-content",
      ".story-body",
      ".content-body",
      "#article-body",
      "#content",
      ".markdown-body",
      // GitHub
      ".wiki-content"
    ];
    const JUNK_SELECTORS = [
      "nav",
      "header",
      "footer",
      "aside",
      "script",
      "style",
      "noscript",
      "iframe",
      ".sidebar",
      ".menu",
      ".nav",
      ".navigation",
      ".cookie",
      ".cookie-banner",
      ".cookie-consent",
      ".popup",
      ".modal",
      ".overlay",
      ".ad",
      ".ads",
      ".advert",
      ".advertisement",
      ".social-share",
      ".share-buttons",
      ".social-buttons",
      ".comments",
      ".comment-section",
      "#comments",
      ".related-posts",
      ".recommended",
      ".newsletter",
      ".subscribe",
      '[role="navigation"]',
      '[role="banner"]',
      '[role="contentinfo"]',
      '[role="complementary"]',
      '[aria-hidden="true"]',
      ".sr-only",
      ".visually-hidden"
    ];
    function findContentRoot() {
      for (const sel of CONTENT_SELECTORS) {
        const el = document.querySelector(sel);
        if (el && el.textContent && el.textContent.trim().length > 200) {
          return el;
        }
      }
      return document.body;
    }
    function cleanClone(root) {
      const clone = root.cloneNode(true);
      for (const sel of JUNK_SELECTORS) {
        clone.querySelectorAll(sel).forEach((el) => el.remove());
      }
      return clone;
    }
    function domToStructuredText(root) {
      const parts = [];
      const BLOCK_TAGS = /* @__PURE__ */ new Set([
        "P",
        "DIV",
        "SECTION",
        "ARTICLE",
        "BLOCKQUOTE",
        "PRE",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "UL",
        "OL",
        "LI",
        "TABLE",
        "TR",
        "DT",
        "DD",
        "FIGURE",
        "FIGCAPTION",
        "HR",
        "BR"
      ]);
      function walk(node) {
        var _a;
        if (node.nodeType === Node.TEXT_NODE) {
          const text = (node.textContent || "").replace(/\s+/g, " ");
          if (text.trim()) parts.push(text);
          return;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        const el = node;
        const tag = el.tagName;
        if (el.hasAttribute("hidden") || ((_a = el.style) == null ? void 0 : _a.display) === "none") return;
        if (/^H[1-6]$/.test(tag)) {
          const level = parseInt(tag[1]);
          const prefix = "#".repeat(Math.min(level, 3)) + " ";
          const headingText = (el.textContent || "").trim();
          if (headingText) {
            parts.push("\n\n" + prefix + headingText + "\n");
          }
          return;
        }
        if (tag === "LI") {
          const text = (el.textContent || "").trim();
          if (text) {
            parts.push("\n- " + text);
          }
          return;
        }
        if (tag === "A") {
          const href = el.href;
          const text = (el.textContent || "").trim();
          if (text && href && !href.startsWith("javascript:")) {
            parts.push(text + " (" + href + ")");
          } else if (text) {
            parts.push(text);
          }
          return;
        }
        if (tag === "HR") {
          parts.push("\n\n---\n\n");
          return;
        }
        if (tag === "BR") {
          parts.push("\n");
          return;
        }
        if (tag === "PRE") {
          const text = (el.textContent || "").trim();
          if (text) parts.push("\n\n```\n" + text + "\n```\n\n");
          return;
        }
        const isBlock = BLOCK_TAGS.has(tag);
        if (isBlock) parts.push("\n\n");
        for (const child of el.childNodes) {
          walk(child);
        }
        if (isBlock) parts.push("\n");
      }
      walk(root);
      return parts.join("").replace(/\n{3,}/g, "\n\n").replace(/[ \t]+/g, " ").trim();
    }
    function extractMainContent() {
      var _a, _b, _c, _d;
      try {
        const url = window.location.href;
        const title = document.title;
        if (url.match(/\.pdf($|\?|#)/i) || document.contentType === "application/pdf") {
          return {
            text: ((_b = (_a = document.body) == null ? void 0 : _a.innerText) == null ? void 0 : _b.slice(0, MAX_TEXT_BYTES)) || "[PDF document]",
            title,
            url,
            wordCount: 0,
            isPdf: true
          };
        }
        if (url.includes("youtube.com/watch") || url.includes("youtu.be/")) {
          return extractYouTubeContent();
        }
        const root = findContentRoot();
        const cleaned = cleanClone(root);
        let text = domToStructuredText(cleaned);
        if (text.length < 100) {
          text = ((_c = document.body) == null ? void 0 : _c.innerText) || "";
        }
        if (text.length > MAX_TEXT_BYTES) {
          text = text.slice(0, MAX_TEXT_BYTES) + "\n\n[...truncated]";
        }
        const wordCount = text.split(/\s+/).filter(Boolean).length;
        return { text, title, url, wordCount };
      } catch (_e) {
        const fallbackText = (((_d = document.body) == null ? void 0 : _d.innerText) || "").slice(0, MAX_TEXT_BYTES);
        return {
          text: fallbackText,
          title: document.title,
          url: window.location.href,
          wordCount: fallbackText.split(/\s+/).filter(Boolean).length
        };
      }
    }
    function extractYouTubeContent() {
      var _a, _b;
      const url = window.location.href;
      const titleEl = document.querySelector(
        "h1.ytd-watch-metadata, h1.ytd-video-primary-info-renderer, #title h1"
      );
      const videoTitle = ((_a = titleEl == null ? void 0 : titleEl.textContent) == null ? void 0 : _a.trim()) || document.title.replace(/ - YouTube$/, "").trim();
      let transcript = "";
      const transcriptSegments = document.querySelectorAll(
        "ytd-transcript-segment-renderer .segment-text, yt-formatted-string.ytd-transcript-segment-renderer, #segments-container ytd-transcript-segment-renderer"
      );
      if (transcriptSegments.length > 0) {
        const lines = [];
        transcriptSegments.forEach((seg) => {
          var _a2;
          const text2 = (_a2 = seg.textContent) == null ? void 0 : _a2.trim();
          if (text2) lines.push(text2);
        });
        transcript = lines.join(" ");
      }
      let description = "";
      const descEl = document.querySelector(
        "ytd-text-inline-expander #plain-snippet-text, #description-inline-expander, ytd-expander .content, #description .content"
      );
      if (descEl) {
        description = ((_b = descEl.textContent) == null ? void 0 : _b.trim()) || "";
      }
      const commentEls = document.querySelectorAll(
        "ytd-comment-thread-renderer #content-text"
      );
      let comments = "";
      if (commentEls.length > 0) {
        const commentLines = [];
        commentEls.forEach((el, i) => {
          var _a2;
          if (i >= 10) return;
          const text2 = (_a2 = el.textContent) == null ? void 0 : _a2.trim();
          if (text2) commentLines.push("- " + text2);
        });
        if (commentLines.length > 0) {
          comments = "\n\n## Top Comments\n" + commentLines.join("\n");
        }
      }
      let text = `# ${videoTitle}

`;
      if (transcript) {
        text += `## Transcript
${transcript}

`;
      }
      if (description) {
        text += `## Description
${description}

`;
      }
      text += comments;
      if (text.length > MAX_TEXT_BYTES) {
        text = text.slice(0, MAX_TEXT_BYTES) + "\n\n[...truncated]";
      }
      const wordCount = text.split(/\s+/).filter(Boolean).length;
      return {
        text,
        title: videoTitle,
        url,
        wordCount,
        isYouTube: true,
        videoTitle,
        transcript: transcript || void 0
      };
    }
    const QUICK_ACTIONS = [
      { label: "Improve", icon: '<path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/>', action: "improve" },
      { label: "Expand", icon: '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>', action: "expand" },
      { label: "Shorten", icon: '<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>', action: "shorten" },
      { label: "Fix grammar", icon: '<polyline points="20 6 9 17 4 12"/>', action: "fix_grammar" },
      { label: "Translate", icon: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>', action: "translate" }
    ];
    const qaHost = document.createElement("div");
    qaHost.id = "aura-quick-action-host";
    Object.assign(qaHost.style, {
      position: "fixed",
      top: "0",
      left: "0",
      zIndex: "2147483646",
      pointerEvents: "none"
    });
    document.documentElement.appendChild(qaHost);
    const qaShadow = qaHost.attachShadow({ mode: "closed" });
    const qaStyle = document.createElement("style");
    qaStyle.textContent = `
    @keyframes qa-icon-in {
      from { opacity: 0; transform: scale(0.7); }
      to   { opacity: 1; transform: scale(1); }
    }
    @keyframes qa-menu-in {
      from { opacity: 0; transform: translateY(4px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes qa-spin {
      to { transform: rotate(360deg); }
    }

    .qa-trigger {
      position: fixed;
      width: 20px;
      height: 20px;
      border-radius: 5px;
      background: rgba(10, 8, 24, 0.75);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(124, 58, 237, 0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      pointer-events: auto;
      animation: qa-icon-in 0.2s ease forwards;
      transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
      padding: 0;
      box-sizing: border-box;
    }
    .qa-trigger:hover {
      border-color: rgba(124, 58, 237, 0.6);
      background: rgba(124, 58, 237, 0.18);
      box-shadow: 0 0 10px rgba(124, 58, 237, 0.25);
    }
    .qa-trigger svg {
      width: 12px;
      height: 12px;
      color: rgba(160, 148, 210, 0.8);
    }

    .qa-menu {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(20px) saturate(1.5);
      -webkit-backdrop-filter: blur(20px) saturate(1.5);
      border: 1px solid rgba(124, 58, 237, 0.25);
      border-radius: 10px;
      padding: 4px;
      pointer-events: auto;
      animation: qa-menu-in 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05) inset;
      min-width: 140px;
    }
    .qa-menu-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      border-radius: 7px;
      background: transparent;
      border: none;
      color: rgba(226, 232, 240, 0.9);
      font-size: 12px;
      font-weight: 500;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      cursor: pointer;
      white-space: nowrap;
      width: 100%;
      text-align: left;
      transition: background 0.12s, color 0.12s;
      line-height: 1;
      box-sizing: border-box;
    }
    .qa-menu-item:hover {
      background: rgba(124, 58, 237, 0.25);
      color: #fff;
    }
    .qa-menu-item:active {
      background: rgba(124, 58, 237, 0.4);
    }
    .qa-menu-item svg {
      width: 14px;
      height: 14px;
      flex-shrink: 0;
      color: rgba(160, 148, 210, 0.7);
    }
    .qa-menu-item:hover svg {
      color: rgba(200, 180, 255, 1);
    }
    .qa-menu-item.loading {
      opacity: 0.6;
      pointer-events: none;
    }
    .qa-menu-item .qa-spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(124, 58, 237, 0.3);
      border-top-color: rgba(160, 148, 210, 0.9);
      border-radius: 50%;
      animation: qa-spin 0.6s linear infinite;
      flex-shrink: 0;
    }

    .qa-translate-sub {
      padding: 2px 4px 4px 4px;
    }
    .qa-translate-sub .qa-menu-item {
      font-size: 11.5px;
      padding: 5px 10px 5px 22px;
    }
  `;
    qaShadow.appendChild(qaStyle);
    const qaContainer = document.createElement("div");
    qaShadow.appendChild(qaContainer);
    let _qaActiveInput = null;
    let _qaTriggerEl = null;
    let _qaMenuEl = null;
    let _qaTranslateSub = null;
    const SKIP_INPUT_TYPES = /* @__PURE__ */ new Set(["password", "hidden", "file", "checkbox", "radio", "range", "color", "date", "datetime-local", "month", "week", "time", "submit", "reset", "button", "image"]);
    const MIN_INPUT_WIDTH = 200;
    function isEligibleInput(el) {
      if (el.tagName === "TEXTAREA") return true;
      if (el.tagName === "INPUT") {
        const inputEl = el;
        const inputType = (inputEl.type || "text").toLowerCase();
        if (SKIP_INPUT_TYPES.has(inputType)) return false;
        return true;
      }
      if (el.isContentEditable && el.getAttribute("contenteditable") === "true") return true;
      return false;
    }
    function removeQaTrigger() {
      if (_qaTriggerEl) {
        _qaTriggerEl.remove();
        _qaTriggerEl = null;
      }
      removeQaMenu();
      _qaActiveInput = null;
    }
    function removeQaMenu() {
      if (_qaMenuEl) {
        _qaMenuEl.remove();
        _qaMenuEl = null;
      }
      if (_qaTranslateSub) {
        _qaTranslateSub.remove();
        _qaTranslateSub = null;
      }
    }
    function positionTrigger(field) {
      const rect = field.getBoundingClientRect();
      if (rect.width < MIN_INPUT_WIDTH) {
        removeQaTrigger();
        return;
      }
      if (!_qaTriggerEl) {
        _qaTriggerEl = document.createElement("div");
        _qaTriggerEl.className = "qa-trigger";
        _qaTriggerEl.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>`;
        _qaTriggerEl.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (_qaMenuEl) {
            removeQaMenu();
            return;
          }
          showQaMenu();
        });
        qaContainer.appendChild(_qaTriggerEl);
      }
      const trigSize = 20;
      const pad = 6;
      _qaTriggerEl.style.top = `${Math.round(rect.top + (rect.height - trigSize) / 2)}px`;
      _qaTriggerEl.style.left = `${Math.round(rect.right - trigSize - pad)}px`;
    }
    function showQaMenu() {
      if (!_qaTriggerEl || !_qaActiveInput) return;
      removeQaMenu();
      _qaMenuEl = document.createElement("div");
      _qaMenuEl.className = "qa-menu";
      QUICK_ACTIONS.forEach((qa) => {
        const item = document.createElement("button");
        item.className = "qa-menu-item";
        item.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${qa.icon}</svg><span>${qa.label}</span>`;
        item.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (qa.action === "translate") {
            toggleTranslateSub(item);
          } else {
            executeQuickAction(qa.action);
          }
        });
        _qaMenuEl.appendChild(item);
      });
      qaContainer.appendChild(_qaMenuEl);
      const trigRect = _qaTriggerEl.getBoundingClientRect();
      const menuGap = 6;
      _qaMenuEl.style.top = `${Math.round(trigRect.bottom + menuGap)}px`;
      _qaMenuEl.style.left = `${Math.round(trigRect.right - 150)}px`;
      requestAnimationFrame(() => {
        if (!_qaMenuEl) return;
        const mRect = _qaMenuEl.getBoundingClientRect();
        if (mRect.right > window.innerWidth - 8) {
          _qaMenuEl.style.left = `${Math.round(window.innerWidth - mRect.width - 8)}px`;
        }
        if (mRect.left < 8) {
          _qaMenuEl.style.left = "8px";
        }
        if (mRect.bottom > window.innerHeight - 8) {
          _qaMenuEl.style.top = `${Math.round(trigRect.top - mRect.height - menuGap)}px`;
        }
      });
    }
    function toggleTranslateSub(anchor) {
      if (_qaTranslateSub) {
        _qaTranslateSub.remove();
        _qaTranslateSub = null;
        return;
      }
      const LANGUAGES = ["English", "Spanish", "French", "German", "Chinese", "Russian", "Japanese", "Arabic", "Portuguese", "Azerbaijani"];
      _qaTranslateSub = document.createElement("div");
      _qaTranslateSub.className = "qa-translate-sub";
      LANGUAGES.forEach((lang) => {
        const item = document.createElement("button");
        item.className = "qa-menu-item";
        item.textContent = lang;
        item.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          executeQuickAction("translate", lang);
        });
        _qaTranslateSub.appendChild(item);
      });
      if (_qaMenuEl && anchor.parentNode === _qaMenuEl) {
        anchor.after(_qaTranslateSub);
      }
    }
    function getInputValue(el) {
      if (el.isContentEditable) {
        return el.innerText || "";
      }
      return el.value || "";
    }
    function setInputValue(el, value) {
      if (el.isContentEditable) {
        el.innerText = value;
      } else {
        el.value = value;
      }
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
    function executeQuickAction(action, language) {
      if (!_qaActiveInput) return;
      const text = getInputValue(_qaActiveInput);
      if (!text.trim()) {
        removeQaMenu();
        return;
      }
      if (_qaMenuEl) {
        _qaMenuEl.querySelectorAll(".qa-menu-item").forEach((item) => {
          item.classList.add("loading");
        });
      }
      const targetField = _qaActiveInput;
      safeSend(
        { type: "QUICK_ACTION", action, text, language },
        (response) => {
          if (response && response.ok && response.result) {
            setInputValue(targetField, response.result);
            showToast("Text updated by AURA");
          } else {
            showToast((response == null ? void 0 : response.error) || "Quick action failed", 3e3);
          }
          removeQaMenu();
        }
      );
    }
    function onFieldFocus(e) {
      const target = e.target;
      if (!target || !isEligibleInput(target)) return;
      const rect = target.getBoundingClientRect();
      if (rect.width < MIN_INPUT_WIDTH) return;
      _qaActiveInput = target;
      positionTrigger(target);
    }
    function onFieldBlur(_e) {
      setTimeout(() => {
        if (_qaMenuEl || _qaTranslateSub) return;
        const active = document.activeElement;
        if (active && active === _qaActiveInput) return;
        removeQaTrigger();
      }, 200);
    }
    function repositionQaTrigger() {
      if (_qaActiveInput) {
        positionTrigger(_qaActiveInput);
      }
    }
    document.addEventListener("mousedown", (e) => {
      if (!_qaMenuEl && !_qaTriggerEl) return;
      const path = e.composedPath();
      if (path.includes(qaHost)) return;
      removeQaMenu();
    }, true);
    document.addEventListener("focusin", onFieldFocus, true);
    document.addEventListener("focusout", onFieldBlur, true);
    window.addEventListener("scroll", repositionQaTrigger, { passive: true });
    window.addEventListener("resize", repositionQaTrigger, { passive: true });
    const qaObserver = new MutationObserver(() => {
      if (_qaActiveInput && !document.body.contains(_qaActiveInput)) {
        removeQaTrigger();
      }
    });
    qaObserver.observe(document.body, { childList: true, subtree: true });
    ext.runtime.onMessage.addListener(
      (msg, _sender, sendResponse) => {
        if (msg.type === "EXTRACT_PAGE") {
          sendResponse(extractMainContent());
          return false;
        }
        if (msg.type === "GET_DOM") {
          sendResponse({ ok: true, dom: serializeDOM(), url: location.href, title: document.title });
          return false;
        }
        if (msg.type === "EXEC_ACTION") {
          sendResponse(execAction(msg.action));
          return false;
        }
        if (msg.type === "SHOW_OCR_OVERLAY") {
          showOcrOverlay(msg.dataUrl, sendResponse);
          return true;
        }
        return void 0;
      }
    );
  })();
})();
