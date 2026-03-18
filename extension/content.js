(function() {
  "use strict";
  const ext = typeof browser !== "undefined" ? browser : chrome;
  function safeSend(msg, cb) {
    var _a, _b, _c, _d, _e, _f;
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
        (_f = document.getElementById("aura-highlight-host")) == null ? void 0 : _f.remove();
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
    const hlHost = document.createElement("div");
    hlHost.id = "aura-highlight-host";
    Object.assign(hlHost.style, {
      position: "fixed",
      top: "0",
      left: "0",
      zIndex: "2147483646",
      pointerEvents: "none"
    });
    document.documentElement.appendChild(hlHost);
    const hlShadow = hlHost.attachShadow({ mode: "closed" });
    const hlStyle = document.createElement("style");
    hlStyle.textContent = `
    @keyframes hl-tooltip-in {
      from { opacity: 0; transform: translateY(4px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    .hl-tooltip {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.3);
      border-radius: 8px;
      padding: 5px 10px;
      display: flex;
      align-items: center;
      gap: 8px;
      pointer-events: auto;
      animation: hl-tooltip-in 0.15s ease forwards;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      z-index: 2147483647;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    }
    .hl-tooltip-text {
      color: rgba(226, 232, 240, 0.9);
      font-size: 11px;
      font-weight: 500;
      white-space: nowrap;
    }
    .hl-tooltip-delete {
      width: 18px; height: 18px; border-radius: 4px;
      background: transparent; border: none;
      color: rgba(226, 232, 240, 0.5);
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      padding: 0; transition: background 0.12s, color 0.12s;
    }
    .hl-tooltip-delete:hover {
      background: rgba(239, 68, 68, 0.25);
      color: rgba(239, 68, 68, 1);
    }
  `;
    hlShadow.appendChild(hlStyle);
    const hlContainer = document.createElement("div");
    hlShadow.appendChild(hlContainer);
    const pageHlStyle = document.createElement("style");
    pageHlStyle.textContent = `
    mark[data-aura-hl] {
      background: rgba(124, 58, 237, 0.15);
      border-bottom: 2px solid rgba(124, 58, 237, 0.5);
      border-radius: 2px;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    mark[data-aura-hl]:hover {
      background: rgba(124, 58, 237, 0.28);
    }
    mark[data-aura-hl].aura-hl-stale {
      background: rgba(124, 58, 237, 0.08);
      border-bottom: 2px dashed rgba(124, 58, 237, 0.35);
    }
    mark[data-aura-hl].aura-hl-flash {
      background: rgba(124, 58, 237, 0.45) !important;
      transition: background 0.3s ease;
    }
  `;
    document.head.appendChild(pageHlStyle);
    let _hlTooltipEl = null;
    let _hlTooltipTimer = null;
    function removeHlTooltip() {
      if (_hlTooltipTimer) {
        clearTimeout(_hlTooltipTimer);
        _hlTooltipTimer = null;
      }
      if (_hlTooltipEl) {
        _hlTooltipEl.remove();
        _hlTooltipEl = null;
      }
    }
    function showHlTooltip(mark, highlightId) {
      removeHlTooltip();
      const rect = mark.getBoundingClientRect();
      _hlTooltipEl = document.createElement("div");
      _hlTooltipEl.className = "hl-tooltip";
      const label = document.createElement("span");
      label.className = "hl-tooltip-text";
      label.textContent = "Saved to AURA";
      _hlTooltipEl.appendChild(label);
      const delBtn = document.createElement("button");
      delBtn.className = "hl-tooltip-delete";
      delBtn.title = "Remove highlight";
      delBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteHighlightFromPage(highlightId);
        removeHlTooltip();
      });
      _hlTooltipEl.appendChild(delBtn);
      _hlTooltipEl.style.top = `${Math.round(rect.top - 34)}px`;
      _hlTooltipEl.style.left = `${Math.round(rect.left + rect.width / 2 - 60)}px`;
      hlContainer.appendChild(_hlTooltipEl);
    }
    function getXPath(node) {
      if (node.nodeType === Node.DOCUMENT_NODE) return "/";
      const parts = [];
      let current = node;
      while (current && current !== document) {
        if (current.nodeType === Node.ELEMENT_NODE) {
          const el = current;
          let tag = el.tagName.toLowerCase();
          const parent = el.parentNode;
          if (parent) {
            const siblings = Array.from(parent.childNodes).filter(
              (n) => n.nodeType === Node.ELEMENT_NODE && n.tagName === el.tagName
            );
            if (siblings.length > 1) {
              const idx = siblings.indexOf(el) + 1;
              tag += `[${idx}]`;
            }
          }
          parts.unshift(tag);
        } else if (current.nodeType === Node.TEXT_NODE) {
          const parent = current.parentNode;
          if (parent) {
            const textNodes = Array.from(parent.childNodes).filter(
              (n) => n.nodeType === Node.TEXT_NODE
            );
            if (textNodes.length > 1) {
              const idx = textNodes.indexOf(current) + 1;
              parts.unshift(`text()[${idx}]`);
            } else {
              parts.unshift("text()");
            }
          }
        }
        current = current.parentNode;
      }
      return "/" + parts.join("/");
    }
    function getHighlightContext(range) {
      const container = range.commonAncestorContainer;
      const fullText = container.nodeType === Node.TEXT_NODE ? container.textContent || "" : container.textContent || "";
      const selectedText = range.toString();
      const idx = fullText.indexOf(selectedText);
      if (idx === -1) return "";
      const before = fullText.slice(Math.max(0, idx - 50), idx);
      const after = fullText.slice(idx + selectedText.length, idx + selectedText.length + 50);
      return before + "|||" + after;
    }
    function generateHighlightId() {
      return "hl_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
    }
    function wrapSelectionWithMark(highlightId) {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return null;
      const range = sel.getRangeAt(0);
      if (range.collapsed) return null;
      try {
        const mark = document.createElement("mark");
        mark.setAttribute("data-aura-hl", highlightId);
        range.surroundContents(mark);
        sel.removeAllRanges();
        attachMarkListeners(mark);
        return mark;
      } catch (_e) {
        try {
          const frag = range.cloneContents();
          const textContent = frag.textContent || "";
          if (!textContent.trim()) return null;
          range.deleteContents();
          const mark = document.createElement("mark");
          mark.setAttribute("data-aura-hl", highlightId);
          mark.textContent = textContent;
          range.insertNode(mark);
          sel.removeAllRanges();
          attachMarkListeners(mark);
          return mark;
        } catch (_e2) {
          return null;
        }
      }
    }
    function attachMarkListeners(mark) {
      const hlId = mark.getAttribute("data-aura-hl") || "";
      mark.addEventListener("mouseenter", () => showHlTooltip(mark, hlId));
      mark.addEventListener("mouseleave", () => {
        _hlTooltipTimer = setTimeout(removeHlTooltip, 300);
      });
    }
    function saveHighlight() {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return false;
      const range = sel.getRangeAt(0);
      const text = range.toString().trim();
      if (!text) return false;
      const highlightId = generateHighlightId();
      const xpath = getXPath(range.startContainer);
      const context = getHighlightContext(range);
      const mark = wrapSelectionWithMark(highlightId);
      if (!mark) return false;
      const highlight = {
        id: highlightId,
        url: window.location.href,
        text,
        xpath,
        context,
        timestamp: Date.now(),
        color: "purple",
        pageTitle: document.title
      };
      safeSend(
        { type: "SAVE_HIGHLIGHT", highlight },
        (response) => {
          if (response && response.ok) {
            showToast("Highlight saved to AURA");
          } else {
            showToast((response == null ? void 0 : response.error) || "Failed to save highlight", 3e3);
          }
        }
      );
      return true;
    }
    function deleteHighlightFromPage(highlightId) {
      const mark = document.querySelector(`mark[data-aura-hl="${highlightId}"]`);
      if (mark) {
        const parent = mark.parentNode;
        while (mark.firstChild) parent == null ? void 0 : parent.insertBefore(mark.firstChild, mark);
        mark.remove();
        parent == null ? void 0 : parent.normalize();
      }
      safeSend(
        { type: "DELETE_HIGHLIGHT", id: highlightId, url: window.location.href },
        (_response) => {
          showToast("Highlight removed");
        }
      );
    }
    function findTextNode(xpath, text, context) {
      try {
        const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
        const node = result.singleNodeValue;
        if (node && node.textContent && node.textContent.includes(text)) {
          const range = document.createRange();
          const idx = node.textContent.indexOf(text);
          if (idx >= 0) {
            range.setStart(node, idx);
            range.setEnd(node, idx + text.length);
            return range;
          }
        }
      } catch (_e) {
      }
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
      const [contextBefore, contextAfter] = context.split("|||");
      let bestNode = null;
      let bestOffset = -1;
      let bestScore = 0;
      while (walker.nextNode()) {
        const tNode = walker.currentNode;
        const nodeText = tNode.textContent || "";
        const idx = nodeText.indexOf(text);
        if (idx === -1) continue;
        let score = 1;
        if (contextBefore) {
          const before = nodeText.slice(Math.max(0, idx - 50), idx);
          if (before.includes(contextBefore.slice(-20))) score += 2;
        }
        if (contextAfter) {
          const after = nodeText.slice(idx + text.length, idx + text.length + 50);
          if (after.includes(contextAfter.slice(0, 20))) score += 2;
        }
        if (score > bestScore) {
          bestScore = score;
          bestNode = tNode;
          bestOffset = idx;
        }
      }
      if (bestNode && bestOffset >= 0) {
        const range = document.createRange();
        range.setStart(bestNode, bestOffset);
        range.setEnd(bestNode, bestOffset + text.length);
        return range;
      }
      return null;
    }
    function restoreHighlight(hl) {
      if (document.querySelector(`mark[data-aura-hl="${hl.id}"]`)) return true;
      const range = findTextNode(hl.xpath, hl.text, hl.context);
      if (!range) return false;
      try {
        const mark = document.createElement("mark");
        mark.setAttribute("data-aura-hl", hl.id);
        if (hl.stale) mark.classList.add("aura-hl-stale");
        range.surroundContents(mark);
        attachMarkListeners(mark);
        return true;
      } catch (_e) {
        try {
          const text = range.toString();
          range.deleteContents();
          const mark = document.createElement("mark");
          mark.setAttribute("data-aura-hl", hl.id);
          if (hl.stale) mark.classList.add("aura-hl-stale");
          mark.textContent = text;
          range.insertNode(mark);
          attachMarkListeners(mark);
          return true;
        } catch (_e2) {
          return false;
        }
      }
    }
    function restoreAllHighlights() {
      safeSend(
        { type: "GET_HIGHLIGHTS", url: window.location.href },
        (response) => {
          if (!response || !response.ok || !response.highlights) return;
          for (const hl of response.highlights) {
            const success = restoreHighlight(hl);
            if (!success) {
              hl.stale = true;
              restoreHighlight(hl);
            }
          }
        }
      );
    }
    function scrollToHighlight(highlightId) {
      const mark = document.querySelector(`mark[data-aura-hl="${highlightId}"]`);
      if (mark) {
        mark.scrollIntoView({ behavior: "smooth", block: "center" });
        mark.classList.add("aura-hl-flash");
        setTimeout(() => mark.classList.remove("aura-hl-flash"), 1500);
      }
    }
    setTimeout(restoreAllHighlights, 1500);
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
        saveHighlight();
        safeSend(
          { type: "SAVE_KNOWLEDGE", text, url, title },
          (_response) => {
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
    document.addEventListener("aura-yt-subtitles", (e) => {
      try {
        const d = e.detail;
        safeSend({
          type: "YT_SUBTITLES",
          videoId: d.videoId || "",
          lang: d.lang || "",
          segments: d.segments || []
        });
      } catch {
      }
    });
    document.addEventListener("aura-yt-metadata", (e) => {
      try {
        const d = e.detail;
        safeSend({
          type: "YT_METADATA",
          videoId: d.videoId || "",
          title: d.title || "",
          duration: d.duration || 0,
          description: d.description || "",
          channelName: d.channelName || "",
          chapters: d.chapters || [],
          captionTracks: d.captionTracks || []
        });
      } catch {
      }
    });
    const TRANSLATABLE_SELECTORS = "p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, figcaption";
    const AURA_TRANSLATE_ATTR = "data-aura-translated";
    const BATCH_SIZE = 10;
    const MAX_CONCURRENT = 10;
    let _translateMode = "bilingual";
    let _translateTargetLang = "English";
    let _translateActive = false;
    let _translateBadge = null;
    let _translatedElements = [];
    let _activeTranslations = 0;
    function getTranslatableElements() {
      const all = document.querySelectorAll(TRANSLATABLE_SELECTORS);
      const results = [];
      for (const el of all) {
        if (el.hasAttribute(AURA_TRANSLATE_ATTR)) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) continue;
        if (el.closest("#aura-host, #aura-dock-shadow, #aura-quick-action-host, .aura-translate-badge")) continue;
        if (el.tagName === "SPAN" && (el.textContent || "").trim().length <= 20) continue;
        const text = (el.textContent || "").trim();
        if (text.length < 5) continue;
        results.push(el);
      }
      return results;
    }
    function createTranslationElement(originalEl) {
      const translationDiv = document.createElement("div");
      translationDiv.className = "aura-page-translation";
      translationDiv.setAttribute("data-aura-translation", "true");
      Object.assign(translationDiv.style, {
        borderLeft: "2px solid rgba(124, 58, 237, 0.6)",
        background: "rgba(124, 58, 237, 0.05)",
        padding: "6px 10px",
        marginTop: "4px",
        marginBottom: "4px",
        fontSize: "0.95em",
        color: "inherit",
        opacity: "0",
        fontFamily: "inherit",
        lineHeight: "1.5",
        borderRadius: "0 4px 4px 0",
        transition: "opacity 0.3s ease",
        fontStyle: "italic"
      });
      translationDiv.textContent = "Translating...";
      translationDiv.style.color = "rgba(124, 58, 237, 0.5)";
      originalEl.setAttribute(AURA_TRANSLATE_ATTR, "true");
      originalEl.after(translationDiv);
      requestAnimationFrame(() => {
        translationDiv.style.opacity = "0.6";
      });
      return translationDiv;
    }
    function fadeInTranslation(el, text) {
      el.style.opacity = "0";
      el.textContent = text;
      el.style.fontStyle = "normal";
      el.style.color = "inherit";
      requestAnimationFrame(() => {
        el.style.opacity = "0.85";
      });
    }
    function translateBatchRequest(texts, lang) {
      return new Promise((resolve) => {
        try {
          ext.runtime.sendMessage(
            { type: "TRANSLATE_BATCH", texts, targetLang: lang },
            (response) => {
              if (ext.runtime.lastError) {
                resolve(texts.map(() => "[Translation failed]"));
                return;
              }
              if ((response == null ? void 0 : response.ok) && response.translations) {
                resolve(response.translations);
              } else {
                resolve(texts.map(() => (response == null ? void 0 : response.error) || "[Translation failed]"));
              }
            }
          );
        } catch {
          resolve(texts.map(() => "[Translation failed]"));
        }
      });
    }
    async function startPageTranslation(targetLang) {
      _translateTargetLang = targetLang;
      _translateActive = true;
      _translateMode = "bilingual";
      _translatedElements = [];
      _activeTranslations = 0;
      showTranslateBadge();
      const elements = getTranslatableElements();
      if (elements.length === 0) return;
      const pairs = [];
      for (const el of elements) {
        const text = (el.textContent || "").trim();
        if (!text) continue;
        const translationDiv = createTranslationElement(el);
        _translatedElements.push({ original: el, translation: translationDiv });
        pairs.push({ original: el, translation: translationDiv, text });
      }
      const batches = [];
      for (let i = 0; i < pairs.length; i += BATCH_SIZE) {
        batches.push(pairs.slice(i, i + BATCH_SIZE));
      }
      const processBatch = async (batch) => {
        while (_activeTranslations >= MAX_CONCURRENT) {
          await new Promise((r) => setTimeout(r, 100));
        }
        if (!_translateActive) return;
        _activeTranslations++;
        try {
          const texts = batch.map((p) => p.text);
          const translations = await translateBatchRequest(texts, _translateTargetLang);
          if (!_translateActive) return;
          batch.forEach((pair, idx) => {
            fadeInTranslation(pair.translation, translations[idx] || "[No translation]");
            if (_translateMode === "translated") {
              pair.original.style.display = "none";
            }
          });
        } finally {
          _activeTranslations--;
        }
      };
      const promises = batches.map((batch) => processBatch(batch));
      await Promise.all(promises);
    }
    function removePageTranslation() {
      _translateActive = false;
      for (const pair of _translatedElements) {
        pair.translation.remove();
        pair.original.removeAttribute(AURA_TRANSLATE_ATTR);
        pair.original.style.display = "";
      }
      _translatedElements = [];
      if (_translateBadge) {
        _translateBadge.remove();
        _translateBadge = null;
      }
    }
    function setTranslateMode(mode) {
      _translateMode = mode;
      for (const pair of _translatedElements) {
        if (mode === "translated") {
          pair.original.style.display = "none";
          pair.translation.style.marginTop = "0";
        } else {
          pair.original.style.display = "";
          pair.translation.style.marginTop = "4px";
        }
      }
      updateBadgeText();
    }
    function updateBadgeText() {
      if (!_translateBadge) return;
      const modeBtn = _translateBadge.querySelector("[data-badge-mode]");
      if (modeBtn) {
        modeBtn.textContent = _translateMode === "bilingual" ? "Bilingual" : "Translated";
      }
    }
    function showTranslateBadge() {
      if (_translateBadge) {
        _translateBadge.remove();
        _translateBadge = null;
      }
      _translateBadge = document.createElement("div");
      _translateBadge.className = "aura-translate-badge";
      Object.assign(_translateBadge.style, {
        position: "fixed",
        bottom: "20px",
        right: "20px",
        zIndex: "2147483646",
        background: "rgba(10, 8, 24, 0.92)",
        backdropFilter: "blur(20px) saturate(1.5)",
        WebkitBackdropFilter: "blur(20px) saturate(1.5)",
        border: "1px solid rgba(124, 58, 237, 0.35)",
        borderRadius: "12px",
        padding: "8px 12px",
        display: "flex",
        alignItems: "center",
        gap: "8px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05) inset",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif",
        fontSize: "12px",
        color: "rgba(226, 232, 240, 0.9)"
      });
      const dot = document.createElement("span");
      Object.assign(dot.style, {
        width: "6px",
        height: "6px",
        borderRadius: "50%",
        background: "#7c3aed",
        flexShrink: "0"
      });
      _translateBadge.appendChild(dot);
      const label = document.createElement("span");
      label.style.color = "rgba(160, 148, 210, 0.8)";
      label.textContent = "Translation active";
      _translateBadge.appendChild(label);
      const sep1 = document.createElement("span");
      Object.assign(sep1.style, { width: "1px", height: "14px", background: "rgba(255,255,255,0.1)", flexShrink: "0" });
      _translateBadge.appendChild(sep1);
      const langSpan = document.createElement("span");
      langSpan.setAttribute("data-badge-lang", "");
      langSpan.textContent = _translateTargetLang;
      langSpan.style.color = "rgba(124, 58, 237, 0.9)";
      langSpan.style.fontWeight = "600";
      _translateBadge.appendChild(langSpan);
      const badgeBtnBase = {
        background: "rgba(124, 58, 237, 0.15)",
        border: "1px solid rgba(124, 58, 237, 0.3)",
        borderRadius: "6px",
        color: "rgba(226, 232, 240, 0.9)",
        padding: "3px 8px",
        cursor: "pointer",
        fontSize: "11px",
        fontFamily: "inherit",
        transition: "background 0.15s, border-color 0.15s"
      };
      const modeBtn = document.createElement("button");
      modeBtn.setAttribute("data-badge-mode", "");
      modeBtn.textContent = "Bilingual";
      Object.assign(modeBtn.style, badgeBtnBase);
      modeBtn.addEventListener("mouseenter", () => {
        modeBtn.style.background = "rgba(124, 58, 237, 0.3)";
      });
      modeBtn.addEventListener("mouseleave", () => {
        modeBtn.style.background = "rgba(124, 58, 237, 0.15)";
      });
      modeBtn.addEventListener("click", () => {
        setTranslateMode(_translateMode === "bilingual" ? "translated" : "bilingual");
      });
      _translateBadge.appendChild(modeBtn);
      const removeBtn = document.createElement("button");
      removeBtn.textContent = "✕";
      Object.assign(removeBtn.style, { ...badgeBtnBase, padding: "3px 6px", color: "rgba(226, 232, 240, 0.6)" });
      removeBtn.title = "Remove translation";
      removeBtn.addEventListener("mouseenter", () => {
        removeBtn.style.background = "rgba(239, 68, 68, 0.2)";
        removeBtn.style.borderColor = "rgba(239, 68, 68, 0.4)";
        removeBtn.style.color = "rgba(239, 68, 68, 0.9)";
      });
      removeBtn.addEventListener("mouseleave", () => {
        removeBtn.style.background = "rgba(124, 58, 237, 0.15)";
        removeBtn.style.borderColor = "rgba(124, 58, 237, 0.3)";
        removeBtn.style.color = "rgba(226, 232, 240, 0.6)";
      });
      removeBtn.addEventListener("click", () => {
        removePageTranslation();
      });
      _translateBadge.appendChild(removeBtn);
      document.body.appendChild(_translateBadge);
    }
    const GMAIL_HOST = "mail.google.com";
    const _gmailTrackedComposes = /* @__PURE__ */ new Map();
    function isGmailPage() {
      return window.location.hostname === GMAIL_HOST;
    }
    function extractGmailThreadText() {
      const bodies = document.querySelectorAll(".a3s.aiL");
      if (bodies.length === 0) return "";
      const parts = [];
      bodies.forEach((body) => {
        var _a;
        const text = (_a = body.innerText) == null ? void 0 : _a.trim();
        if (text) parts.push(text);
      });
      return parts.join("\n\n---\n\n").slice(0, 2e4);
    }
    function getComposeBody(composeEl) {
      return composeEl.querySelector(
        'div[aria-label="Message Body"], div[aria-label="Nachrichtentext"], div[aria-label="Corps du message"], div[g_editable="true"][contenteditable="true"], div.editable[contenteditable="true"], div[contenteditable="true"][role="textbox"]'
      );
    }
    function getComposeText(composeEl) {
      var _a;
      const body = getComposeBody(composeEl);
      if (!body) return "";
      return ((_a = body.innerText) == null ? void 0 : _a.trim()) || "";
    }
    function setComposeText(composeEl, text) {
      const body = getComposeBody(composeEl);
      if (!body) return;
      body.focus();
      const sel = window.getSelection();
      if (sel) {
        const range = document.createRange();
        range.selectNodeContents(body);
        sel.removeAllRanges();
        sel.addRange(range);
      }
      const success = document.execCommand("insertText", false, text);
      if (!success) {
        body.innerHTML = text.split("\n").map(
          (line) => `<div>${line || "<br>"}</div>`
        ).join("");
      }
      body.dispatchEvent(new Event("input", { bubbles: true }));
      body.dispatchEvent(new Event("change", { bubbles: true }));
    }
    function injectGmailAiButton(composeEl) {
      if (_gmailTrackedComposes.has(composeEl)) return;
      const sendBtn = composeEl.querySelector(
        'div[aria-label*="Send"], div[data-tooltip*="Send"], div[aria-label*="Enviar"], div[aria-label*="Envoyer"], div[aria-label*="Senden"], div[aria-label*="Отправить"]'
      );
      const toolbarRow = composeEl.querySelector(
        ".btC, .bAK, tr.btC, .IZ"
      );
      const insertTarget = (sendBtn == null ? void 0 : sendBtn.parentElement) || toolbarRow;
      if (!insertTarget) return;
      const buttonHost = document.createElement("div");
      buttonHost.className = "aura-gmail-ai-host";
      Object.assign(buttonHost.style, {
        display: "inline-flex",
        alignItems: "center",
        verticalAlign: "middle",
        marginLeft: "8px",
        position: "relative",
        zIndex: "1"
      });
      const gmailShadow = buttonHost.attachShadow({ mode: "closed" });
      const gmailStyle = document.createElement("style");
      gmailStyle.textContent = `
      @keyframes gmail-aura-in {
        from { opacity: 0; transform: scale(0.85); }
        to   { opacity: 1; transform: scale(1); }
      }
      @keyframes gmail-aura-spin {
        to { transform: rotate(360deg); }
      }
      @keyframes gmail-aura-menu-in {
        from { opacity: 0; transform: translateY(4px) scale(0.95); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
      }

      :host {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      }

      .gmail-ai-btn {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 18px;
        border: 1px solid rgba(124, 58, 237, 0.35);
        background: rgba(124, 58, 237, 0.08);
        color: #7c3aed;
        font-size: 12px;
        font-weight: 600;
        font-family: inherit;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.15s ease;
        animation: gmail-aura-in 0.25s ease forwards;
        line-height: 1.4;
        letter-spacing: 0.01em;
      }
      .gmail-ai-btn:hover {
        background: rgba(124, 58, 237, 0.15);
        border-color: rgba(124, 58, 237, 0.5);
        box-shadow: 0 0 12px rgba(124, 58, 237, 0.15);
      }
      .gmail-ai-btn:active {
        transform: scale(0.97);
      }
      .gmail-ai-btn .sparkle {
        font-size: 13px;
        line-height: 1;
      }

      .gmail-ai-menu {
        position: absolute;
        bottom: calc(100% + 6px);
        left: 0;
        background: rgba(10, 8, 24, 0.94);
        backdrop-filter: blur(20px) saturate(1.5);
        -webkit-backdrop-filter: blur(20px) saturate(1.5);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 10px;
        padding: 4px;
        min-width: 180px;
        box-shadow: 0 -8px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.05) inset;
        animation: gmail-aura-menu-in 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        z-index: 10000;
      }

      .gmail-ai-menu-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 7px;
        background: transparent;
        border: none;
        color: rgba(226, 232, 240, 0.92);
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        cursor: pointer;
        white-space: nowrap;
        width: 100%;
        text-align: left;
        transition: background 0.12s, color 0.12s;
        line-height: 1;
        box-sizing: border-box;
      }
      .gmail-ai-menu-item:hover {
        background: rgba(124, 58, 237, 0.25);
        color: #fff;
      }
      .gmail-ai-menu-item:active {
        background: rgba(124, 58, 237, 0.4);
      }
      .gmail-ai-menu-item .item-icon {
        font-size: 14px;
        width: 18px;
        text-align: center;
        flex-shrink: 0;
      }
      .gmail-ai-menu-item.loading {
        opacity: 0.55;
        pointer-events: none;
      }

      .gmail-ai-sep {
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 3px 8px;
      }

      .gmail-ai-sub {
        padding: 2px 0 2px 4px;
      }
      .gmail-ai-sub .gmail-ai-menu-item {
        font-size: 11.5px;
        padding: 6px 12px 6px 26px;
      }

      .gmail-ai-spinner {
        display: inline-block;
        width: 14px;
        height: 14px;
        border: 2px solid rgba(124, 58, 237, 0.3);
        border-top-color: rgba(160, 148, 210, 0.9);
        border-radius: 50%;
        animation: gmail-aura-spin 0.6s linear infinite;
        flex-shrink: 0;
      }

      .gmail-ai-toast {
        position: absolute;
        bottom: calc(100% + 6px);
        left: 50%;
        transform: translateX(-50%);
        background: rgba(5, 150, 105, 0.92);
        backdrop-filter: blur(12px);
        color: #fff;
        font-size: 11.5px;
        font-weight: 500;
        font-family: inherit;
        padding: 5px 12px;
        border-radius: 6px;
        white-space: nowrap;
        pointer-events: none;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        animation: gmail-aura-menu-in 0.15s ease forwards;
        z-index: 10001;
      }
    `;
      gmailShadow.appendChild(gmailStyle);
      const gmailContainer = document.createElement("div");
      gmailContainer.style.position = "relative";
      gmailContainer.style.display = "inline-flex";
      gmailContainer.style.alignItems = "center";
      gmailShadow.appendChild(gmailContainer);
      const aiBtn = document.createElement("button");
      aiBtn.className = "gmail-ai-btn";
      aiBtn.innerHTML = `<span class="sparkle">✦</span> AI`;
      gmailContainer.appendChild(aiBtn);
      let gmailMenuEl = null;
      let gmailTranslateSubEl = null;
      let gmailToastEl = null;
      let gmailToastTimer = null;
      function showGmailToast(msg, duration = 2500) {
        if (gmailToastEl) gmailToastEl.remove();
        if (gmailToastTimer) clearTimeout(gmailToastTimer);
        gmailToastEl = document.createElement("div");
        gmailToastEl.className = "gmail-ai-toast";
        gmailToastEl.textContent = msg;
        gmailContainer.appendChild(gmailToastEl);
        gmailToastTimer = setTimeout(() => {
          if (gmailToastEl) {
            gmailToastEl.remove();
            gmailToastEl = null;
          }
          gmailToastTimer = null;
        }, duration);
      }
      function removeGmailMenu() {
        if (gmailMenuEl) {
          gmailMenuEl.remove();
          gmailMenuEl = null;
        }
        gmailTranslateSubEl = null;
      }
      function setGmailMenuLoading(loading) {
        if (!gmailMenuEl) return;
        gmailMenuEl.querySelectorAll(".gmail-ai-menu-item").forEach((item) => {
          item.classList.add("loading");
        });
      }
      function executeGmailAction(action, language) {
        const composeText = getComposeText(composeEl);
        const threadText = extractGmailThreadText();
        if (action === "draft_reply" && !composeText && !threadText) {
          showGmailToast("No email thread found", 3e3);
          removeGmailMenu();
          return;
        }
        if (action !== "draft_reply" && !composeText) {
          showGmailToast("Compose body is empty", 3e3);
          removeGmailMenu();
          return;
        }
        setGmailMenuLoading();
        const outMsg = {
          type: "QUICK_ACTION",
          action,
          text: composeText || "(empty — draft a new reply)",
          ...action === "draft_reply" ? { threadContext: threadText } : {},
          ...language ? { language } : {}
        };
        safeSend(outMsg, (response) => {
          if (response && response.ok && response.result) {
            setComposeText(composeEl, response.result);
            showGmailToast("Updated by AURA");
          } else {
            showGmailToast((response == null ? void 0 : response.error) || "Action failed", 3e3);
          }
          removeGmailMenu();
        });
      }
      const GMAIL_ACTIONS = [
        { icon: "✍️", label: "Draft reply", action: "draft_reply" },
        { icon: "✨", label: "Improve", action: "improve" },
        { icon: "🏢", label: "Make formal", action: "make_formal", separator: true },
        { icon: "😊", label: "Make casual", action: "make_casual" },
        { icon: "✂️", label: "Shorten", action: "shorten" },
        { icon: "🌐", label: "Translate to...", action: "translate_menu", separator: true }
      ];
      const GMAIL_TRANSLATE_LANGS = ["English", "Spanish", "French", "German", "Chinese"];
      function showGmailMenu() {
        removeGmailMenu();
        gmailMenuEl = document.createElement("div");
        gmailMenuEl.className = "gmail-ai-menu";
        GMAIL_ACTIONS.forEach((item) => {
          if (item.separator) {
            const sep = document.createElement("div");
            sep.className = "gmail-ai-sep";
            gmailMenuEl.appendChild(sep);
          }
          const btn = document.createElement("button");
          btn.className = "gmail-ai-menu-item";
          btn.innerHTML = `<span class="item-icon">${item.icon}</span><span>${item.label}</span>`;
          btn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (item.action === "translate_menu") {
              toggleGmailTranslateSub(btn);
            } else {
              executeGmailAction(item.action);
            }
          });
          gmailMenuEl.appendChild(btn);
        });
        gmailContainer.appendChild(gmailMenuEl);
      }
      function toggleGmailTranslateSub(anchor) {
        if (gmailTranslateSubEl) {
          gmailTranslateSubEl.remove();
          gmailTranslateSubEl = null;
          return;
        }
        gmailTranslateSubEl = document.createElement("div");
        gmailTranslateSubEl.className = "gmail-ai-sub";
        GMAIL_TRANSLATE_LANGS.forEach((lang) => {
          const item = document.createElement("button");
          item.className = "gmail-ai-menu-item";
          item.textContent = lang;
          item.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            executeGmailAction("gmail_translate", lang);
          });
          gmailTranslateSubEl.appendChild(item);
        });
        if (gmailMenuEl && anchor.parentNode === gmailMenuEl) {
          anchor.after(gmailTranslateSubEl);
        }
      }
      aiBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (gmailMenuEl) {
          removeGmailMenu();
        } else {
          showGmailMenu();
        }
      });
      const outsideClickHandler = (e) => {
        if (!gmailMenuEl) return;
        const path = e.composedPath();
        if (!path.includes(buttonHost)) {
          removeGmailMenu();
        }
      };
      document.addEventListener("mousedown", outsideClickHandler, true);
      if (sendBtn == null ? void 0 : sendBtn.parentElement) {
        sendBtn.parentElement.insertBefore(buttonHost, sendBtn.nextSibling);
      } else if (toolbarRow) {
        toolbarRow.appendChild(buttonHost);
      }
      const composeObserver = new MutationObserver(() => {
        if (!document.body.contains(composeEl)) {
          composeObserver.disconnect();
          document.removeEventListener("mousedown", outsideClickHandler, true);
          buttonHost.remove();
          _gmailTrackedComposes.delete(composeEl);
        }
      });
      composeObserver.observe(document.body, { childList: true, subtree: true });
      _gmailTrackedComposes.set(composeEl, {
        composeEl,
        buttonHost,
        shadow: gmailShadow,
        observer: composeObserver,
        outsideHandler: outsideClickHandler
      });
    }
    function scanGmailComposeWindows() {
      const composeSelectors = [
        'div[role="dialog"]',
        // Popup compose / reply
        "div.ip.iq",
        // Inline reply
        "div.nH.nn"
        // Another compose variant
      ];
      composeSelectors.forEach((sel) => {
        document.querySelectorAll(sel).forEach((el) => {
          const body = getComposeBody(el);
          if (!body) return;
          if (_gmailTrackedComposes.has(el)) return;
          injectGmailAiButton(el);
        });
      });
    }
    function initGmailIntegration() {
      if (!isGmailPage()) return;
      scanGmailComposeWindows();
      const gmailObserver = new MutationObserver((mutations) => {
        var _a, _b, _c;
        let shouldScan = false;
        for (const mutation of mutations) {
          if (mutation.addedNodes.length > 0) {
            for (const node of mutation.addedNodes) {
              if (node.nodeType !== Node.ELEMENT_NODE) continue;
              const el = node;
              if (((_a = el.matches) == null ? void 0 : _a.call(el, 'div[role="dialog"]')) || ((_b = el.querySelector) == null ? void 0 : _b.call(el, 'div[role="dialog"]')) || ((_c = el.querySelector) == null ? void 0 : _c.call(el, 'div[contenteditable="true"]'))) {
                shouldScan = true;
                break;
              }
            }
          }
          if (shouldScan) break;
        }
        if (shouldScan) {
          setTimeout(scanGmailComposeWindows, 300);
        }
      });
      const gmailRoot = document.querySelector('div[role="main"]') || document.body;
      gmailObserver.observe(gmailRoot, { childList: true, subtree: true });
      let _gmailScanInterval = null;
      _gmailScanInterval = setInterval(() => {
        if (!isGmailPage()) {
          if (_gmailScanInterval) clearInterval(_gmailScanInterval);
          return;
        }
        scanGmailComposeWindows();
      }, 3e3);
    }
    initGmailIntegration();
    const _linkPreviewCache = /* @__PURE__ */ new Map();
    const LP_CACHE_MAX = 50;
    function lpCacheSet(cacheUrl, cacheData) {
      if (_linkPreviewCache.size >= LP_CACHE_MAX) {
        const oldest = _linkPreviewCache.keys().next().value;
        if (oldest) _linkPreviewCache.delete(oldest);
      }
      _linkPreviewCache.set(cacheUrl, cacheData);
    }
    function lpCacheGet(cacheUrl) {
      const d = _linkPreviewCache.get(cacheUrl);
      if (d) {
        _linkPreviewCache.delete(cacheUrl);
        _linkPreviewCache.set(cacheUrl, d);
      }
      return d;
    }
    const lpHost = document.createElement("div");
    lpHost.id = "aura-link-preview-host";
    Object.assign(lpHost.style, { position: "fixed", top: "0", left: "0", zIndex: "2147483646", pointerEvents: "none" });
    document.documentElement.appendChild(lpHost);
    const lpShadow = lpHost.attachShadow({ mode: "closed" });
    const lpCss = document.createElement("style");
    lpCss.textContent = [
      "@keyframes lp-in { from { opacity:0; transform:translateY(4px) scale(0.96); } to { opacity:1; transform:translateY(0) scale(1); } }",
      "@keyframes lp-shimmer { 0% { background-position:-200px 0; } 100% { background-position:200px 0; } }",
      '.lp-popup { position:fixed; width:320px; max-height:280px; background:rgba(10,8,24,0.92); backdrop-filter:blur(20px) saturate(1.5); -webkit-backdrop-filter:blur(20px) saturate(1.5); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:14px 16px 12px; pointer-events:auto; animation:lp-in 0.2s cubic-bezier(0.16,1,0.3,1) forwards; box-shadow:0 8px 32px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.05) inset; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Inter",system-ui,sans-serif; box-sizing:border-box; overflow:hidden; display:flex; flex-direction:column; gap:8px; }',
      ".lp-domain { display:inline-block; background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.25); border-radius:4px; padding:2px 7px; font-size:10.5px; font-weight:600; color:rgba(160,148,210,0.9); letter-spacing:0.3px; max-width:fit-content; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }",
      ".lp-title { font-size:13px; font-weight:600; color:rgba(226,232,240,0.95); line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin:0; }",
      ".lp-description { font-size:12px; font-weight:400; color:rgba(226,232,240,0.65); line-height:1.45; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; margin:0; }",
      ".lp-shimmer { height:12px; border-radius:4px; background:linear-gradient(90deg,rgba(124,58,237,0.08) 25%,rgba(124,58,237,0.18) 50%,rgba(124,58,237,0.08) 75%); background-size:400px 100%; animation:lp-shimmer 1.5s infinite linear; }",
      ".lp-shimmer.short { width:60%; } .lp-shimmer.long { width:90%; } .lp-shimmer+.lp-shimmer { margin-top:6px; }",
      ".lp-loading-label { font-size:11px; color:rgba(160,148,210,0.5); margin-bottom:4px; }",
      ".lp-actions { display:flex; gap:6px; margin-top:4px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06); }",
      ".lp-btn { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.2); border-radius:6px; padding:4px 10px; font-size:11px; font-weight:500; font-family:inherit; color:rgba(200,180,255,0.9); cursor:pointer; transition:background 0.15s,border-color 0.15s,color 0.15s; white-space:nowrap; }",
      ".lp-btn:hover { background:rgba(124,58,237,0.25); border-color:rgba(124,58,237,0.4); color:#fff; }",
      ".lp-btn:active { background:rgba(124,58,237,0.35); }"
    ].join("\n");
    lpShadow.appendChild(lpCss);
    const lpBox = document.createElement("div");
    lpShadow.appendChild(lpBox);
    let _lpPopup = null;
    let _lpHoverTmr = null;
    let _lpDismissTmr = null;
    let _lpCurLink = null;
    let _lpMouseIsDown = false;
    document.addEventListener("mousedown", () => {
      _lpMouseIsDown = true;
    }, true);
    document.addEventListener("mouseup", () => {
      _lpMouseIsDown = false;
    }, true);
    function lpIsExternal(a) {
      try {
        return new URL(a.href, location.href).hostname !== location.hostname;
      } catch {
        return false;
      }
    }
    function lpShouldShow(a) {
      const h = a.href || "";
      if (!h.startsWith("http://") && !h.startsWith("https://")) return false;
      try {
        const u = new URL(h, location.href);
        if (u.hostname === location.hostname && u.pathname === location.pathname && u.hash) return false;
      } catch {
        return false;
      }
      if ((a.textContent || "").trim().length < 10) return false;
      return lpIsExternal(a);
    }
    function lpRemove() {
      if (_lpPopup) {
        _lpPopup.remove();
        _lpPopup = null;
      }
      _lpCurLink = null;
    }
    function lpCancelTimers() {
      if (_lpHoverTmr) {
        clearTimeout(_lpHoverTmr);
        _lpHoverTmr = null;
      }
      if (_lpDismissTmr) {
        clearTimeout(_lpDismissTmr);
        _lpDismissTmr = null;
      }
    }
    function lpStartDismiss() {
      if (_lpDismissTmr) clearTimeout(_lpDismissTmr);
      _lpDismissTmr = setTimeout(() => {
        lpRemove();
        _lpDismissTmr = null;
      }, 300);
    }
    function lpCancelDismiss() {
      if (_lpDismissTmr) {
        clearTimeout(_lpDismissTmr);
        _lpDismissTmr = null;
      }
    }
    function lpPosition(a) {
      if (!_lpPopup) return;
      const r = a.getBoundingClientRect();
      _lpPopup.style.visibility = "hidden";
      _lpPopup.style.display = "flex";
      const ph = _lpPopup.offsetHeight || 180;
      _lpPopup.style.visibility = "";
      let l = r.left + r.width / 2 - 160;
      if (l < 8) l = 8;
      if (l + 320 > window.innerWidth - 8) l = window.innerWidth - 328;
      let t = r.bottom + 8;
      if (t + ph > window.innerHeight - 8) {
        t = r.top - ph - 8;
        if (t < 8) t = 8;
      }
      _lpPopup.style.top = Math.round(t) + "px";
      _lpPopup.style.left = Math.round(l) + "px";
    }
    function lpBuild(a, href) {
      lpRemove();
      _lpCurLink = a;
      let dom = "";
      try {
        dom = new URL(href).hostname;
      } catch {
        dom = href;
      }
      const txt = (a.textContent || "").trim();
      _lpPopup = document.createElement("div");
      _lpPopup.className = "lp-popup";
      const dEl = document.createElement("div");
      dEl.className = "lp-domain";
      dEl.textContent = dom;
      _lpPopup.appendChild(dEl);
      const tEl = document.createElement("div");
      tEl.className = "lp-title";
      tEl.textContent = txt;
      _lpPopup.appendChild(tEl);
      const lw = document.createElement("div");
      const ll = document.createElement("div");
      ll.className = "lp-loading-label";
      ll.textContent = "Loading preview…";
      const s1 = document.createElement("div");
      s1.className = "lp-shimmer long";
      const s2 = document.createElement("div");
      s2.className = "lp-shimmer short";
      lw.appendChild(ll);
      lw.appendChild(s1);
      lw.appendChild(s2);
      _lpPopup.appendChild(lw);
      const acts = document.createElement("div");
      acts.className = "lp-actions";
      const ob = document.createElement("button");
      ob.className = "lp-btn";
      ob.textContent = "Open";
      ob.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        window.open(href, "_blank", "noopener");
        lpRemove();
      });
      const sb = document.createElement("button");
      sb.className = "lp-btn";
      sb.textContent = "Summarize in AURA";
      sb.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        safeSend({ type: "OPEN_WITH_TEXT", action: "summarize", text: "Summarize this page: " + href, url: href, title: txt });
        lpRemove();
      });
      acts.appendChild(ob);
      acts.appendChild(sb);
      _lpPopup.appendChild(acts);
      _lpPopup.addEventListener("mouseenter", lpCancelDismiss);
      _lpPopup.addEventListener("mouseleave", lpStartDismiss);
      lpBox.appendChild(_lpPopup);
      lpPosition(a);
      const c = lpCacheGet(href);
      if (c) {
        lpUpdate(lw, tEl, c);
        return;
      }
      try {
        ext.runtime.sendMessage({ type: "LINK_PREVIEW", url: href }, (rsp) => {
          if (ext.runtime.lastError || !rsp) return;
          if (!_lpPopup || _lpCurLink !== a) return;
          const pd = { title: rsp.title || txt, description: rsp.description || "", domain: rsp.domain || dom };
          lpCacheSet(href, pd);
          lpUpdate(lw, tEl, pd);
        });
      } catch {
      }
    }
    function lpUpdate(lw, te, d) {
      lw.innerHTML = "";
      lw.style.display = "none";
      if (d.title && d.title !== te.textContent) te.textContent = d.title;
      if (d.description) {
        const de = document.createElement("div");
        de.className = "lp-description";
        de.textContent = d.description;
        te.after(de);
      }
      if (_lpPopup && _lpCurLink) lpPosition(_lpCurLink);
    }
    document.addEventListener("mouseover", (me) => {
      if (_lpMouseIsDown) return;
      const a = me.target.closest("a");
      if (!a || !lpShouldShow(a)) return;
      if (_lpCurLink === a && _lpPopup) {
        lpCancelDismiss();
        return;
      }
      lpCancelTimers();
      _lpHoverTmr = setTimeout(() => {
        if (_lpMouseIsDown) return;
        lpBuild(a, a.href);
        _lpHoverTmr = null;
      }, 800);
    }, true);
    document.addEventListener("mouseout", (me) => {
      const a = me.target.closest("a");
      if (a && a === _lpCurLink) {
        const rel = me.relatedTarget;
        if (rel && lpHost.contains(rel)) return;
        lpStartDismiss();
      }
      if (a && _lpHoverTmr) lpCancelTimers();
    }, true);
    window.addEventListener("scroll", () => {
      if (_lpPopup && _lpCurLink) {
        const r = _lpCurLink.getBoundingClientRect();
        if (r.bottom < 0 || r.top > window.innerHeight) {
          lpCancelTimers();
          lpRemove();
        } else {
          lpPosition(_lpCurLink);
        }
      }
    }, { passive: true });
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
        if (msg.type === "PAGE_TRANSLATE") {
          if (_translateActive) removePageTranslation();
          startPageTranslation(msg.targetLang).then(() => {
            sendResponse({ ok: true });
          }).catch((err) => {
            sendResponse({ ok: false, error: err.message });
          });
          return true;
        }
        if (msg.type === "TRANSLATE_TOGGLE_MODE") {
          setTranslateMode(msg.mode);
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "TRANSLATE_REMOVE") {
          removePageTranslation();
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "TRANSLATE_CHANGE_LANG") {
          if (_translateActive) {
            removePageTranslation();
            startPageTranslation(msg.targetLang).then(() => {
              sendResponse({ ok: true });
            }).catch((err) => {
              sendResponse({ ok: false, error: err.message });
            });
            return true;
          }
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "SCROLL_TO_HIGHLIGHT") {
          scrollToHighlight(msg.id);
          sendResponse({ ok: true });
          return false;
        }
        return void 0;
      }
    );
  })();
})();
