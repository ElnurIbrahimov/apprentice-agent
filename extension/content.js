(function() {
  "use strict";
  const ext = typeof browser !== "undefined" ? browser : chrome;
  function safeSend(msg, cb) {
    var _a, _b, _c, _d, _e, _f, _g, _h;
    try {
      if (cb) {
        ext.runtime.sendMessage(msg, cb);
      } else {
        ext.runtime.sendMessage(msg);
      }
    } catch (e) {
      const err = e;
      if (((_a = err == null ? void 0 : err.message) == null ? void 0 : _a.includes("Extension context invalidated")) || ((_b = err == null ? void 0 : err.message) == null ? void 0 : _b.includes("context invalidated"))) {
        (_c = document.getElementById("aura-dock-shadow")) == null ? void 0 : _c.remove();
        (_d = document.getElementById("aura-host")) == null ? void 0 : _d.remove();
        (_e = document.getElementById("aura-quick-action-host")) == null ? void 0 : _e.remove();
        (_f = document.getElementById("aura-highlight-host")) == null ? void 0 : _f.remove();
        (_g = document.getElementById("aura-img-toolbar-host")) == null ? void 0 : _g.remove();
        (_h = document.getElementById("aura-capture-host")) == null ? void 0 : _h.remove();
        window.__auraToolbarMounted = false;
      }
    }
  }
  (function() {
    var _a;
    if (window.__auraToolbarMounted) return;
    window.__auraToolbarMounted = true;
    const _prevDock = document.getElementById("aura-dock-shadow");
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
    function isDarkMode() {
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) return true;
      const bg = window.getComputedStyle(document.documentElement).backgroundColor;
      if (!bg || bg === "rgba(0, 0, 0, 0)") {
        const bodyBg = document.body ? window.getComputedStyle(document.body).backgroundColor : "";
        if (bodyBg && bodyBg !== "rgba(0, 0, 0, 0)") {
          const m2 = bodyBg.match(/\d+/g);
          if (m2) {
            const lum = (parseInt(m2[0]) * 299 + parseInt(m2[1]) * 587 + parseInt(m2[2]) * 114) / 1e3;
            return lum < 128;
          }
        }
        return false;
      }
      const m = bg.match(/\d+/g);
      if (m) {
        const lum = (parseInt(m[0]) * 299 + parseInt(m[1]) * 587 + parseInt(m[2]) * 114) / 1e3;
        return lum < 128;
      }
      return false;
    }
    function themeClass() {
      return isDarkMode() ? "dark" : "light";
    }
    const style = document.createElement("style");
    style.textContent = `
    @keyframes aura-bubble-in { from { opacity: 0; } to { opacity: 1; } }
    @keyframes aura-bubble-out { from { opacity: 1; } to { opacity: 0; } }

    #bubble {
      display: none; position: fixed; padding: 2px; gap: 4px;
      border-radius: 8px; pointer-events: auto; z-index: 2147483647;
      align-items: center;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    }
    #bubble.light {
      background: #ffffff; border: 1px solid rgba(0,0,0,0.08);
      box-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.08), 0 8px 24px rgba(0,0,0,0.06);
    }
    #bubble.dark {
      background: #1e1e2e; border: 1px solid rgba(255,255,255,0.1);
      box-shadow: 0 1px 2px rgba(0,0,0,0.2), 0 4px 12px rgba(0,0,0,0.25), 0 8px 24px rgba(0,0,0,0.3);
    }
    #bubble.visible { display: flex; animation: aura-bubble-in 0.3s ease-in-out forwards; }
    #bubble.hiding { display: flex; animation: aura-bubble-out 0.15s ease-in-out forwards; }

    .bubble-btn {
      width: 26px; height: 26px; border: none; border-radius: 6px;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      padding: 0; transition: background 0.12s ease, transform 0.12s ease; flex-shrink: 0;
    }
    #bubble.light .bubble-btn { background: transparent; color: #4a4a5a; }
    #bubble.light .bubble-btn:hover { background: rgba(0,0,0,0.06); color: #1a1a2e; }
    #bubble.dark .bubble-btn { background: transparent; color: rgba(200,200,220,0.8); }
    #bubble.dark .bubble-btn:hover { background: rgba(255,255,255,0.08); color: #fff; }
    .bubble-btn:active { transform: scale(0.92); }
    .bubble-btn svg { width: 15px; height: 15px; flex-shrink: 0; }
    .bubble-btn.logo-btn svg { width: 16px; height: 16px; }
    #bubble.light .bubble-btn.logo-btn { color: #7c3aed; }
    #bubble.light .bubble-btn.logo-btn:hover { background: rgba(124,58,237,0.1); }
    #bubble.dark .bubble-btn.logo-btn { color: #a78bfa; }
    #bubble.dark .bubble-btn.logo-btn:hover { background: rgba(124,58,237,0.2); }

    .bubble-sep { width: 1px; height: 16px; flex-shrink: 0; }
    #bubble.light .bubble-sep { background: rgba(0,0,0,0.08); }
    #bubble.dark .bubble-sep { background: rgba(255,255,255,0.1); }

    #quick-launch {
      display: none; position: fixed; width: 450px; max-height: 420px;
      border-radius: 12px; pointer-events: auto; z-index: 2147483647;
      flex-direction: column; overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    }
    #quick-launch.light {
      background: #ffffff; border: 1px solid rgba(0,0,0,0.1);
      box-shadow: 0 4px 16px rgba(0,0,0,0.1), 0 12px 40px rgba(0,0,0,0.12);
    }
    #quick-launch.dark {
      background: #1e1e2e; border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 4px 16px rgba(0,0,0,0.3), 0 12px 40px rgba(0,0,0,0.4);
    }
    #quick-launch.visible { display: flex; animation: aura-bubble-in 0.3s ease-in-out forwards; }
    #quick-launch.hiding { display: flex; animation: aura-bubble-out 0.15s ease-in-out forwards; }

    .ql-header { display: flex; align-items: center; gap: 8px; padding: 12px 14px 8px; }
    .ql-header svg { width: 18px; height: 18px; flex-shrink: 0; color: #7c3aed; }
    .ql-header-title { font-size: 13px; font-weight: 600; }
    #quick-launch.light .ql-header-title { color: #1a1a2e; }
    #quick-launch.dark .ql-header-title { color: rgba(226,232,240,0.95); }
    .ql-close {
      margin-left: auto; width: 22px; height: 22px; border: none; border-radius: 5px;
      background: transparent; cursor: pointer; display: flex; align-items: center;
      justify-content: center; padding: 0; transition: background 0.12s;
    }
    #quick-launch.light .ql-close { color: #999; }
    #quick-launch.light .ql-close:hover { background: rgba(0,0,0,0.06); color: #333; }
    #quick-launch.dark .ql-close { color: rgba(200,200,220,0.5); }
    #quick-launch.dark .ql-close:hover { background: rgba(255,255,255,0.08); color: #fff; }

    .ql-preview { padding: 0 14px; margin-bottom: 8px; }
    .ql-preview-text {
      font-size: 12px; line-height: 1.5; max-height: 56px; overflow: hidden;
      display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
      border-radius: 6px; padding: 8px 10px;
    }
    #quick-launch.light .ql-preview-text { background: rgba(0,0,0,0.03); color: #555; border: 1px solid rgba(0,0,0,0.05); }
    #quick-launch.dark .ql-preview-text { background: rgba(255,255,255,0.04); color: rgba(200,200,220,0.7); border: 1px solid rgba(255,255,255,0.06); }

    .ql-input-row { display: flex; gap: 8px; padding: 0 14px; margin-bottom: 10px; }
    .ql-input {
      flex: 1; border-radius: 8px; padding: 8px 12px; font-size: 13px;
      font-family: inherit; outline: none; transition: border-color 0.15s;
    }
    #quick-launch.light .ql-input { background: #fff; border: 1px solid rgba(0,0,0,0.12); color: #1a1a2e; }
    #quick-launch.light .ql-input::placeholder { color: #aaa; }
    #quick-launch.light .ql-input:focus { border-color: #7c3aed; }
    #quick-launch.dark .ql-input { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: rgba(226,232,240,0.95); }
    #quick-launch.dark .ql-input::placeholder { color: rgba(200,200,220,0.35); }
    #quick-launch.dark .ql-input:focus { border-color: #7c3aed; }

    .ql-submit {
      padding: 8px 16px; border-radius: 8px; border: none; background: #7c3aed; color: #fff;
      font-size: 13px; font-weight: 600; font-family: inherit; cursor: pointer;
      white-space: nowrap; transition: background 0.15s, transform 0.12s;
    }
    .ql-submit:hover { background: #6d28d9; }
    .ql-submit:active { transform: scale(0.97); }

    .ql-actions { display: flex; gap: 6px; padding: 0 14px 12px; flex-wrap: wrap; }
    .ql-action-btn {
      padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 500;
      font-family: inherit; cursor: pointer; white-space: nowrap;
      transition: background 0.12s, border-color 0.12s, color 0.12s;
      display: flex; align-items: center; gap: 5px;
    }
    #quick-launch.light .ql-action-btn { background: rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.08); color: #555; }
    #quick-launch.light .ql-action-btn:hover { background: rgba(124,58,237,0.08); border-color: rgba(124,58,237,0.25); color: #7c3aed; }
    #quick-launch.dark .ql-action-btn { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: rgba(200,200,220,0.7); }
    #quick-launch.dark .ql-action-btn:hover { background: rgba(124,58,237,0.15); border-color: rgba(124,58,237,0.3); color: #a78bfa; }

    .ql-model-row { display: flex; align-items: center; gap: 6px; padding: 0 14px 10px; }
    .ql-model-label { font-size: 11px; font-weight: 500; }
    #quick-launch.light .ql-model-label { color: #999; }
    #quick-launch.dark .ql-model-label { color: rgba(200,200,220,0.4); }
    .ql-model-select {
      font-size: 11px; font-family: inherit; border-radius: 5px; padding: 3px 8px;
      cursor: pointer; outline: none;
    }
    #quick-launch.light .ql-model-select { background: #fff; border: 1px solid rgba(0,0,0,0.1); color: #333; }
    #quick-launch.dark .ql-model-select { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); color: rgba(200,200,220,0.8); }

    #toast {
      display: none; position: fixed;
      background: rgba(5, 150, 105, 0.92); backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px); color: #fff; font-size: 12px;
      font-weight: 500; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      padding: 6px 14px; border-radius: 8px; pointer-events: none;
      z-index: 2147483647; box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    #toast.visible { display: block; }
  `;
    shadow.appendChild(style);
    const bubble = document.createElement("div");
    bubble.id = "bubble";
    const BUBBLE_LOGO = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;
    const BUBBLE_COPY = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>`;
    const BUBBLE_EXPLAIN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>`;
    const BUBBLE_SUMMARIZE = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>`;
    const BUBBLE_MORE = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/></svg>`;
    const bubbleButtons = [
      { svg: BUBBLE_LOGO, action: "quick-launch", tip: "AURA", cls: "logo-btn" },
      { svg: BUBBLE_COPY, action: "copy", tip: "Copy" },
      { svg: BUBBLE_EXPLAIN, action: "explain", tip: "Explain" },
      { svg: BUBBLE_SUMMARIZE, action: "summarize", tip: "Summarize" },
      { svg: BUBBLE_MORE, action: "more", tip: "More actions" }
    ];
    bubbleButtons.forEach((def, i) => {
      if (i === 1) {
        const sep = document.createElement("div");
        sep.className = "bubble-sep";
        bubble.appendChild(sep);
      }
      const btn = document.createElement("button");
      btn.className = "bubble-btn" + (def.cls ? " " + def.cls : "");
      btn.innerHTML = def.svg;
      btn.dataset.action = def.action;
      btn.title = def.tip;
      bubble.appendChild(btn);
    });
    shadow.appendChild(bubble);
    const quickLaunch = document.createElement("div");
    quickLaunch.id = "quick-launch";
    quickLaunch.innerHTML = '<div class="ql-header"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg><span class="ql-header-title">AURA</span><button class="ql-close" data-action="ql-close" title="Close"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button></div><div class="ql-preview"><div class="ql-preview-text" id="ql-selected-text"></div></div><div class="ql-input-row"><input class="ql-input" id="ql-prompt-input" type="text" placeholder="Ask anything about this text..." /><button class="ql-submit" id="ql-submit-btn">Send</button></div><div class="ql-actions"><button class="ql-action-btn" data-action="explain">Explain</button><button class="ql-action-btn" data-action="summarize">Summarize</button><button class="ql-action-btn" data-action="ask">Chat with AURA</button><button class="ql-action-btn" data-action="save">Save to Memory</button><button class="ql-action-btn" data-action="translate">Translate</button></div><div class="ql-model-row"><span class="ql-model-label">Model:</span><select class="ql-model-select" id="ql-model-select"><option value="auto">Auto</option><option value="fast">Fast</option><option value="balanced">Balanced</option><option value="powerful">Powerful</option></select></div>';
    shadow.appendChild(quickLaunch);
    const toolbar = bubble;
    const toast = document.createElement("div");
    toast.id = "toast";
    shadow.appendChild(toast);
    const fabShadowHost = document.createElement("div");
    fabShadowHost.id = "aura-dock-shadow";
    Object.assign(fabShadowHost.style, {
      position: "fixed",
      left: "0",
      top: "0",
      width: "100vw",
      height: "100vh",
      zIndex: "2147483647",
      pointerEvents: "none"
    });
    document.body.appendChild(fabShadowHost);
    const fabShadow = fabShadowHost.attachShadow({ mode: "closed" });
    const fabStyle = document.createElement("style");
    fabStyle.textContent = `
    /* ── FAB outer container — fixed to right or left edge ── */
    .aura-fab {
      position: fixed;
      right: 0;
      bottom: 30px;
      z-index: 2147483645;
      user-select: none;
      -webkit-user-select: none;
      visibility: visible !important;
      pointer-events: none;
      transform: translate3d(100%, 0, 0);
      transition: all 0.3s ease 0.2s;
      opacity: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    }
    .aura-fab.show {
      transform: translateZ(0);
      opacity: 1;
    }
    .aura-fab.left {
      left: 0;
      right: auto;
      transform: translate3d(-100%, 0, 0);
    }
    .aura-fab.left.show {
      transform: translateZ(0);
      opacity: 1;
    }

    /* ── The pill button ── */
    .aura-fab .fab-pill {
      padding: 8px 10px;
      border-radius: 999px 0 0 999px;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px) saturate(1.4);
      -webkit-backdrop-filter: blur(16px) saturate(1.4);
      box-shadow: 0 0 1px 0 rgba(124, 58, 237, 0.3),
                  0 6px 24px rgba(12, 13, 25, 0.06),
                  0 12px 48px rgba(12, 13, 25, 0.06),
                  0 24px 96px rgba(12, 13, 25, 0.06);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      pointer-events: auto;
      position: relative;
    }
    .aura-fab .fab-pill:not(.dragging) {
      transition: padding 0.3s, border-radius 0.3s;
    }
    .aura-fab.left .fab-pill {
      border-radius: 0 999px 999px 0;
    }

    /* Hover expand */
    .aura-fab .fab-pill:not(.dragging).hover {
      padding: 8px 14px;
    }

    /* Dragging state: full circle */
    .aura-fab .fab-pill.dragging {
      border-radius: 999px;
      padding: 8px;
      cursor: move;
    }

    /* ── Close button ── */
    .aura-fab .fab-close {
      position: absolute;
      bottom: -4px;
      left: -4px;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.12);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.2s ease, background 0.15s ease;
      pointer-events: auto;
      border: none;
      padding: 0;
      color: rgba(255, 255, 255, 0.7);
    }
    .aura-fab.left .fab-close {
      left: auto;
      right: -4px;
    }
    .aura-fab .fab-pill.hover:not(.dragging) .fab-close {
      opacity: 1;
    }
    .aura-fab .fab-close:hover {
      background: rgba(255, 255, 255, 0.22);
    }

    /* ── Logo icon ── */
    .aura-fab .fab-logo {
      width: 20px;
      height: 20px;
      pointer-events: none;
      flex-shrink: 0;
    }

    /* ── Popout toolbar ── */
    .aura-fab .fab-popout {
      position: absolute;
      right: 0;
      bottom: 100%;
      padding: 8px 0;
      pointer-events: auto;
      transition: padding 0.3s, opacity 0.3s;
    }
    .aura-fab .fab-popout.hidden {
      padding: 0;
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
    }
    .aura-fab .fab-popout.hidden .fab-popout-inner {
      transform: scale(0.95);
    }
    .aura-fab .fab-popout.reverse {
      bottom: auto;
      top: 100%;
      padding: 8px 0;
    }
    .aura-fab.left .fab-popout {
      right: auto;
      left: 0;
    }

    .aura-fab .fab-popout-inner {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 3px;
      padding: 3px;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px) saturate(1.4);
      -webkit-backdrop-filter: blur(16px) saturate(1.4);
      border-radius: 999px;
      box-shadow: 0 0 1px 0 rgba(124, 58, 237, 0.3),
                  0 6px 24px rgba(12, 13, 25, 0.06),
                  0 12px 48px rgba(12, 13, 25, 0.06),
                  0 24px 96px rgba(12, 13, 25, 0.06);
      transition: transform 0.2s ease;
    }
    .aura-fab .fab-popout.reverse .fab-popout-inner {
      flex-direction: column-reverse;
    }

    .fab-popout-sep {
      width: 14px;
      height: 1px;
      background: rgba(255, 255, 255, 0.08);
      flex-shrink: 0;
    }

    .fab-action-btn {
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: transparent;
      border: none;
      padding: 0;
      margin: 0;
      color: rgba(160, 148, 210, 0.7);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      box-sizing: border-box;
      outline: none;
      transition: background 0.15s ease, color 0.15s ease, transform 0.1s ease;
    }
    .fab-action-btn:hover {
      background: rgba(124, 58, 237, 0.3);
      color: rgba(224, 214, 255, 1);
    }
    .fab-action-btn:active {
      transform: scale(0.9);
    }
  `;
    fabShadow.appendChild(fabStyle);
    let _fabSide = "right";
    let _fabOffset = 0;
    let _fabVisible = true;
    let _fabHiddenOnce = false;
    let _fabHovering = false;
    let _fabDragging = false;
    let _fabPopupOpen = false;
    ext.storage.local.get(["auraFabSide", "auraFabOffset"], (data) => {
      if (data.auraFabSide === "left" || data.auraFabSide === "right") _fabSide = data.auraFabSide;
      if (typeof data.auraFabOffset === "number") _fabOffset = data.auraFabOffset;
      applyFabState();
    });
    const fabContainer = document.createElement("div");
    fabContainer.className = "aura-fab";
    fabShadow.appendChild(fabContainer);
    const fabTranslateWrap = document.createElement("div");
    fabTranslateWrap.style.position = "relative";
    fabContainer.appendChild(fabTranslateWrap);
    const fabPointerWrap = document.createElement("div");
    fabPointerWrap.style.pointerEvents = "auto";
    fabTranslateWrap.appendChild(fabPointerWrap);
    const fabPopout = document.createElement("div");
    fabPopout.className = "fab-popout hidden";
    fabPointerWrap.appendChild(fabPopout);
    const fabPopoutInner = document.createElement("div");
    fabPopoutInner.className = "fab-popout-inner";
    fabPopout.appendChild(fabPopoutInner);
    const _fabItems = [
      { svg: '<path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>', action: "dock-chat", tip: "Chat with AURA" },
      { svg: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>', action: "dock-search", tip: "Search" },
      null,
      { svg: '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><path d="M9 21V9"/>', action: "dock-thispage", tip: "This Page" },
      { svg: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>', action: "dock-translate", tip: "Translate" },
      null,
      { svg: '<path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>', action: "dock-save", tip: "Save to Memory" }
    ];
    _fabItems.forEach((item) => {
      if (!item) {
        const sep = document.createElement("div");
        sep.className = "fab-popout-sep";
        fabPopoutInner.appendChild(sep);
        return;
      }
      const btn = document.createElement("button");
      btn.className = "fab-action-btn";
      btn.dataset.action = item.action;
      btn.title = item.tip;
      btn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${item.svg}</svg>`;
      fabPopoutInner.appendChild(btn);
    });
    const fabPill = document.createElement("div");
    fabPill.className = "fab-pill";
    fabPointerWrap.appendChild(fabPill);
    const fabLogo = document.createElement("div");
    fabLogo.className = "fab-logo";
    fabLogo.innerHTML = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="rgba(160, 148, 210, 0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;
    fabPill.appendChild(fabLogo);
    const fabClose = document.createElement("button");
    fabClose.className = "fab-close";
    fabClose.innerHTML = `<svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
    fabPill.appendChild(fabClose);
    function applyFabState() {
      fabContainer.classList.toggle("left", _fabSide === "left");
      if (_fabVisible && !_fabHiddenOnce) {
        fabContainer.classList.add("show");
      } else {
        fabContainer.classList.remove("show");
      }
      fabTranslateWrap.style.transform = `translate3d(0, ${_fabOffset + (_fabDragging ? 0.01 : 0)}px, 0)`;
      const isNearTop = _fabOffset < -0.7 * window.innerHeight;
      fabPopout.classList.toggle("reverse", isNearTop);
      const showPopout = (_fabHovering || _fabPopupOpen) && !_fabDragging;
      fabPopout.classList.toggle("hidden", !showPopout);
      fabPill.classList.toggle("hover", _fabHovering && !_fabDragging);
      fabPill.classList.toggle("dragging", _fabDragging);
    }
    function saveFabState() {
      try {
        ext.storage.local.set({ auraFabSide: _fabSide, auraFabOffset: _fabOffset });
      } catch (_e) {
      }
    }
    let _downPos = [0, 0];
    let _movePos = [0, 0];
    let _downTranslateY = 0;
    let _totalMoveDist = 0;
    let _ptrDown = false;
    let _ptrCaptured = false;
    function parseTranslateY(el) {
      const m = el.style.transform.match(/translate3d\(\s*[\d.eE+-]+px\s*,\s*([\d.eE+-]+)px/);
      return m ? parseFloat(m[1]) : 0;
    }
    function clampOffset(offset) {
      const pillH = fabPill.getBoundingClientRect().height || 36;
      const maxUp = -(window.innerHeight - 30 - 10 - pillH / 2);
      const maxDown = 30 - 10 - pillH / 2;
      return Math.max(maxUp, Math.min(offset, maxDown));
    }
    function onFabPointerDown(e) {
      if (e.button === 2) return;
      _downPos = [e.clientX, e.clientY];
      _movePos = [e.clientX, e.clientY];
      _ptrDown = true;
      _ptrCaptured = false;
      _downTranslateY = parseTranslateY(fabTranslateWrap);
      _totalMoveDist = 0;
      fabPill.dataset.move = "0";
      fabPill.style.touchAction = "none";
    }
    function onFabPointerMove(e) {
      if (!_ptrDown) return;
      _movePos = [e.clientX, e.clientY];
      const dx = _movePos[0] - _downPos[0];
      const dy = _movePos[1] - _downPos[1];
      _totalMoveDist = Math.abs(dx) + Math.abs(dy);
      fabPill.dataset.move = String(_totalMoveDist);
      if (_totalMoveDist > 3 && !_fabDragging) {
        _fabDragging = true;
        if (!_ptrCaptured) {
          fabPill.setPointerCapture(e.pointerId);
          _ptrCaptured = true;
        }
        applyFabState();
      }
      if (!_fabDragging) return;
      const newY = clampOffset(_downTranslateY + dy);
      fabTranslateWrap.style.transform = `translate3d(0, ${newY}px, 0)`;
      const currentSide = _downPos[0] + dx > window.innerWidth / 2 ? "right" : "left";
      if (currentSide !== _fabSide) {
        _fabSide = currentSide;
        fabContainer.classList.toggle("left", _fabSide === "left");
      }
    }
    function onFabPointerUp(e) {
      if (!_ptrDown) return;
      _ptrDown = false;
      if (_ptrCaptured) {
        fabPill.releasePointerCapture(e.pointerId);
        _ptrCaptured = false;
      }
      if (_fabDragging) {
        const dy = _movePos[1] - _downPos[1];
        const dx = _movePos[0] - _downPos[0];
        let finalY = clampOffset(_downTranslateY + dy);
        if (finalY > 0) finalY = 0;
        _fabSide = _downPos[0] + dx > window.innerWidth / 2 ? "right" : "left";
        _fabOffset = finalY;
        _fabDragging = false;
        applyFabState();
        saveFabState();
      }
    }
    fabPill.addEventListener("pointerdown", onFabPointerDown);
    fabPill.addEventListener("pointermove", onFabPointerMove);
    fabPill.addEventListener("pointerup", onFabPointerUp);
    fabPill.addEventListener("pointercancel", onFabPointerUp);
    let _hoverTimer = null;
    function setFabHover(val) {
      if (_hoverTimer) {
        clearTimeout(_hoverTimer);
        _hoverTimer = null;
      }
      if (val) {
        _fabHovering = true;
        applyFabState();
      } else {
        _hoverTimer = setTimeout(() => {
          {
            _fabHovering = false;
            applyFabState();
          }
        }, 0);
      }
    }
    fabPill.addEventListener("mouseenter", () => setFabHover(true));
    fabPill.addEventListener("mouseleave", () => setFabHover(false));
    fabPopout.addEventListener("mouseenter", () => setFabHover(true));
    fabPopout.addEventListener("mouseleave", () => setFabHover(false));
    fabPill.addEventListener("click", (e) => {
      if (_totalMoveDist > 10) return;
      const target = e.target;
      if (target.closest(".fab-close")) return;
      safeSend({ type: "OPEN_PANEL", panel: "chat" });
    });
    fabClose.addEventListener("click", (e) => {
      e.stopPropagation();
      _fabHiddenOnce = true;
      _fabVisible = false;
      applyFabState();
    });
    fabPopoutInner.addEventListener("click", (e) => {
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
            if (response && response.ok) showToast("Saved to AURA memory");
            else showToast("Save failed — is backend running?", 3e3);
          }
        );
      }
    });
    function clampFabOnResize() {
      if (Math.abs(_fabOffset) > window.innerHeight) {
        _fabOffset = -(window.innerHeight - 66);
        if (_fabOffset > 0) _fabOffset = 0;
        saveFabState();
      }
      applyFabState();
    }
    window.addEventListener("resize", clampFabOnResize);
    window.addEventListener("focus", () => setTimeout(clampFabOnResize, 100));
    function showDock() {
      _fabHiddenOnce = false;
      _fabVisible = true;
      applyFabState();
    }
    setTimeout(() => {
      if (!_fabHiddenOnce) {
        _fabVisible = true;
        applyFabState();
      }
    }, 50);
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
    let _lastMouseX = 0;
    let _lastMouseY = 0;
    let _mouseDownY = 0;
    let _bubbleShowTime = 0;
    let _qlVisible = false;
    document.addEventListener("mousedown", (e) => {
      _mouseDownY = e.clientY;
      _lastMouseX = e.clientX;
      _lastMouseY = e.clientY;
    }, true);
    function positionBubble() {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      const theme = themeClass();
      bubble.classList.remove("light", "dark");
      bubble.classList.add(theme);
      bubble.style.visibility = "hidden";
      bubble.style.display = "flex";
      const bRect = bubble.getBoundingClientRect();
      const bw = bRect.width || 160;
      const bh = bRect.height || 34;
      bubble.style.visibility = "";
      bubble.style.display = "";
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      let left = _lastMouseX - bw / 2;
      if (left < 8) left = 8;
      if (left + bw > vw - 8) left = vw - bw - 8;
      const selectionWentDown = _lastMouseY >= _mouseDownY;
      let top;
      if (selectionWentDown) {
        top = _lastMouseY + 15;
        if (top + bh > vh - 8) top = _lastMouseY - bh - 15;
      } else {
        top = _lastMouseY - bh - 40;
        if (top < 8) top = _lastMouseY + 15;
      }
      bubble.style.top = `${Math.round(top)}px`;
      bubble.style.left = `${Math.round(left)}px`;
    }
    function showBubble() {
      if (_hideAnimTimer) {
        clearTimeout(_hideAnimTimer);
        _hideAnimTimer = null;
      }
      bubble.classList.remove("hiding");
      bubble.classList.add("visible");
      positionBubble();
      host.style.pointerEvents = "auto";
      _bubbleShowTime = Date.now();
    }
    function hideBubble() {
      if (!bubble.classList.contains("visible") && !_qlVisible) return;
      hideQuickLaunch();
      bubble.classList.remove("visible");
      bubble.classList.add("hiding");
      if (_hideAnimTimer) clearTimeout(_hideAnimTimer);
      _hideAnimTimer = setTimeout(() => {
        bubble.classList.remove("hiding");
        host.style.pointerEvents = "none";
        _hideAnimTimer = null;
      }, 150);
    }
    function showQuickLaunch() {
      const text = getSelectionText();
      if (!text) return;
      const theme = themeClass();
      quickLaunch.classList.remove("light", "dark");
      quickLaunch.classList.add(theme);
      const previewEl = shadow.getElementById("ql-selected-text");
      if (previewEl) previewEl.textContent = text;
      const inputEl = shadow.getElementById("ql-prompt-input");
      if (inputEl) inputEl.value = "";
      const bubbleRect = bubble.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const qlW = 450;
      const qlH = 400;
      let left = bubbleRect.left;
      if (left + qlW > vw - 16) left = vw - qlW - 16;
      if (left < 16) left = 16;
      let top = bubbleRect.bottom + 8;
      if (top + qlH > vh - 16) {
        top = bubbleRect.top - qlH - 8;
        if (top < 16) top = 16;
      }
      quickLaunch.style.top = `${Math.round(top)}px`;
      quickLaunch.style.left = `${Math.round(left)}px`;
      quickLaunch.classList.remove("hiding");
      quickLaunch.classList.add("visible");
      _qlVisible = true;
      setTimeout(() => {
        const inp = shadow.getElementById("ql-prompt-input");
        if (inp) inp.focus();
      }, 100);
    }
    function hideQuickLaunch() {
      if (!_qlVisible) return;
      quickLaunch.classList.remove("visible");
      quickLaunch.classList.add("hiding");
      _qlVisible = false;
      setTimeout(() => {
        quickLaunch.classList.remove("hiding");
      }, 150);
    }
    document.addEventListener("mouseup", (e) => {
      _lastMouseX = e.clientX;
      _lastMouseY = e.clientY;
      setTimeout(() => {
        const text = getSelectionText();
        if (text.length > 0) showBubble();
        else hideBubble();
      }, 50);
    });
    document.addEventListener("mousedown", (e) => {
      if (Date.now() - _bubbleShowTime < 100) return;
      if (!host.contains(e.target)) hideBubble();
    });
    document.addEventListener("contextmenu", () => {
      hideBubble();
    });
    window.addEventListener("scroll", () => {
      hideBubble();
    }, { passive: true });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") hideBubble();
    });
    document.addEventListener("selectionchange", () => {
      if (getSelectionText().length === 0 && !_qlVisible) hideBubble();
    });
    bubble.addEventListener("click", (e) => {
      const target = e.target;
      const btn = target.closest(".bubble-btn");
      if (!btn) return;
      e.stopPropagation();
      const action = btn.dataset.action;
      const text = getSelectionText();
      const url = window.location.href;
      const title = document.title;
      if (action === "quick-launch") {
        showQuickLaunch();
        return;
      }
      if (action === "more") {
        if (text) safeSend({ type: "OPEN_WITH_TEXT", action: "ask", text, url, title });
        hideBubble();
        return;
      }
      if (!text) return;
      if (action === "copy") {
        navigator.clipboard.writeText(text).then(() => {
          showToast("Copied to clipboard");
        }).catch(() => {
          showToast("Copy failed", 3e3);
        });
        hideBubble();
      } else if (action === "save") {
        saveHighlight();
        safeSend(
          { type: "SAVE_KNOWLEDGE", text, url, title },
          (_response) => {
          }
        );
        hideBubble();
      } else {
        safeSend({ type: "OPEN_WITH_TEXT", action, text, url, title });
        hideBubble();
      }
    });
    quickLaunch.addEventListener("click", (e) => {
      var _a2;
      const target = e.target;
      e.stopPropagation();
      if (target.closest('[data-action="ql-close"]')) {
        hideQuickLaunch();
        return;
      }
      if (target.id === "ql-submit-btn" || target.closest("#ql-submit-btn")) {
        const inputEl = shadow.getElementById("ql-prompt-input");
        const prompt = ((_a2 = inputEl == null ? void 0 : inputEl.value) == null ? void 0 : _a2.trim()) || "";
        const text = getSelectionText();
        if (!text && !prompt) return;
        safeSend({ type: "OPEN_WITH_TEXT", action: "ask", text: prompt ? `${prompt}

Context:
${text}` : text, url: window.location.href, title: document.title });
        hideBubble();
        return;
      }
      const actionBtn = target.closest(".ql-action-btn");
      if (actionBtn) {
        const action = actionBtn.dataset.action;
        const text = getSelectionText();
        const url = window.location.href;
        const title = document.title;
        if (!text) return;
        if (action === "save") {
          saveHighlight();
          safeSend(
            { type: "SAVE_KNOWLEDGE", text, url, title },
            (_response) => {
            }
          );
          hideBubble();
        } else if (action === "translate") {
          safeSend({ type: "OPEN_PANEL", panel: "translate" });
          hideBubble();
        } else {
          safeSend({ type: "OPEN_WITH_TEXT", action, text, url, title });
          hideBubble();
        }
        return;
      }
    });
    quickLaunch.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const submitBtn = shadow.getElementById("ql-submit-btn");
        if (submitBtn) submitBtn.click();
      }
    });
    function bestSelector(el) {
      var _a2;
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
        const siblings = [...((_a2 = cur.parentElement) == null ? void 0 : _a2.children) || []];
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
        var _a2;
        if (node.nodeType === Node.TEXT_NODE) {
          const text = (node.textContent || "").replace(/\s+/g, " ");
          if (text.trim()) parts.push(text);
          return;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        const el = node;
        const tag = el.tagName;
        if (el.hasAttribute("hidden") || ((_a2 = el.style) == null ? void 0 : _a2.display) === "none") return;
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
      var _a2, _b, _c, _d;
      try {
        const url = window.location.href;
        const title = document.title;
        if (url.match(/\.pdf($|\?|#)/i) || document.contentType === "application/pdf") {
          return {
            text: ((_b = (_a2 = document.body) == null ? void 0 : _a2.innerText) == null ? void 0 : _b.slice(0, MAX_TEXT_BYTES)) || "[PDF document]",
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
      var _a2, _b;
      const url = window.location.href;
      const titleEl = document.querySelector(
        "h1.ytd-watch-metadata, h1.ytd-video-primary-info-renderer, #title h1"
      );
      const videoTitle = ((_a2 = titleEl == null ? void 0 : titleEl.textContent) == null ? void 0 : _a2.trim()) || document.title.replace(/ - YouTube$/, "").trim();
      let transcript = "";
      const transcriptSegments = document.querySelectorAll(
        "ytd-transcript-segment-renderer .segment-text, yt-formatted-string.ytd-transcript-segment-renderer, #segments-container ytd-transcript-segment-renderer"
      );
      if (transcriptSegments.length > 0) {
        const lines = [];
        transcriptSegments.forEach((seg) => {
          var _a3;
          const text2 = (_a3 = seg.textContent) == null ? void 0 : _a3.trim();
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
          var _a3;
          if (i >= 10) return;
          const text2 = (_a3 = el.textContent) == null ? void 0 : _a3.trim();
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
      const offscreen = rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth;
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
      _qaTriggerEl.style.display = offscreen ? "none" : "";
      if (offscreen) {
        removeQaMenu();
        return;
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
    document.addEventListener("aura-netflix-subtitles", (e) => {
      try {
        const d = e.detail;
        safeSend({
          type: "NETFLIX_SUBTITLES",
          movieId: d.movieId || "",
          lang: d.lang || "",
          trackId: d.trackId || "",
          segments: d.segments || []
        });
      } catch {
      }
    });
    document.addEventListener("aura-netflix-metadata", (e) => {
      try {
        const d = e.detail;
        safeSend({
          type: "NETFLIX_METADATA",
          movieId: d.movieId || "",
          title: d.title || "",
          episodeTitle: d.episodeTitle || "",
          seasonNumber: d.seasonNumber || 0,
          episodeNumber: d.episodeNumber || 0,
          duration: d.duration || 0
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
            if (!_translateActive) return;
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
        var _a2;
        const text = (_a2 = body.innerText) == null ? void 0 : _a2.trim();
        if (text) parts.push(text);
      });
      return parts.join("\n\n---\n\n").slice(0, 2e4);
    }
    function getComposeBody(composeEl) {
      const ariaLabels = [
        "Message Body",
        // English
        "Nachrichtentext",
        // German
        "Corps du message",
        // French
        "Cuerpo del mensaje",
        // Spanish
        "Corpo da mensagem",
        // Portuguese
        "Corpo del messaggio",
        // Italian
        "Текст сообщения",
        // Russian
        "Mesaj Metni",
        // Turkish
        "メッセージ本文",
        // Japanese
        "메시지 본문",
        // Korean
        "邮件正文",
        // Chinese Simplified
        "نص الرسالة",
        // Arabic
        "Berichttekst",
        // Dutch
        "Treść wiadomości",
        // Polish
        "संदेश का मुख्य भाग",
        // Hindi
        "Mesaj mətni"
        // Azerbaijani
      ];
      const ariaSelector = ariaLabels.map((l) => `div[aria-label="${l}"]`).join(", ");
      const result = composeEl.querySelector(
        ariaSelector + ', div[g_editable="true"][contenteditable="true"], div.editable[contenteditable="true"]'
      );
      if (result) return result;
      return composeEl.querySelector('div[contenteditable="true"][role="textbox"]');
    }
    function getComposeText(composeEl) {
      var _a2;
      const body = getComposeBody(composeEl);
      if (!body) return "";
      return ((_a2 = body.innerText) == null ? void 0 : _a2.trim()) || "";
    }
    function escapeHtml(s) {
      return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
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
          (line) => `<div>${escapeHtml(line) || "<br>"}</div>`
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
        var _a2, _b, _c;
        let shouldScan = false;
        for (const mutation of mutations) {
          if (mutation.addedNodes.length > 0) {
            for (const node of mutation.addedNodes) {
              if (node.nodeType !== Node.ELEMENT_NODE) continue;
              const el = node;
              if (((_a2 = el.matches) == null ? void 0 : _a2.call(el, 'div[role="dialog"]')) || ((_b = el.querySelector) == null ? void 0 : _b.call(el, 'div[role="dialog"]')) || ((_c = el.querySelector) == null ? void 0 : _c.call(el, 'div[contenteditable="true"]'))) {
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
    const imgTbHost = document.createElement("div");
    imgTbHost.id = "aura-img-toolbar-host";
    Object.assign(imgTbHost.style, {
      position: "fixed",
      top: "0",
      left: "0",
      zIndex: "2147483646",
      pointerEvents: "none"
    });
    document.documentElement.appendChild(imgTbHost);
    const imgTbShadow = imgTbHost.attachShadow({ mode: "closed" });
    const imgTbCss = document.createElement("style");
    imgTbCss.textContent = `
    @keyframes aura-imgtb-in {
      from { opacity: 0; transform: translateY(4px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes aura-imgtb-out {
      from { opacity: 1; transform: translateY(0) scale(1); }
      to   { opacity: 0; transform: translateY(4px) scale(0.95); }
    }

    .imgtb {
      position: fixed;
      display: flex;
      align-items: center;
      gap: 3px;
      padding: 4px 6px;
      background: rgba(10, 8, 24, 0.88);
      backdrop-filter: blur(20px) saturate(1.5);
      -webkit-backdrop-filter: blur(20px) saturate(1.5);
      border: 1px solid rgba(124, 58, 237, 0.25);
      border-radius: 10px;
      box-shadow: 0 6px 24px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.05) inset;
      pointer-events: auto;
      animation: aura-imgtb-in 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    }
    .imgtb.hiding {
      animation: aura-imgtb-out 0.12s ease forwards;
    }

    .imgtb-btn {
      display: flex;
      align-items: center;
      gap: 4px;
      background: transparent;
      border: none;
      color: rgba(226, 232, 240, 0.85);
      font-size: 11px;
      font-weight: 500;
      font-family: inherit;
      padding: 4px 8px;
      border-radius: 6px;
      cursor: pointer;
      white-space: nowrap;
      transition: background 0.12s ease, color 0.12s ease;
      line-height: 1;
    }
    .imgtb-btn:hover {
      background: rgba(124, 58, 237, 0.3);
      color: #fff;
    }
    .imgtb-btn:active {
      background: rgba(124, 58, 237, 0.45);
    }
    .imgtb-btn svg { flex-shrink: 0; }

    .imgtb-sep {
      width: 1px;
      height: 14px;
      background: rgba(255, 255, 255, 0.08);
      flex-shrink: 0;
    }
  `;
    imgTbShadow.appendChild(imgTbCss);
    const imgTbContainer = document.createElement("div");
    imgTbShadow.appendChild(imgTbContainer);
    let _imgTbEl = null;
    let _imgTbHoverTmr = null;
    let _imgTbDismissTmr = null;
    let _imgTbCurImg = null;
    const IMG_TB_MIN_SIZE = 80;
    const IMGTB_ICON_EYE = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;
    const IMGTB_ICON_EDIT = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>`;
    const IMGTB_ICON_SAVE = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;
    function imgTbShouldShow(img) {
      if (!img.src && !img.currentSrc) return false;
      const src = img.currentSrc || img.src;
      if (src.startsWith("data:") && src.length < 200) return false;
      if (src.startsWith("data:image/svg")) return false;
      const rect = img.getBoundingClientRect();
      if (rect.width < IMG_TB_MIN_SIZE || rect.height < IMG_TB_MIN_SIZE) return false;
      return true;
    }
    function imgTbRemove() {
      if (_imgTbEl) {
        _imgTbEl.remove();
        _imgTbEl = null;
      }
      _imgTbCurImg = null;
    }
    function imgTbCancelTimers() {
      if (_imgTbHoverTmr) {
        clearTimeout(_imgTbHoverTmr);
        _imgTbHoverTmr = null;
      }
      if (_imgTbDismissTmr) {
        clearTimeout(_imgTbDismissTmr);
        _imgTbDismissTmr = null;
      }
    }
    function imgTbStartDismiss() {
      if (_imgTbDismissTmr) clearTimeout(_imgTbDismissTmr);
      _imgTbDismissTmr = setTimeout(() => {
        imgTbRemove();
        _imgTbDismissTmr = null;
      }, 400);
    }
    function imgTbCancelDismiss() {
      if (_imgTbDismissTmr) {
        clearTimeout(_imgTbDismissTmr);
        _imgTbDismissTmr = null;
      }
    }
    function imgTbPosition(img) {
      if (!_imgTbEl) return;
      const r = img.getBoundingClientRect();
      const tbH = 32;
      let top = r.top - tbH - 6;
      if (top < 4) top = r.top + 6;
      let left = r.left + r.width / 2 - 75;
      if (left < 4) left = 4;
      if (left + 150 > window.innerWidth - 4) left = window.innerWidth - 154;
      _imgTbEl.style.top = Math.round(top) + "px";
      _imgTbEl.style.left = Math.round(left) + "px";
    }
    function imgTbGetImageSrc(img) {
      return img.currentSrc || img.src;
    }
    function imgTbBuild(img) {
      imgTbRemove();
      _imgTbCurImg = img;
      _imgTbEl = document.createElement("div");
      _imgTbEl.className = "imgtb";
      const items = [
        { label: "Describe", icon: IMGTB_ICON_EYE, action: "describe" },
        { label: "Edit in AURA", icon: IMGTB_ICON_EDIT, action: "edit" },
        { label: "Save", icon: IMGTB_ICON_SAVE, action: "save" }
      ];
      items.forEach((item, i) => {
        if (i > 0) {
          const sep = document.createElement("div");
          sep.className = "imgtb-sep";
          _imgTbEl.appendChild(sep);
        }
        const btn = document.createElement("button");
        btn.className = "imgtb-btn";
        btn.innerHTML = item.icon + `<span>${item.label}</span>`;
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          const imgSrc = imgTbGetImageSrc(img);
          if (item.action === "describe") {
            safeSend({ type: "IMAGE_DESCRIBE", imageUrl: imgSrc });
          } else if (item.action === "edit") {
            safeSend({ type: "IMAGE_EDIT_OPEN", imageUrl: imgSrc });
          } else if (item.action === "save") {
            safeSend({ type: "IMAGE_SAVE", imageUrl: imgSrc });
            showToast("Image saved");
          }
          imgTbRemove();
        });
        _imgTbEl.appendChild(btn);
      });
      _imgTbEl.addEventListener("mouseenter", imgTbCancelDismiss);
      _imgTbEl.addEventListener("mouseleave", imgTbStartDismiss);
      imgTbContainer.appendChild(_imgTbEl);
      imgTbPosition(img);
    }
    document.addEventListener("mouseover", (me) => {
      const target = me.target;
      const img = target.tagName === "IMG" ? target : target.closest("img");
      if (!img || !imgTbShouldShow(img)) return;
      if (_imgTbCurImg === img && _imgTbEl) {
        imgTbCancelDismiss();
        return;
      }
      imgTbCancelTimers();
      _imgTbHoverTmr = setTimeout(() => {
        imgTbBuild(img);
        _imgTbHoverTmr = null;
      }, 1e3);
    }, true);
    document.addEventListener("mouseout", (me) => {
      const target = me.target;
      const img = target.tagName === "IMG" ? target : target.closest("img");
      if (img && img === _imgTbCurImg) {
        const rel = me.relatedTarget;
        if (rel && imgTbHost.contains(rel)) return;
        imgTbStartDismiss();
      }
      if (img && _imgTbHoverTmr) imgTbCancelTimers();
    }, true);
    window.addEventListener("scroll", () => {
      if (_imgTbEl && _imgTbCurImg) {
        const r = _imgTbCurImg.getBoundingClientRect();
        if (r.bottom < 0 || r.top > window.innerHeight) {
          imgTbCancelTimers();
          imgTbRemove();
        } else {
          imgTbPosition(_imgTbCurImg);
        }
      }
    }, { passive: true });
    const captureHost = document.createElement("div");
    captureHost.id = "aura-capture-host";
    Object.assign(captureHost.style, {
      position: "fixed",
      top: "0",
      left: "0",
      width: "0",
      height: "0",
      zIndex: "2147483647",
      pointerEvents: "none"
    });
    document.documentElement.appendChild(captureHost);
    const captureShadow = captureHost.attachShadow({ mode: "closed" });
    const captureStyle = document.createElement("style");
    captureStyle.textContent = `
    @keyframes capture-pulse {
      0%, 100% { opacity: 0.6; }
      50% { opacity: 1; }
    }
    .capture-overlay {
      position: fixed;
      pointer-events: none;
      border: 2px solid rgba(124, 58, 237, 0.8);
      background: rgba(124, 58, 237, 0.08);
      border-radius: 3px;
      transition: top 0.05s ease, left 0.05s ease, width 0.05s ease, height 0.05s ease;
      box-shadow: 0 0 0 1px rgba(124, 58, 237, 0.2),
                  0 0 20px rgba(124, 58, 237, 0.15),
                  inset 0 0 20px rgba(124, 58, 237, 0.05);
      z-index: 2147483647;
    }
    .capture-tooltip {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.35);
      border-radius: 6px;
      padding: 4px 10px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      font-size: 11px;
      color: rgba(226, 232, 240, 0.9);
      white-space: nowrap;
      pointer-events: none;
      z-index: 2147483647;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .capture-tooltip .tag {
      color: #a78bfa;
      font-weight: 600;
    }
    .capture-tooltip .cls {
      color: rgba(167, 139, 250, 0.6);
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .capture-tooltip .dims {
      color: rgba(226, 232, 240, 0.5);
      font-size: 10px;
    }
    .capture-banner {
      position: fixed;
      top: 8px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.4);
      border-radius: 10px;
      padding: 8px 16px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      font-size: 12px;
      color: rgba(226, 232, 240, 0.9);
      z-index: 2147483647;
      pointer-events: none;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      gap: 8px;
      animation: capture-pulse 2s ease-in-out infinite;
    }
    .capture-banner .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #a78bfa;
      box-shadow: 0 0 8px rgba(124, 58, 237, 0.6);
    }
  `;
    captureShadow.appendChild(captureStyle);
    const captureContainer = document.createElement("div");
    captureShadow.appendChild(captureContainer);
    let _captureActive = false;
    let _captureOverlay = null;
    let _captureTooltip = null;
    let _captureBanner = null;
    let _captureHoveredEl = null;
    const CAPTURE_CSS_PROPS = [
      "display",
      "position",
      "flex-direction",
      "align-items",
      "justify-content",
      "gap",
      "flex-wrap",
      "flex",
      "flex-grow",
      "flex-shrink",
      "width",
      "height",
      "min-width",
      "min-height",
      "max-width",
      "max-height",
      "padding",
      "padding-top",
      "padding-right",
      "padding-bottom",
      "padding-left",
      "margin",
      "margin-top",
      "margin-right",
      "margin-bottom",
      "margin-left",
      "border",
      "border-radius",
      "border-color",
      "border-width",
      "border-style",
      "background",
      "background-color",
      "background-image",
      "background-size",
      "color",
      "font-size",
      "font-weight",
      "font-family",
      "line-height",
      "letter-spacing",
      "text-align",
      "text-decoration",
      "text-transform",
      "box-shadow",
      "opacity",
      "overflow",
      "z-index",
      "grid-template-columns",
      "grid-template-rows",
      "grid-gap",
      "transform",
      "transition"
    ];
    function extractComputedStyles(el) {
      const styles = window.getComputedStyle(el);
      const result = {};
      for (const prop of CAPTURE_CSS_PROPS) {
        const val = styles.getPropertyValue(prop);
        if (val && val !== "none" && val !== "normal" && val !== "auto" && val !== "0px" && val !== "rgba(0, 0, 0, 0)") {
          result[prop] = val;
        }
      }
      return result;
    }
    function buildCssSelector(el) {
      const tag = el.tagName.toLowerCase();
      const cls = el.className && typeof el.className === "string" ? "." + el.className.trim().split(/\s+/).slice(0, 2).join(".") : "";
      return tag + cls;
    }
    function captureElementData(el) {
      const rect = el.getBoundingClientRect();
      const styles = window.getComputedStyle(el);
      const html = el.outerHTML;
      const cssMap = {};
      cssMap[buildCssSelector(el)] = extractComputedStyles(el);
      const children = el.querySelectorAll("*");
      let count = 0;
      for (const child of children) {
        if (count >= 50) break;
        const childStyles = extractComputedStyles(child);
        if (Object.keys(childStyles).length > 0) {
          const selector = buildCssSelector(child);
          const key = cssMap[selector] ? `${selector}:nth(${count})` : selector;
          cssMap[key] = childStyles;
        }
        count++;
      }
      return {
        html,
        css: cssMap,
        dimensions: {
          width: rect.width,
          height: rect.height,
          padding: `${styles.paddingTop} ${styles.paddingRight} ${styles.paddingBottom} ${styles.paddingLeft}`,
          margin: `${styles.marginTop} ${styles.marginRight} ${styles.marginBottom} ${styles.marginLeft}`
        },
        textContent: (el.textContent || "").slice(0, 2e3).trim(),
        tagName: el.tagName.toLowerCase(),
        className: (typeof el.className === "string" ? el.className : "").trim()
      };
    }
    function extractFullPageData() {
      const clone = document.documentElement.cloneNode(true);
      const removeSelectors = [
        "script",
        "noscript",
        'iframe[src*="ads"]',
        'iframe[src*="track"]',
        'iframe[src*="pixel"]',
        'iframe[width="0"]',
        'iframe[height="0"]',
        'img[src*="pixel"]',
        'img[src*="track"]',
        'img[width="1"]',
        'img[height="1"]',
        '[id*="cookie"]',
        '[class*="cookie"]',
        '[id*="consent"]',
        '[class*="consent"]',
        '[id*="gdpr"]',
        '[class*="gdpr"]',
        '[id*="onetrust"]',
        '[class*="onetrust"]',
        '[id*="CybotCookiebot"]',
        '[data-testid*="cookie"]',
        '[id*="ad-"]',
        '[class*="ad-container"]',
        '[class*="ad-wrapper"]',
        'link[rel="preconnect"]',
        'link[rel="dns-prefetch"]',
        'meta[http-equiv="Content-Security-Policy"]',
        "style[data-emotion]"
        // runtime CSS-in-JS noise
      ];
      for (const sel of removeSelectors) {
        try {
          clone.querySelectorAll(sel).forEach((el) => el.remove());
        } catch (_e) {
        }
      }
      clone.querySelectorAll("*").forEach((el) => {
        const attrs = el.getAttributeNames();
        for (const attr of attrs) {
          if (attr.startsWith("on") || attr === "data-analytics" || attr === "data-tracking") {
            el.removeAttribute(attr);
          }
        }
      });
      const cleanHtml = clone.outerHTML;
      const cssMap = {};
      const keySelectors = [
        "body",
        "header",
        "nav",
        "main",
        "footer",
        "aside",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "a",
        "button",
        "input",
        "textarea",
        "ul",
        "ol",
        "li",
        "img",
        "form",
        "table",
        "th",
        "td",
        '[class*="hero"]',
        '[class*="card"]',
        '[class*="btn"]',
        '[class*="nav"]',
        '[class*="header"]',
        '[class*="footer"]',
        '[class*="sidebar"]',
        '[class*="container"]',
        '[class*="wrapper"]',
        '[class*="grid"]',
        '[class*="flex"]',
        '[class*="modal"]',
        '[class*="banner"]'
      ];
      let styleCount = 0;
      for (const sel of keySelectors) {
        if (styleCount >= 200) break;
        try {
          const els = document.querySelectorAll(sel);
          for (const el of els) {
            if (styleCount >= 200) break;
            const styles = extractComputedStyles(el);
            if (Object.keys(styles).length > 0) {
              const key = buildCssSelector(el);
              const finalKey = cssMap[key] ? `${key}:nth(${styleCount})` : key;
              cssMap[finalKey] = styles;
              styleCount++;
            }
          }
        } catch (_e) {
        }
      }
      const cssLines = [];
      for (const [selector, props] of Object.entries(cssMap)) {
        cssLines.push(`${selector} {`);
        for (const [prop, val] of Object.entries(props)) {
          cssLines.push(`  ${prop}: ${val};`);
        }
        cssLines.push("}");
        cssLines.push("");
      }
      const cssString = cssLines.join("\n");
      const colorSet = /* @__PURE__ */ new Set();
      const colorProps = ["color", "background-color", "border-color", "outline-color"];
      const sampleEls = document.querySelectorAll("*");
      let sampleCount = 0;
      for (const el of sampleEls) {
        if (sampleCount >= 500) break;
        const cs = window.getComputedStyle(el);
        for (const cp of colorProps) {
          const val = cs.getPropertyValue(cp);
          if (val && val !== "rgba(0, 0, 0, 0)" && val !== "transparent" && val !== "inherit" && val !== "initial") {
            colorSet.add(val);
          }
        }
        sampleCount++;
      }
      const colors = Array.from(colorSet).slice(0, 50);
      const fontSet = /* @__PURE__ */ new Set();
      for (const el of sampleEls) {
        if (fontSet.size >= 20) break;
        const cs = window.getComputedStyle(el);
        const ff = cs.getPropertyValue("font-family");
        if (ff) {
          const fonts2 = ff.split(",").map((f) => f.trim().replace(/^["']|["']$/g, ""));
          for (const font of fonts2) {
            if (font && !font.includes("inherit") && !font.includes("initial") && font.length < 50) {
              fontSet.add(font);
            }
          }
        }
      }
      const fonts = Array.from(fontSet).slice(0, 20);
      const getMeta = (name) => {
        const el = document.querySelector(`meta[property="${name}"], meta[name="${name}"]`);
        return (el == null ? void 0 : el.getAttribute("content")) || "";
      };
      const faviconEl = document.querySelector('link[rel="icon"], link[rel="shortcut icon"]');
      const metadata = {
        title: document.title || "",
        description: getMeta("description"),
        og_image: getMeta("og:image"),
        og_title: getMeta("og:title"),
        og_description: getMeta("og:description"),
        og_type: getMeta("og:type"),
        og_site_name: getMeta("og:site_name"),
        favicon: (faviconEl == null ? void 0 : faviconEl.getAttribute("href")) || ""
      };
      const viewport = {
        width: window.innerWidth,
        height: window.innerHeight
      };
      const mediaQueries = [];
      try {
        for (const sheet of document.styleSheets) {
          try {
            const rules = sheet.cssRules || sheet.rules;
            if (!rules) continue;
            for (const rule of rules) {
              if (rule instanceof CSSMediaRule && rule.conditionText) {
                if (!mediaQueries.includes(rule.conditionText)) {
                  mediaQueries.push(rule.conditionText);
                }
                if (mediaQueries.length >= 20) break;
              }
            }
          } catch (_e) {
          }
          if (mediaQueries.length >= 20) break;
        }
      } catch (_e) {
      }
      const images = [];
      document.querySelectorAll("img[src]").forEach((img) => {
        const src = img.getAttribute("src");
        if (src && !src.startsWith("data:") && images.length < 50) {
          try {
            images.push(new URL(src, location.href).href);
          } catch (_e) {
            images.push(src);
          }
        }
      });
      const stylesheets = [];
      document.querySelectorAll('link[rel="stylesheet"][href]').forEach((link) => {
        const href = link.getAttribute("href");
        if (href && stylesheets.length < 20) {
          try {
            stylesheets.push(new URL(href, location.href).href);
          } catch (_e) {
            stylesheets.push(href);
          }
        }
      });
      const elementCount = document.querySelectorAll("*").length;
      return {
        html: cleanHtml,
        css: cssString,
        css_map: cssMap,
        colors,
        fonts,
        metadata,
        source_url: location.href,
        viewport,
        asset_urls: { images, stylesheets },
        responsive_info: { viewport_width: viewport.width, media_queries: mediaQueries },
        element_count: elementCount
      };
    }
    function startCaptureMode() {
      if (_captureActive) return;
      _captureActive = true;
      _captureBanner = document.createElement("div");
      _captureBanner.className = "capture-banner";
      _captureBanner.innerHTML = '<span class="dot"></span> AURA Capture Mode — Click any element • Esc to exit';
      captureContainer.appendChild(_captureBanner);
      _captureOverlay = document.createElement("div");
      _captureOverlay.className = "capture-overlay";
      _captureOverlay.style.display = "none";
      captureContainer.appendChild(_captureOverlay);
      _captureTooltip = document.createElement("div");
      _captureTooltip.className = "capture-tooltip";
      _captureTooltip.style.display = "none";
      captureContainer.appendChild(_captureTooltip);
      captureHost.style.width = "100vw";
      captureHost.style.height = "100vh";
      document.addEventListener("mousemove", onCaptureMouseMove, true);
      document.addEventListener("click", onCaptureClick, true);
      document.addEventListener("keydown", onCaptureKeydown, true);
    }
    function stopCaptureMode() {
      if (!_captureActive) return;
      _captureActive = false;
      _captureHoveredEl = null;
      if (_captureOverlay) {
        _captureOverlay.remove();
        _captureOverlay = null;
      }
      if (_captureTooltip) {
        _captureTooltip.remove();
        _captureTooltip = null;
      }
      if (_captureBanner) {
        _captureBanner.remove();
        _captureBanner = null;
      }
      captureHost.style.width = "0";
      captureHost.style.height = "0";
      document.removeEventListener("mousemove", onCaptureMouseMove, true);
      document.removeEventListener("click", onCaptureClick, true);
      document.removeEventListener("keydown", onCaptureKeydown, true);
      safeSend({ type: "OPEN_PANEL", panel: "capture" });
    }
    function onCaptureMouseMove(e) {
      if (!_captureActive) return;
      const elements = document.elementsFromPoint(e.clientX, e.clientY);
      let target = null;
      for (const el of elements) {
        if (el === captureHost || captureHost.contains(el)) continue;
        if (el.id === "aura-host" || el.id === "aura-dock-shadow" || el.id === "aura-quick-action-host" || el.id === "aura-highlight-host" || el.id === "aura-img-toolbar-host" || el.id === "aura-capture-host") continue;
        if (el === document.documentElement || el === document.body) continue;
        target = el;
        break;
      }
      if (!target) {
        if (_captureOverlay) _captureOverlay.style.display = "none";
        if (_captureTooltip) _captureTooltip.style.display = "none";
        _captureHoveredEl = null;
        return;
      }
      _captureHoveredEl = target;
      const rect = target.getBoundingClientRect();
      if (_captureOverlay) {
        _captureOverlay.style.display = "block";
        _captureOverlay.style.top = rect.top + "px";
        _captureOverlay.style.left = rect.left + "px";
        _captureOverlay.style.width = rect.width + "px";
        _captureOverlay.style.height = rect.height + "px";
      }
      if (_captureTooltip) {
        const tag = target.tagName.toLowerCase();
        const cls = target.className && typeof target.className === "string" ? target.className.trim().split(/\s+/).slice(0, 3).join(" ") : "";
        const w = Math.round(rect.width);
        const h = Math.round(rect.height);
        _captureTooltip.innerHTML = `<span class="tag">&lt;${tag}&gt;</span>` + (cls ? `<span class="cls">.${cls.split(" ").join(".")}</span>` : "") + `<span class="dims">${w}x${h}</span>`;
        let tooltipTop = rect.top - 30;
        if (tooltipTop < 4) tooltipTop = rect.bottom + 6;
        let tooltipLeft = rect.left;
        if (tooltipLeft < 4) tooltipLeft = 4;
        _captureTooltip.style.display = "flex";
        _captureTooltip.style.top = tooltipTop + "px";
        _captureTooltip.style.left = tooltipLeft + "px";
      }
    }
    function onCaptureClick(e) {
      if (!_captureActive || !_captureHoveredEl) return;
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      const el = _captureHoveredEl;
      const rect = el.getBoundingClientRect();
      const data = captureElementData(el);
      try {
        ext.runtime.sendMessage(
          {
            type: "CAPTURE_ELEMENT",
            rect: {
              x: Math.round(rect.left),
              y: Math.round(rect.top),
              w: Math.round(rect.width),
              h: Math.round(rect.height)
            },
            elementData: data
          },
          (_response) => {
          }
        );
      } catch (_e) {
      }
      stopCaptureMode();
    }
    function onCaptureKeydown(e) {
      if (e.key === "Escape" && _captureActive) {
        e.preventDefault();
        e.stopPropagation();
        stopCaptureMode();
        try {
          ext.runtime.sendMessage({ type: "CAPTURE_MODE_EXITED" }).catch(() => {
          });
        } catch (_e) {
        }
      }
    }
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
        if (msg.type === "SHOW_DOCK") {
          showDock();
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "START_CAPTURE_MODE") {
          startCaptureMode();
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "STOP_CAPTURE_MODE") {
          stopCaptureMode();
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "EXTRACT_FULL_PAGE") {
          try {
            const data = extractFullPageData();
            sendResponse({ ok: true, data });
          } catch (err) {
            sendResponse({ ok: false, error: err.message || "Extraction failed" });
          }
          return false;
        }
        return void 0;
      }
    );
    const SERP_DEFAULT_BACKEND = "http://89.167.107.134";
    const SERP_DEFAULT_API_KEY = "i-L5ShpMkY2B7loNb8VS4EAAT-Ronh-K8cIgRILGjnQ";
    let SERP_BACKEND = SERP_DEFAULT_BACKEND;
    let SERP_API_KEY = SERP_DEFAULT_API_KEY;
    if ((_a = ext == null ? void 0 : ext.storage) == null ? void 0 : _a.local) {
      ext.storage.local.get(["backendUrl", "apiKey"], (d) => {
        var _a2, _b;
        if ((_a2 = d == null ? void 0 : d.backendUrl) == null ? void 0 : _a2.trim()) SERP_BACKEND = d.backendUrl.trim().replace(/\/+$/, "");
        if ((_b = d == null ? void 0 : d.apiKey) == null ? void 0 : _b.trim()) SERP_API_KEY = d.apiKey.trim();
      });
    }
    function isGoogleSearchPage() {
      const hostname = window.location.hostname;
      const pathname = window.location.pathname;
      const params = new URLSearchParams(window.location.search);
      if (!hostname.match(/^(www\.)?google\./)) return false;
      if (pathname !== "/search") return false;
      if (!params.get("q")) return false;
      const tbm = params.get("tbm");
      if (tbm && ["isch", "lcl", "vid", "shop", "nws", "bks", "fin"].includes(tbm)) return false;
      const udm = params.get("udm");
      if (udm && ["2", "14"].includes(udm)) return false;
      return true;
    }
    function getSearchQuery() {
      const params = new URLSearchParams(window.location.search);
      const qParam = params.get("q") || "";
      if (qParam) return qParam;
      const input = document.querySelector('input[name="q"]');
      return (input == null ? void 0 : input.value) || "";
    }
    function detectGoogleTheme() {
      const bg = window.getComputedStyle(document.body).backgroundColor;
      if (!bg || bg === "transparent") return "light";
      const rgbMatch = bg.match(/\d+/g);
      if (rgbMatch && rgbMatch.length >= 3) {
        const [r, g, b] = rgbMatch.map(Number);
        const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        return luminance < 128 ? "dark" : "light";
      }
      return "light";
    }
    async function initGoogleSerpIntegration() {
      if (!isGoogleSearchPage()) return;
      const stored = await new Promise((resolve) => {
        ext.storage.local.get(["aura_serp_hidden"], resolve);
      });
      if (stored.aura_serp_hidden) return;
      const query = getSearchQuery();
      if (!query) return;
      const serpHost = document.createElement("div");
      serpHost.id = "aura-serp-host";
      Object.assign(serpHost.style, {
        position: "fixed",
        top: "80px",
        right: "16px",
        width: "340px",
        maxHeight: "calc(100vh - 100px)",
        zIndex: "2147483640",
        pointerEvents: "auto"
      });
      document.documentElement.appendChild(serpHost);
      const serpShadow = serpHost.attachShadow({ mode: "closed" });
      const theme = detectGoogleTheme();
      const isDark = theme === "dark";
      const serpStyle = document.createElement("style");
      serpStyle.textContent = `
      @keyframes serp-fade-in {
        from { opacity: 0; transform: translateY(-8px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes serp-pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
      }

      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

      :host {
        display: block;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      }

      .serp-card {
        background: ${isDark ? "rgba(30, 27, 48, 0.92)" : "rgba(255, 255, 255, 0.95)"};
        backdrop-filter: blur(24px) saturate(1.4);
        -webkit-backdrop-filter: blur(24px) saturate(1.4);
        border-radius: 16px;
        overflow-y: auto;
        max-height: calc(100vh - 120px);
        box-shadow: ${isDark ? "0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06)" : "0 8px 40px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06)"};
        border: 1px solid ${isDark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.15)"};
        border-radius: 16px;
        padding: 20px 24px 16px;
        animation: serp-fade-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        box-shadow: ${isDark ? "0 4px 24px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255,255,255,0.04) inset" : "0 2px 16px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(0,0,0,0.03) inset"};
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease;
      }
      .serp-card:hover {
        border-color: ${isDark ? "rgba(124, 58, 237, 0.35)" : "rgba(124, 58, 237, 0.3)"};
      }

      .serp-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
      }
      .serp-header-left {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .serp-logo {
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: ${isDark ? "rgba(160, 148, 210, 0.9)" : "rgba(124, 58, 237, 0.85)"};
        background: ${isDark ? "rgba(124, 58, 237, 0.12)" : "rgba(124, 58, 237, 0.08)"};
        border-radius: 8px;
        flex-shrink: 0;
      }
      .serp-title {
        font-size: 14px;
        font-weight: 600;
        color: ${isDark ? "rgba(226, 232, 240, 0.9)" : "rgba(30, 27, 48, 0.9)"};
        letter-spacing: -0.01em;
      }
      .serp-title-sub {
        font-size: 11px;
        font-weight: 400;
        color: ${isDark ? "rgba(160, 148, 210, 0.5)" : "rgba(100, 90, 140, 0.6)"};
        margin-left: 6px;
      }

      .serp-controls {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .serp-ctrl-btn {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        border: none;
        background: transparent;
        color: ${isDark ? "rgba(160, 148, 210, 0.5)" : "rgba(100, 90, 140, 0.5)"};
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s, color 0.15s;
        padding: 0;
      }
      .serp-ctrl-btn:hover {
        background: ${isDark ? "rgba(124, 58, 237, 0.15)" : "rgba(124, 58, 237, 0.1)"};
        color: ${isDark ? "rgba(224, 214, 255, 1)" : "rgba(124, 58, 237, 0.9)"};
      }
      .serp-ctrl-btn[title="Hide AURA answers"]:hover {
        background: rgba(239, 68, 68, 0.12);
        color: rgba(239, 68, 68, 0.9);
      }

      .serp-body {
        font-size: 14px;
        line-height: 1.7;
        color: ${isDark ? "rgba(226, 232, 240, 0.85)" : "rgba(30, 27, 48, 0.85)"};
        overflow: hidden;
        transition: max-height 0.3s ease;
      }
      .serp-body.collapsed {
        max-height: 0 !important;
        margin: 0;
        padding: 0;
      }

      .serp-loading {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0;
      }
      .serp-loading-dots {
        display: flex;
        gap: 4px;
      }
      .serp-loading-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: ${isDark ? "rgba(124, 58, 237, 0.6)" : "rgba(124, 58, 237, 0.5)"};
        animation: serp-pulse 1.2s ease-in-out infinite;
      }
      .serp-loading-dots span:nth-child(2) { animation-delay: 0.2s; }
      .serp-loading-dots span:nth-child(3) { animation-delay: 0.4s; }
      .serp-loading-text {
        font-size: 13px;
        color: ${isDark ? "rgba(160, 148, 210, 0.6)" : "rgba(100, 90, 140, 0.6)"};
      }

      .serp-answer {
        white-space: pre-wrap;
        word-break: break-word;
      }
      .serp-answer p { margin-bottom: 8px; }
      .serp-answer p:last-child { margin-bottom: 0; }
      .serp-answer strong, .serp-answer b {
        font-weight: 600;
        color: ${isDark ? "rgba(226, 232, 240, 0.95)" : "rgba(30, 27, 48, 0.95)"};
      }
      .serp-answer code {
        background: ${isDark ? "rgba(124, 58, 237, 0.1)" : "rgba(124, 58, 237, 0.06)"};
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 13px;
        font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
      }
      .serp-answer ul, .serp-answer ol {
        padding-left: 20px;
        margin-bottom: 8px;
      }
      .serp-answer li { margin-bottom: 4px; }
      .serp-answer a {
        color: ${isDark ? "rgba(160, 148, 255, 0.9)" : "rgba(100, 58, 237, 0.9)"};
        text-decoration: none;
      }
      .serp-answer a:hover { text-decoration: underline; }

      .serp-citations {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"};
      }
      .serp-citations-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: ${isDark ? "rgba(160, 148, 210, 0.5)" : "rgba(100, 90, 140, 0.5)"};
        margin-bottom: 6px;
      }
      .serp-citation-list {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .serp-citation-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: ${isDark ? "rgba(124, 58, 237, 0.1)" : "rgba(124, 58, 237, 0.06)"};
        border: 1px solid ${isDark ? "rgba(124, 58, 237, 0.15)" : "rgba(124, 58, 237, 0.1)"};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: ${isDark ? "rgba(200, 180, 255, 0.8)" : "rgba(100, 58, 237, 0.8)"};
        text-decoration: none;
        transition: background 0.15s, border-color 0.15s;
        max-width: 280px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .serp-citation-chip:hover {
        background: ${isDark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.12)"};
        border-color: ${isDark ? "rgba(124, 58, 237, 0.3)" : "rgba(124, 58, 237, 0.2)"};
      }
      .serp-citation-num {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        background: ${isDark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.1)"};
        font-size: 10px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }

      .serp-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"};
      }
      .serp-followup-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: ${isDark ? "rgba(124, 58, 237, 0.12)" : "rgba(124, 58, 237, 0.08)"};
        border: 1px solid ${isDark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.15)"};
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        color: ${isDark ? "rgba(200, 180, 255, 0.9)" : "rgba(100, 58, 237, 0.9)"};
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.15s;
      }
      .serp-followup-btn:hover {
        background: ${isDark ? "rgba(124, 58, 237, 0.22)" : "rgba(124, 58, 237, 0.15)"};
        border-color: ${isDark ? "rgba(124, 58, 237, 0.35)" : "rgba(124, 58, 237, 0.3)"};
        transform: scale(1.01);
      }
      .serp-followup-btn:active { transform: scale(0.98); }
      .serp-powered {
        font-size: 11px;
        color: ${isDark ? "rgba(160, 148, 210, 0.35)" : "rgba(100, 90, 140, 0.35)"};
      }

      .serp-offline {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
      }
      .serp-offline-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(239, 68, 68, 0.6);
        flex-shrink: 0;
      }
      .serp-offline-text {
        font-size: 13px;
        color: ${isDark ? "rgba(226, 232, 240, 0.5)" : "rgba(30, 27, 48, 0.5)"};
      }

      .serp-error {
        font-size: 13px;
        color: ${isDark ? "rgba(239, 150, 150, 0.8)" : "rgba(200, 50, 50, 0.7)"};
        padding: 4px 0;
      }
    `;
      serpShadow.appendChild(serpStyle);
      const card = document.createElement("div");
      card.className = "serp-card";
      const serpHeader = document.createElement("div");
      serpHeader.className = "serp-header";
      const headerLeft = document.createElement("div");
      headerLeft.className = "serp-header-left";
      const serpLogo = document.createElement("div");
      serpLogo.className = "serp-logo";
      serpLogo.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;
      const titleWrap = document.createElement("div");
      const titleText = document.createElement("span");
      titleText.className = "serp-title";
      titleText.textContent = "AI Answer";
      const titleSub = document.createElement("span");
      titleSub.className = "serp-title-sub";
      titleSub.textContent = "by AURA";
      titleWrap.appendChild(titleText);
      titleWrap.appendChild(titleSub);
      headerLeft.appendChild(serpLogo);
      headerLeft.appendChild(titleWrap);
      const controls = document.createElement("div");
      controls.className = "serp-controls";
      const collapseBtn = document.createElement("button");
      collapseBtn.className = "serp-ctrl-btn";
      collapseBtn.title = "Collapse";
      collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`;
      const hideBtn = document.createElement("button");
      hideBtn.className = "serp-ctrl-btn";
      hideBtn.title = "Hide AURA answers";
      hideBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
      controls.appendChild(collapseBtn);
      controls.appendChild(hideBtn);
      serpHeader.appendChild(headerLeft);
      serpHeader.appendChild(controls);
      card.appendChild(serpHeader);
      const serpBody = document.createElement("div");
      serpBody.className = "serp-body";
      const serpLoading = document.createElement("div");
      serpLoading.className = "serp-loading";
      const serpDots = document.createElement("div");
      serpDots.className = "serp-loading-dots";
      serpDots.innerHTML = "<span></span><span></span><span></span>";
      const serpLoadingText = document.createElement("span");
      serpLoadingText.className = "serp-loading-text";
      serpLoadingText.textContent = `Thinking about "${query.slice(0, 60)}${query.length > 60 ? "..." : ""}"`;
      serpLoading.appendChild(serpDots);
      serpLoading.appendChild(serpLoadingText);
      serpBody.appendChild(serpLoading);
      card.appendChild(serpBody);
      serpShadow.appendChild(card);
      let isSerpCollapsed = false;
      collapseBtn.addEventListener("click", () => {
        isSerpCollapsed = !isSerpCollapsed;
        if (isSerpCollapsed) {
          serpBody.classList.add("collapsed");
          collapseBtn.title = "Expand";
          collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;
        } else {
          serpBody.classList.remove("collapsed");
          collapseBtn.title = "Collapse";
          collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`;
        }
      });
      hideBtn.addEventListener("click", () => {
        ext.storage.local.set({ aura_serp_hidden: true });
        serpHost.remove();
      });
      try {
        const fetchBody = JSON.stringify({
          message: query,
          conversation_id: "__serp_answer__",
          stream: false,
          system_context: `The user searched Google for: "${query}". Provide a concise, direct answer to their query. Be helpful and factual. Use markdown formatting sparingly — bold for emphasis, lists where appropriate. If you reference sources, format them as [Source Title](URL) and they will be rendered as citation chips. Keep the answer focused and under 200 words unless the topic requires more detail.`
        });
        const proxyResult = await new Promise((resolve) => {
          ext.runtime.sendMessage(
            { type: "SERP_FETCH", url: `${SERP_BACKEND}/api/chat`, body: fetchBody, apiKey: SERP_API_KEY },
            (response) => resolve(response)
          );
        });
        if (!(proxyResult == null ? void 0 : proxyResult.ok)) {
          throw new Error((proxyResult == null ? void 0 : proxyResult.error) || "Backend unreachable");
        }
        serpLoading.remove();
        const answerEl = document.createElement("div");
        answerEl.className = "serp-answer";
        serpBody.appendChild(answerEl);
        let fullText = "";
        const responseText = proxyResult.text || "";
        const lines = responseText.split("\n").filter((l) => l.trim());
        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.chunk) fullText += parsed.chunk;
            else if (parsed.response) fullText = parsed.response;
            else if (parsed.content) fullText = parsed.content;
          } catch {
            fullText += line;
          }
        }
        if (!fullText.trim() && responseText.trim()) {
          fullText = responseText;
        }
        answerEl.innerHTML = serpRenderMarkdown(fullText);
        if (!fullText.trim()) {
          answerEl.innerHTML = '<span class="serp-error">No response from AI.</span>';
          return;
        }
        serpAddCitations(serpBody, fullText);
        serpAddFooter(card, query, fullText);
      } catch (_err) {
        serpLoading.remove();
        const offline = document.createElement("div");
        offline.className = "serp-offline";
        const offDot = document.createElement("div");
        offDot.className = "serp-offline-dot";
        const offText = document.createElement("span");
        offText.className = "serp-offline-text";
        offText.textContent = "AURA is offline — start the backend to see AI answers";
        offline.appendChild(offDot);
        offline.appendChild(offText);
        serpBody.appendChild(offline);
      }
    }
    function serpAddCitations(bodyEl, fullText) {
      const citationRegex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
      const citations = [];
      let citMatch;
      while ((citMatch = citationRegex.exec(fullText)) !== null) {
        citations.push({ title: citMatch[1], url: citMatch[2] });
      }
      if (citations.length === 0) return;
      const citationsContainer = document.createElement("div");
      citationsContainer.className = "serp-citations";
      const citLabel = document.createElement("div");
      citLabel.className = "serp-citations-label";
      citLabel.textContent = "Sources";
      citationsContainer.appendChild(citLabel);
      const citList = document.createElement("div");
      citList.className = "serp-citation-list";
      citations.forEach((cit, idx) => {
        const chip = document.createElement("a");
        chip.className = "serp-citation-chip";
        chip.href = cit.url;
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
        const num = document.createElement("span");
        num.className = "serp-citation-num";
        num.textContent = String(idx + 1);
        chip.appendChild(num);
        const chipText = document.createTextNode(" " + cit.title);
        chip.appendChild(chipText);
        citList.appendChild(chip);
      });
      citationsContainer.appendChild(citList);
      bodyEl.appendChild(citationsContainer);
    }
    function serpAddFooter(cardEl, query, fullText) {
      const footer = document.createElement("div");
      footer.className = "serp-footer";
      const followupBtn = document.createElement("button");
      followupBtn.className = "serp-followup-btn";
      followupBtn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg> Ask follow-up`;
      followupBtn.addEventListener("click", () => {
        safeSend({
          type: "OPEN_WITH_TEXT",
          action: "ask",
          text: `I searched for "${query}" and got the following AI answer:

${fullText}

I have a follow-up question: `,
          url: window.location.href,
          title: document.title
        });
      });
      const powered = document.createElement("span");
      powered.className = "serp-powered";
      powered.textContent = "Powered by AURA";
      footer.appendChild(followupBtn);
      footer.appendChild(powered);
      cardEl.appendChild(footer);
    }
    function serpRenderMarkdown(text) {
      let html = serpEscapeHtml(text);
      html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      html = html.replace(/__(.+?)__/g, "<strong>$1</strong>");
      html = html.replace(new RegExp("(?<!\\*)\\*(?!\\*)(.+?)(?<!\\*)\\*(?!\\*)", "g"), "<em>$1</em>");
      html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
      html = html.replace(
        /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );
      html = html.replace(/^[\s]*[-*]\s+(.+)$/gm, "<li>$1</li>");
      html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");
      html = html.replace(/^[\s]*\d+\.\s+(.+)$/gm, "<li>$1</li>");
      html = html.replace(/\n\n+/g, "</p><p>");
      html = "<p>" + html + "</p>";
      html = html.replace(/\n/g, "<br>");
      html = html.replace(/<p>\s*<\/p>/g, "");
      return html;
    }
    function serpEscapeHtml(text) {
      return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
    initGoogleSerpIntegration();
  })();
})();
