import type { ContentModule, ContextStore, ContextSignal } from './types';
import { crossFade } from './animator';
import { FAB, ANIM, Z_TOP, FONT_STACK } from './tokens';

// ── Action button definitions ─────────────────────────────────────────────────

interface ActionDef {
  action: string;
  tip: string;
  svg: string;
}

const DEFAULT_ACTIONS: ActionDef[] = [
  {
    action: 'chat',
    tip: 'Chat',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M2 2h12a1 1 0 011 1v8a1 1 0 01-1 1H5l-3 3V3a1 1 0 011-1z"/>
    </svg>`,
  },
  {
    action: 'search',
    tip: 'Search',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M6.5 1a5.5 5.5 0 014.23 9.02l3.12 3.12-1.06 1.06-3.12-3.12A5.5 5.5 0 116.5 1zm0 1.5a4 4 0 100 8 4 4 0 000-8z"/>
    </svg>`,
  },
  {
    action: 'page',
    tip: 'This Page',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 2h10a1 1 0 011 1v10a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1zm1 3v1h8V5H4zm0 3v1h8V8H4zm0 3v1h5v-1H4z"/>
    </svg>`,
  },
  {
    action: 'translate',
    tip: 'Translate',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M1 2h7v1.5H5.5v1H8v1.5H5.5c-.2 1-.7 2-1.5 2.7l1.7 1.8-1.1 1-1.6-1.8C2.5 10 2 10.2 1.5 10.3L1 8.8c.5-.1.9-.3 1.3-.5L1 6.8l1.1-1 1.2 1.4c.5-.5.9-1.1 1.1-1.7H1V2zm10 3l3 8h-1.5l-.6-1.7h-2.8L8.5 13H7l3-8h1zm-.5 2.5l-1 2.8h2l-1-2.8z"/>
    </svg>`,
  },
  {
    action: 'save',
    tip: 'Save to Memory',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 1h10a1 1 0 011 1v13l-6-3-6 3V2a1 1 0 011-1z"/>
    </svg>`,
  },
];

// ── Inline styles helper ──────────────────────────────────────────────────────

// Keyframes for glow pulse are defined in styles.ts (shadow DOM stylesheet)

// ── createFab ─────────────────────────────────────────────────────────────────

export function createFab(): ContentModule & { showDock(): void } {
  let _root: HTMLElement | null = null;
  let _pill: HTMLElement | null = null;
  let _glow: HTMLElement | null = null;
  let _logo: HTMLElement | null = null;
  let _popout: HTMLElement | null = null;
  let _closeBtn: HTMLElement | null = null;

  let _unsub: (() => void) | null = null;
  let _ext: typeof chrome | null = null;

  // Drag state
  let _side: 'left' | 'right' = 'right';
  let _offset: number = 40; // % from top
  let _dragStartX = 0;
  let _dragStartY = 0;
  let _isDragging = false;
  let _totalMove = 0;

  // Hover / visibility
  let _hiddenOnce = false;
  let _hoverTimer: ReturnType<typeof setTimeout> | null = null;

  // Cross-fade guard
  let _morphing = false;

  // ── DOM builders ────────────────────────────────────────────────────────────

  function buildGlow(): HTMLElement {
    const glow = document.createElement('div');
    glow.className = 'fab-glow';
    Object.assign(glow.style, {
      position: 'absolute',
      inset: '-8px',
      borderRadius: '50px',
      background: 'var(--aura-glow)',
      filter: 'blur(12px)',
      animation: 'aura-glow-pulse 3s ease-in-out infinite',
      pointerEvents: 'none',
      zIndex: '-1',
    });
    return glow;
  }

  function buildLogo(iconSvg: string): HTMLElement {
    const logo = document.createElement('div');
    logo.className = 'fab-logo';
    Object.assign(logo.style, {
      width: `${FAB.logoSize}px`,
      height: `${FAB.logoSize}px`,
      color: 'var(--aura-accent)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: '0',
      transition: 'color 0.3s ease',
    });
    logo.innerHTML = iconSvg;
    // Ensure SVG fills container
    const svg = logo.querySelector('svg');
    if (svg) {
      svg.style.width = '100%';
      svg.style.height = '100%';
    }
    return logo;
  }

  function buildCloseBtn(): HTMLElement {
    const btn = document.createElement('button');
    btn.className = 'fab-close';
    btn.setAttribute('aria-label', 'Close Aura');
    btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12" fill="currentColor" width="10" height="10">
      <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    </svg>`;
    Object.assign(btn.style, {
      position: 'absolute',
      top: '-6px',
      right: '-6px',
      width: '16px',
      height: '16px',
      borderRadius: '50%',
      background: 'rgba(10,8,24,0.9)',
      border: '1px solid rgba(255,255,255,0.15)',
      color: 'rgba(255,255,255,0.6)',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '0',
      opacity: '0',
      transition: 'opacity 0.2s',
      pointerEvents: 'all',
    });
    return btn;
  }

  function buildActionBtn(def: ActionDef): HTMLElement {
    const btn = document.createElement('button');
    btn.className = 'fab-action-btn';
    btn.dataset.action = def.action;
    btn.setAttribute('aria-label', def.tip);
    btn.innerHTML = def.svg;
    Object.assign(btn.style, {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '4px',
      background: 'transparent',
      border: 'none',
      color: 'rgba(255,255,255,0.75)',
      cursor: 'pointer',
      padding: '6px 8px',
      borderRadius: '8px',
      fontSize: '10px',
      fontFamily: FONT_STACK,
      transition: 'background 0.15s, color 0.15s',
    });
    const svg = btn.querySelector('svg');
    if (svg) {
      svg.setAttribute('width', '16');
      svg.setAttribute('height', '16');
    }

    // Tooltip
    const tip = document.createElement('span');
    tip.textContent = def.tip;
    tip.style.fontSize = '10px';
    btn.appendChild(tip);

    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(255,255,255,0.08)';
      btn.style.color = 'var(--aura-accent)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'transparent';
      btn.style.color = 'rgba(255,255,255,0.75)';
    });

    return btn;
  }

  function buildPopout(): HTMLElement {
    const popout = document.createElement('div');
    popout.className = 'fab-popout hidden';
    Object.assign(popout.style, {
      position: 'absolute',
      display: 'flex',
      flexDirection: 'row',
      gap: '4px',
      padding: '8px',
      background: 'rgba(10,8,24,0.92)',
      backdropFilter: 'blur(20px) saturate(1.5)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: '14px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
      zIndex: String(Z_TOP),
      transition: 'opacity 0.2s, transform 0.2s',
      opacity: '0',
      pointerEvents: 'none',
    });

    for (const def of DEFAULT_ACTIONS) {
      popout.appendChild(buildActionBtn(def));
    }

    return popout;
  }

  function buildPill(signal: ContextSignal): HTMLElement {
    const pill = document.createElement('div');
    pill.className = 'fab-pill';
    Object.assign(pill.style, {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: FAB.pillPadding,
      background: 'rgba(10,8,24,0.88)',
      backdropFilter: 'blur(20px) saturate(1.5)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: '50px',
      cursor: 'pointer',
      position: 'relative',
      boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
      transition: `padding ${FAB.expandDuration}ms ease, border-radius ${FAB.expandDuration}ms ease`,
      userSelect: 'none',
      touchAction: 'none',
    });

    _glow = buildGlow();
    pill.appendChild(_glow);

    _logo = buildLogo(signal.icon);
    pill.appendChild(_logo);

    _closeBtn = buildCloseBtn();
    pill.appendChild(_closeBtn);

    return pill;
  }

  // ── Positioning ──────────────────────────────────────────────────────────────

  function applyPosition(): void {
    if (!_root || !_pill) return;
    const margin = FAB.edgeMargin;
    Object.assign(_root.style, {
      position: 'fixed',
      top: `${_offset}%`,
      [_side === 'right' ? 'right' : 'left']: `${margin}px`,
      [_side === 'right' ? 'left' : 'right']: 'auto',
      zIndex: String(Z_TOP),
      transform: '',
    });
    positionPopout();
  }

  function positionPopout(): void {
    if (!_popout || !_root) return;
    const isRight = _side === 'right';

    Object.assign(_popout.style, {
      top: '50%',
      transform: 'translateY(-50%)',
      [isRight ? 'right' : 'left']: `calc(100% + 8px)`,
      [isRight ? 'left' : 'right']: 'auto',
    });
  }

  // ── Hover handling ───────────────────────────────────────────────────────────

  function showPopout(): void {
    if (!_popout || !_pill) return;
    if (_hoverTimer) { clearTimeout(_hoverTimer); _hoverTimer = null; }

    _popout.classList.remove('hidden');
    _popout.style.opacity = '1';
    _popout.style.pointerEvents = 'all';
    _pill.style.borderBottomRightRadius = _side === 'right' ? '50px' : '50px';
    _closeBtn && (_closeBtn.style.opacity = '1');
  }

  function hidePopout(): void {
    if (!_popout || !_pill) return;
    _popout.style.opacity = '0';
    _popout.style.pointerEvents = 'none';
    _hoverTimer = setTimeout(() => {
      _popout!.classList.add('hidden');
      _closeBtn && (_closeBtn.style.opacity = '0');
    }, 200);
  }

  function setupHover(pill: HTMLElement, popout: HTMLElement): void {
    let insidePill = false;
    let insidePopout = false;

    function maybeHide() {
      setTimeout(() => {
        if (!insidePill && !insidePopout) hidePopout();
      }, 0);
    }

    pill.addEventListener('mouseenter', () => { insidePill = true; showPopout(); });
    pill.addEventListener('mouseleave', () => { insidePill = false; maybeHide(); });
    popout.addEventListener('mouseenter', () => { insidePopout = true; showPopout(); });
    popout.addEventListener('mouseleave', () => { insidePopout = false; maybeHide(); });
  }

  // ── Drag handling ────────────────────────────────────────────────────────────

  function setupDrag(pill: HTMLElement): void {
    pill.addEventListener('pointerdown', (e: PointerEvent) => {
      if ((e.target as HTMLElement).closest('.fab-close')) return;
      _dragStartX = e.clientX;
      _dragStartY = e.clientY;
      _isDragging = false;
      _totalMove = 0;
      pill.setPointerCapture(e.pointerId);
    });

    pill.addEventListener('pointermove', (e: PointerEvent) => {
      if (!pill.hasPointerCapture(e.pointerId)) return;
      const dx = e.clientX - _dragStartX;
      const dy = e.clientY - _dragStartY;
      _totalMove = Math.sqrt(dx * dx + dy * dy);

      if (_totalMove > FAB.dragThreshold) {
        _isDragging = true;
        pill.classList.add('dragging');
        pill.style.borderRadius = '50%';

        // Move vertically: compute new offset %
        const vpHeight = window.innerHeight;
        const newTopPx = e.clientY;
        const clampedPct = Math.min(Math.max((newTopPx / vpHeight) * 100, 5), 90);
        if (_root) {
          _root.style.top = `${clampedPct}%`;
        }
      }
    });

    pill.addEventListener('pointerup', (e: PointerEvent) => {
      if (!_isDragging) return;

      pill.classList.remove('dragging');
      pill.style.borderRadius = '50px';

      // Snap to nearest edge
      const vpWidth = window.innerWidth;
      _side = e.clientX > vpWidth / 2 ? 'right' : 'left';

      // Compute final offset %
      const vpHeight = window.innerHeight;
      _offset = Math.min(Math.max((e.clientY / vpHeight) * 100, 5), 90);

      _isDragging = false;
      applyPosition();
      savePersistence();
    });
  }

  // ── Click handling ───────────────────────────────────────────────────────────

  function setupClicks(pill: HTMLElement, popout: HTMLElement): void {
    pill.addEventListener('click', (e: MouseEvent) => {
      if ((e.target as HTMLElement).closest('.fab-close')) return;
      if (_isDragging || _totalMove > FAB.dragThreshold) return; // was a drag
      if (!_ext) return;
      _ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'chat' });
    });

    _closeBtn?.addEventListener('click', (e: MouseEvent) => {
      e.stopPropagation();
      _hiddenOnce = true;
      if (_root) _root.style.display = 'none';
    });

    popout.addEventListener('click', (e: MouseEvent) => {
      const btn = (e.target as HTMLElement).closest<HTMLElement>('.fab-action-btn');
      if (!btn || !_ext) return;
      const action = btn.dataset.action ?? '';
      dispatchAction(action);
    });
  }

  function getSelectionText(): string {
    const sel = window.getSelection();
    return sel ? sel.toString().trim() : '';
  }

  function dispatchAction(action: string): void {
    if (!_ext) return;
    const url = location.href;
    const title = document.title;
    switch (action) {
      case 'chat':
        _ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'chat' });
        break;
      case 'search':
        _ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'search' });
        break;
      case 'page':
        // extractMainContent is handled by page-services; send OPEN_WITH_TEXT
        // and let sidebar request content extraction via EXTRACT_PAGE
        _ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'ask' });
        break;
      case 'translate':
        _ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'translate' });
        break;
      case 'save': {
        const selText = getSelectionText();
        const textToSave = selText || `${title}\n${url}`;
        _ext.runtime.sendMessage(
          { type: 'SAVE_KNOWLEDGE', text: textToSave, url, title },
          (response: any) => {
            // Toast is handled by coordinator if wired
          }
        );
        break;
      }
    }
  }

  // ── Context subscription ─────────────────────────────────────────────────────

  function handleContextChange(signal: ContextSignal): void {
    if (!_root || !_logo) return;

    // Update CSS vars
    _root.style.setProperty('--aura-accent', signal.accent);
    _root.style.setProperty('--aura-glow', signal.glow);

    // Cross-fade the logo icon
    if (!_morphing && _logo) {
      _morphing = true;
      const oldLogo = _logo;
      const newLogo = buildLogo(signal.icon);
      newLogo.style.position = 'absolute';
      newLogo.style.inset = '0';
      newLogo.style.opacity = '0';

      oldLogo.parentElement?.appendChild(newLogo);

      crossFade(oldLogo, newLogo, {
        duration: ANIM.crossFadeDuration,
        easing: 'ease',
      }).then(() => {
        oldLogo.remove();
        newLogo.style.position = '';
        newLogo.style.inset = '';
        newLogo.style.opacity = '1';
        _logo = newLogo;
        _morphing = false;
      });
    }
  }

  // ── Persistence ──────────────────────────────────────────────────────────────

  function loadPersistence(): void {
    if (!_ext) return;
    _ext.storage.local.get(['auraFabSide', 'auraFabOffset'], (result: Record<string, any>) => {
      if (result.auraFabSide === 'left' || result.auraFabSide === 'right') {
        _side = result.auraFabSide;
      }
      if (typeof result.auraFabOffset === 'number') {
        _offset = result.auraFabOffset;
      }
      applyPosition();
    });
  }

  function savePersistence(): void {
    if (!_ext) return;
    _ext.storage.local.set({ auraFabSide: _side, auraFabOffset: _offset });
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  return {
    init(container: HTMLElement, store: ContextStore, ext: typeof chrome): void {
      _ext = ext;

      const signal = store.get();

      // Root container
      const root = document.createElement('div');
      root.className = 'aura-fab';
      Object.assign(root.style, {
        position: 'fixed',
        zIndex: String(Z_TOP),
        fontFamily: FONT_STACK,
        // CSS custom props for theming
        '--aura-accent': signal.accent,
        '--aura-glow': signal.glow,
      } as any);
      _root = root;

      // Build pill
      const pill = buildPill(signal);
      _pill = pill;
      root.appendChild(pill);

      // Build popout (sibling to pill inside root)
      const popout = buildPopout();
      _popout = popout;
      root.appendChild(popout);

      container.appendChild(root);

      // Wire up interactions
      setupHover(pill, popout);
      setupDrag(pill);
      setupClicks(pill, popout);

      // Load saved position
      loadPersistence();

      // Subscribe to context changes
      _unsub = store.subscribe(handleContextChange);
    },

    destroy(): void {
      if (_unsub) { _unsub(); _unsub = null; }
      if (_hoverTimer) { clearTimeout(_hoverTimer); _hoverTimer = null; }
      _root?.remove();
      _root = null;
      _pill = null;
      _glow = null;
      _logo = null;
      _popout = null;
      _closeBtn = null;
      _ext = null;
    },

    showDock(): void {
      _hiddenOnce = false;
      if (_root) _root.style.display = '';
    },
  };
}
