/**
 * Quick Actions on Input Fields — extracted from page-services.ts
 * Self-contained DOM manipulation module for inline AI actions on text inputs.
 */

interface QuickActionDef {
  label: string;
  icon: string;
  action: string;
  language?: string;
}

interface QuickActionResponse {
  ok: boolean;
  result?: string;
  error?: string;
}

const QUICK_ACTIONS_DEF: QuickActionDef[] = [
  { label: 'Improve', icon: '<path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/>', action: 'improve' },
  { label: 'Expand', icon: '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>', action: 'expand' },
  { label: 'Shorten', icon: '<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>', action: 'shorten' },
  { label: 'Fix grammar', icon: '<polyline points="20 6 9 17 4 12"/>', action: 'fix_grammar' },
  { label: 'Translate', icon: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>', action: 'translate' },
];

// Input types to skip
const SKIP_INPUT_TYPES = new Set(['password', 'hidden', 'file', 'checkbox', 'radio', 'range', 'color', 'date', 'datetime-local', 'month', 'week', 'time', 'submit', 'reset', 'button', 'image']);
const MIN_INPUT_WIDTH = 200;

export function initQuickActionsOnInputs(ext: typeof chrome, safeSend: (msg: any, cb?: (r: any) => void) => void, showToast: (msg: string, duration?: number) => void): void {
  // Shadow DOM host for all quick-action UI
  const qaHost: HTMLDivElement = document.createElement('div');
  qaHost.id = 'aura-quick-action-host';
  Object.assign(qaHost.style, {
    position: 'fixed',
    top: '0',
    left: '0',
    zIndex: '2147483646',
    pointerEvents: 'none',
  });
  document.documentElement.appendChild(qaHost);
  const qaShadow: ShadowRoot = qaHost.attachShadow({ mode: 'closed' });

  const qaStyle: HTMLStyleElement = document.createElement('style');
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

  // Container inside the shadow for dynamic elements
  const qaContainer: HTMLDivElement = document.createElement('div');
  qaShadow.appendChild(qaContainer);

  // Track currently attached input and UI elements
  let _qaActiveInput: HTMLInputElement | HTMLTextAreaElement | null = null;
  let _qaTriggerEl: HTMLDivElement | null = null;
  let _qaMenuEl: HTMLDivElement | null = null;
  let _qaTranslateSub: HTMLDivElement | null = null;

  function isEligibleInput(el: Element): el is HTMLInputElement | HTMLTextAreaElement {
    if (el.tagName === 'TEXTAREA') return true;
    if (el.tagName === 'INPUT') {
      const inputEl = el as HTMLInputElement;
      const inputType = (inputEl.type || 'text').toLowerCase();
      if (SKIP_INPUT_TYPES.has(inputType)) return false;
      return true;
    }
    // contenteditable
    if ((el as HTMLElement).isContentEditable && el.getAttribute('contenteditable') === 'true') return true;
    return false;
  }

  function removeQaMenu(): void {
    if (_qaMenuEl) { _qaMenuEl.remove(); _qaMenuEl = null; }
    if (_qaTranslateSub) { _qaTranslateSub.remove(); _qaTranslateSub = null; }
  }

  function removeQaTrigger(): void {
    if (_qaTriggerEl) { _qaTriggerEl.remove(); _qaTriggerEl = null; }
    removeQaMenu();
    _qaActiveInput = null;
  }

  function positionTrigger(field: HTMLElement): void {
    const rect = field.getBoundingClientRect();
    if (rect.width < MIN_INPUT_WIDTH) { removeQaTrigger(); return; }

    // Hide trigger when the input is scrolled offscreen
    const offscreen = rect.bottom < 0 || rect.top > window.innerHeight ||
                      rect.right < 0 || rect.left > window.innerWidth;

    if (!_qaTriggerEl) {
      _qaTriggerEl = document.createElement('div');
      _qaTriggerEl.className = 'qa-trigger';
      _qaTriggerEl.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>`;
      _qaTriggerEl.addEventListener('click', (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (_qaMenuEl) { removeQaMenu(); return; }
        showQaMenu();
      });
      qaContainer.appendChild(_qaTriggerEl);
    }

    // Toggle visibility based on whether the field is in the viewport
    _qaTriggerEl.style.display = offscreen ? 'none' : '';

    if (offscreen) {
      removeQaMenu(); // also hide the menu if open
      return;
    }

    // Position at the right edge of the field, vertically centered
    const trigSize = 20;
    const pad = 6;
    _qaTriggerEl.style.top = `${Math.round(rect.top + (rect.height - trigSize) / 2)}px`;
    _qaTriggerEl.style.left = `${Math.round(rect.right - trigSize - pad)}px`;
  }

  function getInputValue(el: HTMLInputElement | HTMLTextAreaElement): string {
    if ((el as HTMLElement).isContentEditable) {
      return (el as HTMLElement).innerText || '';
    }
    return el.value || '';
  }

  function setInputValue(el: HTMLInputElement | HTMLTextAreaElement, value: string): void {
    if ((el as HTMLElement).isContentEditable) {
      (el as HTMLElement).innerText = value;
    } else {
      el.value = value;
    }
    // Fire events so frameworks pick up the change
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function executeQuickAction(action: string, language?: string): void {
    if (!_qaActiveInput) return;
    const text = getInputValue(_qaActiveInput);
    if (!text.trim()) { removeQaMenu(); return; }

    // Show loading state on all menu items
    if (_qaMenuEl) {
      _qaMenuEl.querySelectorAll('.qa-menu-item').forEach((item) => {
        item.classList.add('loading');
      });
    }

    const targetField = _qaActiveInput;

    safeSend(
      { type: 'QUICK_ACTION', action, text, language },
      (response: QuickActionResponse) => {
        if (response && response.ok && response.result) {
          setInputValue(targetField, response.result);
          showToast('Text updated by AURA');
        } else {
          showToast(response?.error || 'Quick action failed', 3000);
        }
        removeQaMenu();
      }
    );
  }

  function toggleTranslateSub(anchor: HTMLButtonElement): void {
    if (_qaTranslateSub) { _qaTranslateSub.remove(); _qaTranslateSub = null; return; }

    const LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Chinese', 'Russian', 'Japanese', 'Arabic', 'Portuguese', 'Azerbaijani'];

    _qaTranslateSub = document.createElement('div');
    _qaTranslateSub.className = 'qa-translate-sub';

    LANGUAGES.forEach((lang) => {
      const item = document.createElement('button');
      item.className = 'qa-menu-item';
      item.textContent = lang;
      item.addEventListener('click', (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        executeQuickAction('translate', lang);
      });
      _qaTranslateSub!.appendChild(item);
    });

    // Insert after the translate button inside the menu
    if (_qaMenuEl && anchor.parentNode === _qaMenuEl) {
      anchor.after(_qaTranslateSub);
    }
  }

  function showQaMenu(): void {
    if (!_qaTriggerEl || !_qaActiveInput) return;
    removeQaMenu();

    _qaMenuEl = document.createElement('div');
    _qaMenuEl.className = 'qa-menu';

    QUICK_ACTIONS_DEF.forEach((qa) => {
      const item = document.createElement('button');
      item.className = 'qa-menu-item';
      item.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${qa.icon}</svg><span>${qa.label}</span>`;
      item.addEventListener('click', (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (qa.action === 'translate') {
          toggleTranslateSub(item as HTMLButtonElement);
        } else {
          executeQuickAction(qa.action);
        }
      });
      _qaMenuEl!.appendChild(item);
    });

    qaContainer.appendChild(_qaMenuEl);

    // Position below the trigger
    const trigRect = _qaTriggerEl.getBoundingClientRect();
    const menuGap = 6;
    _qaMenuEl.style.top = `${Math.round(trigRect.bottom + menuGap)}px`;
    _qaMenuEl.style.left = `${Math.round(trigRect.right - 150)}px`; // right-align roughly

    // Clamp to viewport
    requestAnimationFrame(() => {
      if (!_qaMenuEl) return;
      const mRect = _qaMenuEl.getBoundingClientRect();
      if (mRect.right > window.innerWidth - 8) {
        _qaMenuEl.style.left = `${Math.round(window.innerWidth - mRect.width - 8)}px`;
      }
      if (mRect.left < 8) {
        _qaMenuEl.style.left = '8px';
      }
      if (mRect.bottom > window.innerHeight - 8) {
        // Show above trigger instead
        _qaMenuEl.style.top = `${Math.round(trigRect.top - mRect.height - menuGap)}px`;
      }
    });
  }

  // Attach trigger on focus
  function onFieldFocus(e: FocusEvent): void {
    const target = e.target as Element;
    if (!target || !isEligibleInput(target)) return;
    const rect = target.getBoundingClientRect();
    if (rect.width < MIN_INPUT_WIDTH) return;

    _qaActiveInput = target as HTMLInputElement | HTMLTextAreaElement;
    positionTrigger(target as HTMLElement);
  }

  // Remove trigger on blur (with small delay so menu clicks register)
  function onFieldBlur(_e: FocusEvent): void {
    setTimeout(() => {
      // If focus moved to our menu, don't remove
      if (_qaMenuEl || _qaTranslateSub) return;
      // Check if the new active element is the same input
      const active = document.activeElement;
      if (active && active === _qaActiveInput) return;
      removeQaTrigger();
    }, 200);
  }

  // Reposition trigger on scroll/resize
  function repositionQaTrigger(): void {
    if (_qaActiveInput) {
      positionTrigger(_qaActiveInput as HTMLElement);
    }
  }

  // Close menu when clicking outside
  document.addEventListener('mousedown', (e: MouseEvent) => {
    if (!_qaMenuEl && !_qaTriggerEl) return;
    // Check if click is inside our shadow DOM
    const path = e.composedPath();
    if (path.includes(qaHost)) return;
    removeQaMenu();
  }, true);

  // Listen for focus/blur on the document (delegation)
  document.addEventListener('focusin', onFieldFocus, true);
  document.addEventListener('focusout', onFieldBlur, true);
  window.addEventListener('scroll', repositionQaTrigger, { passive: true });
  window.addEventListener('resize', repositionQaTrigger, { passive: true });

  // MutationObserver: detect if input is removed while trigger is visible
  const qaObserver = new MutationObserver(() => {
    if (_qaActiveInput && !document.body.contains(_qaActiveInput)) {
      removeQaTrigger();
    }
  });
  qaObserver.observe(document.body, { childList: true, subtree: true });
}