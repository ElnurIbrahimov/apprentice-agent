/**
 * Gmail module — Gmail AI Compose Integration
 * Migrated from content.ts lines ~2987-3565
 */

import type { ContentModule } from './types';

interface QuickActionOutMessage {
  type: 'QUICK_ACTION';
  action: string;
  text: string;
  language?: string;
  threadContext?: string;
}

interface QuickActionResponse {
  ok: boolean;
  result?: string;
  error?: string;
}

interface GmailComposeTracker {
  composeEl: HTMLElement;
  buttonHost: HTMLDivElement;
  shadow: ShadowRoot;
  observer: MutationObserver;
  outsideHandler: (e: MouseEvent) => void;
}

interface GmailMenuItemDef {
  icon: string;
  label: string;
  action: string;
  separator?: boolean;
}

const GMAIL_HOST = 'mail.google.com';

export function createGmail(): ContentModule {
  let _ext!: typeof chrome;
  let _safeSend: (msg: any, cb?: (r: any) => void) => void = () => {};

  const _gmailTrackedComposes = new Map<HTMLElement, GmailComposeTracker>();

  function isGmailPage(): boolean {
    return window.location.hostname === GMAIL_HOST;
  }

  /**
   * Extracts the email thread text from the Gmail DOM.
   * Gmail renders email bodies inside `.a3s.aiL` elements.
   */
  function extractGmailThreadText(): string {
    const bodies = document.querySelectorAll<HTMLElement>('.a3s.aiL');
    if (bodies.length === 0) return '';
    const parts: string[] = [];
    bodies.forEach((body) => {
      const text = body.innerText?.trim();
      if (text) parts.push(text);
    });
    return parts.join('\n\n---\n\n').slice(0, 20000);
  }

  /**
   * Gets the compose body's contenteditable div within a compose window.
   */
  function getComposeBody(composeEl: HTMLElement): HTMLElement | null {
    // Locale-specific aria-labels for Gmail compose body
    const ariaLabels = [
      'Message Body',           // English
      'Nachrichtentext',        // German
      'Corps du message',       // French
      'Cuerpo del mensaje',     // Spanish
      'Corpo da mensagem',      // Portuguese
      'Corpo del messaggio',    // Italian
      '\u0422\u0435\u043a\u0441\u0442 \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u044f',          // Russian
      'Mesaj Metni',            // Turkish
      '\u30e1\u30c3\u30bb\u30fc\u30b8\u672c\u6587',            // Japanese
      '\uba54\uc2dc\uc9c0 \ubcf8\ubb38',              // Korean
      '\u90ae\u4ef6\u6b63\u6587',                // Chinese Simplified
      '\u0646\u0635 \u0627\u0644\u0631\u0633\u0627\u0644\u0629',            // Arabic
      'Berichttekst',           // Dutch
      'Tre\u015b\u0107 wiadomo\u015bci',      // Polish
      '\u0938\u0902\u0926\u0947\u0936 \u0915\u093e \u092e\u0941\u0916\u094d\u092f \u092d\u093e\u0917',    // Hindi
      'Mesaj m\u0259tni',           // Azerbaijani
    ];
    const ariaSelector = ariaLabels.map(l => `div[aria-label="${l}"]`).join(', ');

    // Try aria-label matches first, then generic Gmail selectors
    const result = composeEl.querySelector<HTMLElement>(
      ariaSelector + ', ' +
      'div[g_editable="true"][contenteditable="true"], ' +
      'div.editable[contenteditable="true"]'
    );
    if (result) return result;

    // Fallback: contenteditable textbox inside the compose dialog (covers unlisted locales)
    return composeEl.querySelector<HTMLElement>('div[contenteditable="true"][role="textbox"]');
  }

  /**
   * Gets text content from a Gmail compose body.
   */
  function getComposeText(composeEl: HTMLElement): string {
    const body = getComposeBody(composeEl);
    if (!body) return '';
    return body.innerText?.trim() || '';
  }

  /** Escape HTML special characters to prevent XSS in innerHTML contexts. */
  function escapeHtml(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /**
   * Sets content in a Gmail compose body, undo-friendly.
   */
  function setComposeText(composeEl: HTMLElement, text: string): void {
    const body = getComposeBody(composeEl);
    if (!body) return;

    body.focus();

    // Select all existing content
    const sel = window.getSelection();
    if (sel) {
      const range = document.createRange();
      range.selectNodeContents(body);
      sel.removeAllRanges();
      sel.addRange(range);
    }

    // Try execCommand for undo-friendly insertion
    const success = document.execCommand('insertText', false, text);
    if (!success) {
      // Fallback: set innerHTML with line breaks (escape each line to prevent XSS)
      body.innerHTML = text.split('\n').map((line) =>
        `<div>${escapeHtml(line) || '<br>'}</div>`
      ).join('');
    }

    // Trigger Gmail's internal change detection
    body.dispatchEvent(new Event('input', { bubbles: true }));
    body.dispatchEvent(new Event('change', { bubbles: true }));
  }

  /**
   * Creates the Aura AI button Shadow DOM host and injects it into
   * a Gmail compose window's toolbar area.
   */
  function injectGmailAiButton(composeEl: HTMLElement): void {
    if (_gmailTrackedComposes.has(composeEl)) return;

    // Find the Send button (multi-language support for aria-label)
    const sendBtn = composeEl.querySelector<HTMLElement>(
      'div[aria-label*="Send"], div[data-tooltip*="Send"], ' +
      'div[aria-label*="Enviar"], div[aria-label*="Envoyer"], ' +
      'div[aria-label*="Senden"], div[aria-label*="\u041E\u0442\u043F\u0440\u0430\u0432\u0438\u0442\u044C"]'
    );

    // Alternative: find the bottom toolbar row
    const toolbarRow = composeEl.querySelector<HTMLElement>(
      '.btC, .bAK, tr.btC, .IZ'
    );

    const insertTarget = sendBtn?.parentElement || toolbarRow;
    if (!insertTarget) return;

    // Create shadow DOM host for our button
    const buttonHost = document.createElement('div');
    buttonHost.className = 'aura-gmail-ai-host';
    Object.assign(buttonHost.style, {
      display: 'inline-flex',
      alignItems: 'center',
      verticalAlign: 'middle',
      marginLeft: '8px',
      position: 'relative',
      zIndex: '1',
    });

    const gmailShadow = buttonHost.attachShadow({ mode: 'closed' });

    // Styles inside shadow
    const gmailStyle = document.createElement('style');
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

    // Container
    const gmailContainer = document.createElement('div');
    gmailContainer.style.position = 'relative';
    gmailContainer.style.display = 'inline-flex';
    gmailContainer.style.alignItems = 'center';
    gmailShadow.appendChild(gmailContainer);

    // AI button
    const aiBtn = document.createElement('button');
    aiBtn.className = 'gmail-ai-btn';
    aiBtn.innerHTML = `<span class="sparkle">\u2726</span> AI`;
    gmailContainer.appendChild(aiBtn);

    let gmailMenuEl: HTMLDivElement | null = null;
    let gmailTranslateSubEl: HTMLDivElement | null = null;
    let gmailToastEl: HTMLDivElement | null = null;
    let gmailToastTimer: ReturnType<typeof setTimeout> | null = null;

    function showGmailToast(msg: string, duration = 2500): void {
      if (gmailToastEl) gmailToastEl.remove();
      if (gmailToastTimer) clearTimeout(gmailToastTimer);
      gmailToastEl = document.createElement('div');
      gmailToastEl.className = 'gmail-ai-toast';
      gmailToastEl.textContent = msg;
      gmailContainer.appendChild(gmailToastEl);
      gmailToastTimer = setTimeout(() => {
        if (gmailToastEl) { gmailToastEl.remove(); gmailToastEl = null; }
        gmailToastTimer = null;
      }, duration);
    }

    function removeGmailMenu(): void {
      if (gmailMenuEl) { gmailMenuEl.remove(); gmailMenuEl = null; }
      gmailTranslateSubEl = null;
    }

    function setGmailMenuLoading(loading: boolean): void {
      if (!gmailMenuEl) return;
      gmailMenuEl.querySelectorAll('.gmail-ai-menu-item').forEach((item) => {
        if (loading) item.classList.add('loading');
        else item.classList.remove('loading');
      });
    }

    function executeGmailAction(action: string, language?: string): void {
      const composeText = getComposeText(composeEl);
      const threadText = extractGmailThreadText();

      if (action === 'draft_reply' && !composeText && !threadText) {
        showGmailToast('No email thread found', 3000);
        removeGmailMenu();
        return;
      }
      if (action !== 'draft_reply' && !composeText) {
        showGmailToast('Compose body is empty', 3000);
        removeGmailMenu();
        return;
      }

      setGmailMenuLoading(true);

      const outMsg: QuickActionOutMessage = {
        type: 'QUICK_ACTION',
        action,
        text: composeText || '(empty \u2014 draft a new reply)',
        ...(action === 'draft_reply' ? { threadContext: threadText } : {}),
        ...(language ? { language } : {}),
      };

      _safeSend(outMsg, (response: QuickActionResponse) => {
        if (response && response.ok && response.result) {
          setComposeText(composeEl, response.result);
          showGmailToast('Updated by AURA');
        } else {
          showGmailToast(response?.error || 'Action failed', 3000);
        }
        removeGmailMenu();
      });
    }

    const GMAIL_ACTIONS: GmailMenuItemDef[] = [
      { icon: '\u270D\uFE0F', label: 'Draft reply', action: 'draft_reply' },
      { icon: '\u2728',       label: 'Improve', action: 'improve' },
      { icon: '\uD83C\uDFE2', label: 'Make formal', action: 'make_formal', separator: true },
      { icon: '\uD83D\uDE0A', label: 'Make casual', action: 'make_casual' },
      { icon: '\u2702\uFE0F', label: 'Shorten', action: 'shorten' },
      { icon: '\uD83C\uDF10', label: 'Translate to...', action: 'translate_menu', separator: true },
    ];

    const GMAIL_TRANSLATE_LANGS = ['English', 'Spanish', 'French', 'German', 'Chinese'];

    function showGmailMenu(): void {
      removeGmailMenu();

      gmailMenuEl = document.createElement('div');
      gmailMenuEl.className = 'gmail-ai-menu';

      GMAIL_ACTIONS.forEach((item) => {
        if (item.separator) {
          const sep = document.createElement('div');
          sep.className = 'gmail-ai-sep';
          gmailMenuEl!.appendChild(sep);
        }
        const btn = document.createElement('button');
        btn.className = 'gmail-ai-menu-item';
        btn.innerHTML = `<span class="item-icon">${item.icon}</span><span>${item.label}</span>`;
        btn.addEventListener('click', (e: MouseEvent) => {
          e.preventDefault();
          e.stopPropagation();
          if (item.action === 'translate_menu') {
            toggleGmailTranslateSub(btn);
          } else {
            executeGmailAction(item.action);
          }
        });
        gmailMenuEl!.appendChild(btn);
      });

      gmailContainer.appendChild(gmailMenuEl);
    }

    function toggleGmailTranslateSub(anchor: HTMLButtonElement): void {
      if (gmailTranslateSubEl) {
        gmailTranslateSubEl.remove();
        gmailTranslateSubEl = null;
        return;
      }
      gmailTranslateSubEl = document.createElement('div');
      gmailTranslateSubEl.className = 'gmail-ai-sub';

      GMAIL_TRANSLATE_LANGS.forEach((lang) => {
        const item = document.createElement('button');
        item.className = 'gmail-ai-menu-item';
        item.textContent = lang;
        item.addEventListener('click', (e: MouseEvent) => {
          e.preventDefault();
          e.stopPropagation();
          executeGmailAction('gmail_translate', lang);
        });
        gmailTranslateSubEl!.appendChild(item);
      });

      if (gmailMenuEl && anchor.parentNode === gmailMenuEl) {
        anchor.after(gmailTranslateSubEl);
      }
    }

    // Toggle menu on AI button click
    aiBtn.addEventListener('click', (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (gmailMenuEl) {
        removeGmailMenu();
      } else {
        showGmailMenu();
      }
    });

    // Close menu when clicking outside the shadow
    const outsideClickHandler = (e: MouseEvent) => {
      if (!gmailMenuEl) return;
      const path = e.composedPath();
      if (!path.includes(buttonHost)) {
        removeGmailMenu();
      }
    };
    document.addEventListener('mousedown', outsideClickHandler, true);

    // Insert the button next to the Send button
    if (sendBtn?.parentElement) {
      sendBtn.parentElement.insertBefore(buttonHost, sendBtn.nextSibling);
    } else if (toolbarRow) {
      toolbarRow.appendChild(buttonHost);
    }

    // Track for cleanup — observe if compose window gets removed
    const composeObserver = new MutationObserver(() => {
      if (!document.body.contains(composeEl)) {
        composeObserver.disconnect();
        document.removeEventListener('mousedown', outsideClickHandler, true);
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
      outsideHandler: outsideClickHandler,
    });
  }

  /**
   * Scans for Gmail compose windows and injects the AI button.
   */
  function scanGmailComposeWindows(): void {
    const composeSelectors = [
      'div[role="dialog"]',  // Popup compose / reply
      'div.ip.iq',           // Inline reply
      'div.nH.nn',           // Another compose variant
    ];

    composeSelectors.forEach((sel) => {
      document.querySelectorAll<HTMLElement>(sel).forEach((el) => {
        const body = getComposeBody(el);
        if (!body) return;
        if (_gmailTrackedComposes.has(el)) return;
        injectGmailAiButton(el);
      });
    });
  }

  /**
   * Initializes Gmail-specific observers. Only runs on mail.google.com.
   */
  function initGmailIntegration(): void {
    if (!isGmailPage()) return;

    // Initial scan (in case compose is already open)
    scanGmailComposeWindows();

    // Watch for new compose windows via MutationObserver
    const gmailObserver = new MutationObserver((mutations: MutationRecord[]) => {
      let shouldScan = false;
      for (const mutation of mutations) {
        if (mutation.addedNodes.length > 0) {
          for (const node of mutation.addedNodes) {
            if (node.nodeType !== Node.ELEMENT_NODE) continue;
            const el = node as HTMLElement;
            if (
              el.matches?.('div[role="dialog"]') ||
              el.querySelector?.('div[role="dialog"]') ||
              el.querySelector?.('div[contenteditable="true"]')
            ) {
              shouldScan = true;
              break;
            }
          }
        }
        if (shouldScan) break;
      }
      if (shouldScan) {
        // Small delay to let Gmail finish rendering the compose DOM
        setTimeout(scanGmailComposeWindows, 300);
      }
    });

    // Observe Gmail's main content container (not entire body for efficiency)
    const gmailRoot = document.querySelector<HTMLElement>('div[role="main"]') || document.body;
    gmailObserver.observe(gmailRoot, { childList: true, subtree: true });

    // Fallback periodic scan for edge cases (Gmail SPA navigation, inline replies)
    let _gmailScanInterval: ReturnType<typeof setInterval> | null = null;
    _gmailScanInterval = setInterval(() => {
      if (!isGmailPage()) {
        if (_gmailScanInterval) clearInterval(_gmailScanInterval);
        return;
      }
      scanGmailComposeWindows();
    }, 3000);
  }

  return {
    init(_container: HTMLElement, _store: any, ext: typeof chrome) {
      _ext = ext;
      _safeSend = (msg: any, cb?: (r: any) => void) => {
        try {
          if (cb) {
            _ext.runtime.sendMessage(msg, cb);
          } else {
            _ext.runtime.sendMessage(msg);
          }
        } catch (_e) { /* context invalidated */ }
      };

      // Boot Gmail integration
      initGmailIntegration();
    },

    destroy() {
      // Clean up all tracked compose buttons
      for (const tracker of _gmailTrackedComposes.values()) {
        tracker.observer.disconnect();
        document.removeEventListener('mousedown', tracker.outsideHandler, true);
        tracker.buttonHost.remove();
      }
      _gmailTrackedComposes.clear();
    },
  };
}
