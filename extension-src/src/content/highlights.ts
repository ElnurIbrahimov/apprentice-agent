/**
 * Highlights module — Highlight System (Shadow DOM isolated)
 * Migrated from content.ts lines ~1111-1480
 */

import type { ContentModule, HighlightData } from './types';

interface SaveHighlightMessage {
  type: 'SAVE_HIGHLIGHT';
  highlight: HighlightData;
}

interface SaveHighlightResponse {
  ok: boolean;
  error?: string;
}

interface DeleteHighlightMessage {
  type: 'DELETE_HIGHLIGHT';
  id: string;
  url: string;
}

interface DeleteHighlightResponse {
  ok: boolean;
}

interface GetHighlightsMessage {
  type: 'GET_HIGHLIGHTS';
  url: string;
}

interface GetHighlightsResponse {
  ok: boolean;
  highlights: HighlightData[];
}

export interface HighlightsModule extends ContentModule {
  scrollTo(id: string): void;
  saveHighlight(): boolean;
  /** Coordinator injects showToast after init */
  setShowToast(fn: (msg: string, duration?: number) => void): void;
}

export function createHighlights(): HighlightsModule {
  let _ext!: typeof chrome;
  let _showToast: (msg: string, duration?: number) => void = () => {};

  function safeSend(msg: any, cb?: (r: any) => void): void {
    try {
      if (cb) {
        _ext.runtime.sendMessage(msg, cb);
      } else {
        _ext.runtime.sendMessage(msg);
      }
    } catch (_e) { /* context invalidated */ }
  }

  // Shadow DOM host for highlight tooltips
  const hlHost: HTMLDivElement = document.createElement('div');
  hlHost.id = 'aura-highlight-host';
  Object.assign(hlHost.style, {
    position: 'fixed', top: '0', left: '0',
    zIndex: '2147483646', pointerEvents: 'none',
  });
  document.documentElement.appendChild(hlHost);
  const hlShadow: ShadowRoot = hlHost.attachShadow({ mode: 'closed' });

  const hlStyle: HTMLStyleElement = document.createElement('style');
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

  const hlContainer: HTMLDivElement = document.createElement('div');
  hlShadow.appendChild(hlContainer);

  // Page-level highlight styles (must be in page DOM, not shadow)
  const pageHlStyle: HTMLStyleElement = document.createElement('style');
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

  let _hlTooltipEl: HTMLDivElement | null = null;
  let _hlTooltipTimer: ReturnType<typeof setTimeout> | null = null;

  function removeHlTooltip(): void {
    if (_hlTooltipTimer) { clearTimeout(_hlTooltipTimer); _hlTooltipTimer = null; }
    if (_hlTooltipEl) { _hlTooltipEl.remove(); _hlTooltipEl = null; }
  }

  function showHlTooltip(mark: HTMLElement, highlightId: string): void {
    removeHlTooltip();
    const rect = mark.getBoundingClientRect();
    _hlTooltipEl = document.createElement('div');
    _hlTooltipEl.className = 'hl-tooltip';

    const label = document.createElement('span');
    label.className = 'hl-tooltip-text';
    label.textContent = 'Saved to AURA';
    _hlTooltipEl.appendChild(label);

    const delBtn = document.createElement('button');
    delBtn.className = 'hl-tooltip-delete';
    delBtn.title = 'Remove highlight';
    delBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
    delBtn.addEventListener('click', (e: MouseEvent) => {
      e.stopPropagation();
      deleteHighlightFromPage(highlightId);
      removeHlTooltip();
    });
    _hlTooltipEl.appendChild(delBtn);

    _hlTooltipEl.style.top = `${Math.round(rect.top - 34)}px`;
    _hlTooltipEl.style.left = `${Math.round(rect.left + rect.width / 2 - 60)}px`;
    hlContainer.appendChild(_hlTooltipEl);
  }

  function getXPath(node: Node): string {
    if (node.nodeType === Node.DOCUMENT_NODE) return '/';
    const parts: string[] = [];
    let current: Node | null = node;
    while (current && current !== document) {
      if (current.nodeType === Node.ELEMENT_NODE) {
        const el = current as Element;
        let tag = el.tagName.toLowerCase();
        const parent = el.parentNode;
        if (parent) {
          const siblings = Array.from(parent.childNodes).filter(
            (n): n is Element => n.nodeType === Node.ELEMENT_NODE && (n as Element).tagName === el.tagName
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
            const idx = textNodes.indexOf(current as ChildNode) + 1;
            parts.unshift(`text()[${idx}]`);
          } else {
            parts.unshift('text()');
          }
        }
      }
      current = current.parentNode;
    }
    return '/' + parts.join('/');
  }

  function getHighlightContext(range: Range): string {
    const container = range.commonAncestorContainer;
    const fullText = (container.nodeType === Node.TEXT_NODE)
      ? (container.textContent || '')
      : (container as Element).textContent || '';
    const selectedText = range.toString();
    const idx = fullText.indexOf(selectedText);
    if (idx === -1) return '';
    const before = fullText.slice(Math.max(0, idx - 50), idx);
    const after = fullText.slice(idx + selectedText.length, idx + selectedText.length + 50);
    return before + '|||' + after;
  }

  function generateHighlightId(): string {
    return 'hl_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
  }

  function attachMarkListeners(mark: HTMLElement): void {
    const hlId = mark.getAttribute('data-aura-hl') || '';
    mark.addEventListener('mouseenter', () => showHlTooltip(mark, hlId));
    mark.addEventListener('mouseleave', () => {
      _hlTooltipTimer = setTimeout(removeHlTooltip, 300);
    });
  }

  function wrapSelectionWithMark(highlightId: string): HTMLElement | null {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return null;
    const range = sel.getRangeAt(0);
    if (range.collapsed) return null;

    try {
      const mark = document.createElement('mark');
      mark.setAttribute('data-aura-hl', highlightId);
      range.surroundContents(mark);
      sel.removeAllRanges();
      attachMarkListeners(mark);
      return mark;
    } catch (_e) {
      try {
        const frag = range.cloneContents();
        const textContent = frag.textContent || '';
        if (!textContent.trim()) return null;
        range.deleteContents();
        const mark = document.createElement('mark');
        mark.setAttribute('data-aura-hl', highlightId);
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

  function saveHighlight(): boolean {
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

    const highlight: HighlightData = {
      id: highlightId,
      url: window.location.href,
      text,
      xpath,
      context,
      timestamp: Date.now(),
      color: 'purple',
      pageTitle: document.title,
    };

    safeSend(
      { type: 'SAVE_HIGHLIGHT', highlight } as SaveHighlightMessage,
      (response: SaveHighlightResponse) => {
        if (response && response.ok) {
          _showToast('Highlight saved to AURA');
        } else {
          _showToast(response?.error || 'Failed to save highlight', 3000);
        }
      }
    );
    return true;
  }

  function deleteHighlightFromPage(highlightId: string): void {
    const mark = document.querySelector(`mark[data-aura-hl="${highlightId}"]`);
    if (mark) {
      const parent = mark.parentNode;
      while (mark.firstChild) parent?.insertBefore(mark.firstChild, mark);
      mark.remove();
      parent?.normalize();
    }
    safeSend(
      { type: 'DELETE_HIGHLIGHT', id: highlightId, url: window.location.href } as DeleteHighlightMessage,
      (_response: DeleteHighlightResponse) => { _showToast('Highlight removed'); }
    );
  }

  function findTextNode(xpath: string, text: string, context: string): Range | null {
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
    } catch (_e) { /* XPath may be stale */ }

    // Fallback: TreeWalker text search with context scoring
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
    const [contextBefore, contextAfter] = context.split('|||');
    let bestNode: Text | null = null;
    let bestOffset = -1;
    let bestScore = 0;

    while (walker.nextNode()) {
      const tNode = walker.currentNode as Text;
      const nodeText = tNode.textContent || '';
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

  function restoreHighlight(hl: HighlightData): boolean {
    if (document.querySelector(`mark[data-aura-hl="${hl.id}"]`)) return true;
    const range = findTextNode(hl.xpath, hl.text, hl.context);
    if (!range) return false;
    try {
      const mark = document.createElement('mark');
      mark.setAttribute('data-aura-hl', hl.id);
      if (hl.stale) mark.classList.add('aura-hl-stale');
      range.surroundContents(mark);
      attachMarkListeners(mark);
      return true;
    } catch (_e) {
      try {
        const text = range.toString();
        range.deleteContents();
        const mark = document.createElement('mark');
        mark.setAttribute('data-aura-hl', hl.id);
        if (hl.stale) mark.classList.add('aura-hl-stale');
        mark.textContent = text;
        range.insertNode(mark);
        attachMarkListeners(mark);
        return true;
      } catch (_e2) {
        return false;
      }
    }
  }

  function restoreAllHighlights(): void {
    safeSend(
      { type: 'GET_HIGHLIGHTS', url: window.location.href } as GetHighlightsMessage,
      (response: GetHighlightsResponse) => {
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

  function scrollTo(highlightId: string): void {
    const mark = document.querySelector(`mark[data-aura-hl="${highlightId}"]`);
    if (mark) {
      mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
      mark.classList.add('aura-hl-flash');
      setTimeout(() => mark.classList.remove('aura-hl-flash'), 1500);
    }
  }

  return {
    init(_container: HTMLElement, _store: any, ext: typeof chrome) {
      _ext = ext;
      // Restore highlights after page content settles
      setTimeout(restoreAllHighlights, 1500);
    },

    destroy() {
      hlHost.remove();
      pageHlStyle.remove();
    },

    scrollTo,
    saveHighlight,
    setShowToast(fn: (msg: string, duration?: number) => void) {
      _showToast = fn;
    },
  };
}
