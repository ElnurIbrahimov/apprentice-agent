/**
 * Capture module — Component Capture Mode (Shadow DOM isolated)
 * Migrated from content.ts lines ~3948-4131
 */

import type { ContentModule } from './types';

// CSS properties worth capturing for UI recreation
const CAPTURE_CSS_PROPS: string[] = [
  'display', 'position', 'flex-direction', 'align-items', 'justify-content',
  'gap', 'flex-wrap', 'flex', 'flex-grow', 'flex-shrink',
  'width', 'height', 'min-width', 'min-height', 'max-width', 'max-height',
  'padding', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
  'margin', 'margin-top', 'margin-right', 'margin-bottom', 'margin-left',
  'border', 'border-radius', 'border-color', 'border-width', 'border-style',
  'background', 'background-color', 'background-image', 'background-size',
  'color', 'font-size', 'font-weight', 'font-family', 'line-height',
  'letter-spacing', 'text-align', 'text-decoration', 'text-transform',
  'box-shadow', 'opacity', 'overflow', 'z-index',
  'grid-template-columns', 'grid-template-rows', 'grid-gap',
  'transform', 'transition',
];

export interface CaptureModule extends ContentModule {
  start(): void;
  stop(): void;
}

export function createCapture(): CaptureModule {
  let _ext!: typeof chrome;

  // Shadow DOM host
  const captureHost: HTMLDivElement = document.createElement('div');
  captureHost.id = 'aura-capture-host';
  Object.assign(captureHost.style, {
    position: 'fixed', top: '0', left: '0', width: '0', height: '0',
    zIndex: '2147483647', pointerEvents: 'none',
  });
  document.documentElement.appendChild(captureHost);
  const captureShadow: ShadowRoot = captureHost.attachShadow({ mode: 'closed' });

  const captureStyle: HTMLStyleElement = document.createElement('style');
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

  const captureContainer: HTMLDivElement = document.createElement('div');
  captureShadow.appendChild(captureContainer);

  let _captureActive = false;
  let _captureOverlay: HTMLDivElement | null = null;
  let _captureTooltip: HTMLDivElement | null = null;
  let _captureBanner: HTMLDivElement | null = null;
  let _captureHoveredEl: Element | null = null;

  function extractComputedStyles(el: Element): Record<string, string> {
    const styles = window.getComputedStyle(el);
    const result: Record<string, string> = {};
    for (const prop of CAPTURE_CSS_PROPS) {
      const val = styles.getPropertyValue(prop);
      if (val && val !== 'none' && val !== 'normal' && val !== 'auto' && val !== '0px' && val !== 'rgba(0, 0, 0, 0)') {
        result[prop] = val;
      }
    }
    return result;
  }

  function buildCssSelector(el: Element): string {
    const tag = el.tagName.toLowerCase();
    const cls = el.className && typeof el.className === 'string'
      ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.')
      : '';
    return tag + cls;
  }

  function captureElementData(el: Element): {
    html: string;
    css: Record<string, Record<string, string>>;
    dimensions: { width: number; height: number; padding: string; margin: string };
    textContent: string;
    tagName: string;
    className: string;
  } {
    const rect = el.getBoundingClientRect();
    const styles = window.getComputedStyle(el);
    const html = (el as HTMLElement).outerHTML;

    // Collect styles for root + up to 50 children
    const cssMap: Record<string, Record<string, string>> = {};
    cssMap[buildCssSelector(el)] = extractComputedStyles(el);

    const children = el.querySelectorAll('*');
    let count = 0;
    for (const child of children) {
      if (count >= 50) break;
      const childStyles = extractComputedStyles(child);
      if (Object.keys(childStyles).length > 0) {
        const selector = buildCssSelector(child);
        // Append index to avoid duplicates
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
        margin: `${styles.marginTop} ${styles.marginRight} ${styles.marginBottom} ${styles.marginLeft}`,
      },
      textContent: ((el as HTMLElement).textContent || '').slice(0, 2000).trim(),
      tagName: el.tagName.toLowerCase(),
      className: (typeof el.className === 'string' ? el.className : '').trim(),
    };
  }

  function start(): void {
    if (_captureActive) return;
    _captureActive = true;

    // Show banner
    _captureBanner = document.createElement('div');
    _captureBanner.className = 'capture-banner';
    _captureBanner.innerHTML = '<span class="dot"></span> AURA Capture Mode — Click any element • Esc to exit';
    captureContainer.appendChild(_captureBanner);

    // Create overlay
    _captureOverlay = document.createElement('div');
    _captureOverlay.className = 'capture-overlay';
    _captureOverlay.style.display = 'none';
    captureContainer.appendChild(_captureOverlay);

    // Create tooltip
    _captureTooltip = document.createElement('div');
    _captureTooltip.className = 'capture-tooltip';
    _captureTooltip.style.display = 'none';
    captureContainer.appendChild(_captureTooltip);

    // Enable pointer events on the host for the overlay to be visible
    captureHost.style.width = '100vw';
    captureHost.style.height = '100vh';

    document.addEventListener('mousemove', onCaptureMouseMove, true);
    document.addEventListener('click', onCaptureClick, true);
    document.addEventListener('keydown', onCaptureKeydown, true);
  }

  function stop(): void {
    if (!_captureActive) return;
    _captureActive = false;
    _captureHoveredEl = null;

    if (_captureOverlay) { _captureOverlay.remove(); _captureOverlay = null; }
    if (_captureTooltip) { _captureTooltip.remove(); _captureTooltip = null; }
    if (_captureBanner) { _captureBanner.remove(); _captureBanner = null; }

    captureHost.style.width = '0';
    captureHost.style.height = '0';

    document.removeEventListener('mousemove', onCaptureMouseMove, true);
    document.removeEventListener('click', onCaptureClick, true);
    document.removeEventListener('keydown', onCaptureKeydown, true);

    try { _ext.runtime.sendMessage({ type: 'OPEN_PANEL', panel: 'capture' }); } catch (_e) { /* ignore */ }
  }

  function onCaptureMouseMove(e: MouseEvent): void {
    if (!_captureActive) return;

    // Find element under cursor, ignoring our overlay elements
    const elements = document.elementsFromPoint(e.clientX, e.clientY);
    let target: Element | null = null;
    for (const el of elements) {
      if (el === captureHost || captureHost.contains(el)) continue;
      if (el.id === 'aura-host' || el.id === 'aura-dock-shadow' ||
          el.id === 'aura-quick-action-host' || el.id === 'aura-highlight-host' ||
          el.id === 'aura-img-toolbar-host' || el.id === 'aura-capture-host') continue;
      // Skip document root elements
      if (el === document.documentElement || el === document.body) continue;
      target = el;
      break;
    }

    if (!target) {
      if (_captureOverlay) _captureOverlay.style.display = 'none';
      if (_captureTooltip) _captureTooltip.style.display = 'none';
      _captureHoveredEl = null;
      return;
    }

    _captureHoveredEl = target;
    const rect = target.getBoundingClientRect();

    // Position overlay
    if (_captureOverlay) {
      _captureOverlay.style.display = 'block';
      _captureOverlay.style.top = rect.top + 'px';
      _captureOverlay.style.left = rect.left + 'px';
      _captureOverlay.style.width = rect.width + 'px';
      _captureOverlay.style.height = rect.height + 'px';
    }

    // Position tooltip
    if (_captureTooltip) {
      const tag = target.tagName.toLowerCase();
      const cls = target.className && typeof target.className === 'string'
        ? target.className.trim().split(/\s+/).slice(0, 3).join(' ')
        : '';
      const w = Math.round(rect.width);
      const h = Math.round(rect.height);

      // Build tooltip DOM safely (avoid innerHTML with page-supplied class names)
      _captureTooltip.textContent = '';
      const tagSpan = document.createElement('span');
      tagSpan.className = 'tag';
      tagSpan.textContent = `<${tag}>`;
      _captureTooltip.appendChild(tagSpan);
      if (cls) {
        const clsSpan = document.createElement('span');
        clsSpan.className = 'cls';
        clsSpan.textContent = '.' + cls.split(' ').join('.');
        _captureTooltip.appendChild(clsSpan);
      }
      const dimsSpan = document.createElement('span');
      dimsSpan.className = 'dims';
      dimsSpan.textContent = `${w}x${h}`;
      _captureTooltip.appendChild(dimsSpan);

      let tooltipTop = rect.top - 30;
      if (tooltipTop < 4) tooltipTop = rect.bottom + 6;
      let tooltipLeft = rect.left;
      if (tooltipLeft < 4) tooltipLeft = 4;

      _captureTooltip.style.display = 'flex';
      _captureTooltip.style.top = tooltipTop + 'px';
      _captureTooltip.style.left = tooltipLeft + 'px';
    }
  }

  function onCaptureClick(e: MouseEvent): void {
    if (!_captureActive || !_captureHoveredEl) return;
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();

    const el = _captureHoveredEl;
    const rect = el.getBoundingClientRect();
    const data = captureElementData(el);

    // Request screenshot from background
    try {
      _ext.runtime.sendMessage(
        {
          type: 'CAPTURE_ELEMENT',
          rect: {
            x: Math.round(rect.left),
            y: Math.round(rect.top),
            w: Math.round(rect.width),
            h: Math.round(rect.height),
          },
          elementData: data,
        },
        (_response: any) => {
          // Background will relay COMPONENT_CAPTURED to sidebar
        }
      );
    } catch (_e) { /* context may be invalidated */ }

    stop();
  }

  function onCaptureKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape' && _captureActive) {
      e.preventDefault();
      e.stopPropagation();
      stop();
      try {
        _ext.runtime.sendMessage({ type: 'CAPTURE_MODE_EXITED' }).catch(() => {});
      } catch (_e) { /* ignore */ }
    }
  }

  return {
    init(_container: HTMLElement, _store: any, ext: typeof chrome) {
      _ext = ext;
    },

    destroy() {
      if (_captureActive) stop();
      captureHost.remove();
    },

    start,
    stop,
  };
}
