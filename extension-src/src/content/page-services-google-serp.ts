/**
 * Google SERP "AI Answer" card — floats a shadow-DOM card on google.com/search
 * pages with an AURA-generated answer above the native results.
 *
 * Extracted from page-services.ts (was ~700 lines of that 2400-line megamodule).
 * Self-contained: all helpers live inside the closure, only takes chrome APIs +
 * message-send helper as arguments.
 */
export function initGoogleSerp(ext: typeof chrome, safeSend: (msg: any, cb?: (r: any) => void) => void): void {
  let SERP_BACKEND = 'https://aura-elnur.duckdns.org';
  let SERP_API_KEY = '';

  /** Read backend URL and API key from chrome.storage.local. */
  function loadSerpConfig(): Promise<void> {
    return new Promise((resolve) => {
      if (!ext?.storage?.local) {
        resolve();
        return;
      }
      ext.storage.local.get(['backendUrl', 'apiKey'], (d: any) => {
        if (d?.backendUrl?.trim()) SERP_BACKEND = d.backendUrl.trim().replace(/\/+$/, '');
        if (d?.apiKey?.trim()) SERP_API_KEY = d.apiKey.trim();
        resolve();
      });
    });
  }

  // Load config on init and listen for storage changes
  loadSerpConfig();
  if (ext?.storage?.onChanged) {
    ext.storage.onChanged.addListener((changes: any, area: string) => {
      if (area !== 'local') return;
      if (changes.backendUrl?.newValue) {
        SERP_BACKEND = changes.backendUrl.newValue.trim().replace(/\/+$/, '');
      }
      if (changes.apiKey?.newValue !== undefined) {
        SERP_API_KEY = changes.apiKey.newValue?.trim() || '';
      }
    });
  }

  function isGoogleSearchPage(): boolean {
    const hostname = window.location.hostname;
    const pathname = window.location.pathname;
    const params = new URLSearchParams(window.location.search);
    if (!hostname.match(/^(www\.)?google\./)) return false;
    if (pathname !== '/search') return false;
    if (!params.get('q')) return false;
    const tbm = params.get('tbm');
    if (tbm && ['isch', 'lcl', 'vid', 'shop', 'nws', 'bks', 'fin'].includes(tbm)) return false;
    const udm = params.get('udm');
    if (udm && ['2', '14'].includes(udm)) return false;
    return true;
  }

  function getSearchQuery(): string {
    const params = new URLSearchParams(window.location.search);
    const qParam = params.get('q') || '';
    if (qParam) return qParam;
    const input = document.querySelector<HTMLInputElement>('input[name="q"]');
    return input?.value || '';
  }

  function detectGoogleTheme(): 'dark' | 'light' {
    const bg = window.getComputedStyle(document.body).backgroundColor;
    if (!bg || bg === 'transparent') return 'light';
    const rgbMatch = bg.match(/\d+/g);
    if (rgbMatch && rgbMatch.length >= 3) {
      const [r, g, b] = rgbMatch.map(Number);
      const luminance = (0.299 * r + 0.587 * g + 0.114 * b);
      return luminance < 128 ? 'dark' : 'light';
    }
    return 'light';
  }

  function serpEscapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /**
   * Minimal markdown-to-HTML renderer for SERP answers.
   * Security: input is escaped first, then only safe structural tags are
   * introduced by regex. Link hrefs are constrained to https?:// by the
   * regex pattern. Final output is sanitized to strip any unexpected tags
   * or attributes as defense-in-depth against backend compromise.
   */
  function serpRenderMarkdown(text: string): string {
    let html = serpEscapeHtml(text);

    // Bold: **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Italic: *text* or _text_
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

    // Inline code: `text`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Links: [text](url) — href constrained to https?:// only
    html = html.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)"]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // Unordered lists: lines starting with - or *
    html = html.replace(/^[\s]*[-*]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Ordered lists: lines starting with 1. 2. etc.
    html = html.replace(/^[\s]*\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // Paragraphs: double newlines
    html = html.replace(/\n\n+/g, '</p><p>');
    html = '<p>' + html + '</p>';

    // Single newlines to line breaks
    html = html.replace(/\n/g, '<br>');

    // Clean empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');

    // Defense-in-depth: strip any tags that aren't in our safe list
    html = html.replace(/<\/?(?!(?:strong|em|code|a|li|ul|ol|p|br)\b)[^>]*>/gi, '');

    return html;
  }

  function serpAddCitations(bodyEl: HTMLElement, fullText: string): void {
    const citationRegex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
    const citations: Array<{ title: string; url: string }> = [];
    let citMatch: RegExpExecArray | null;
    while ((citMatch = citationRegex.exec(fullText)) !== null) {
      citations.push({ title: citMatch[1], url: citMatch[2] });
    }
    if (citations.length === 0) return;

    const citationsContainer = document.createElement('div');
    citationsContainer.className = 'serp-citations';
    const citLabel = document.createElement('div');
    citLabel.className = 'serp-citations-label';
    citLabel.textContent = 'Sources';
    citationsContainer.appendChild(citLabel);

    const citList = document.createElement('div');
    citList.className = 'serp-citation-list';
    citations.forEach((cit, idx) => {
      const chip = document.createElement('a');
      chip.className = 'serp-citation-chip';
      chip.href = cit.url;
      chip.target = '_blank';
      chip.rel = 'noopener noreferrer';
      const num = document.createElement('span');
      num.className = 'serp-citation-num';
      num.textContent = String(idx + 1);
      chip.appendChild(num);
      const chipText = document.createTextNode(' ' + cit.title);
      chip.appendChild(chipText);
      citList.appendChild(chip);
    });
    citationsContainer.appendChild(citList);
    bodyEl.appendChild(citationsContainer);
  }

  function serpAddFooter(cardEl: HTMLElement, query: string, fullText: string): void {
    const footer = document.createElement('div');
    footer.className = 'serp-footer';

    const followupBtn = document.createElement('button');
    followupBtn.className = 'serp-followup-btn';
    followupBtn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg> Ask follow-up`;
    followupBtn.addEventListener('click', () => {
      safeSend({
        type: 'OPEN_WITH_TEXT',
        action: 'ask',
        text: `I searched for "${query}" and got the following AI answer:\n\n${fullText}\n\nI have a follow-up question: `,
        url: window.location.href,
        title: document.title,
      });
    });

    const powered = document.createElement('span');
    powered.className = 'serp-powered';
    powered.textContent = 'Powered by AURA';

    footer.appendChild(followupBtn);
    footer.appendChild(powered);
    cardEl.appendChild(footer);
  }

  async function initGoogleSerpIntegration(): Promise<void> {
    if (!isGoogleSearchPage()) return;

    // Check user preference
    const stored = await new Promise<Record<string, any>>((resolve) => {
      ext.storage.local.get(['aura_serp_hidden'], resolve);
    });
    if (stored.aura_serp_hidden) return;

    const query = getSearchQuery();
    if (!query) return;

    // Create floating panel on the RIGHT side of the page (like Sider)
    const serpHost = document.createElement('div');
    serpHost.id = 'aura-serp-host';
    Object.assign(serpHost.style, {
      position: 'fixed',
      top: '80px',
      right: '16px',
      width: '340px',
      maxHeight: 'calc(100vh - 100px)',
      zIndex: '2147483640',
      pointerEvents: 'auto',
    });
    document.documentElement.appendChild(serpHost);

    const serpShadow = serpHost.attachShadow({ mode: 'closed' });
    const theme = detectGoogleTheme();
    const isDark = theme === 'dark';

    // Styles
    const serpStyle = document.createElement('style');
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
        background: ${isDark
          ? 'rgba(30, 27, 48, 0.92)'
          : 'rgba(255, 255, 255, 0.95)'};
        backdrop-filter: blur(24px) saturate(1.4);
        -webkit-backdrop-filter: blur(24px) saturate(1.4);
        border-radius: 16px;
        overflow-y: auto;
        max-height: calc(100vh - 120px);
        box-shadow: ${isDark
          ? '0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06)'
          : '0 8px 40px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06)'};
        border: 1px solid ${isDark
          ? 'rgba(124, 58, 237, 0.2)'
          : 'rgba(124, 58, 237, 0.15)'};
        padding: 20px 24px 16px;
        animation: serp-fade-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease;
      }
      .serp-card:hover {
        border-color: ${isDark
          ? 'rgba(124, 58, 237, 0.35)'
          : 'rgba(124, 58, 237, 0.3)'};
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
        color: ${isDark ? 'rgba(160, 148, 210, 0.9)' : 'rgba(124, 58, 237, 0.85)'};
        background: ${isDark
          ? 'rgba(124, 58, 237, 0.12)'
          : 'rgba(124, 58, 237, 0.08)'};
        border-radius: 8px;
        flex-shrink: 0;
      }
      .serp-title {
        font-size: 14px;
        font-weight: 600;
        color: ${isDark ? 'rgba(226, 232, 240, 0.9)' : 'rgba(30, 27, 48, 0.9)'};
        letter-spacing: -0.01em;
      }
      .serp-title-sub {
        font-size: 11px;
        font-weight: 400;
        color: ${isDark ? 'rgba(160, 148, 210, 0.5)' : 'rgba(100, 90, 140, 0.6)'};
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
        color: ${isDark ? 'rgba(160, 148, 210, 0.5)' : 'rgba(100, 90, 140, 0.5)'};
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s, color 0.15s;
        padding: 0;
      }
      .serp-ctrl-btn:hover {
        background: ${isDark ? 'rgba(124, 58, 237, 0.15)' : 'rgba(124, 58, 237, 0.1)'};
        color: ${isDark ? 'rgba(224, 214, 255, 1)' : 'rgba(124, 58, 237, 0.9)'};
      }
      .serp-ctrl-btn[title="Hide AURA answers"]:hover {
        background: rgba(239, 68, 68, 0.12);
        color: rgba(239, 68, 68, 0.9);
      }

      .serp-body {
        font-size: 14px;
        line-height: 1.7;
        color: ${isDark ? 'rgba(226, 232, 240, 0.85)' : 'rgba(30, 27, 48, 0.85)'};
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
        background: ${isDark ? 'rgba(124, 58, 237, 0.6)' : 'rgba(124, 58, 237, 0.5)'};
        animation: serp-pulse 1.2s ease-in-out infinite;
      }
      .serp-loading-dots span:nth-child(2) { animation-delay: 0.2s; }
      .serp-loading-dots span:nth-child(3) { animation-delay: 0.4s; }
      .serp-loading-text {
        font-size: 13px;
        color: ${isDark ? 'rgba(160, 148, 210, 0.6)' : 'rgba(100, 90, 140, 0.6)'};
      }

      .serp-answer {
        white-space: pre-wrap;
        word-break: break-word;
      }
      .serp-answer p { margin-bottom: 8px; }
      .serp-answer p:last-child { margin-bottom: 0; }
      .serp-answer strong, .serp-answer b {
        font-weight: 600;
        color: ${isDark ? 'rgba(226, 232, 240, 0.95)' : 'rgba(30, 27, 48, 0.95)'};
      }
      .serp-answer code {
        background: ${isDark ? 'rgba(124, 58, 237, 0.1)' : 'rgba(124, 58, 237, 0.06)'};
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
        color: ${isDark ? 'rgba(160, 148, 255, 0.9)' : 'rgba(100, 58, 237, 0.9)'};
        text-decoration: none;
      }
      .serp-answer a:hover { text-decoration: underline; }

      .serp-citations {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'};
      }
      .serp-citations-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: ${isDark ? 'rgba(160, 148, 210, 0.5)' : 'rgba(100, 90, 140, 0.5)'};
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
        background: ${isDark ? 'rgba(124, 58, 237, 0.1)' : 'rgba(124, 58, 237, 0.06)'};
        border: 1px solid ${isDark ? 'rgba(124, 58, 237, 0.15)' : 'rgba(124, 58, 237, 0.1)'};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: ${isDark ? 'rgba(200, 180, 255, 0.8)' : 'rgba(100, 58, 237, 0.8)'};
        text-decoration: none;
        transition: background 0.15s, border-color 0.15s;
        max-width: 280px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .serp-citation-chip:hover {
        background: ${isDark ? 'rgba(124, 58, 237, 0.2)' : 'rgba(124, 58, 237, 0.12)'};
        border-color: ${isDark ? 'rgba(124, 58, 237, 0.3)' : 'rgba(124, 58, 237, 0.2)'};
      }
      .serp-citation-num {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        background: ${isDark ? 'rgba(124, 58, 237, 0.2)' : 'rgba(124, 58, 237, 0.1)'};
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
        border-top: 1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'};
      }
      .serp-followup-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: ${isDark ? 'rgba(124, 58, 237, 0.12)' : 'rgba(124, 58, 237, 0.08)'};
        border: 1px solid ${isDark ? 'rgba(124, 58, 237, 0.2)' : 'rgba(124, 58, 237, 0.15)'};
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        color: ${isDark ? 'rgba(200, 180, 255, 0.9)' : 'rgba(100, 58, 237, 0.9)'};
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.15s;
      }
      .serp-followup-btn:hover {
        background: ${isDark ? 'rgba(124, 58, 237, 0.22)' : 'rgba(124, 58, 237, 0.15)'};
        border-color: ${isDark ? 'rgba(124, 58, 237, 0.35)' : 'rgba(124, 58, 237, 0.3)'};
        transform: scale(1.01);
      }
      .serp-followup-btn:active { transform: scale(0.98); }
      .serp-powered {
        font-size: 11px;
        color: ${isDark ? 'rgba(160, 148, 210, 0.35)' : 'rgba(100, 90, 140, 0.35)'};
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
        color: ${isDark ? 'rgba(226, 232, 240, 0.5)' : 'rgba(30, 27, 48, 0.5)'};
      }

      .serp-error {
        font-size: 13px;
        color: ${isDark ? 'rgba(239, 150, 150, 0.8)' : 'rgba(200, 50, 50, 0.7)'};
        padding: 4px 0;
      }
    `;
    serpShadow.appendChild(serpStyle);

    // Card container
    const card = document.createElement('div');
    card.className = 'serp-card';

    // Header
    const serpHeader = document.createElement('div');
    serpHeader.className = 'serp-header';

    const headerLeft = document.createElement('div');
    headerLeft.className = 'serp-header-left';

    const serpLogo = document.createElement('div');
    serpLogo.className = 'serp-logo';
    serpLogo.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;

    const titleWrap = document.createElement('div');
    const titleText = document.createElement('span');
    titleText.className = 'serp-title';
    titleText.textContent = 'AI Answer';
    const titleSub = document.createElement('span');
    titleSub.className = 'serp-title-sub';
    titleSub.textContent = 'by AURA';
    titleWrap.appendChild(titleText);
    titleWrap.appendChild(titleSub);

    headerLeft.appendChild(serpLogo);
    headerLeft.appendChild(titleWrap);

    const controls = document.createElement('div');
    controls.className = 'serp-controls';

    // Collapse/expand toggle
    const collapseBtn = document.createElement('button');
    collapseBtn.className = 'serp-ctrl-btn';
    collapseBtn.title = 'Collapse';
    collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`;

    // Hide toggle
    const hideBtn = document.createElement('button');
    hideBtn.className = 'serp-ctrl-btn';
    hideBtn.title = 'Hide AURA answers';
    hideBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;

    controls.appendChild(collapseBtn);
    controls.appendChild(hideBtn);

    serpHeader.appendChild(headerLeft);
    serpHeader.appendChild(controls);
    card.appendChild(serpHeader);

    // Body
    const serpBody = document.createElement('div');
    serpBody.className = 'serp-body';

    // Loading state
    const serpLoading = document.createElement('div');
    serpLoading.className = 'serp-loading';
    const serpDots = document.createElement('div');
    serpDots.className = 'serp-loading-dots';
    serpDots.innerHTML = '<span></span><span></span><span></span>';
    const serpLoadingText = document.createElement('span');
    serpLoadingText.className = 'serp-loading-text';
    serpLoadingText.textContent = `Thinking about "${query.slice(0, 60)}${query.length > 60 ? '...' : ''}"`;
    serpLoading.appendChild(serpDots);
    serpLoading.appendChild(serpLoadingText);
    serpBody.appendChild(serpLoading);

    card.appendChild(serpBody);
    serpShadow.appendChild(card);

    // Collapse toggle logic
    let isSerpCollapsed = false;
    collapseBtn.addEventListener('click', () => {
      isSerpCollapsed = !isSerpCollapsed;
      if (isSerpCollapsed) {
        serpBody.classList.add('collapsed');
        collapseBtn.title = 'Expand';
        collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;
      } else {
        serpBody.classList.remove('collapsed');
        collapseBtn.title = 'Collapse';
        collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`;
      }
    });

    // Hide toggle: persist and remove card
    hideBtn.addEventListener('click', () => {
      ext.storage.local.set({ aura_serp_hidden: true });
      serpHost.remove();
    });

    // Fetch AI answer via background script (avoids CORS)
    try {
      const fetchBody = JSON.stringify({
        message: query,
        conversation_id: '__serp_answer__',
        stream: false,
        system_context: `The user searched Google for: "${query}". Provide a concise, direct answer to their query. Be helpful and factual. Use markdown formatting sparingly — bold for emphasis, lists where appropriate. If you reference sources, format them as [Source Title](URL) and they will be rendered as citation chips. Keep the answer focused and under 200 words unless the topic requires more detail.`,
      });

      // Try background proxy first, fall back to direct fetch
      let proxyResult: any = null;
      (async () => {
        try {
          proxyResult = await new Promise((resolve, reject) => {
            ext.runtime.sendMessage(
              { type: 'SERP_FETCH', url: `${SERP_BACKEND}/api/chat`, body: fetchBody, apiKey: SERP_API_KEY },
              (response: any) => {
                if (ext.runtime.lastError) {
                  reject(new Error(ext.runtime.lastError.message));
                } else {
                  resolve(response);
                }
              }
            );
          });
        } catch {
          // Background unavailable — try direct fetch as fallback
          const serpHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
          if (SERP_API_KEY) serpHeaders['X-API-Key'] = SERP_API_KEY;
          const directResp = await fetch(`${SERP_BACKEND}/api/chat`, {
            method: 'POST',
            headers: serpHeaders,
            body: fetchBody,
            signal: AbortSignal.timeout(30000),
          });
          if (!directResp.ok) throw new Error(`HTTP ${directResp.status}`);
          proxyResult = { ok: true, text: await directResp.text() };
        }

        if (!proxyResult?.ok) {
          throw new Error(proxyResult?.error || 'Backend unreachable');
        }

        // Clear loading
        serpLoading.remove();

        const answerEl = document.createElement('div');
        answerEl.className = 'serp-answer';
        serpBody.appendChild(answerEl);

        // Parse the response (may be NDJSON or plain JSON)
        let fullText = '';
        const responseText = proxyResult.text || '';
        const lines = responseText.split('\n').filter((l: string) => l.trim());
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

        // Render the response
        answerEl.innerHTML = serpRenderMarkdown(fullText);

        if (!fullText.trim()) {
          answerEl.innerHTML = '<span class="serp-error">No response from AI.</span>';
          return;
        }

        serpAddCitations(serpBody, fullText);
        serpAddFooter(card, query, fullText);
      })().catch((_err: unknown) => {
        serpLoading.remove();
        const offline = document.createElement('div');
        offline.className = 'serp-offline';
        const offDot = document.createElement('div');
        offDot.className = 'serp-offline-dot';
        const offText = document.createElement('span');
        offText.className = 'serp-offline-text';
        offText.textContent = `AURA is offline — backend did not respond (${(_err as Error)?.message || 'timeout'})`;
        offline.appendChild(offDot);
        offline.appendChild(offText);
        serpBody.appendChild(offline);
      });

    } catch (_err: unknown) {
      serpLoading.remove();
      const offline = document.createElement('div');
      offline.className = 'serp-offline';
      const offDot = document.createElement('div');
      offDot.className = 'serp-offline-dot';
      const offText = document.createElement('span');
      offText.className = 'serp-offline-text';
      offText.textContent = `AURA is offline — backend did not respond`;
      offline.appendChild(offDot);
      offline.appendChild(offText);
      serpBody.appendChild(offline);
    }
  }

  // Boot Google SERP integration
  initGoogleSerpIntegration();
}
