/// <reference types="chrome" />

/**
 * AURA Chrome Extension — Service Worker (background.ts)
 * Handles: side panel toggle, context menu, message routing
 */

// ── Firefox compatibility shim ───────────────────────────────────────────────

declare const browser: typeof chrome | undefined;

const ext: typeof chrome =
  typeof browser !== 'undefined' ? (browser as typeof chrome) : chrome;

// Dynamic backend URL — reads from storage, falls back to localhost
let BACKEND = 'http://localhost:8000';
let BACKEND_API_KEY = '';

// Load saved backend URL and API key from extension storage
(async () => {
  try {
    const data = await ext.storage.local.get(['backendUrl', 'apiKey']);
    if (data.backendUrl) BACKEND = data.backendUrl.replace(/\/+$/, '');
    if (data.apiKey) BACKEND_API_KEY = data.apiKey;
  } catch {}
})();

// Listen for settings changes
ext.storage.onChanged.addListener((changes, area) => {
  if (area === 'local') {
    if (changes.backendUrl?.newValue) BACKEND = changes.backendUrl.newValue.replace(/\/+$/, '');
    if (changes.apiKey?.newValue) BACKEND_API_KEY = changes.apiKey.newValue;
  }
});

function backendHeaders(): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' };
  if (BACKEND_API_KEY) h['X-API-Key'] = BACKEND_API_KEY;
  return h;
}

// ── URL validation helpers ──────────────────────────────────────────────────

/** Returns true if the URL uses a safe scheme (http/https only). */
function isSafeScheme(url: string): boolean {
  try {
    const u = new URL(url);
    return u.protocol === 'http:' || u.protocol === 'https:';
  } catch {
    return false;
  }
}

/** Returns true if hostname resolves to a private/internal IP range. */
function isPrivateHost(url: string): boolean {
  try {
    const hostname = new URL(url).hostname.toLowerCase();
    if (
      hostname === 'localhost' ||
      hostname === '0.0.0.0' ||
      hostname === '[::1]' ||
      hostname === '::1'
    ) return true;
    // Check numeric IPv4 patterns
    const parts = hostname.split('.');
    if (parts.length === 4 && parts.every(p => /^\d{1,3}$/.test(p))) {
      const [a, b] = parts.map(Number);
      if (a === 127) return true;                        // 127.0.0.0/8
      if (a === 10) return true;                         // 10.0.0.0/8
      if (a === 172 && b >= 16 && b <= 31) return true;  // 172.16.0.0/12
      if (a === 192 && b === 168) return true;           // 192.168.0.0/16
      if (a === 169 && b === 254) return true;           // 169.254.0.0/16
      if (a === 0) return true;                          // 0.0.0.0/8
    }
    return false;
  } catch {
    return true; // If we can't parse it, block it
  }
}

/** Rate limiter for link preview fetches. */
let _linkPreviewActive = 0;
const LINK_PREVIEW_MAX_CONCURRENT = 5;
const LINK_PREVIEW_MAX_BYTES = 500000; // 500 KB

// ── Message type definitions ─────────────────────────────────────────────────

interface SidebarReadyMessage {
  type: 'SIDEBAR_READY';
}

interface SaveKnowledgeMessage {
  type: 'SAVE_KNOWLEDGE';
  text: string;
  url: string;
  title: string;
  tags?: string[];
}

interface GetCurrentTabMessage {
  type: 'GET_CURRENT_TAB';
}

interface GetPageContentMessage {
  type: 'GET_PAGE_CONTENT';
}

interface OpenPanelMessage {
  type: 'OPEN_PANEL';
  panel: string;
}

interface OpenWithTextMessage {
  type: 'OPEN_WITH_TEXT';
  text: string;
  action: string;
  url?: string;
  title?: string;
}

interface OcrStartMessage {
  type: 'OCR_START';
}

interface AgentDomMessage {
  type: 'AGENT_DOM';
}

interface AgentExecMessage {
  type: 'AGENT_EXEC';
  action: unknown;
}

interface AgentNavMessage {
  type: 'AGENT_NAV';
  url: string;
}

interface QuickActionMessage {
  type: 'QUICK_ACTION';
  action: 'improve' | 'expand' | 'shorten' | 'fix_grammar' | 'translate'
    | 'draft_reply' | 'make_formal' | 'make_casual' | 'gmail_translate';
  text: string;
  language?: string;
  threadContext?: string;
}

interface QuickActionResponse {
  ok: boolean;
  result?: string;
  error?: string;
}

interface YtSubtitlesMessage {
  type: 'YT_SUBTITLES';
  videoId: string;
  lang: string;
  segments: Array<{ start: number; dur: number; text: string }>;
}

interface YtMetadataMessage {
  type: 'YT_METADATA';
  videoId: string;
  title: string;
  duration: number;
  description: string;
  channelName: string;
  chapters: Array<{ title: string; startMs: number }>;
  captionTracks: Array<{ baseUrl: string; languageCode: string; name: string }>;
}

interface NetflixSubtitlesMessage {
  type: 'NETFLIX_SUBTITLES';
  movieId: string;
  lang: string;
  trackId: string;
  segments: Array<{ start: number; dur: number; text: string }>;
}

interface NetflixMetadataMessage {
  type: 'NETFLIX_METADATA';
  movieId: string;
  title: string;
  episodeTitle: string;
  seasonNumber: number;
  episodeNumber: number;
  duration: number;
}

interface NetflixTabDetectedMessage {
  type: 'NETFLIX_TAB_DETECTED';
  url: string;
  title: string;
}

interface TranslateBatchMessage {
  type: 'TRANSLATE_BATCH';
  texts: string[];
  targetLang: string;
}

interface TranslateBatchResponse {
  ok: boolean;
  translations?: string[];
  error?: string;
}

// ── Highlight message types ──────────────────────────────────────────────────

interface HighlightData {
  id: string;
  url: string;
  text: string;
  xpath: string;
  context: string;
  timestamp: number;
  color: string;
  pageTitle: string;
  stale?: boolean;
}

interface SaveHighlightMessage {
  type: 'SAVE_HIGHLIGHT';
  highlight: HighlightData;
}

interface GetHighlightsMessage {
  type: 'GET_HIGHLIGHTS';
  url: string;
}

interface DeleteHighlightMessage {
  type: 'DELETE_HIGHLIGHT';
  id: string;
  url: string;
}

interface SearchHighlightsMessage {
  type: 'SEARCH_HIGHLIGHTS';
  query: string;
}

interface GetAllHighlightsMessage {
  type: 'GET_ALL_HIGHLIGHTS';
}

interface ClearUrlHighlightsMessage {
  type: 'CLEAR_URL_HIGHLIGHTS';
  url: string;
}

interface ScrollToHighlightMessage {
  type: 'SCROLL_TO_HIGHLIGHT_PAGE';
  id: string;
  url: string;
}

interface LinkPreviewMessage {
  type: 'LINK_PREVIEW';
  url: string;
}

interface LinkPreviewResponse {
  ok: boolean;
  title?: string;
  description?: string;
  domain?: string;
  error?: string;
}

interface ImageEditOpenMessage {
  type: 'IMAGE_EDIT_OPEN';
  imageUrl: string;
}

interface ImageDescribeMessage {
  type: 'IMAGE_DESCRIBE';
  imageUrl: string;
}

interface ImageSaveMessage {
  type: 'IMAGE_SAVE';
  imageUrl: string;
}

interface ImageEditLoadMessage {
  type: 'IMAGE_EDIT_LOAD';
  dataUrl: string;
}

interface OpenSidebarMessage {
  type: 'OPEN_SIDEBAR';
  panel?: string;
  message?: string;
  conversationId?: string;
}

interface SerpFetchMessage {
  type: 'SERP_FETCH';
  url: string;
  body: string;
  apiKey?: string;
}

type ExtensionMessage =
  | SerpFetchMessage
  | SidebarReadyMessage
  | SaveKnowledgeMessage
  | GetCurrentTabMessage
  | GetPageContentMessage
  | OpenPanelMessage
  | OpenWithTextMessage
  | OcrStartMessage
  | AgentDomMessage
  | AgentExecMessage
  | AgentNavMessage
  | QuickActionMessage
  | YtSubtitlesMessage
  | YtMetadataMessage
  | NetflixSubtitlesMessage
  | NetflixMetadataMessage
  | TranslateBatchMessage
  | SaveHighlightMessage
  | GetHighlightsMessage
  | DeleteHighlightMessage
  | SearchHighlightsMessage
  | GetAllHighlightsMessage
  | ClearUrlHighlightsMessage
  | ScrollToHighlightMessage
  | LinkPreviewMessage
  | ImageEditOpenMessage
  | ImageDescribeMessage
  | ImageSaveMessage
  | OpenSidebarMessage
  | CaptureElementMessage
  | StartCaptureModeMessage
  | StopCaptureModeMessage
  | CaptureExitedMessage
  | FullPageCaptureMessage
  | FullPageDataMessage
  | SaveToCliFeedMessage;

// ── Full page capture message types ─────────────────────────────────────────

interface FullPageCaptureMessage {
  type: 'FULL_PAGE_CAPTURE';
}

interface FullPageDataMessage {
  type: 'FULL_PAGE_DATA';
  data: {
    html: string;
    css: string;
    css_map: Record<string, Record<string, string>>;
    colors: string[];
    fonts: string[];
    metadata: {
      title: string;
      description: string;
      og_image: string;
      og_title: string;
      og_description: string;
      og_type: string;
      og_site_name: string;
      favicon: string;
    };
    source_url: string;
    viewport: { width: number; height: number };
    asset_urls: { images: string[]; stylesheets: string[] };
    responsive_info: { viewport_width: number; media_queries: string[] };
    element_count: number;
  };
}

interface SaveToCliFeedMessage {
  type: 'SAVE_TO_CLI_FEED';
  payload: Record<string, unknown>;
}

// ── Component capture message types ──────────────────────────────────────────

interface CaptureElementMessage {
  type: 'CAPTURE_ELEMENT';
  rect: { x: number; y: number; w: number; h: number };
  elementData: {
    html: string;
    css: Record<string, Record<string, string>>;
    dimensions: { width: number; height: number; padding: string; margin: string };
    textContent: string;
    tagName: string;
    className: string;
  };
}

interface StartCaptureModeMessage {
  type: 'START_CAPTURE_MODE';
}

interface StopCaptureModeMessage {
  type: 'STOP_CAPTURE_MODE';
}

interface CaptureExitedMessage {
  type: 'CAPTURE_MODE_EXITED';
}

interface ComponentCapturedMessage {
  type: 'COMPONENT_CAPTURED';
  data: {
    html: string;
    css: Record<string, Record<string, string>>;
    screenshot_b64: string;
    dimensions: { width: number; height: number; padding: string; margin: string };
    textContent: string;
    tagName: string;
    className: string;
  };
}

// ── Outbound / internal message types ────────────────────────────────────────

interface PrefillTextMessage {
  type: 'PREFILL_TEXT';
  text: string;
  action: string;
  url: string;
  title: string;
}

interface SwitchPanelMessage {
  type: 'SWITCH_PANEL';
  panel: string;
}

interface PdfTabDetectedMessage {
  type: 'PDF_TAB_DETECTED';
  url: string;
  title: string;
}

interface YtTabDetectedMessage {
  type: 'YT_TAB_DETECTED';
  url: string;
  title: string;
}

interface OcrResultMessage {
  type: 'OCR_RESULT';
  text?: string;
  error?: string;
}

interface OcrOverlayMessage {
  type: 'SHOW_OCR_OVERLAY';
  dataUrl: string;
}

interface OcrRegionResponse {
  ok: boolean;
  x: number;
  y: number;
  w: number;
  h: number;
  dpr: number;
}

interface PendingStorage {
  pendingQuery?: string;
  pendingAction?: string;
  pendingUrl?: string;
  pendingTitle?: string;
  pendingPanelSwitch?: string;
  pendingImageDataUrl?: string;
}

interface SaveKnowledgeResponse {
  ok: boolean;
  data?: unknown;
  error?: string;
}

interface TabInfoResponse {
  ok: boolean;
  url?: string;
  title?: string;
}

interface PageContentResponse {
  ok: boolean;
  text?: string;
  url?: string;
  title?: string;
  wordCount?: number;
  isPdf?: boolean;
  isYouTube?: boolean;
  videoTitle?: string;
  transcript?: string;
  error?: string;
}

interface ExtractPageResponse {
  text: string;
  url: string;
  title: string;
  wordCount: number;
  isPdf?: boolean;
  isYouTube?: boolean;
  videoTitle?: string;
  transcript?: string;
}

// ── Context Menu IDs ─────────────────────────────────────────────────────────

const MENU_IDS = {
  parent: 'aura-menu',
  ask: 'aura-ask',
  explain: 'aura-explain',
  summarize: 'aura-summarize',
  translate: 'aura-translate',
  translateEn: 'aura-translate-en',
  translateEs: 'aura-translate-es',
  translateFr: 'aura-translate-fr',
  translateDe: 'aura-translate-de',
  translateZh: 'aura-translate-zh',
  improve: 'aura-improve',
  saveMemory: 'aura-save-memory',
} as const;

type MenuId = (typeof MENU_IDS)[keyof typeof MENU_IDS];

/** Maps a menu ID to the action string sent to the sidebar */
const MENU_ACTION_MAP: Record<string, { action: string; prefix?: string }> = {
  [MENU_IDS.ask]: { action: 'ask' },
  [MENU_IDS.explain]: { action: 'explain', prefix: 'Explain this:\n\n' },
  [MENU_IDS.summarize]: { action: 'summarize', prefix: 'Summarize this:\n\n' },
  [MENU_IDS.improve]: { action: 'improve', prefix: 'Improve this writing:\n\n' },
  [MENU_IDS.translateEn]: { action: 'translate', prefix: 'Translate to English:\n\n' },
  [MENU_IDS.translateEs]: { action: 'translate', prefix: 'Translate to Spanish:\n\n' },
  [MENU_IDS.translateFr]: { action: 'translate', prefix: 'Translate to French:\n\n' },
  [MENU_IDS.translateDe]: { action: 'translate', prefix: 'Translate to German:\n\n' },
  [MENU_IDS.translateZh]: { action: 'translate', prefix: 'Translate to Chinese:\n\n' },
  [MENU_IDS.saveMemory]: { action: 'save' },
};

// ── Startup ──────────────────────────────────────────────────────────────────

ext.runtime.onInstalled.addListener((): void => {
  // Open side panel when toolbar icon is clicked (Chrome only)
  if (chrome.sidePanel) {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
  }

  // ── Build context menu tree ───────────────────────────────────────────────
  // Remove any stale menus first (handles extension updates cleanly)
  ext.contextMenus.removeAll(() => {
    // Parent menu
    ext.contextMenus.create({
      id: MENU_IDS.parent,
      title: 'AURA',
      contexts: ['selection'],
    });

    // Ask AURA about "%s"
    ext.contextMenus.create({
      id: MENU_IDS.ask,
      parentId: MENU_IDS.parent,
      title: 'Ask AURA about "%s"',
      contexts: ['selection'],
    });

    // Explain this
    ext.contextMenus.create({
      id: MENU_IDS.explain,
      parentId: MENU_IDS.parent,
      title: 'Explain this',
      contexts: ['selection'],
    });

    // Summarize this
    ext.contextMenus.create({
      id: MENU_IDS.summarize,
      parentId: MENU_IDS.parent,
      title: 'Summarize this',
      contexts: ['selection'],
    });

    // Translate to... (sub-menu)
    ext.contextMenus.create({
      id: MENU_IDS.translate,
      parentId: MENU_IDS.parent,
      title: 'Translate to...',
      contexts: ['selection'],
    });

    ext.contextMenus.create({
      id: MENU_IDS.translateEn,
      parentId: MENU_IDS.translate,
      title: 'English',
      contexts: ['selection'],
    });

    ext.contextMenus.create({
      id: MENU_IDS.translateEs,
      parentId: MENU_IDS.translate,
      title: 'Spanish',
      contexts: ['selection'],
    });

    ext.contextMenus.create({
      id: MENU_IDS.translateFr,
      parentId: MENU_IDS.translate,
      title: 'French',
      contexts: ['selection'],
    });

    ext.contextMenus.create({
      id: MENU_IDS.translateDe,
      parentId: MENU_IDS.translate,
      title: 'German',
      contexts: ['selection'],
    });

    ext.contextMenus.create({
      id: MENU_IDS.translateZh,
      parentId: MENU_IDS.translate,
      title: 'Chinese',
      contexts: ['selection'],
    });

    // Separator before utility actions
    ext.contextMenus.create({
      id: 'aura-sep-1',
      parentId: MENU_IDS.parent,
      type: 'separator',
      contexts: ['selection'],
    });

    // Improve writing
    ext.contextMenus.create({
      id: MENU_IDS.improve,
      parentId: MENU_IDS.parent,
      title: 'Improve writing',
      contexts: ['selection'],
    });

    // Save to memory
    ext.contextMenus.create({
      id: MENU_IDS.saveMemory,
      parentId: MENU_IDS.parent,
      title: 'Save to memory',
      contexts: ['selection'],
    });
  });
});

// ── Context Menu Click Handler ────────────────────────────────────────────────

ext.contextMenus.onClicked.addListener(
  (info: chrome.contextMenus.OnClickData, tab?: chrome.tabs.Tab): void => {
    const menuId = info.menuItemId as string;
    const mapping = MENU_ACTION_MAP[menuId];
    if (!mapping) return; // Clicked the parent or separator — ignore

    const selectedText: string = info.selectionText || '';
    const pageUrl: string = tab?.url || '';
    const pageTitle: string = tab?.title || '';

    // "Save to memory" goes directly to the backend, no sidebar needed
    if (menuId === MENU_IDS.saveMemory) {
      fetch(`${BACKEND}/api/knowledge/save`, {
        method: 'POST',
        headers: backendHeaders(),
        body: JSON.stringify({
          text: selectedText,
          url: pageUrl,
          title: pageTitle,
          tags: [],
          source_type: 'selection',
        }),
      }).catch(() => {}); // fire-and-forget
      return;
    }

    // For all other actions, open sidebar with pre-filled text
    const queryText: string = mapping.prefix
      ? mapping.prefix + selectedText
      : selectedText;

    ext.storage.local.set({
      pendingQuery: queryText,
      pendingAction: mapping.action,
      pendingUrl: pageUrl,
      pendingTitle: pageTitle,
    });

    // Open the side panel
    if (chrome.sidePanel && tab) {
      chrome.sidePanel.open({ windowId: tab.windowId! });
    } else if (typeof browser !== 'undefined' && (browser as any)?.sidebarAction) {
      (browser as any).sidebarAction.open();
    }
  }
);

// ── PDF + YouTube Tab Detection ───────────────────────────────────────────────

ext.tabs.onUpdated.addListener(
  (tabId: number, changeInfo: chrome.tabs.OnUpdatedInfo, tab: chrome.tabs.Tab): void => {
    if (changeInfo.status !== 'complete') return;
    const url: string = tab.url || '';

    if (/\.pdf($|\?)/i.test(url)) {
      ext.runtime.sendMessage({
        type: 'PDF_TAB_DETECTED',
        url,
        title: tab.title || url,
      } satisfies PdfTabDetectedMessage).catch(() => {});
    }

    if (url.includes('youtube.com/watch')) {
      ext.runtime.sendMessage({
        type: 'YT_TAB_DETECTED',
        url,
        title: tab.title || url,
      } satisfies YtTabDetectedMessage).catch(() => {});

      // Inject YouTube subtitle interceptor into MAIN world
      chrome.scripting.executeScript({
        target: { tabId },
        files: ['youtube-inject.js'],
        world: 'MAIN' as any,
      }).catch(() => {});
    }

    if (url.includes('netflix.com/watch')) {
      ext.runtime.sendMessage({
        type: 'NETFLIX_TAB_DETECTED',
        url,
        title: tab.title || url,
      } satisfies NetflixTabDetectedMessage).catch(() => {});

      // Inject Netflix subtitle interceptor into MAIN world
      chrome.scripting.executeScript({
        target: { tabId },
        files: ['netflix-inject.js'],
        world: 'MAIN' as any,
      }).catch(() => {});
    }
  }
);

// ── Message Router ────────────────────────────────────────────────────────────

ext.runtime.onMessage.addListener(
  (
    msg: ExtensionMessage,
    sender: chrome.runtime.MessageSender,
    sendResponse: (response?: unknown) => void
  ): boolean => {
    switch (msg.type) {
      // Sidebar has loaded — send any pending prefill text or panel switch
      case 'SIDEBAR_READY': {
        ext.storage.local.get(
          ['pendingQuery', 'pendingAction', 'pendingUrl', 'pendingTitle', 'pendingPanelSwitch', 'pendingImageDataUrl'],
          (data: PendingStorage) => {
            if (data.pendingQuery) {
              ext.runtime.sendMessage({
                type: 'PREFILL_TEXT',
                text: data.pendingQuery,
                action: data.pendingAction || 'ask',
                url: data.pendingUrl || '',
                title: data.pendingTitle || '',
              } satisfies PrefillTextMessage).catch(() => {});
              ext.storage.local.remove(['pendingQuery', 'pendingAction', 'pendingUrl', 'pendingTitle']);
            } else if (data.pendingImageDataUrl) {
              // Image edit was requested before sidebar was open
              ext.runtime.sendMessage({
                type: 'SWITCH_PANEL',
                panel: 'image',
              } satisfies SwitchPanelMessage).catch(() => {});
              ext.runtime.sendMessage({
                type: 'IMAGE_EDIT_LOAD',
                dataUrl: data.pendingImageDataUrl,
              } satisfies ImageEditLoadMessage).catch(() => {});
              ext.storage.local.remove(['pendingImageDataUrl']);
            } else if (data.pendingPanelSwitch) {
              ext.runtime.sendMessage({
                type: 'SWITCH_PANEL',
                panel: data.pendingPanelSwitch,
              } satisfies SwitchPanelMessage).catch(() => {});
              ext.storage.local.remove(['pendingPanelSwitch']);
            }
          }
        );
        return false;
      }

      // Content script → background → backend: save knowledge clip
      case 'SAVE_KNOWLEDGE': {
        fetch(`${BACKEND}/api/knowledge/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: msg.text,
            url: msg.url,
            title: msg.title,
            tags: msg.tags || [],
            source_type: 'selection',
          }),
        })
          .then((r) => r.json())
          .then((data: unknown) =>
            sendResponse({ ok: true, data } satisfies SaveKnowledgeResponse)
          )
          .catch((err: Error) =>
            sendResponse({ ok: false, error: err.message } satisfies SaveKnowledgeResponse)
          );
        return true;
      }

      // Sidebar asks background to extract page content from active tab
      case 'GET_CURRENT_TAB': {
        ext.tabs.query(
          { active: true, currentWindow: true },
          (tabs: chrome.tabs.Tab[]): void => {
            const t = tabs[0];
            sendResponse(
              t
                ? ({ ok: true, url: t.url || '', title: t.title || '' } satisfies TabInfoResponse)
                : ({ ok: false } satisfies TabInfoResponse)
            );
          }
        );
        return true;
      }

      case 'GET_PAGE_CONTENT': {
        ext.tabs.query(
          { active: true, currentWindow: true },
          async (tabs: chrome.tabs.Tab[]): Promise<void> => {
            const activeTab = tabs[0];
            if (!activeTab) {
              sendResponse({ ok: false, error: 'No active tab' } satisfies PageContentResponse);
              return;
            }

            const tabId: number = activeTab.id!;
            const url: string = activeTab.url || '';

            // Block protected pages extensions cannot touch
            if (
              !url ||
              url.startsWith('chrome://') ||
              url.startsWith('chrome-extension://') ||
              url.startsWith('about:') ||
              url.startsWith('edge://') ||
              url.startsWith('moz-extension://')
            ) {
              sendResponse({ ok: false, error: 'Protected page' } satisfies PageContentResponse);
              return;
            }

            // Try content script first
            ext.tabs.sendMessage(
              tabId,
              { type: 'EXTRACT_PAGE' },
              (response?: ExtractPageResponse): void => {
                if (!ext.runtime.lastError && response) {
                  sendResponse({ ok: true, ...response } satisfies PageContentResponse);
                  return;
                }

                // Content script not running in this tab (was opened before extension loaded).
                // Inject it on-the-fly using scripting API then extract.
                // Guard with a 10-second timeout so sendResponse is always called.
                let responded = false;
                const extractTimer: ReturnType<typeof setTimeout> = setTimeout(() => {
                  if (!responded) {
                    responded = true;
                    sendResponse({
                      ok: false,
                      error: 'Extraction timed out',
                    } satisfies PageContentResponse);
                  }
                }, 10000);

                ext.scripting
                  .executeScript({
                    target: { tabId },
                    func: (): { text: string; url: string; title: string } => ({
                      text: (document.body?.innerText || '').slice(0, 50000),
                      url: location.href,
                      title: document.title,
                    }),
                  })
                  .then(
                    (
                      results: chrome.scripting.InjectionResult<{
                        text: string;
                        url: string;
                        title: string;
                      }>[]
                    ) => {
                      clearTimeout(extractTimer);
                      if (responded) return;
                      responded = true;
                      const r = results?.[0]?.result;
                      if (r?.text) {
                        sendResponse({ ok: true, ...r } satisfies PageContentResponse);
                      } else {
                        sendResponse({
                          ok: false,
                          error: 'Could not extract page text',
                        } satisfies PageContentResponse);
                      }
                    }
                  )
                  .catch((err: Error) => {
                    clearTimeout(extractTimer);
                    if (responded) return;
                    responded = true;
                    sendResponse({
                      ok: false,
                      error: err.message,
                    } satisfies PageContentResponse);
                  });
              }
            );
          }
        );
        return true;
      }

      // Dock button: open sidebar and switch to a specific panel
      case 'OPEN_PANEL': {
        if (sender.tab) {
          if (chrome.sidePanel) {
            chrome.sidePanel.open({ windowId: sender.tab.windowId! });
          } else if (typeof browser !== 'undefined' && (browser as any)?.sidebarAction) {
            (browser as any).sidebarAction.open();
          }
        }
        // Attempt to notify sidebar immediately (works if already open)
        ext.runtime
          .sendMessage({ type: 'SWITCH_PANEL', panel: msg.panel } satisfies SwitchPanelMessage)
          .catch(() => {
            // Sidebar not yet open — store for delivery on SIDEBAR_READY
            ext.storage.local.set({ pendingPanelSwitch: msg.panel });
          });
        return false;
      }

      // New tab page: open sidebar with optional panel, message, or conversationId
      case 'OPEN_SIDEBAR': {
        // Open the sidebar in the current window
        ext.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          const tab = tabs?.[0];
          if (tab?.windowId != null && chrome.sidePanel) {
            chrome.sidePanel.open({ windowId: tab.windowId });
          }
        });

        if (msg.message) {
          // Pre-fill chat with the message
          ext.storage.local.set({
            pendingQuery: msg.message,
            pendingAction: 'ask',
          });
        } else if (msg.panel) {
          ext.runtime
            .sendMessage({ type: 'SWITCH_PANEL', panel: msg.panel } satisfies SwitchPanelMessage)
            .catch(() => {
              ext.storage.local.set({ pendingPanelSwitch: msg.panel });
            });
        }
        return false;
      }

      // Toolbar button in content script: open sidebar with pre-filled action
      case 'OPEN_WITH_TEXT': {
        ext.storage.local.set({
          pendingQuery: msg.text,
          pendingAction: msg.action,
          pendingUrl: msg.url || '',
          pendingTitle: msg.title || '',
        });
        if (sender.tab) {
          if (chrome.sidePanel) {
            chrome.sidePanel.open({ windowId: sender.tab.windowId! });
          } else if (typeof browser !== 'undefined' && (browser as any)?.sidebarAction) {
            (browser as any).sidebarAction.open();
          }
        }
        return false;
      }

      // OCR: capture visible tab then ask content script for region selection
      case 'OCR_START': {
        // captureVisibleTab is Chrome-only; no standard ext.* equivalent
        chrome.tabs.captureVisibleTab(
          null as unknown as number,
          { format: 'png' },
          (dataUrl: string): void => {
            if (chrome.runtime.lastError || !dataUrl) {
              ext.runtime.sendMessage({
                type: 'OCR_RESULT',
                error: 'Screenshot failed',
              } satisfies OcrResultMessage).catch(() => {});
              return;
            }
            ext.tabs.query(
              { active: true, currentWindow: true },
              ([tab]: chrome.tabs.Tab[]): void => {
                if (!tab) {
                  ext.runtime.sendMessage({
                    type: 'OCR_RESULT',
                    error: 'No active tab',
                  } satisfies OcrResultMessage).catch(() => {});
                  return;
                }
                ext.tabs.sendMessage(
                  tab.id!,
                  { type: 'SHOW_OCR_OVERLAY', dataUrl } satisfies OcrOverlayMessage,
                  async (region?: OcrRegionResponse): Promise<void> => {
                    if (!region?.ok) {
                      ext.runtime.sendMessage({
                        type: 'OCR_RESULT',
                        error: 'Cancelled',
                      } satisfies OcrResultMessage).catch(() => {});
                      return;
                    }
                    try {
                      const { x, y, w, h, dpr } = region;
                      const imgBlob: Blob = await fetch(dataUrl).then((r) => r.blob());
                      const bmp: ImageBitmap = await createImageBitmap(imgBlob);
                      const cw: number = Math.max(1, Math.round(w * dpr));
                      const ch: number = Math.max(1, Math.round(h * dpr));
                      const oc = new OffscreenCanvas(cw, ch);
                      const ctx = oc.getContext('2d')!;
                      ctx.drawImage(bmp, x * dpr, y * dpr, w * dpr, h * dpr, 0, 0, cw, ch);
                      bmp.close(); // Release ImageBitmap memory
                      const blob: Blob = await oc.convertToBlob({ type: 'image/png' });
                      const arrayBuf = await blob.arrayBuffer();
                      const b64: string = btoa(
                        String.fromCharCode(...new Uint8Array(arrayBuf))
                      );
                      const resp: Response = await fetch(`${BACKEND}/api/ocr`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_b64: b64 }),
                      });
                      if (!resp.ok) {
                        ext.runtime.sendMessage({
                          type: 'OCR_RESULT',
                          error: `OCR server error ${resp.status}`,
                        } satisfies OcrResultMessage).catch(() => {});
                        return;
                      }
                      const d: { text?: string; detail?: string } = await resp.json();
                      ext.runtime.sendMessage({
                        type: 'OCR_RESULT',
                        text: d.text || '',
                        error: d.detail || '',
                      } satisfies OcrResultMessage).catch(() => {});
                    } catch (e) {
                      ext.runtime.sendMessage({
                        type: 'OCR_RESULT',
                        error: String(e),
                      } satisfies OcrResultMessage).catch(() => {});
                    }
                  }
                );
              }
            );
          }
        );
        return false;
      }

      // Component Capture: relay START/STOP to active tab's content script
      case 'START_CAPTURE_MODE': {
        ext.tabs.query(
          { active: true, currentWindow: true },
          ([tab]: chrome.tabs.Tab[]): void => {
            if (!tab?.id) return;
            ext.tabs.sendMessage(tab.id, { type: 'START_CAPTURE_MODE' });
          }
        );
        return false;
      }

      case 'STOP_CAPTURE_MODE': {
        ext.tabs.query(
          { active: true, currentWindow: true },
          ([tab]: chrome.tabs.Tab[]): void => {
            if (!tab?.id) return;
            ext.tabs.sendMessage(tab.id, { type: 'STOP_CAPTURE_MODE' });
          }
        );
        return false;
      }

      // Content script signals capture mode was exited (Esc key)
      case 'CAPTURE_MODE_EXITED': {
        ext.runtime.sendMessage({ type: 'CAPTURE_MODE_EXITED' }).catch(() => {});
        return false;
      }

      // Content script captured an element — take screenshot, crop, relay to sidebar
      case 'CAPTURE_ELEMENT': {
        const captureRect = msg.rect;
        const elementData = msg.elementData;

        chrome.tabs.captureVisibleTab(
          null as unknown as number,
          { format: 'png' },
          async (dataUrl: string): Promise<void> => {
            let screenshot_b64 = '';

            if (!chrome.runtime.lastError && dataUrl) {
              try {
                const dpr = 1; // captureVisibleTab already uses device pixels on most systems
                const imgBlob: Blob = await fetch(dataUrl).then((r) => r.blob());
                const bmp: ImageBitmap = await createImageBitmap(imgBlob);

                // Calculate crop coordinates — captureVisibleTab may return at device pixel ratio
                const scaleX = bmp.width / (typeof screen !== 'undefined' ? screen.width : bmp.width);
                const scaleY = bmp.height / (typeof screen !== 'undefined' ? screen.height : bmp.height);
                const scale = Math.max(scaleX, scaleY, 1);

                const cx = Math.max(0, Math.round(captureRect.x * scale));
                const cy = Math.max(0, Math.round(captureRect.y * scale));
                const cw = Math.max(1, Math.min(Math.round(captureRect.w * scale), bmp.width - cx));
                const ch = Math.max(1, Math.min(Math.round(captureRect.h * scale), bmp.height - cy));

                const oc = new OffscreenCanvas(cw, ch);
                const ctx = oc.getContext('2d')!;
                ctx.drawImage(bmp, cx, cy, cw, ch, 0, 0, cw, ch);
                bmp.close();

                const blob: Blob = await oc.convertToBlob({ type: 'image/png' });
                const arrayBuf = await blob.arrayBuffer();
                screenshot_b64 = btoa(
                  String.fromCharCode(...new Uint8Array(arrayBuf))
                );
              } catch (_e) {
                // Proceed without screenshot
              }
            }

            // Relay captured component data to sidebar
            const captured: ComponentCapturedMessage = {
              type: 'COMPONENT_CAPTURED',
              data: {
                ...elementData,
                screenshot_b64,
              },
            };

            ext.runtime.sendMessage(captured).catch(() => {});
            sendResponse({ ok: true });
          }
        );
        return true; // async
      }

      // Full Page Capture: tell content script to extract page data, take screenshot
      case 'FULL_PAGE_CAPTURE': {
        ext.tabs.query(
          { active: true, currentWindow: true },
          ([tab]: chrome.tabs.Tab[]): void => {
            if (!tab?.id) {
              ext.runtime.sendMessage({
                type: 'FULL_PAGE_CAPTURE_ERROR',
                error: 'No active tab found',
              }).catch(() => {});
              sendResponse({ ok: false });
              return;
            }
            const tabId = tab.id;

            // First take a screenshot of the visible tab
            chrome.tabs.captureVisibleTab(
              null as unknown as number,
              { format: 'png' },
              async (dataUrl: string): Promise<void> => {
                let screenshot_b64 = '';

                if (!chrome.runtime.lastError && dataUrl) {
                  try {
                    // Convert data URL to base64
                    screenshot_b64 = dataUrl.replace(/^data:image\/png;base64,/, '');
                  } catch (_e) {
                    // Continue without screenshot
                  }
                }

                // Now ask content script to extract full page data
                try {
                  ext.tabs.sendMessage(
                    tabId,
                    { type: 'EXTRACT_FULL_PAGE' },
                    (response: any) => {
                      if (chrome.runtime.lastError || !response?.ok) {
                        ext.runtime.sendMessage({
                          type: 'FULL_PAGE_CAPTURE_ERROR',
                          error: chrome.runtime.lastError?.message || 'Page extraction failed',
                        }).catch(() => {});
                        return;
                      }

                      // Relay to sidebar with screenshot
                      ext.runtime.sendMessage({
                        type: 'FULL_PAGE_CAPTURED',
                        data: {
                          ...response.data,
                          screenshot_b64,
                          timestamp: Date.now() / 1000,
                        },
                      }).catch(() => {});
                    }
                  );
                } catch (_e) {
                  ext.runtime.sendMessage({
                    type: 'FULL_PAGE_CAPTURE_ERROR',
                    error: 'Failed to communicate with content script',
                  }).catch(() => {});
                }
              }
            );
            sendResponse({ ok: true });
          }
        );
        return true; // async
      }

      // Content script sends full page extracted data
      case 'FULL_PAGE_DATA': {
        ext.runtime.sendMessage({
          type: 'FULL_PAGE_CAPTURED',
          data: msg.data,
        }).catch(() => {});
        sendResponse({ ok: true });
        return false;
      }

      // Save capture data to CLI feed via backend API
      case 'SAVE_TO_CLI_FEED': {
        const payload = (msg as SaveToCliFeedMessage).payload;
        fetch(`${BACKEND}/api/feed/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })
          .then((r) => r.json())
          .then((data) => {
            sendResponse({ ok: true, ...data });
          })
          .catch((err) => {
            sendResponse({ ok: false, error: err.message });
          });
        return true; // async
      }

      // Quick-action: content script asks background to call LLM for inline text edits
      case 'QUICK_ACTION': {
        const PROMPT_MAP: Record<string, string> = {
          improve: 'Improve this text. Return ONLY the improved text, no explanation:\n\n',
          expand: 'Expand this text to be longer and more detailed. Return ONLY the expanded text, no explanation:\n\n',
          shorten: 'Make this text more concise. Return ONLY the shortened text, no explanation:\n\n',
          fix_grammar: 'Fix all grammar and spelling errors. Return ONLY the corrected text, no explanation:\n\n',
          translate: 'Translate to {lang}. Return ONLY the translation, no explanation:\n\n',
          draft_reply: 'You are an email assistant. Based on the email thread context below, draft a professional and helpful reply. Return ONLY the reply text, no explanation, no subject line, no greeting preamble like "Here is a draft".\n\nEmail thread:\n{thread}\n\nCurrent draft (if any):\n',
          make_formal: 'Rewrite this email text in a professional, formal tone. Keep the same meaning and intent. Return ONLY the rewritten text, no explanation:\n\n',
          make_casual: 'Rewrite this email text in a friendly, casual tone. Keep the same meaning and intent. Return ONLY the rewritten text, no explanation:\n\n',
          gmail_translate: 'Translate this email text to {lang}. Return ONLY the translation, no explanation:\n\n',
        };

        let prompt = PROMPT_MAP[msg.action] || PROMPT_MAP.improve;
        if ((msg.action === 'translate' || msg.action === 'gmail_translate') && msg.language) {
          prompt = prompt.replace('{lang}', msg.language);
        } else if (msg.action === 'translate' || msg.action === 'gmail_translate') {
          prompt = prompt.replace('{lang}', 'English');
        }
        if (msg.action === 'draft_reply' && (msg as QuickActionMessage).threadContext) {
          prompt = prompt.replace('{thread}', (msg as QuickActionMessage).threadContext || '');
        } else if (msg.action === 'draft_reply') {
          prompt = prompt.replace('{thread}', '(no thread context available)');
        }
        prompt += msg.text;

        fetch(`${BACKEND}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: prompt,
            conversation_id: '__quick_action__',
            stream: false,
          }),
        })
          .then((r) => r.json())
          .then((data: { response?: string; message?: string }) => {
            sendResponse({
              ok: true,
              result: data.response || data.message || '',
            } satisfies QuickActionResponse);
          })
          .catch((err: Error) => {
            sendResponse({
              ok: false,
              error: err.message,
            } satisfies QuickActionResponse);
          });
        return true; // async
      }

      // Browser Agent relay handlers
      case 'AGENT_DOM':
      case 'AGENT_EXEC':
      case 'AGENT_NAV': {
        ext.tabs.query(
          { active: true, currentWindow: true },
          ([tab]: chrome.tabs.Tab[]): void => {
            if (!tab) {
              sendResponse({ ok: false, error: 'No active tab' });
              return;
            }
            if (msg.type === 'AGENT_DOM') {
              ext.tabs.sendMessage(tab.id!, { type: 'GET_DOM' }, (r: unknown) =>
                sendResponse(r)
              );
            } else if (msg.type === 'AGENT_EXEC') {
              ext.tabs.sendMessage(
                tab.id!,
                { type: 'EXEC_ACTION', action: msg.action },
                (r: unknown) => sendResponse(r)
              );
            } else {
              // AGENT_NAV — only allow http/https schemes
              if (!isSafeScheme(msg.url)) {
                sendResponse({ ok: false, error: 'Blocked URL scheme' });
                return;
              }
              ext.tabs.update(tab.id!, { url: msg.url }, () =>
                sendResponse({ ok: true })
              );
            }
          }
        );
        return true;
      }

      // YouTube subtitle/metadata relay: content script → sidebar
      case 'YT_SUBTITLES':
      case 'YT_METADATA': {
        ext.runtime.sendMessage(msg).catch(() => {});
        return false;
      }

      // Netflix subtitle/metadata relay: content script → sidebar
      case 'NETFLIX_SUBTITLES':
      case 'NETFLIX_METADATA': {
        ext.runtime.sendMessage(msg).catch(() => {});
        return false;
      }

      // Full page translation: batch translate text blocks via LLM
      case 'TRANSLATE_BATCH': {
        const batchMsg = msg as TranslateBatchMessage;
        const { texts, targetLang } = batchMsg;

        // Build a numbered prompt so we can parse results back
        const numbered = texts.map((t, i) => `[${i + 1}] ${t}`).join('\n\n');
        const prompt = `Translate the following numbered text blocks to ${targetLang}. Return ONLY the translations, keeping the same [N] numbering format. Preserve paragraph structure within each block. Do not add explanations.\n\n${numbered}`;

        fetch(`${BACKEND}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: prompt,
            conversation_id: '__page_translate__',
            stream: false,
          }),
        })
          .then((r) => r.json())
          .then((data: { response?: string; message?: string }) => {
            const raw = data.response || data.message || '';
            // Parse numbered translations back out
            const translations: string[] = [];
            for (let i = 0; i < texts.length; i++) {
              const marker = `[${i + 1}]`;
              const nextMarker = `[${i + 2}]`;
              const startIdx = raw.indexOf(marker);
              if (startIdx === -1) {
                translations.push(raw && i === 0 ? raw.trim() : '[Translation unavailable]');
                continue;
              }
              const contentStart = startIdx + marker.length;
              const endIdx = i < texts.length - 1 ? raw.indexOf(nextMarker, contentStart) : -1;
              const segment = endIdx === -1
                ? raw.slice(contentStart)
                : raw.slice(contentStart, endIdx);
              translations.push(segment.trim());
            }
            sendResponse({ ok: true, translations } satisfies TranslateBatchResponse);
          })
          .catch((err: Error) => {
            sendResponse({ ok: false, error: err.message } satisfies TranslateBatchResponse);
          });
        return true; // async
      }

      // ── Highlight handlers ──────────────────────────────────────────────────

      case 'SAVE_HIGHLIGHT': {
        const hl = msg.highlight;
        ext.storage.local.get(['aura_highlights'], (data: Record<string, unknown>) => {
          const store: Record<string, HighlightData[]> = (data.aura_highlights as Record<string, HighlightData[]>) || {};

          // Check total count limit (1000)
          let total = 0;
          for (const url of Object.keys(store)) total += store[url].length;
          if (total >= 1000) {
            sendResponse({ ok: false, error: 'Highlight limit reached (1000). Please delete some highlights first.' });
            return;
          }

          if (!store[hl.url]) store[hl.url] = [];
          store[hl.url].push(hl);
          ext.storage.local.set({ aura_highlights: store }, () => {
            sendResponse({ ok: true });
          });
        });
        return true;
      }

      case 'GET_HIGHLIGHTS': {
        const url = (msg as GetHighlightsMessage).url;
        ext.storage.local.get(['aura_highlights'], (data: Record<string, unknown>) => {
          const store: Record<string, HighlightData[]> = (data.aura_highlights as Record<string, HighlightData[]>) || {};
          sendResponse({ ok: true, highlights: store[url] || [] });
        });
        return true;
      }

      case 'DELETE_HIGHLIGHT': {
        const { id, url } = msg as DeleteHighlightMessage;
        ext.storage.local.get(['aura_highlights'], (data: Record<string, unknown>) => {
          const store: Record<string, HighlightData[]> = (data.aura_highlights as Record<string, HighlightData[]>) || {};
          if (store[url]) {
            store[url] = store[url].filter((h: HighlightData) => h.id !== id);
            if (store[url].length === 0) delete store[url];
          }
          ext.storage.local.set({ aura_highlights: store }, () => {
            sendResponse({ ok: true });
          });
        });
        return true;
      }

      case 'SEARCH_HIGHLIGHTS': {
        const query = (msg as SearchHighlightsMessage).query.toLowerCase();
        ext.storage.local.get(['aura_highlights'], (data: Record<string, unknown>) => {
          const store: Record<string, HighlightData[]> = (data.aura_highlights as Record<string, HighlightData[]>) || {};
          const results: HighlightData[] = [];
          for (const url of Object.keys(store)) {
            for (const hl of store[url]) {
              if (
                hl.text.toLowerCase().includes(query) ||
                hl.pageTitle.toLowerCase().includes(query) ||
                url.toLowerCase().includes(query)
              ) {
                results.push(hl);
              }
            }
          }
          results.sort((a, b) => b.timestamp - a.timestamp);
          sendResponse({ ok: true, highlights: results });
        });
        return true;
      }

      case 'GET_ALL_HIGHLIGHTS': {
        ext.storage.local.get(['aura_highlights'], (data: Record<string, unknown>) => {
          const store: Record<string, HighlightData[]> = (data.aura_highlights as Record<string, HighlightData[]>) || {};
          sendResponse({ ok: true, store });
        });
        return true;
      }

      case 'CLEAR_URL_HIGHLIGHTS': {
        const url = (msg as ClearUrlHighlightsMessage).url;
        ext.storage.local.get(['aura_highlights'], (data: Record<string, unknown>) => {
          const store: Record<string, HighlightData[]> = (data.aura_highlights as Record<string, HighlightData[]>) || {};
          delete store[url];
          ext.storage.local.set({ aura_highlights: store }, () => {
            sendResponse({ ok: true });
          });
        });
        return true;
      }

      case 'SCROLL_TO_HIGHLIGHT_PAGE': {
        const { id, url } = msg as ScrollToHighlightMessage;
        ext.tabs.query({ active: true, currentWindow: true }, ([tab]: chrome.tabs.Tab[]) => {
          if (!tab) { sendResponse({ ok: false }); return; }
          const tabUrl = tab.url || '';
          // If already on the right page, just scroll
          if (tabUrl === url || tabUrl.split('#')[0] === url.split('#')[0]) {
            ext.tabs.sendMessage(tab.id!, { type: 'SCROLL_TO_HIGHLIGHT', id }, () => {
              sendResponse({ ok: true });
            });
          } else {
            // Navigate to the page, then scroll after load
            ext.tabs.update(tab.id!, { url }, () => {
              // Wait for page load then scroll
              const listener = (tabId: number, info: chrome.tabs.OnUpdatedInfo) => {
                if (tabId === tab.id && info.status === 'complete') {
                  ext.tabs.onUpdated.removeListener(listener);
                  setTimeout(() => {
                    ext.tabs.sendMessage(tab.id!, { type: 'SCROLL_TO_HIGHLIGHT', id }, () => {
                      // Ignore errors — content script may not be ready yet
                      void chrome.runtime.lastError;
                    });
                  }, 2000); // Wait for content script + highlights to restore
                  sendResponse({ ok: true });
                }
              };
              ext.tabs.onUpdated.addListener(listener);
              // Safety timeout
              setTimeout(() => {
                ext.tabs.onUpdated.removeListener(listener);
              }, 15000);
            });
          }
        });
        return true;
      }

      // Link preview: fetch page title + meta description for hover popup
      case 'LINK_PREVIEW': {
        const previewUrl = (msg as LinkPreviewMessage).url;
        let previewDomain = '';
        try { previewDomain = new URL(previewUrl).hostname; } catch { previewDomain = previewUrl; }

        // Security: validate URL scheme and block private/internal hosts
        if (!isSafeScheme(previewUrl) || isPrivateHost(previewUrl)) {
          sendResponse({ ok: false, error: 'Blocked URL' } satisfies LinkPreviewResponse);
          return true;
        }

        // Rate limit: max concurrent preview fetches
        if (_linkPreviewActive >= LINK_PREVIEW_MAX_CONCURRENT) {
          sendResponse({ ok: false, error: 'Too many concurrent previews' } satisfies LinkPreviewResponse);
          return true;
        }
        _linkPreviewActive++;

        fetch(previewUrl, {
          signal: AbortSignal.timeout(3000),
          headers: { 'Accept': 'text/html' },
        })
          .then((r) => {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            // Check Content-Length to enforce size limit
            const contentLength = r.headers.get('Content-Length');
            if (contentLength && parseInt(contentLength, 10) > LINK_PREVIEW_MAX_BYTES) {
              throw new Error('Response too large');
            }
            return r.text();
          })
          .then((html: string) => {
            // Enforce size limit on actual body (Content-Length may be absent)
            if (html.length > LINK_PREVIEW_MAX_BYTES) {
              html = html.slice(0, LINK_PREVIEW_MAX_BYTES);
            }

            // Parse title
            let title = '';
            const titleMatch = html.match(/<title[^>]*>([^<]*)<\/title>/i);
            if (titleMatch) title = titleMatch[1].trim();

            // Parse meta description
            let description = '';
            const descMatch = html.match(/<meta\s+name=["']description["']\s+content=["']([^"']*)["']/i)
              || html.match(/<meta\s+content=["']([^"']*)["']\s+name=["']description["']/i);
            if (descMatch) description = descMatch[1].trim();

            // Try og:description as fallback
            if (!description) {
              const ogMatch = html.match(/<meta\s+property=["']og:description["']\s+content=["']([^"']*)["']/i)
                || html.match(/<meta\s+content=["']([^"']*)["']\s+property=["']og:description["']/i);
              if (ogMatch) description = ogMatch[1].trim();
            }

            // Try og:title as fallback for title
            if (!title) {
              const ogTitleMatch = html.match(/<meta\s+property=["']og:title["']\s+content=["']([^"']*)["']/i)
                || html.match(/<meta\s+content=["']([^"']*)["']\s+property=["']og:title["']/i);
              if (ogTitleMatch) title = ogTitleMatch[1].trim();
            }

            // Decode HTML entities in title/description
            const decodeEntities = (s: string): string =>
              s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&#x27;/g, "'");

            sendResponse({
              ok: true,
              title: decodeEntities(title),
              description: decodeEntities(description),
              domain: previewDomain,
            } satisfies LinkPreviewResponse);
          })
          .catch(() => {
            // CORS, timeout, or other fetch failure — return just the domain
            sendResponse({
              ok: true,
              title: '',
              description: '',
              domain: previewDomain,
            } satisfies LinkPreviewResponse);
          })
          .finally(() => {
            _linkPreviewActive--;
          });
        return true; // async
      }

      // Image hover toolbar: "Edit in AURA" — fetch image, convert to data URL, send to sidebar
      case 'IMAGE_EDIT_OPEN': {
        const imgUrl = (msg as ImageEditOpenMessage).imageUrl;
        // Open sidebar + switch to image panel
        if (sender.tab) {
          if (chrome.sidePanel) {
            chrome.sidePanel.open({ windowId: sender.tab.windowId! });
          } else if (typeof browser !== 'undefined' && (browser as any)?.sidebarAction) {
            (browser as any).sidebarAction.open();
          }
        }
        // Switch to image panel
        ext.runtime
          .sendMessage({ type: 'SWITCH_PANEL', panel: 'image' } satisfies SwitchPanelMessage)
          .catch(() => {
            ext.storage.local.set({ pendingPanelSwitch: 'image' });
          });

        // Fetch image and convert to data URL, then send to sidebar
        fetch(imgUrl, { signal: AbortSignal.timeout(10000) })
          .then((r) => {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.blob();
          })
          .then(async (blob) => {
            const arrayBuf = await blob.arrayBuffer();
            const bytes = new Uint8Array(arrayBuf);
            let binary = '';
            for (let i = 0; i < bytes.length; i++) {
              binary += String.fromCharCode(bytes[i]);
            }
            const b64 = btoa(binary);
            const mime = blob.type || 'image/png';
            const dataUrl = `data:${mime};base64,${b64}`;
            // Send data URL to sidebar for the edit panel
            ext.runtime
              .sendMessage({ type: 'IMAGE_EDIT_LOAD', dataUrl } satisfies ImageEditLoadMessage)
              .catch(() => {
                // Sidebar may not be ready yet; store for later
                ext.storage.local.set({ pendingImageDataUrl: dataUrl });
              });
          })
          .catch(() => {
            // If fetch fails (CORS etc.), try sending the URL directly for the sidebar to handle
            ext.runtime
              .sendMessage({ type: 'IMAGE_EDIT_LOAD', dataUrl: imgUrl } satisfies ImageEditLoadMessage)
              .catch(() => {
                ext.storage.local.set({ pendingImageDataUrl: imgUrl });
              });
          });
        return false;
      }

      // Image hover toolbar: "Describe" — open sidebar, send image URL to chat
      case 'IMAGE_DESCRIBE': {
        const descImgUrl = (msg as ImageDescribeMessage).imageUrl;
        // Store as pending prefill so sidebar loads with the describe request
        ext.storage.local.set({
          pendingQuery: `Describe this image in detail: ${descImgUrl}`,
          pendingAction: 'ask',
          pendingUrl: descImgUrl,
          pendingTitle: 'Image Description',
        });
        if (sender.tab) {
          if (chrome.sidePanel) {
            chrome.sidePanel.open({ windowId: sender.tab.windowId! });
          } else if (typeof browser !== 'undefined' && (browser as any)?.sidebarAction) {
            (browser as any).sidebarAction.open();
          }
        }
        return false;
      }

      // Image hover toolbar: "Save" — download image via the active tab
      case 'IMAGE_SAVE': {
        const saveImgUrl = (msg as ImageSaveMessage).imageUrl;
        // Use chrome.downloads API to save the image
        if (chrome.downloads) {
          const filename = 'aura-saved-' + Date.now() + '.png';
          chrome.downloads.download({
            url: saveImgUrl,
            filename,
            saveAs: false,
          });
        }
        return false;
      }

      // SERP AI Answer — proxy fetch through background to avoid CORS
      case 'SERP_FETCH': {
        (async () => {
          try {
            const headers: Record<string, string> = { 'Content-Type': 'application/json' };
            if (msg.apiKey) headers['X-API-Key'] = msg.apiKey;
            const resp = await fetch(msg.url, {
              method: 'POST',
              headers,
              body: msg.body,
              signal: AbortSignal.timeout(30000),
            });
            if (!resp.ok) {
              sendResponse({ ok: false, error: `HTTP ${resp.status}` });
              return;
            }
            const text = await resp.text();
            sendResponse({ ok: true, text });
          } catch (e: any) {
            sendResponse({ ok: false, error: e?.message || 'fetch failed' });
          }
        })();
        return true; // async sendResponse
      }

      default:
        return false;
    }
  }
);
