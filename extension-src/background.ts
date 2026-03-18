/// <reference types="chrome" />

/**
 * AURA Chrome Extension — Service Worker (background.ts)
 * Handles: side panel toggle, context menu, message routing
 */

// ── Firefox compatibility shim ───────────────────────────────────────────────

declare const browser: typeof chrome | undefined;

const ext: typeof chrome =
  typeof browser !== 'undefined' ? (browser as typeof chrome) : chrome;

const BACKEND = 'http://localhost:8000' as const;

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
  action: 'improve' | 'expand' | 'shorten' | 'fix_grammar' | 'translate';
  text: string;
  language?: string;
}

interface QuickActionResponse {
  ok: boolean;
  result?: string;
  error?: string;
}

type ExtensionMessage =
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
  | QuickActionMessage;

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
        headers: { 'Content-Type': 'application/json' },
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
          ['pendingQuery', 'pendingAction', 'pendingUrl', 'pendingTitle', 'pendingPanelSwitch'],
          (data: PendingStorage) => {
            if (data.pendingQuery) {
              ext.runtime.sendMessage({
                type: 'PREFILL_TEXT',
                text: data.pendingQuery,
                action: data.pendingAction || 'ask',
                url: data.pendingUrl || '',
                title: data.pendingTitle || '',
              } satisfies PrefillTextMessage);
              ext.storage.local.remove(['pendingQuery', 'pendingAction', 'pendingUrl', 'pendingTitle']);
            } else if (data.pendingPanelSwitch) {
              ext.runtime.sendMessage({
                type: 'SWITCH_PANEL',
                panel: data.pendingPanelSwitch,
              } satisfies SwitchPanelMessage);
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
              } satisfies OcrResultMessage);
              return;
            }
            ext.tabs.query(
              { active: true, currentWindow: true },
              ([tab]: chrome.tabs.Tab[]): void => {
                if (!tab) {
                  ext.runtime.sendMessage({
                    type: 'OCR_RESULT',
                    error: 'No active tab',
                  } satisfies OcrResultMessage);
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
                      } satisfies OcrResultMessage);
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
                        } satisfies OcrResultMessage);
                        return;
                      }
                      const d: { text?: string; detail?: string } = await resp.json();
                      ext.runtime.sendMessage({
                        type: 'OCR_RESULT',
                        text: d.text || '',
                        error: d.detail || '',
                      } satisfies OcrResultMessage);
                    } catch (e) {
                      ext.runtime.sendMessage({
                        type: 'OCR_RESULT',
                        error: String(e),
                      } satisfies OcrResultMessage);
                    }
                  }
                );
              }
            );
          }
        );
        return false;
      }

      // Quick-action: content script asks background to call LLM for inline text edits
      case 'QUICK_ACTION': {
        const PROMPT_MAP: Record<string, string> = {
          improve: 'Improve this text. Return ONLY the improved text, no explanation:\n\n',
          expand: 'Expand this text to be longer and more detailed. Return ONLY the expanded text, no explanation:\n\n',
          shorten: 'Make this text more concise. Return ONLY the shortened text, no explanation:\n\n',
          fix_grammar: 'Fix all grammar and spelling errors. Return ONLY the corrected text, no explanation:\n\n',
          translate: 'Translate to {lang}. Return ONLY the translation, no explanation:\n\n',
        };

        let prompt = PROMPT_MAP[msg.action] || PROMPT_MAP.improve;
        if (msg.action === 'translate' && msg.language) {
          prompt = prompt.replace('{lang}', msg.language);
        } else if (msg.action === 'translate') {
          prompt = prompt.replace('{lang}', 'English');
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
              // AGENT_NAV
              ext.tabs.update(tab.id!, { url: msg.url }, () =>
                sendResponse({ ok: true })
              );
            }
          }
        );
        return true;
      }

      default:
        return false;
    }
  }
);
