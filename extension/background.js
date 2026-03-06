/**
 * AURA Chrome Extension — Service Worker (background.js)
 * Handles: side panel toggle, context menu, message routing
 */

// Firefox compatibility shim
const ext = typeof browser !== 'undefined' ? browser : chrome;

const BACKEND = 'http://localhost:8000';

// ── Startup ──────────────────────────────────────────────────────────────────

ext.runtime.onInstalled.addListener(() => {
  // Open side panel when toolbar icon is clicked (Chrome only)
  if (chrome.sidePanel) {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
  }

  // Right-click context menu for selected text
  ext.contextMenus.create({
    id: 'ask-aura',
    title: 'Ask AURA about "%s"',
    contexts: ['selection'],
  });
});

// ── Context Menu ──────────────────────────────────────────────────────────────

ext.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== 'ask-aura') return;

  const selectedText = info.selectionText || '';
  const pageUrl = tab?.url || '';
  const pageTitle = tab?.title || '';

  ext.storage.local.set({
    pendingQuery: selectedText,
    pendingAction: 'ask',
    pendingUrl: pageUrl,
    pendingTitle: pageTitle,
  });

  // Open the side panel
  if (chrome.sidePanel) {
    chrome.sidePanel.open({ windowId: tab.windowId });
  } else if (typeof browser !== 'undefined' && browser.sidebarAction) {
    browser.sidebarAction.open();
  }
});

// ── PDF Tab Detection ─────────────────────────────────────────────────────────

ext.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && /\.pdf($|\?)/i.test(tab.url || '')) {
    ext.runtime.sendMessage({
      type: 'PDF_TAB_DETECTED',
      url: tab.url,
      title: tab.title || tab.url,
    });
  }
});

// ── Message Router ────────────────────────────────────────────────────────────

ext.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  switch (msg.type) {

    // Sidebar has loaded — send any pending prefill text or panel switch
    case 'SIDEBAR_READY': {
      ext.storage.local.get(
        ['pendingQuery', 'pendingAction', 'pendingUrl', 'pendingTitle', 'pendingPanelSwitch'],
        (data) => {
          if (data.pendingQuery) {
            ext.runtime.sendMessage({
              type: 'PREFILL_TEXT',
              text: data.pendingQuery,
              action: data.pendingAction || 'ask',
              url: data.pendingUrl || '',
              title: data.pendingTitle || '',
            });
            ext.storage.local.remove(['pendingQuery', 'pendingAction', 'pendingUrl', 'pendingTitle']);
          } else if (data.pendingPanelSwitch) {
            ext.runtime.sendMessage({ type: 'SWITCH_PANEL', panel: data.pendingPanelSwitch });
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
        .then((data) => sendResponse({ ok: true, data }))
        .catch((err) => sendResponse({ ok: false, error: err.message }));
      return true;
    }

    // Sidebar asks background to extract page content from active tab
    case 'GET_PAGE_CONTENT': {
      ext.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const activeTab = tabs[0];
        if (!activeTab) {
          sendResponse({ ok: false, error: 'No active tab' });
          return;
        }
        ext.tabs.sendMessage(
          activeTab.id,
          { type: 'EXTRACT_PAGE' },
          (response) => {
            if (ext.runtime.lastError) {
              sendResponse({ ok: false, error: ext.runtime.lastError.message });
            } else {
              sendResponse({ ok: true, ...response });
            }
          }
        );
      });
      return true;
    }

    // Dock button: open sidebar and switch to a specific panel
    case 'OPEN_PANEL': {
      if (sender.tab) {
        if (chrome.sidePanel) {
          chrome.sidePanel.open({ windowId: sender.tab.windowId });
        } else if (typeof browser !== 'undefined' && browser.sidebarAction) {
          browser.sidebarAction.open();
        }
      }
      // Attempt to notify sidebar immediately (works if already open)
      ext.runtime.sendMessage({ type: 'SWITCH_PANEL', panel: msg.panel })
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
          chrome.sidePanel.open({ windowId: sender.tab.windowId });
        } else if (typeof browser !== 'undefined' && browser.sidebarAction) {
          browser.sidebarAction.open();
        }
      }
      return false;
    }

    // OCR: capture visible tab then ask content script for region selection
    case 'OCR_START': {
      // captureVisibleTab is Chrome-only; no standard ext.* equivalent
      chrome.tabs.captureVisibleTab(null, { format: 'png' }, (dataUrl) => {
        if (chrome.runtime.lastError || !dataUrl) {
          ext.runtime.sendMessage({ type: 'OCR_RESULT', error: 'Screenshot failed' });
          return;
        }
        ext.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
          if (!tab) { ext.runtime.sendMessage({ type: 'OCR_RESULT', error: 'No active tab' }); return; }
          ext.tabs.sendMessage(tab.id, { type: 'SHOW_OCR_OVERLAY', dataUrl }, async (region) => {
            if (!region?.ok) {
              ext.runtime.sendMessage({ type: 'OCR_RESULT', error: 'Cancelled' });
              return;
            }
            try {
              const { x, y, w, h, dpr } = region;
              const imgBlob = await fetch(dataUrl).then(r => r.blob());
              const bmp = await createImageBitmap(imgBlob);
              const cw = Math.max(1, Math.round(w * dpr));
              const ch = Math.max(1, Math.round(h * dpr));
              const oc = new OffscreenCanvas(cw, ch);
              oc.getContext('2d').drawImage(bmp, x * dpr, y * dpr, w * dpr, h * dpr, 0, 0, cw, ch);
              const blob = await oc.convertToBlob({ type: 'image/png' });
              const b64 = btoa(String.fromCharCode(...new Uint8Array(await blob.arrayBuffer())));
              const resp = await fetch(`${BACKEND}/api/ocr`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_b64: b64 }),
              });
              const d = await resp.json();
              ext.runtime.sendMessage({
                type: 'OCR_RESULT',
                text: d.text || '',
                error: d.detail || '',
              });
            } catch (e) {
              ext.runtime.sendMessage({ type: 'OCR_RESULT', error: String(e) });
            }
          });
        });
      });
      return false;
    }

    // Browser Agent relay handlers
    case 'AGENT_DOM': {
      ext.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
        if (!tab) { sendResponse({ ok: false, error: 'No active tab' }); return; }
        ext.tabs.sendMessage(tab.id, { type: 'GET_DOM' }, r => sendResponse(r));
      });
      return true;
    }

    case 'AGENT_EXEC': {
      ext.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
        if (!tab) { sendResponse({ ok: false, error: 'No active tab' }); return; }
        ext.tabs.sendMessage(tab.id, { type: 'EXEC_ACTION', action: msg.action }, r => sendResponse(r));
      });
      return true;
    }

    case 'AGENT_NAV': {
      ext.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
        if (!tab) { sendResponse({ ok: false, error: 'No active tab' }); return; }
        ext.tabs.update(tab.id, { url: msg.url }, () => sendResponse({ ok: true }));
      });
      return true;
    }

    default:
      return false;
  }
});
