/**
 * Page Services module — Barrel re-export
 * All implementation has been extracted into focused sub-modules:
 *   - page-services-dom.ts        — DOM serializer, execAction
 *   - page-services-ocr.ts        — OCR overlay
 *   - page-services-extraction.ts  — Smart content extraction
 *   - page-services-quick-actions.ts — Quick actions on input fields
 *   - page-services-youtube.ts    — YouTube subtitle relay
 *   - page-services-netflix.ts    — Netflix subtitle relay
 *   - page-services-translate.ts  — Full page translation
 *   - page-services-full-page.ts   — Full page extraction
 *   - page-services-google-serp.ts — Google SERP AI answer card (existing)
 *
 * This file preserves all original exports so downstream consumers
 * (content.ts, sidebar, etc.) continue to work unchanged.
 */

// ── Re-exports from sub-modules ──────────────────────────────────────────────

export type {
  SerializedElement,
  ExecActionParams,
  ExecActionResult,
} from './page-services-dom';

export {
  serializeDOM,
  execAction,
} from './page-services-dom';

export type {
  OcrOverlayResult,
} from './page-services-ocr';

export {
  showOcrOverlay,
} from './page-services-ocr';

export type {
  ExtractPageResponse,
} from './page-services-extraction';

export {
  extractMainContent,
} from './page-services-extraction';

export {
  initQuickActionsOnInputs,
} from './page-services-quick-actions';

export {
  initYoutubeRelay,
} from './page-services-youtube';

export {
  initNetflixRelay,
} from './page-services-netflix';

export {
  initTranslation,
} from './page-services-translate';

export {
  extractFullPageData,
} from './page-services-full-page';

export { initGoogleSerp } from './page-services-google-serp';

// ── Message Listener Setup ────────────────────────────────────────────────────

import { SerializedElement, ExecActionParams, ExecActionResult } from './page-services-dom';
import { OcrOverlayResult } from './page-services-ocr';
import { ExtractPageResponse } from './page-services-extraction';

export interface MessageHandlers {
  extractMainContent(): any;
  serializeDOM(): SerializedElement[];
  execAction(params: ExecActionParams): ExecActionResult;
  showOcrOverlay(dataUrl: string, sendResponse: (r: any) => void): void;
  startPageTranslation(targetLang: string): Promise<void>;
  removePageTranslation(): void;
  setTranslateMode(mode: 'bilingual' | 'translated'): void;
  scrollToHighlight(id: string): void;
  showDock(): void;
  startCaptureMode(): void;
  stopCaptureMode(): void;
  extractFullPageData(): any;
  translateActive: boolean;
}

export function setupMessageListener(ext: typeof chrome, handlers: MessageHandlers): void {
  ext.runtime.onMessage.addListener(
    (
      msg: any,
      _sender: chrome.runtime.MessageSender,
      sendResponse: (response: any) => void
    ): boolean | undefined => {
      // Sender gate: only accept messages from our own extension. Content
      // scripts + UI + background all share ext.runtime.id. Foreign
      // extensions would have a different id; web pages can't reach us at
      // all (no externally_connectable in the manifest), but this keeps the
      // door shut if that ever changes.
      if (_sender && _sender.id && _sender.id !== ext.runtime.id) {
        return false;
      }
      if (msg.type === 'EXTRACT_PAGE') {
        sendResponse(handlers.extractMainContent());
        return false;
      }

      if (msg.type === 'GET_DOM') {
        sendResponse({ ok: true, dom: handlers.serializeDOM(), url: location.href, title: document.title });
        return false;
      }

      if (msg.type === 'EXEC_ACTION') {
        sendResponse(handlers.execAction(msg.action));
        return false;
      }

      if (msg.type === 'FILL_FORM') {
        const fields = msg.fields as Array<{ selector: string; value: string }>;
        let filled = 0;
        for (const field of fields || []) {
          const result = handlers.execAction({ action: 'type', selector: field.selector, text: field.value });
          if (result.ok) filled++;
        }
        sendResponse({ ok: true, filled, total: fields?.length || 0 });
        return false;
      }

      if (msg.type === 'SHOW_OCR_OVERLAY') {
        handlers.showOcrOverlay(msg.dataUrl, sendResponse);
        return true; // async
      }

      if (msg.type === 'PAGE_TRANSLATE') {
        if (handlers.translateActive) handlers.removePageTranslation();
        handlers.startPageTranslation(msg.targetLang).then(() => {
          sendResponse({ ok: true });
        }).catch((err: Error) => {
          sendResponse({ ok: false, error: err.message });
        });
        return true; // async
      }

      if (msg.type === 'TRANSLATE_TOGGLE_MODE') {
        handlers.setTranslateMode(msg.mode);
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'TRANSLATE_REMOVE') {
        handlers.removePageTranslation();
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'TRANSLATE_CHANGE_LANG') {
        if (handlers.translateActive) {
          handlers.removePageTranslation();
          handlers.startPageTranslation(msg.targetLang).then(() => {
            sendResponse({ ok: true });
          }).catch((err: Error) => {
            sendResponse({ ok: false, error: err.message });
          });
          return true;
        }
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'SCROLL_TO_HIGHLIGHT') {
        handlers.scrollToHighlight(msg.id);
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'SHOW_DOCK') {
        handlers.showDock();
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'START_CAPTURE_MODE') {
        handlers.startCaptureMode();
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'STOP_CAPTURE_MODE') {
        handlers.stopCaptureMode();
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'EXTRACT_FULL_PAGE') {
        try {
          const data = handlers.extractFullPageData();
          sendResponse({ ok: true, data });
        } catch (err: any) {
          sendResponse({ ok: false, error: err.message || 'Extraction failed' });
        }
        return false;
      }

      return undefined;
    }
  );
}