/**
 * AURA — YouTube Subtitle Interceptor (MAIN world)
 *
 * Runs in the page context (not the extension isolated world).
 * Monkey-patches XMLHttpRequest and fetch to intercept YouTube's
 * timedtext API calls and ytInitialPlayerResponse for video metadata.
 *
 * Dispatches CustomEvent 'aura-yt-subtitles' on document with subtitle data.
 * Dispatches CustomEvent 'aura-yt-metadata' on document with video metadata.
 */

(function () {
  'use strict';

  // Guard against double-injection
  if ((window as any).__auraYtInjected) return;
  (window as any).__auraYtInjected = true;

  // ── Types ───────────────────────────────────────────────────────────────

  interface SubtitleEvent {
    url: string;
    lang: string;
    text: string;          // raw XML/JSON response
    segments: Segment[];   // parsed segments
    videoId: string;
  }

  interface Segment {
    start: number;   // seconds
    dur: number;     // seconds
    text: string;
  }

  interface VideoMeta {
    videoId: string;
    title: string;
    duration: number;       // seconds
    description: string;
    channelName: string;
    chapters: Chapter[];
    captionTracks: CaptionTrack[];
  }

  interface Chapter {
    title: string;
    startMs: number;
  }

  interface CaptionTrack {
    baseUrl: string;
    languageCode: string;
    name: string;
    kind?: string;
  }

  // ── Helpers ─────────────────────────────────────────────────────────────

  function getVideoId(): string {
    const params = new URLSearchParams(location.search);
    return params.get('v') || '';
  }

  function extractLang(url: string): string {
    try {
      const u = new URL(url);
      return u.searchParams.get('lang') || u.searchParams.get('tlang') || 'unknown';
    } catch {
      return 'unknown';
    }
  }

  function isTimedTextUrl(url: string): boolean {
    return url.includes('timedtext') || url.includes('api/timedtext');
  }

  /**
   * Parse YouTube subtitle XML format:
   * <transcript><text start="0" dur="5.2">Hello</text>...</transcript>
   * Also handles JSON3 format.
   */
  function parseSubtitleResponse(text: string): Segment[] {
    const segments: Segment[] = [];

    // Try JSON3 format first
    try {
      const json = JSON.parse(text);
      if (json.events) {
        for (const evt of json.events) {
          if (evt.segs) {
            const segText = evt.segs.map((s: any) => s.utf8 || '').join('');
            if (segText.trim()) {
              segments.push({
                start: (evt.tStartMs || 0) / 1000,
                dur: (evt.dDurationMs || 0) / 1000,
                text: segText.trim(),
              });
            }
          }
        }
        if (segments.length > 0) return segments;
      }
    } catch {
      // Not JSON — try XML
    }

    // XML format
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, 'text/xml');
    const textNodes = doc.querySelectorAll('text');
    textNodes.forEach((node) => {
      const startAttr = node.getAttribute('start');
      const durAttr = node.getAttribute('dur');
      const content = node.textContent || '';
      if (content.trim()) {
        segments.push({
          start: parseFloat(startAttr || '0'),
          dur: parseFloat(durAttr || '0'),
          text: content.replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').trim(),
        });
      }
    });

    return segments;
  }

  function dispatch(name: string, detail: any): void {
    document.dispatchEvent(new CustomEvent(name, { detail }));
  }

  // ── Extract metadata from ytInitialPlayerResponse ──────────────────────

  function extractMetadata(playerResponse: any): VideoMeta | null {
    if (!playerResponse) return null;
    try {
      const vd = playerResponse.videoDetails || {};
      const microformat = playerResponse.microformat?.playerMicroformatRenderer || {};
      const captions = playerResponse.captions?.playerCaptionsTracklistRenderer?.captionTracks || [];

      // Chapters from engagement panels or description
      const chapters: Chapter[] = [];
      try {
        const panels = playerResponse.engagementPanels || [];
        for (const panel of panels) {
          const macro = panel.engagementPanelSectionListRenderer?.content?.macroMarkersListRenderer?.contents;
          if (macro) {
            for (const item of macro) {
              const marker = item.macroMarkersListItemRenderer;
              if (marker) {
                chapters.push({
                  title: marker.title?.simpleText || marker.title?.runs?.[0]?.text || '',
                  startMs: parseInt(marker.timeDescription?.simpleText?.replace(/:/g, '') || '0', 10),
                });
              }
            }
          }
        }
      } catch { /* chapters are optional */ }

      const captionTracks: CaptionTrack[] = captions.map((t: any) => ({
        baseUrl: t.baseUrl || '',
        languageCode: t.languageCode || '',
        name: t.name?.simpleText || t.name?.runs?.[0]?.text || '',
        kind: t.kind,
      }));

      return {
        videoId: vd.videoId || getVideoId(),
        title: vd.title || microformat.title?.simpleText || '',
        duration: parseInt(vd.lengthSeconds || '0', 10),
        description: vd.shortDescription || microformat.description?.simpleText || '',
        channelName: vd.author || '',
        chapters,
        captionTracks,
      };
    } catch {
      return null;
    }
  }

  // ── Scan for ytInitialPlayerResponse on page load ──────────────────────

  function scanInitialData(): void {
    try {
      const w = window as any;
      const pr = w.ytInitialPlayerResponse;
      if (pr) {
        const meta = extractMetadata(pr);
        if (meta) dispatch('aura-yt-metadata', meta);
      }
    } catch { /* no-op */ }
  }

  // Run scan after a small delay to let YouTube hydrate
  setTimeout(scanInitialData, 1500);

  // Listen for YouTube's native SPA navigation event instead of expensive MutationObserver
  function onSpaNavigate(): void {
    if (location.href.includes('youtube.com/watch')) {
      setTimeout(scanInitialData, 2000);
    }
  }
  document.addEventListener('yt-navigate-finish', onSpaNavigate);
  window.addEventListener('popstate', onSpaNavigate);

  // ── Monkey-patch XMLHttpRequest ────────────────────────────────────────

  const OrigXHR = XMLHttpRequest;
  const origOpen = OrigXHR.prototype.open;
  const origSend = OrigXHR.prototype.send;

  OrigXHR.prototype.open = function (
    this: XMLHttpRequest,
    method: string,
    url: string | URL,
    ...rest: any[]
  ) {
    (this as any).__auraUrl = String(url);
    (this as any).__auraMethod = method;
    return origOpen.apply(this, [method, url, ...rest] as any);
  };

  OrigXHR.prototype.send = function (this: XMLHttpRequest, body?: any) {
    const url = (this as any).__auraUrl as string;

    if (url && isTimedTextUrl(url)) {
      this.addEventListener('load', function () {
        try {
          const responseText = this.responseText || '';
          if (!responseText) return;
          const segments = parseSubtitleResponse(responseText);
          if (segments.length === 0) return;

          const detail: SubtitleEvent = {
            url,
            lang: extractLang(url),
            text: responseText,
            segments,
            videoId: getVideoId(),
          };
          dispatch('aura-yt-subtitles', detail);
        } catch { /* don't break the page */ }
      });
    }

    // Intercept player response from innertube API
    if (url && (url.includes('/youtubei/v1/player') || url.includes('/youtubei/v1/next'))) {
      this.addEventListener('load', function () {
        try {
          const json = JSON.parse(this.responseText);
          if (json.videoDetails || json.captions) {
            const meta = extractMetadata(json);
            if (meta) dispatch('aura-yt-metadata', meta);
          }
        } catch { /* ignore parse errors */ }
      });
    }

    return origSend.call(this, body);
  };

  // ── Monkey-patch fetch ─────────────────────────────────────────────────

  const origFetch = window.fetch;
  window.fetch = function (input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = typeof input === 'string'
      ? input
      : input instanceof URL
        ? input.href
        : input instanceof Request
          ? input.url
          : '';

    const promise = origFetch.call(this, input, init);

    if (url && isTimedTextUrl(url)) {
      promise.then((response) => {
        // Clone so the original consumer still gets the body
        response.clone().text().then((text) => {
          try {
            const segments = parseSubtitleResponse(text);
            if (segments.length === 0) return;
            const detail: SubtitleEvent = {
              url,
              lang: extractLang(url),
              text,
              segments,
              videoId: getVideoId(),
            };
            dispatch('aura-yt-subtitles', detail);
          } catch { /* don't break the page */ }
        }).catch(() => {});
      }).catch(() => {});
    }

    if (url && (url.includes('/youtubei/v1/player') || url.includes('/youtubei/v1/next'))) {
      promise.then((response) => {
        response.clone().json().then((json: any) => {
          try {
            if (json.videoDetails || json.captions) {
              const meta = extractMetadata(json);
              if (meta) dispatch('aura-yt-metadata', meta);
            }
          } catch { /* ignore */ }
        }).catch(() => {});
      }).catch(() => {});
    }

    return promise;
  };

  // ── Auto-fetch subtitles if caption tracks are available ───────────────

  // Listen for our own metadata event and try to fetch the first caption track
  document.addEventListener('aura-yt-metadata', ((e: CustomEvent) => {
    const meta = e.detail as VideoMeta;
    if (!meta.captionTracks || meta.captionTracks.length === 0) return;

    // Prefer English, fallback to first track
    const enTrack = meta.captionTracks.find(
      (t) => t.languageCode === 'en' || t.languageCode.startsWith('en')
    );
    const track = enTrack || meta.captionTracks[0];
    if (!track.baseUrl) return;

    // Fetch the subtitle track — use fmt=json3 for structured data
    const fetchUrl = track.baseUrl + (track.baseUrl.includes('?') ? '&' : '?') + 'fmt=json3';
    origFetch(fetchUrl)
      .then((r) => r.text())
      .then((text) => {
        const segments = parseSubtitleResponse(text);
        if (segments.length > 0) {
          dispatch('aura-yt-subtitles', {
            url: fetchUrl,
            lang: track.languageCode,
            text,
            segments,
            videoId: meta.videoId,
          } as SubtitleEvent);
        }
      })
      .catch(() => {});
  }) as EventListener);

})();
