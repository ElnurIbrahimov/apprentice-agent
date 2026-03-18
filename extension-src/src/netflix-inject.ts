/**
 * AURA — Netflix Subtitle Interceptor (MAIN world)
 *
 * Runs in the page context (not the extension isolated world).
 * Monkey-patches JSON.parse to intercept Netflix's timedtext/subtitle
 * responses as they are parsed by the player.
 *
 * Dispatches CustomEvent 'aura-netflix-subtitles' on document with subtitle data.
 * Dispatches CustomEvent 'aura-netflix-metadata' on document with show/episode info.
 */

(function () {
  'use strict';

  // Guard against double-injection
  if ((window as any).__auraNetflixInjected) return;
  (window as any).__auraNetflixInjected = true;

  // ── Types ───────────────────────────────────────────────────────────────

  interface Segment {
    start: number;   // seconds
    dur: number;     // seconds
    text: string;
  }

  interface NetflixSubtitleEvent {
    segments: Segment[];
    lang: string;
    trackId: string;
    movieId: string;
  }

  interface NetflixMetadataEvent {
    movieId: string;
    title: string;
    episodeTitle: string;
    seasonNumber: number;
    episodeNumber: number;
    duration: number;  // seconds
  }

  // ── Helpers ─────────────────────────────────────────────────────────────

  function dispatch(name: string, detail: any): void {
    document.dispatchEvent(new CustomEvent(name, { detail }));
  }

  /**
   * Extract the Netflix video/movie ID from the current URL.
   * Netflix URLs look like: /watch/81234567
   */
  function getMovieId(): string {
    const match = location.pathname.match(/\/watch\/(\d+)/);
    return match ? match[1] : '';
  }

  /**
   * Parse Netflix's timedtext subtitle format.
   * Netflix uses a JSON structure with cues containing startTime/endTime in ms
   * and text content, or the DFXP/TTML XML format.
   *
   * Common JSON structures observed:
   * 1. { result: { timedtexttracks: [...] } } — track listing
   * 2. Individual track data with cue arrays
   * 3. Webvtt-like JSON with "cues" or "subtitles" arrays
   */
  function parseNetflixTimedText(obj: any): Segment[] | null {
    const segments: Segment[] = [];

    // Pattern 1: Array of cues with startTime/endTime (milliseconds)
    // Netflix uses this in their cadmium player payload
    if (Array.isArray(obj)) {
      for (const cue of obj) {
        if (cue.startTime != null && cue.endTime != null && cue.text) {
          const startMs = Number(cue.startTime);
          const endMs = Number(cue.endTime);
          const text = stripHtmlTags(String(cue.text));
          if (text.trim()) {
            segments.push({
              start: startMs / 1000,
              dur: (endMs - startMs) / 1000,
              text: text.trim(),
            });
          }
        }
      }
      if (segments.length > 0) return segments;
    }

    // Pattern 2: Object with "cues" array
    if (obj && Array.isArray(obj.cues)) {
      for (const cue of obj.cues) {
        const startMs = Number(cue.startTime ?? cue.start ?? 0);
        const endMs = Number(cue.endTime ?? cue.end ?? 0);
        const raw = cue.text ?? cue.content ?? '';
        const text = stripHtmlTags(String(raw));
        if (text.trim() && endMs > startMs) {
          segments.push({
            start: startMs / 1000,
            dur: (endMs - startMs) / 1000,
            text: text.trim(),
          });
        }
      }
      if (segments.length > 0) return segments;
    }

    // Pattern 3: Netflix timedtext track with downloadable URLs
    // { result: { timedtexttracks: [{ ttDownloadables: { ... } }] } }
    // This is the track listing — we handle actual subtitle data above.
    // We'll extract track info from this for metadata purposes.

    return segments.length > 0 ? segments : null;
  }

  /**
   * Check if a parsed JSON object looks like Netflix timedtext track metadata.
   * This is the response to ?o=https://...nflx...timedtext...
   */
  function extractTimedTextTracks(obj: any): any[] | null {
    // Netflix manifest responses contain timedtexttracks array
    if (obj?.result?.timedtexttracks && Array.isArray(obj.result.timedtexttracks)) {
      return obj.result.timedtexttracks;
    }
    return null;
  }

  /**
   * Check if object contains Netflix video metadata.
   */
  function extractNetflixMetadata(obj: any): NetflixMetadataEvent | null {
    try {
      // Pattern: { value: { videos: { "12345": { ... } } } }
      if (obj?.value?.videos) {
        const videos = obj.value.videos;
        const keys = Object.keys(videos);
        if (keys.length > 0) {
          const v = videos[keys[0]];
          if (v?.title || v?.type) {
            return {
              movieId: keys[0],
              title: v.title || v.name || '',
              episodeTitle: v.episodeTitle || '',
              seasonNumber: v.seasonNumber || 0,
              episodeNumber: v.episodeNumber || 0,
              duration: (v.runtime || v.duration || 0) / 1000,
            };
          }
        }
      }
    } catch { /* ignore */ }
    return null;
  }

  function stripHtmlTags(s: string): string {
    return s.replace(/<[^>]*>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'")
      .replace(/&nbsp;/g, ' ');
  }

  // ── Monkey-patch JSON.parse ─────────────────────────────────────────────

  const origParse = JSON.parse;

  JSON.parse = function (this: any, text: string, ...rest: any[]): any {
    const result = origParse.call(this, text, ...rest);

    // Only process on Netflix watch pages
    if (!location.hostname.includes('netflix.com')) return result;

    try {
      if (typeof result === 'object' && result !== null) {
        processNetflixData(result);
      }
    } catch { /* never break the page */ }

    return result;
  } as typeof JSON.parse;

  /**
   * Inspect a parsed JSON object for Netflix subtitle or metadata payloads.
   * Called on every JSON.parse result — must be fast for non-matching data.
   */
  function processNetflixData(obj: any): void {
    // Quick bail-out: skip primitives and tiny objects
    if (obj === null || typeof obj !== 'object') return;

    // Check for timedtext track listings in manifest responses
    const tracks = extractTimedTextTracks(obj);
    if (tracks && tracks.length > 0) {
      // Dispatch track listing info so content script knows what's available
      dispatch('aura-netflix-tracks', {
        movieId: getMovieId(),
        tracks: tracks.map((t: any) => ({
          language: t.language || '',
          languageDescription: t.languageDescription || '',
          trackType: t.trackType || '',
          isForced: t.isForcedNarrative || false,
        })),
      });
    }

    // Check for subtitle cue data
    // Netflix sends subtitle segments in various nested formats
    const subtitleData = findSubtitleCues(obj);
    if (subtitleData) {
      dispatch('aura-netflix-subtitles', subtitleData);
    }

    // Check for video metadata
    const metadata = extractNetflixMetadata(obj);
    if (metadata) {
      dispatch('aura-netflix-metadata', metadata);
    }
  }

  /**
   * Recursively search (up to 3 levels) for arrays that look like subtitle cues.
   * Netflix nests subtitle data in various places depending on the API version.
   */
  function findSubtitleCues(obj: any, depth: number = 0): NetflixSubtitleEvent | null {
    if (depth > 4 || !obj || typeof obj !== 'object') return null;

    // Direct array of cues
    if (Array.isArray(obj)) {
      const segments = parseNetflixTimedText(obj);
      if (segments && segments.length >= 3) { // Minimum 3 cues to avoid false positives
        return {
          segments,
          lang: 'unknown',
          trackId: '',
          movieId: getMovieId(),
        };
      }
      return null;
    }

    // Object with "cues" property
    if (obj.cues) {
      const segments = parseNetflixTimedText(obj);
      if (segments && segments.length >= 3) {
        return {
          segments,
          lang: obj.language || obj.locale || 'unknown',
          trackId: obj.trackId || obj.id || '',
          movieId: getMovieId(),
        };
      }
    }

    // Look for timedtexttracks with downloadable subtitle data embedded
    if (obj.result?.timedtexttracks) {
      for (const track of obj.result.timedtexttracks) {
        if (track.ttDownloadables) {
          // Check each downloadable format for embedded cue data
          for (const format of Object.values(track.ttDownloadables) as any[]) {
            if (format?.cues || format?.downloadUrls) {
              const segments = parseNetflixTimedText(format);
              if (segments && segments.length >= 3) {
                return {
                  segments,
                  lang: track.language || 'unknown',
                  trackId: track.trackId || track.new_track_id || '',
                  movieId: getMovieId(),
                };
              }
            }
          }
        }
      }
    }

    // Recurse into object keys (limit to avoid performance issues)
    const keys = Object.keys(obj);
    if (keys.length > 50) return null; // Skip huge objects
    for (const key of keys) {
      const val = obj[key];
      if (val && typeof val === 'object') {
        const found = findSubtitleCues(val, depth + 1);
        if (found) return found;
      }
    }

    return null;
  }

  // ── Intercept fetch for TTML/WebVTT subtitle downloads ─────────────────

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

    // Netflix fetches subtitle files as TTML/DFXP XML or WebVTT
    if (url && isSubtitleUrl(url)) {
      promise.then((response) => {
        response.clone().text().then((text) => {
          try {
            const segments = parseTtmlSubtitles(text);
            if (segments && segments.length >= 3) {
              dispatch('aura-netflix-subtitles', {
                segments,
                lang: extractLangFromUrl(url),
                trackId: '',
                movieId: getMovieId(),
              } as NetflixSubtitleEvent);
            }
          } catch { /* don't break the page */ }
        }).catch(() => {});
      }).catch(() => {});
    }

    return promise;
  };

  function isSubtitleUrl(url: string): boolean {
    return url.includes('timedtext') ||
           url.includes('?o=') && url.includes('dfxp') ||
           url.includes('webvtt') ||
           /nflx.*text/.test(url);
  }

  function extractLangFromUrl(url: string): string {
    try {
      const u = new URL(url);
      return u.searchParams.get('lang') || u.searchParams.get('locale') || 'unknown';
    } catch {
      return 'unknown';
    }
  }

  /**
   * Parse TTML/DFXP XML subtitle format used by Netflix.
   * <tt><body><div><p begin="00:01:23.456" end="00:01:25.789">Text</p></div></body></tt>
   */
  function parseTtmlSubtitles(text: string): Segment[] | null {
    // Quick check — is this XML?
    if (!text.includes('<tt') && !text.includes('<WEBVTT') && !text.includes('WEBVTT')) {
      return null;
    }

    const segments: Segment[] = [];

    // Try TTML/DFXP
    if (text.includes('<tt') || text.includes('<p ')) {
      try {
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/xml');
        const pNodes = doc.querySelectorAll('p[begin]');
        pNodes.forEach((node) => {
          const begin = node.getAttribute('begin') || '';
          const end = node.getAttribute('end') || '';
          const dur = node.getAttribute('dur') || '';
          const content = node.textContent || '';
          if (content.trim()) {
            const startSec = timeToSeconds(begin);
            const endSec = end ? timeToSeconds(end) : startSec + timeToSeconds(dur);
            segments.push({
              start: startSec,
              dur: endSec - startSec,
              text: stripHtmlTags(content.trim()),
            });
          }
        });
      } catch { /* not valid XML */ }
    }

    // Try WebVTT
    if (segments.length === 0 && text.includes('WEBVTT')) {
      const lines = text.split('\n');
      let i = 0;
      while (i < lines.length) {
        const line = lines[i].trim();
        // Look for timestamp lines: 00:01:23.456 --> 00:01:25.789
        const match = line.match(/(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})/);
        if (match) {
          const startSec = timeToSeconds(match[1]);
          const endSec = timeToSeconds(match[2]);
          const textLines: string[] = [];
          i++;
          while (i < lines.length && lines[i].trim() !== '') {
            textLines.push(lines[i].trim());
            i++;
          }
          const content = stripHtmlTags(textLines.join(' '));
          if (content.trim()) {
            segments.push({
              start: startSec,
              dur: endSec - startSec,
              text: content.trim(),
            });
          }
        }
        i++;
      }
    }

    return segments.length > 0 ? segments : null;
  }

  /**
   * Convert a TTML/WebVTT timestamp to seconds.
   * Handles: HH:MM:SS.mmm, HH:MM:SS:FF, MM:SS.mmm, SS.mmm
   */
  function timeToSeconds(ts: string): number {
    if (!ts) return 0;
    // Replace tick-based timestamps: 1234567890t -> seconds
    if (ts.endsWith('t')) {
      return parseInt(ts, 10) / 10000000;
    }
    const parts = ts.split(':');
    if (parts.length === 3) {
      const h = parseInt(parts[0], 10);
      const m = parseInt(parts[1], 10);
      const s = parseFloat(parts[2]);
      return h * 3600 + m * 60 + s;
    } else if (parts.length === 2) {
      const m = parseInt(parts[0], 10);
      const s = parseFloat(parts[1]);
      return m * 60 + s;
    }
    return parseFloat(ts) || 0;
  }

})();
