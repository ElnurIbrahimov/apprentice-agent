/**
 * YouTube Subtitle Interception Relay — extracted from page-services.ts
 * Sanitizes and forwards YouTube custom events to the background script.
 */

/** Sanitize a value expected to be a short string — clamp length, coerce type. */
export function safeStr(v: unknown, max = 500): string {
  return typeof v === 'string' ? v.slice(0, max) : '';
}

/** Sanitize a number expected to be an integer — clamp to sane range. */
export function safeInt(v: unknown, min = 0, max = 1e9): number {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(Math.max(Math.round(n), min), max) : 0;
}

/** Sanitize a segments array — each segment must be a plain object with a text string. */
export function safeSegments(v: unknown, maxLen = 500, maxItems = 5000): Array<{ start: number; dur: number; text: string }> {
  if (!Array.isArray(v)) return [];
  const out: Array<{ start: number; dur: number; text: string }> = [];
  for (let i = 0; i < Math.min(v.length, maxItems); i++) {
    const s = v[i];
    if (!s || typeof s !== 'object' || Array.isArray(s)) continue;
    out.push({
      start: safeInt((s as any).start, 0, 1e8),
      dur: safeInt((s as any).dur, 0, 1e8),
      text: safeStr((s as any).text, maxLen),
    });
  }
  return out;
}

export function initYoutubeRelay(safeSend: (msg: any, cb?: (r: any) => void) => void): void {
  document.addEventListener('aura-yt-subtitles', ((e: CustomEvent) => {
    try {
      const d = e.detail;
      if (!d || typeof d !== 'object') return;
      safeSend({
        type: 'YT_SUBTITLES',
        videoId: safeStr(d.videoId, 20),
        lang: safeStr(d.lang, 10),
        segments: safeSegments(d.segments),
      });
    } catch { /* extension context may be invalidated */ }
  }) as EventListener);

  document.addEventListener('aura-yt-metadata', ((e: CustomEvent) => {
    try {
      const d = e.detail;
      if (!d || typeof d !== 'object') return;
      safeSend({
        type: 'YT_METADATA',
        videoId: safeStr(d.videoId, 20),
        title: safeStr(d.title, 500),
        duration: safeInt(d.duration, 0, 86400),
        description: safeStr(d.description, 5000),
        channelName: safeStr(d.channelName, 200),
        chapters: Array.isArray(d.chapters) ? d.chapters.slice(0, 200).map((c: any) => ({
          title: safeStr(c?.title, 200),
          start: safeInt(c?.start, 0, 86400),
        })) : [],
        captionTracks: Array.isArray(d.captionTracks) ? d.captionTracks.slice(0, 50).map((t: any) => ({
          lang: safeStr(t?.lang, 10),
          url: safeStr(t?.url, 2000),
        })) : [],
      });
    } catch { /* extension context may be invalidated */ }
  }) as EventListener);
}