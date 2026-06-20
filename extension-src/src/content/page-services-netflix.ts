/**
 * Netflix Subtitle Interception Relay — extracted from page-services.ts
 * Forwards Netflix custom events to the background script.
 * Relies on safeStr, safeInt, safeSegments from page-services-youtube.ts.
 */

import { safeStr, safeInt, safeSegments } from './page-services-youtube';

export function initNetflixRelay(safeSend: (msg: any, cb?: (r: any) => void) => void): void {
  document.addEventListener('aura-netflix-subtitles', ((e: CustomEvent) => {
    try {
      const d = e.detail;
      if (!d || typeof d !== 'object') return;
      safeSend({
        type: 'NETFLIX_SUBTITLES',
        movieId: safeStr(d.movieId, 20),
        lang: safeStr(d.lang, 10),
        trackId: safeStr(d.trackId, 50),
        segments: safeSegments(d.segments),
      });
    } catch { /* extension context may be invalidated */ }
  }) as EventListener);

  document.addEventListener('aura-netflix-metadata', ((e: CustomEvent) => {
    try {
      const d = e.detail;
      if (!d || typeof d !== 'object') return;
      safeSend({
        type: 'NETFLIX_METADATA',
        movieId: safeStr(d.movieId, 20),
        title: safeStr(d.title, 500),
        episodeTitle: safeStr(d.episodeTitle, 500),
        seasonNumber: safeInt(d.seasonNumber, 0, 100),
        episodeNumber: safeInt(d.episodeNumber, 0, 100),
        duration: safeInt(d.duration, 0, 86400),
      });
    } catch { /* extension context may be invalidated */ }
  }) as EventListener);
}