// Shared constants for the Aura extension. Keep this file tiny — it's
// imported by entry points across several vite bundles (background.ts,
// offscreen.ts, newtab.ts, sidebar src/, content scripts).

/** Default Aura backend URL used when the user hasn't set one in settings. */
export const DEFAULT_BACKEND_URL = 'https://aura-elnur.duckdns.org';
