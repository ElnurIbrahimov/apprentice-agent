import DOMPurify from 'dompurify';

/**
 * Sanitize HTML from untrusted sources (Pyodide output, user-generated content).
 * Allows safe HTML elements but strips scripts, event handlers, and dangerous attrs.
 */
export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    ALLOW_ARIA_ATTR: true,
    ALLOW_DATA_ATTR: false,
  });
}
