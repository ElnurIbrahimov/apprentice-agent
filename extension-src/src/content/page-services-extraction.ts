/**
 * Smart Content Extraction — extracted from page-services.ts
 * Provides content selectors, findContentRoot, cleanClone, domToStructuredText,
 * YouTube extraction, and the main extractMainContent function.
 */

export interface ExtractPageResponse {
  text: string;
  url: string;
  title: string;
  wordCount: number;
  isPdf?: boolean;
  isYouTube?: boolean;
  videoTitle?: string;
  transcript?: string;
}

const MAX_TEXT_BYTES = 50000; // ~50KB limit

/** Selectors for elements likely to be the main content area */
const CONTENT_SELECTORS = [
  'article',
  'main',
  '[role="main"]',
  '.post-content',
  '.article-body',
  '.entry-content',
  '.post-body',
  '.article-content',
  '.story-body',
  '.content-body',
  '#article-body',
  '#content',
  '.markdown-body', // GitHub
  '.wiki-content',
];

/** Selectors for junk elements to strip from the clone */
const JUNK_SELECTORS = [
  'nav', 'header', 'footer', 'aside',
  'script', 'style', 'noscript', 'iframe',
  '.sidebar', '.menu', '.nav', '.navigation',
  '.cookie', '.cookie-banner', '.cookie-consent',
  '.popup', '.modal', '.overlay',
  '.ad', '.ads', '.advert', '.advertisement',
  '.social-share', '.share-buttons', '.social-buttons',
  '.comments', '.comment-section', '#comments',
  '.related-posts', '.recommended',
  '.newsletter', '.subscribe',
  '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
  '[role="complementary"]', '[aria-hidden="true"]',
  '.sr-only', '.visually-hidden',
];

/**
 * Finds the best content root element on the page.
 * Tries semantic selectors first, falls back to document.body.
 */
function findContentRoot(): Element {
  for (const sel of CONTENT_SELECTORS) {
    const el = document.querySelector(sel);
    if (el && el.textContent && el.textContent.trim().length > 200) {
      return el;
    }
  }
  return document.body;
}

/**
 * Clones the content root and strips junk elements from the clone.
 * Never modifies the live DOM.
 */
function cleanClone(root: Element): Element {
  const clone = root.cloneNode(true) as Element;
  for (const sel of JUNK_SELECTORS) {
    clone.querySelectorAll(sel).forEach((el) => el.remove());
  }
  return clone;
}

/**
 * Walks a cleaned DOM tree and produces structured plain text.
 */
function domToStructuredText(root: Element): string {
  const parts: string[] = [];
  const BLOCK_TAGS = new Set([
    'P', 'DIV', 'SECTION', 'ARTICLE', 'BLOCKQUOTE', 'PRE',
    'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
    'UL', 'OL', 'LI', 'TABLE', 'TR', 'DT', 'DD',
    'FIGURE', 'FIGCAPTION', 'HR', 'BR',
  ]);

  function walk(node: Node): void {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = (node.textContent || '').replace(/\s+/g, ' ');
      if (text.trim()) parts.push(text);
      return;
    }

    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const el = node as Element;
    const tag = el.tagName;

    // Skip hidden elements
    if (el.hasAttribute('hidden') || (el as HTMLElement).style?.display === 'none') return;

    // Headings
    if (/^H[1-6]$/.test(tag)) {
      const level = parseInt(tag[1]);
      const prefix = '#'.repeat(Math.min(level, 3)) + ' ';
      const headingText = (el.textContent || '').trim();
      if (headingText) {
        parts.push('\n\n' + prefix + headingText + '\n');
      }
      return; // Don't walk children again
    }

    // List items
    if (tag === 'LI') {
      const text = (el.textContent || '').trim();
      if (text) {
        parts.push('\n- ' + text);
      }
      return;
    }

    // Links — inline, preserve URL
    if (tag === 'A') {
      const href = (el as HTMLAnchorElement).href;
      const text = (el.textContent || '').trim();
      if (text && href && !href.startsWith('javascript:')) {
        parts.push(text + ' (' + href + ')');
      } else if (text) {
        parts.push(text);
      }
      return;
    }

    // HR → separator
    if (tag === 'HR') {
      parts.push('\n\n---\n\n');
      return;
    }

    // BR → newline
    if (tag === 'BR') {
      parts.push('\n');
      return;
    }

    // Pre/code → preserve formatting
    if (tag === 'PRE') {
      const text = (el.textContent || '').trim();
      if (text) parts.push('\n\n```\n' + text + '\n```\n\n');
      return;
    }

    // Block element → paragraph break before
    const isBlock = BLOCK_TAGS.has(tag);
    if (isBlock) parts.push('\n\n');

    // Walk children
    for (const child of el.childNodes) {
      walk(child);
    }

    // Block element → paragraph break after
    if (isBlock) parts.push('\n');
  }

  walk(root);

  // Clean up excessive whitespace
  return parts
    .join('')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+/g, ' ')
    .trim();
}

/**
 * YouTube-specific extraction.
 */
function extractYouTubeContent(): ExtractPageResponse {
  const url = window.location.href;

  // Video title
  const titleEl = document.querySelector<HTMLElement>(
    'h1.ytd-watch-metadata, h1.ytd-video-primary-info-renderer, #title h1'
  );
  const videoTitle = titleEl?.textContent?.trim() || document.title.replace(/ - YouTube$/, '').trim();

  // Transcript segments
  let transcript = '';
  const transcriptSegments = document.querySelectorAll<HTMLElement>(
    'ytd-transcript-segment-renderer .segment-text, ' +
    'yt-formatted-string.ytd-transcript-segment-renderer, ' +
    '#segments-container ytd-transcript-segment-renderer'
  );
  if (transcriptSegments.length > 0) {
    const lines: string[] = [];
    transcriptSegments.forEach((seg) => {
      const text = seg.textContent?.trim();
      if (text) lines.push(text);
    });
    transcript = lines.join(' ');
  }

  // Description text
  let description = '';
  const descEl = document.querySelector<HTMLElement>(
    'ytd-text-inline-expander #plain-snippet-text, ' +
    '#description-inline-expander, ' +
    'ytd-expander .content, ' +
    '#description .content'
  );
  if (descEl) {
    description = descEl.textContent?.trim() || '';
  }

  // Comments (first few)
  const commentEls = document.querySelectorAll<HTMLElement>(
    'ytd-comment-thread-renderer #content-text'
  );
  let comments = '';
  if (commentEls.length > 0) {
    const commentLines: string[] = [];
    commentEls.forEach((el, i) => {
      if (i >= 10) return;
      const text = el.textContent?.trim();
      if (text) commentLines.push('- ' + text);
    });
    if (commentLines.length > 0) {
      comments = '\n\n## Top Comments\n' + commentLines.join('\n');
    }
  }

  // Build structured text
  let text = `# ${videoTitle}\n\n`;
  if (transcript) {
    text += `## Transcript\n${transcript}\n\n`;
  }
  if (description) {
    text += `## Description\n${description}\n\n`;
  }
  text += comments;

  if (text.length > MAX_TEXT_BYTES) {
    text = text.slice(0, MAX_TEXT_BYTES) + '\n\n[...truncated]';
  }

  const wordCount = text.split(/\s+/).filter(Boolean).length;

  return {
    text,
    title: videoTitle,
    url,
    wordCount,
    isYouTube: true,
    videoTitle,
    transcript: transcript || undefined,
  };
}

/**
 * Main extraction: finds content, cleans it, converts to structured text.
 * Falls back to raw innerText on any error.
 */
export function extractMainContent(): ExtractPageResponse {
  try {
    const url = window.location.href;
    const title = document.title;

    // PDF detection
    if (
      url.match(/\.pdf($|\?|#)/i) ||
      document.contentType === 'application/pdf'
    ) {
      return {
        text: document.body?.innerText?.slice(0, MAX_TEXT_BYTES) || '[PDF document]',
        title,
        url,
        wordCount: 0,
        isPdf: true,
      };
    }

    // YouTube detection
    if (url.includes('youtube.com/watch') || url.includes('youtu.be/')) {
      return extractYouTubeContent();
    }

    // General page extraction
    const root = findContentRoot();
    const cleaned = cleanClone(root);
    let text = domToStructuredText(cleaned);

    // If smart extraction yielded very little, fall back to body innerText
    if (text.length < 100) {
      text = document.body?.innerText || '';
    }

    // Truncate
    if (text.length > MAX_TEXT_BYTES) {
      text = text.slice(0, MAX_TEXT_BYTES) + '\n\n[...truncated]';
    }

    const wordCount = text.split(/\s+/).filter(Boolean).length;

    return { text, title, url, wordCount };
  } catch (_e) {
    // Fallback: raw innerText
    const fallbackText = (document.body?.innerText || '').slice(0, MAX_TEXT_BYTES);
    return {
      text: fallbackText,
      title: document.title,
      url: window.location.href,
      wordCount: fallbackText.split(/\s+/).filter(Boolean).length,
    };
  }
}