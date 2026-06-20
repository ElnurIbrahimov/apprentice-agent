/**
 * Full Page Extraction — extracted from page-services.ts
 * Provides comprehensive page data extraction for AI analysis.
 */

// CSS properties for style extraction (shared with capture module)
const PAGE_CSS_PROPS: string[] = [
  'display', 'position', 'flex-direction', 'align-items', 'justify-content',
  'gap', 'flex-wrap', 'flex', 'flex-grow', 'flex-shrink',
  'width', 'height', 'min-width', 'min-height', 'max-width', 'max-height',
  'padding', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
  'margin', 'margin-top', 'margin-right', 'margin-bottom', 'margin-left',
  'border', 'border-radius', 'border-color', 'border-width', 'border-style',
  'background', 'background-color', 'background-image', 'background-size',
  'color', 'font-size', 'font-weight', 'font-family', 'line-height',
  'letter-spacing', 'text-align', 'text-decoration', 'text-transform',
  'box-shadow', 'opacity', 'overflow', 'z-index',
  'grid-template-columns', 'grid-template-rows', 'grid-gap',
  'transform', 'transition',
];

function extractComputedStylesForPage(el: Element): Record<string, string> {
  const styles = window.getComputedStyle(el);
  const result: Record<string, string> = {};
  for (const prop of PAGE_CSS_PROPS) {
    const val = styles.getPropertyValue(prop);
    if (val && val !== 'none' && val !== 'normal' && val !== 'auto' && val !== '0px' && val !== 'rgba(0, 0, 0, 0)') {
      result[prop] = val;
    }
  }
  return result;
}

function buildPageCssSelector(el: Element): string {
  const tag = el.tagName.toLowerCase();
  const cls = el.className && typeof el.className === 'string'
    ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.')
    : '';
  return tag + cls;
}

export function extractFullPageData(): {
  html: string;
  css: string;
  css_map: Record<string, Record<string, string>>;
  colors: string[];
  fonts: string[];
  metadata: {
    title: string;
    description: string;
    og_image: string;
    og_title: string;
    og_description: string;
    og_type: string;
    og_site_name: string;
    favicon: string;
  };
  source_url: string;
  viewport: { width: number; height: number };
  asset_urls: { images: string[]; stylesheets: string[] };
  responsive_info: { viewport_width: number; media_queries: string[] };
  element_count: number;
} {
  // 1. Clean full page HTML
  const clone = document.documentElement.cloneNode(true) as HTMLElement;

  // Remove scripts, tracking pixels, ads, cookie banners, noscript
  const removeSelectors = [
    'script', 'noscript', 'iframe[src*="ads"]', 'iframe[src*="track"]',
    'iframe[src*="pixel"]', 'iframe[width="0"]', 'iframe[height="0"]',
    'img[src*="pixel"]', 'img[src*="track"]', 'img[width="1"]', 'img[height="1"]',
    '[id*="cookie"]', '[class*="cookie"]', '[id*="consent"]', '[class*="consent"]',
    '[id*="gdpr"]', '[class*="gdpr"]', '[id*="onetrust"]', '[class*="onetrust"]',
    '[id*="CybotCookiebot"]', '[data-testid*="cookie"]',
    '[id*="ad-"]', '[class*="ad-container"]', '[class*="ad-wrapper"]',
    'link[rel="preconnect"]', 'link[rel="dns-prefetch"]',
    'meta[http-equiv="Content-Security-Policy"]',
    'style[data-emotion]', // runtime CSS-in-JS noise
  ];
  for (const sel of removeSelectors) {
    try {
      clone.querySelectorAll(sel).forEach(el => el.remove());
    } catch (_e) { /* invalid selector, skip */ }
  }
  // Remove all inline event handlers
  clone.querySelectorAll('*').forEach(el => {
    const attrs = el.getAttributeNames();
    for (const attr of attrs) {
      if (attr.startsWith('on') || attr === 'data-analytics' || attr === 'data-tracking') {
        el.removeAttribute(attr);
      }
    }
  });
  const cleanHtml = clone.outerHTML;

  // 2. Extract computed styles for key elements (up to 200)
  const cssMap: Record<string, Record<string, string>> = {};
  const keySelectors = [
    'body', 'header', 'nav', 'main', 'footer', 'aside', 'section', 'article',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'button', 'input', 'textarea',
    'ul', 'ol', 'li', 'img', 'form', 'table', 'th', 'td',
    '[class*="hero"]', '[class*="card"]', '[class*="btn"]', '[class*="nav"]',
    '[class*="header"]', '[class*="footer"]', '[class*="sidebar"]',
    '[class*="container"]', '[class*="wrapper"]', '[class*="grid"]',
    '[class*="flex"]', '[class*="modal"]', '[class*="banner"]',
  ];
  let styleCount = 0;
  for (const sel of keySelectors) {
    if (styleCount >= 200) break;
    try {
      const els = document.querySelectorAll(sel);
      for (const el of els) {
        if (styleCount >= 200) break;
        const styles = extractComputedStylesForPage(el);
        if (Object.keys(styles).length > 0) {
          const key = buildPageCssSelector(el);
          const finalKey = cssMap[key] ? `${key}:nth(${styleCount})` : key;
          cssMap[finalKey] = styles;
          styleCount++;
        }
      }
    } catch (_e) { /* invalid selector */ }
  }

  // Build CSS string
  const cssLines: string[] = [];
  for (const [selector, props] of Object.entries(cssMap)) {
    cssLines.push(`${selector} {`);
    for (const [prop, val] of Object.entries(props)) {
      cssLines.push(`  ${prop}: ${val};`);
    }
    cssLines.push('}');
    cssLines.push('');
  }
  const cssString = cssLines.join('\n');

  // 3. Extract color palette from computed styles
  const colorSet = new Set<string>();
  const colorProps = ['color', 'background-color', 'border-color', 'outline-color'];
  const sampleEls = document.querySelectorAll('*');
  let sampleCount = 0;
  for (const el of sampleEls) {
    if (sampleCount >= 500) break;
    const cs = window.getComputedStyle(el);
    for (const cp of colorProps) {
      const val = cs.getPropertyValue(cp);
      if (val && val !== 'rgba(0, 0, 0, 0)' && val !== 'transparent' && val !== 'inherit' && val !== 'initial') {
        colorSet.add(val);
      }
    }
    sampleCount++;
  }
  const colors = Array.from(colorSet).slice(0, 50);

  // 4. Extract font stack
  const fontSet = new Set<string>();
  for (const el of sampleEls) {
    if (fontSet.size >= 20) break;
    const cs = window.getComputedStyle(el);
    const ff = cs.getPropertyValue('font-family');
    if (ff) {
      // Extract individual font names
      const fonts = ff.split(',').map(f => f.trim().replace(/^["']|["']$/g, ''));
      for (const font of fonts) {
        if (font && !font.includes('inherit') && !font.includes('initial') && font.length < 50) {
          fontSet.add(font);
        }
      }
    }
  }
  const fonts = Array.from(fontSet).slice(0, 20);

  // 5. Page metadata
  const getMeta = (name: string): string => {
    const el = document.querySelector(`meta[property="${name}"], meta[name="${name}"]`);
    return el?.getAttribute('content') || '';
  };
  const faviconEl = document.querySelector('link[rel="icon"], link[rel="shortcut icon"]');
  const metadata = {
    title: document.title || '',
    description: getMeta('description'),
    og_image: getMeta('og:image'),
    og_title: getMeta('og:title'),
    og_description: getMeta('og:description'),
    og_type: getMeta('og:type'),
    og_site_name: getMeta('og:site_name'),
    favicon: faviconEl?.getAttribute('href') || '',
  };

  // 6. Responsive info
  const viewport = {
    width: window.innerWidth,
    height: window.innerHeight,
  };

  // Try to detect media queries from stylesheets
  const mediaQueries: string[] = [];
  try {
    for (const sheet of document.styleSheets) {
      try {
        const rules = sheet.cssRules || sheet.rules;
        if (!rules) continue;
        for (const rule of rules) {
          if (rule instanceof CSSMediaRule && rule.conditionText) {
            if (!mediaQueries.includes(rule.conditionText)) {
              mediaQueries.push(rule.conditionText);
            }
            if (mediaQueries.length >= 20) break;
          }
        }
      } catch (_e) { /* cross-origin stylesheet */ }
      if (mediaQueries.length >= 20) break;
    }
  } catch (_e) { /* no access */ }

  // 7. Asset URLs
  const images: string[] = [];
  document.querySelectorAll('img[src]').forEach(img => {
    const src = img.getAttribute('src');
    if (src && !src.startsWith('data:') && images.length < 50) {
      try {
        images.push(new URL(src, location.href).href);
      } catch (_e) {
        images.push(src);
      }
    }
  });

  const stylesheets: string[] = [];
  document.querySelectorAll('link[rel="stylesheet"][href]').forEach(link => {
    const href = link.getAttribute('href');
    if (href && stylesheets.length < 20) {
      try {
        stylesheets.push(new URL(href, location.href).href);
      } catch (_e) {
        stylesheets.push(href);
      }
    }
  });

  // 8. Element count
  const elementCount = document.querySelectorAll('*').length;

  return {
    html: cleanHtml,
    css: cssString,
    css_map: cssMap,
    colors,
    fonts,
    metadata,
    source_url: location.href,
    viewport,
    asset_urls: { images, stylesheets },
    responsive_info: { viewport_width: viewport.width, media_queries: mediaQueries },
    element_count: elementCount,
  };
}