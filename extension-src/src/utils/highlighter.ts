/**
 * Lightweight multi-language syntax highlighter.
 * Supports: HTML, CSS, JavaScript, Python, JSON.
 * Uses placeholder-based single-pass to prevent cross-rule corruption.
 * No external dependencies — ~3KB.
 */

const RULES: Record<string, Array<[RegExp, string]>> = {
  html: [
    [/(&lt;!--[\s\S]*?--&gt;)/g, 'comment'],
    [/(&lt;\/?[a-zA-Z][a-zA-Z0-9-]*)/g, 'tag'],
    [/(\s[a-zA-Z-]+=)/g, 'attr'],
    [/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, 'string'],
    [/(&gt;)/g, 'tag'],
  ],
  javascript: [
    [/(\/\/.*$)/gm, 'comment'],
    [/(\/\*[\s\S]*?\*\/)/g, 'comment'],
    [/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/g, 'string'],
    [/\b(const|let|var|function|return|if|else|for|while|do|switch|case|break|continue|new|this|class|extends|import|export|from|default|async|await|try|catch|finally|throw|typeof|instanceof|in|of|yield|null|undefined|true|false|void|delete)\b/g, 'keyword'],
    [/\b(console|document|window|Math|Array|Object|String|Number|Boolean|Promise|Map|Set|Date|JSON|Error|RegExp)\b/g, 'builtin'],
    [/\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g, 'number'],
  ],
  css: [
    [/(\/\*[\s\S]*?\*\/)/g, 'comment'],
    [/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, 'string'],
    [/([.#][a-zA-Z_-][a-zA-Z0-9_-]*)/g, 'selector'],
    [/\b([a-z-]+)\s*:/g, 'property'],
    [/(#[0-9a-fA-F]{3,8})\b/g, 'number'],
    [/\b(\d+\.?\d*(?:px|em|rem|%|vh|vw|s|ms|deg|fr)?)\b/g, 'number'],
    [/(@[a-z-]+)/g, 'keyword'],
  ],
  python: [
    [/(#.*$)/gm, 'comment'],
    [/("""[\s\S]*?"""|'''[\s\S]*?''')/g, 'string'],
    [/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, 'string'],
    [/\b(import|from|as|def|class|return|if|elif|else|for|while|in|not|and|or|is|with|try|except|finally|raise|yield|lambda|pass|break|continue|True|False|None|print|len|range|list|dict|set|tuple|int|float|str|bool|type|self|async|await)\b/g, 'keyword'],
    [/\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g, 'number'],
  ],
  json: [
    [/("(?:[^"\\]|\\.)*")\s*:/g, 'property'],
    [/("(?:[^"\\]|\\.)*")/g, 'string'],
    [/\b(true|false|null)\b/g, 'keyword'],
    [/\b(-?\d+\.?\d*(?:e[+-]?\d+)?)\b/g, 'number'],
  ],
};

// Color scheme (GitHub dark)
const COLORS: Record<string, string> = {
  comment: '#8b949e',
  tag: '#7ee787',
  attr: '#79c0ff',
  string: '#a5d6ff',
  keyword: '#ff7b72',
  builtin: '#ffa657',
  number: '#79c0ff',
  selector: '#7ee787',
  property: '#79c0ff',
};

/**
 * Detect language from code content.
 */
export function detectLanguage(code: string, hint?: string): string {
  if (hint) {
    const h = hint.toLowerCase();
    if (h === 'html' || h === 'svg' || h === 'react') return 'html';
    if (h === 'css') return 'css';
    if (h === 'python') return 'python';
    if (h === 'json' || h === 'chart') return 'json';
    if (h === 'mermaid' || h === 'markdown') return 'plain';
  }
  if (code.includes('<!DOCTYPE') || code.includes('<html') || code.includes('<div')) return 'html';
  if (code.includes('import ') && code.includes('def ')) return 'python';
  if (code.trimStart().startsWith('{') || code.trimStart().startsWith('[')) return 'json';
  if (code.includes('{') && (code.includes(':') && code.includes(';'))) return 'css';
  return 'javascript';
}

/**
 * Highlight code string and return HTML.
 * Uses placeholder replacement to prevent multi-pass regex corruption.
 */
export function highlightCode(code: string, language?: string): string {
  const lang = language || detectLanguage(code);
  const rules = RULES[lang];
  if (!rules) return escapeHtml(code);

  // Escape HTML entities first
  let html = escapeHtml(code);

  // Placeholder-based approach: replace matches with unique markers,
  // then substitute markers with colored spans at the end.
  const placeholders: string[] = [];

  for (const [regex, cls] of rules) {
    const color = COLORS[cls] || '#e6edf3';
    html = html.replace(regex, (match) => {
      // Don't highlight inside an existing placeholder
      if (match.includes('\x00')) return match;
      const idx = placeholders.length;
      placeholders.push(`<span style="color:${color}">${match}</span>`);
      return `\x00${idx}\x00`;
    });
  }

  // Replace placeholders with actual spans
  html = html.replace(/\x00(\d+)\x00/g, (_, idx) => placeholders[parseInt(idx, 10)]);

  return html;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
