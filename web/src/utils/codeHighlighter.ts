import { createHighlighter, type Highlighter } from 'shiki';

let highlighterPromise: Promise<Highlighter> | null = null;

const LANGUAGES = [
  'javascript', 'typescript', 'python', 'html', 'css', 'json', 'bash',
  'sql', 'jsx', 'tsx', 'markdown', 'yaml', 'rust', 'go', 'java', 'c',
  'cpp', 'shell', 'plaintext',
];

function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: ['github-dark', 'github-light'],
      langs: LANGUAGES,
    });
  }
  return highlighterPromise;
}

export async function highlightCode(
  code: string,
  language: string,
  theme: 'dark' | 'light',
): Promise<string> {
  const h = await getHighlighter();
  const themeName = theme === 'dark' ? 'github-dark' : 'github-light';
  // Normalize language name
  let lang = language.toLowerCase();
  if (lang === 'sh' || lang === 'zsh') lang = 'bash';
  if (lang === 'js') lang = 'javascript';
  if (lang === 'ts') lang = 'typescript';
  if (lang === 'py') lang = 'python';
  if (lang === 'yml') lang = 'yaml';
  if (lang === 'c++') lang = 'cpp';
  if (!LANGUAGES.includes(lang)) lang = 'plaintext';

  return h.codeToHtml(code, { lang, theme: themeName });
}
