import type { Extension } from '@codemirror/state';

export type CodeEditorLanguage =
  | 'html'
  | 'javascript'
  | 'typescript'
  | 'jsx'
  | 'tsx'
  | 'python'
  | 'css'
  | 'json'
  | 'markdown'
  | 'svg';

const languageExtensionCache = new Map<CodeEditorLanguage, Promise<Extension>>();

function createLanguageExtension(language: CodeEditorLanguage): Promise<Extension> {
  switch (language) {
    case 'html':
    case 'svg':
      return import('@codemirror/lang-html').then(({ html }) => html());
    case 'javascript':
      return import('@codemirror/lang-javascript').then(({ javascript }) => javascript());
    case 'typescript':
      return import('@codemirror/lang-javascript').then(({ javascript }) => javascript({ typescript: true }));
    case 'jsx':
      return import('@codemirror/lang-javascript').then(({ javascript }) => javascript({ jsx: true }));
    case 'tsx':
      return import('@codemirror/lang-javascript').then(({ javascript }) =>
        javascript({ jsx: true, typescript: true }),
      );
    case 'python':
      return import('@codemirror/lang-python').then(({ python }) => python());
    case 'css':
      return import('@codemirror/lang-css').then(({ css }) => css());
    case 'json':
      return import('@codemirror/lang-json').then(({ json }) => json());
    case 'markdown':
      return import('@codemirror/lang-markdown').then(({ markdown }) => markdown());
    default:
      return import('@codemirror/lang-html').then(({ html }) => html());
  }
}

export function loadCodeEditorLanguageExtension(language: CodeEditorLanguage): Promise<Extension> {
  const cached = languageExtensionCache.get(language);
  if (cached) return cached;

  const next = createLanguageExtension(language);
  languageExtensionCache.set(language, next);
  return next;
}
