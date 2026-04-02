import { useEffect, useRef, useState } from 'react';
import { EditorState, type Extension } from '@codemirror/state';
import { EditorView, keymap, lineNumbers } from '@codemirror/view';
import { defaultKeymap } from '@codemirror/commands';
import { foldGutter, foldKeymap } from '@codemirror/language';
import { highlightSelectionMatches, searchKeymap } from '@codemirror/search';
import { MergeView } from '@codemirror/merge';
import { oneDark } from '@codemirror/theme-one-dark';
import {
  loadCodeEditorLanguageExtension,
  type CodeEditorLanguage,
} from '../utils/codeEditorLanguage';

interface DiffEditorProps {
  language: CodeEditorLanguage;
  modified: string;
  original: string;
}

function buildReadonlyExtensions(languageExtension: Extension): Extension[] {
  return [
    lineNumbers(),
    foldGutter(),
    EditorState.readOnly.of(true),
    EditorView.editable.of(false),
    EditorState.tabSize.of(2),
    oneDark,
    languageExtension,
    keymap.of([...defaultKeymap, ...searchKeymap, ...foldKeymap]),
    highlightSelectionMatches(),
    EditorView.theme({
      '&': {
        height: '100%',
        backgroundColor: '#0d0d14',
        color: 'var(--tx)',
        fontSize: '12px',
      },
      '.cm-scroller': {
        fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
        lineHeight: '1.6',
      },
      '.cm-content': {
        padding: '12px 14px',
      },
      '.cm-gutters': {
        backgroundColor: '#10111a',
        borderRight: '1px solid rgba(255,255,255,0.05)',
        color: 'rgba(255,255,255,0.28)',
      },
      '.cm-activeLine, .cm-activeLineGutter': {
        backgroundColor: 'transparent',
      },
      '.cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection': {
        backgroundColor: 'rgba(124,58,237,0.28)',
      },
      '.cm-cursor': {
        display: 'none',
      },
      '.cm-searchMatch': {
        backgroundColor: 'rgba(245,158,11,0.14)',
        outline: '1px solid rgba(245,158,11,0.28)',
      },
    }),
  ];
}

export default function DiffEditor({ language, modified, original }: DiffEditorProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mergeRef = useRef<MergeView | null>(null);
  const [languageExtension, setLanguageExtension] = useState<Extension | null>(null);

  useEffect(() => {
    let cancelled = false;

    loadCodeEditorLanguageExtension(language).then((extension) => {
      if (!cancelled) {
        setLanguageExtension(extension);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [language]);

  useEffect(() => {
    if (!containerRef.current || !languageExtension) return;

    mergeRef.current?.destroy();
    mergeRef.current = new MergeView({
      parent: containerRef.current,
      orientation: 'a-b',
      highlightChanges: true,
      gutter: true,
      collapseUnchanged: { margin: 2, minSize: 4 },
      diffConfig: { scanLimit: 500, timeout: 1000 },
      a: {
        doc: original,
        extensions: buildReadonlyExtensions(languageExtension),
      },
      b: {
        doc: modified,
        extensions: buildReadonlyExtensions(languageExtension),
      },
    });

    return () => {
      mergeRef.current?.destroy();
      mergeRef.current = null;
    };
  }, [languageExtension, modified, original]);

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 14,
        overflow: 'hidden',
        background: '#0d0d14',
        position: 'relative',
      }}
    >
      {!languageExtension && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--mu)',
            fontSize: 12,
            background: '#0d0d14',
            zIndex: 1,
          }}
        >
          Loading diff...
        </div>
      )}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
