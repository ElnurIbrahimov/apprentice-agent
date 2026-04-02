import { useEffect, useRef } from 'react';
import { Compartment, EditorState, type Extension } from '@codemirror/state';
import {
  EditorView,
  drawSelection,
  highlightActiveLine,
  highlightActiveLineGutter,
  keymap,
  lineNumbers,
} from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import {
  bracketMatching,
  defaultHighlightStyle,
  foldGutter,
  foldKeymap,
  indentOnInput,
  indentUnit,
  syntaxHighlighting,
} from '@codemirror/language';
import { autocompletion, closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';
import { highlightSelectionMatches, searchKeymap } from '@codemirror/search';
import { lintGutter, setDiagnostics, type Diagnostic } from '@codemirror/lint';
import { oneDark } from '@codemirror/theme-one-dark';
import {
  loadCodeEditorLanguageExtension,
  type CodeEditorLanguage,
} from '../utils/codeEditorLanguage';
export type { CodeEditorLanguage } from '../utils/codeEditorLanguage';

export interface CodeEditorDiagnostic {
  line: number;
  message: string;
  severity?: 'error' | 'warning' | 'info';
}

interface CodeEditorProps {
  code: string;
  diagnostics?: CodeEditorDiagnostic[];
  height?: string;
  language: CodeEditorLanguage;
  lineWrapping?: boolean;
  onChange?: (code: string) => void;
  readOnly?: boolean;
}

export default function CodeEditor({
  code,
  diagnostics = [],
  height = '100%',
  language,
  lineWrapping = true,
  onChange,
  readOnly = false,
}: CodeEditorProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const latestCodeRef = useRef(code);
  const onChangeRef = useRef(onChange);
  const languageRequestRef = useRef(0);

  const languageCompartmentRef = useRef(new Compartment());
  const readOnlyCompartmentRef = useRef(new Compartment());
  const editableCompartmentRef = useRef(new Compartment());
  const wrappingCompartmentRef = useRef(new Compartment());
  const themeCompartmentRef = useRef(new Compartment());

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  useEffect(() => {
    latestCodeRef.current = code;
  }, [code]);

  useEffect(() => {
    if (!containerRef.current || viewRef.current) return;

    const themeExtension = EditorView.theme({
      '&': {
        height,
        color: 'var(--tx)',
        backgroundColor: '#0d0d14',
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
      '.cm-activeLineGutter': {
        backgroundColor: 'rgba(124,58,237,0.08)',
      },
      '.cm-activeLine': {
        backgroundColor: 'rgba(124,58,237,0.07)',
      },
      '.cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection': {
        backgroundColor: 'rgba(124,58,237,0.32)',
      },
      '.cm-cursor, .cm-dropCursor': {
        borderLeftColor: '#c4b5fd',
      },
      '.cm-panels': {
        backgroundColor: '#12131c',
        color: 'var(--tx)',
      },
      '.cm-searchMatch': {
        backgroundColor: 'rgba(245,158,11,0.14)',
        outline: '1px solid rgba(245,158,11,0.28)',
      },
      '.cm-searchMatch.cm-searchMatch-selected': {
        backgroundColor: 'rgba(245,158,11,0.24)',
      },
      '.cm-tooltip': {
        backgroundColor: '#171827',
        border: '1px solid rgba(255,255,255,0.08)',
      },
      '.cm-tooltip-autocomplete > ul > li[aria-selected]': {
        backgroundColor: 'rgba(124,58,237,0.22)',
      },
      '.cm-foldPlaceholder': {
        backgroundColor: 'rgba(255,255,255,0.06)',
        border: '1px solid rgba(255,255,255,0.08)',
        color: 'var(--mu)',
      },
    });

    const state = EditorState.create({
      doc: code,
      extensions: [
        lineNumbers(),
        foldGutter(),
        drawSelection(),
        EditorState.allowMultipleSelections.of(true),
        EditorState.tabSize.of(2),
        indentUnit.of('  '),
        indentOnInput(),
        bracketMatching(),
        closeBrackets(),
        autocompletion(),
        history(),
        highlightActiveLine(),
        highlightActiveLineGutter(),
        highlightSelectionMatches(),
        lintGutter(),
        syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
        keymap.of([
          indentWithTab,
          ...closeBracketsKeymap,
          ...defaultKeymap,
          ...historyKeymap,
          ...searchKeymap,
          ...foldKeymap,
        ]),
        oneDark,
        languageCompartmentRef.current.of([]),
        readOnlyCompartmentRef.current.of(EditorState.readOnly.of(readOnly)),
        editableCompartmentRef.current.of(EditorView.editable.of(!readOnly)),
        wrappingCompartmentRef.current.of(lineWrapping ? EditorView.lineWrapping : []),
        themeCompartmentRef.current.of(themeExtension),
        EditorView.updateListener.of((update) => {
          if (!update.docChanged) return;
          const nextCode = update.state.doc.toString();
          latestCodeRef.current = nextCode;
          onChangeRef.current?.(nextCode);
        }),
      ],
    });

    viewRef.current = new EditorView({
      state,
      parent: containerRef.current,
    });

    return () => {
      viewRef.current?.destroy();
      viewRef.current = null;
    };
  }, [code, height, language, lineWrapping, readOnly]);

  useEffect(() => {
    const requestId = ++languageRequestRef.current;

    loadCodeEditorLanguageExtension(language).then((extension) => {
      if (languageRequestRef.current !== requestId) return;
      const view = viewRef.current;
      if (!view) return;

      view.dispatch({
        effects: languageCompartmentRef.current.reconfigure(extension),
      });
    });
  }, [language]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;

    view.dispatch({
      effects: [
        readOnlyCompartmentRef.current.reconfigure(EditorState.readOnly.of(readOnly)),
        editableCompartmentRef.current.reconfigure(EditorView.editable.of(!readOnly)),
      ],
    });
  }, [readOnly]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;

    view.dispatch({
      effects: wrappingCompartmentRef.current.reconfigure(lineWrapping ? EditorView.lineWrapping : []),
    });
  }, [lineWrapping]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;

    view.dispatch({
      effects: themeCompartmentRef.current.reconfigure(EditorView.theme({
        '&': {
          height,
          color: 'var(--tx)',
          backgroundColor: '#0d0d14',
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
        '.cm-activeLineGutter': {
          backgroundColor: 'rgba(124,58,237,0.08)',
        },
        '.cm-activeLine': {
          backgroundColor: 'rgba(124,58,237,0.07)',
        },
        '.cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection': {
          backgroundColor: 'rgba(124,58,237,0.32)',
        },
        '.cm-cursor, .cm-dropCursor': {
          borderLeftColor: '#c4b5fd',
        },
        '.cm-panels': {
          backgroundColor: '#12131c',
          color: 'var(--tx)',
        },
        '.cm-searchMatch': {
          backgroundColor: 'rgba(245,158,11,0.14)',
          outline: '1px solid rgba(245,158,11,0.28)',
        },
        '.cm-searchMatch.cm-searchMatch-selected': {
          backgroundColor: 'rgba(245,158,11,0.24)',
        },
        '.cm-tooltip': {
          backgroundColor: '#171827',
          border: '1px solid rgba(255,255,255,0.08)',
        },
        '.cm-tooltip-autocomplete > ul > li[aria-selected]': {
          backgroundColor: 'rgba(124,58,237,0.22)',
        },
        '.cm-foldPlaceholder': {
          backgroundColor: 'rgba(255,255,255,0.06)',
          border: '1px solid rgba(255,255,255,0.08)',
          color: 'var(--mu)',
        },
      })),
    });
  }, [height]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;

    const doc = view.state.doc;
    const nextDiagnostics: Diagnostic[] = diagnostics
      .filter((item) => item.line >= 1 && !!item.message.trim())
      .map((item) => {
        const lineNumber = Math.min(Math.max(item.line, 1), doc.lines);
        const line = doc.line(lineNumber);
        const from = line.from;
        const to = line.to > line.from ? line.to : Math.min(line.from + 1, doc.length);

        return {
          from,
          to,
          severity: item.severity || 'error',
          message: item.message,
          source: 'Runtime',
        };
      });

    view.dispatch(setDiagnostics(view.state, nextDiagnostics));
  }, [diagnostics]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    if (code === latestCodeRef.current) return;

    latestCodeRef.current = code;
    view.dispatch({
      changes: {
        from: 0,
        to: view.state.doc.length,
        insert: code,
      },
    });
  }, [code]);

  return <div ref={containerRef} style={{ width: '100%', height }} />;
}
