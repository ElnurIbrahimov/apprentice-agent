import { useState, useRef, useEffect, useCallback } from 'react';
import { ClipboardDocumentIcon, CheckIcon, ChevronDownIcon, ChevronUpIcon, PlayIcon, EyeIcon } from '@heroicons/react/24/outline';
import { highlightCode } from '../utils/codeHighlighter';
import { executeInline } from '../utils/pyodideExecutor';
import { detectArtifactType, buildSrcdoc, type ArtifactType } from '../utils/artifactRenderer';
import { sanitizeHtml } from '../utils/sanitize';
import { copyText } from '../utils/clipboard';
import { useSettingsStore } from '../store/settingsStore';

const COLLAPSE_THRESHOLD = 20;
const RUNNABLE_LANGS = new Set(['python', 'py']);

interface InlineOutput {
  type: 'stdout' | 'stderr' | 'image' | 'html' | 'error';
  text?: string;
  data?: string;
  mime?: string;
  content?: string;
  ename?: string;
  evalue?: string;
}

interface CodeBlockProps {
  language: string;
  children: string;
  onOpenArtifact?: (code: string, type: ArtifactType) => void;
}

export function CodeBlock({ language, children, onOpenArtifact }: CodeBlockProps) {
  const [codeCopied, setCodeCopied] = useState(false);
  const [highlighted, setHighlighted] = useState('');
  const [collapsed, setCollapsed] = useState(() => children.split('\n').length > COLLAPSE_THRESHOLD);
  const [runOutputs, setRunOutputs] = useState<InlineOutput[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [previewHeight, setPreviewHeight] = useState(300);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const copyTimer = useRef<ReturnType<typeof setTimeout>>();
  const lineCount = children.split('\n').length;
  const canRun = RUNNABLE_LANGS.has(language.toLowerCase());
  const artifactType = detectArtifactType(children, language);
  const theme = useSettingsStore((s) => s.settings.theme);

  useEffect(() => () => { clearTimeout(copyTimer.current); }, []);

  useEffect(() => {
    // Determine actual rendered theme (handle 'system' by checking OS preference)
    const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    highlightCode(children, language, isDark ? 'dark' : 'light')
      .then(setHighlighted).catch(() => {});
  }, [children, language, theme]);

  useEffect(() => {
    if (!showPreview) return;
    const handler = (e: MessageEvent) => {
      if (e.data?.type === 'artifact-resize' && e.source === iframeRef.current?.contentWindow) {
        setPreviewHeight(Math.min(e.data.height + 10, 500));
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [showPreview]);

  const handleCopy = useCallback(async () => {
    if (await copyText(children)) {
      clearTimeout(copyTimer.current);
      setCodeCopied(true);
      copyTimer.current = setTimeout(() => setCodeCopied(false), 2000);
    }
  }, [children]);

  const handleRun = useCallback(() => {
    if (isRunning || !canRun) return;
    setIsRunning(true);
    setRunOutputs([]);
    executeInline(
      children,
      (block) => setRunOutputs((prev) => [...prev, block as InlineOutput]),
      () => setIsRunning(false),
    );
  }, [children, canRun, isRunning]);

  return (
    <div className="relative rounded-lg border border-chat-border overflow-hidden bg-surface-1">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-chat-border">
        <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-purple-500/25 text-purple-300">
          {language}
        </span>
        <div className="flex items-center gap-1">
          {canRun && (
            <button onClick={handleRun} disabled={isRunning}
              className="flex items-center gap-1 px-2 py-1 text-xs text-green-400 hover:text-green-300 hover:bg-green-500/10 disabled:opacity-50 transition-colors rounded">
              <PlayIcon className="w-3.5 h-3.5" />
              {isRunning ? 'Running...' : 'Run'}
            </button>
          )}
          {artifactType && onOpenArtifact && (
            <button onClick={() => onOpenArtifact(children, artifactType)}
              className="flex items-center gap-1 px-2 py-1 text-xs text-blue-400 hover:text-blue-300 hover:bg-blue-500/10 transition-colors rounded">
              <EyeIcon className="w-3.5 h-3.5" />Preview
            </button>
          )}
          {lineCount > COLLAPSE_THRESHOLD && (
            <button onClick={() => setCollapsed(!collapsed)}
              className="flex items-center gap-1 px-2 py-1 text-xs text-chat-text-secondary hover:text-chat-text hover:bg-surface-2 transition-colors rounded">
              {collapsed ? <><ChevronDownIcon className="w-3 h-3" />{lineCount} lines</> : <><ChevronUpIcon className="w-3 h-3" />Collapse</>}
            </button>
          )}
          <button onClick={handleCopy} aria-label="Copy code"
            className="flex items-center gap-1 px-2 py-1 text-xs text-chat-text-secondary hover:text-chat-text hover:bg-surface-2 transition-colors rounded code-copy-btn touch-target">
            {codeCopied
              ? <><CheckIcon className="w-3.5 h-3.5 text-green-400" /><span className="text-green-400">Copied</span></>
              : <><ClipboardDocumentIcon className="w-3.5 h-3.5" /><span>Copy</span></>
            }
          </button>
        </div>
      </div>

      {/* Preview tabs */}
      {artifactType && (
        <div className="flex gap-1 px-3 py-1 border-b border-chat-border">
          <button onClick={() => setShowPreview(false)} className={`text-[10px] px-2 py-0.5 rounded ${!showPreview ? 'bg-chat-accent text-white' : 'text-chat-text-secondary'}`}>Code</button>
          <button onClick={() => setShowPreview(true)} className={`text-[10px] px-2 py-0.5 rounded ${showPreview ? 'bg-chat-accent text-white' : 'text-chat-text-secondary'}`}>Preview</button>
        </div>
      )}

      {/* Code content or inline preview */}
      {showPreview && artifactType ? (
        <div className="p-2">
          <iframe
            ref={iframeRef}
            srcDoc={buildSrcdoc(artifactType, children)}
            sandbox="allow-scripts"
            className="w-full border-none rounded"
            style={{ height: previewHeight }}
            title="Inline preview"
          />
        </div>
      ) : (
        <div className="overflow-x-auto transition-[max-height] duration-300"
          style={{ maxHeight: collapsed ? '240px' : 'none', overflow: collapsed ? 'hidden' : 'auto' }}>
          {highlighted ? (
            <div dangerouslySetInnerHTML={{ __html: highlighted }}
              className="shiki-block p-4 m-0 text-sm [&_pre]:!bg-transparent [&_pre]:!m-0 [&_pre]:!p-0 [&_code]:!bg-transparent" />
          ) : (
            <pre className="p-4 overflow-x-auto m-0 text-sm bg-surface-1">
              <code className={`language-${language}`}>{children}</code>
            </pre>
          )}
        </div>
      )}

      {collapsed && lineCount > COLLAPSE_THRESHOLD && (
        <div className="absolute bottom-0 left-0 right-0 h-12 pointer-events-none"
          style={{ background: 'linear-gradient(transparent, var(--surface-1))' }} />
      )}

      {/* Inline run output */}
      {runOutputs.length > 0 && (
        <div className="border-t border-chat-border">
          {runOutputs.map((block, i) => {
            if (block.type === 'stdout' || block.type === 'stderr')
              return <pre key={i} className={`text-xs font-mono whitespace-pre-wrap px-3 py-1 ${block.type === 'stderr' ? 'text-yellow-400' : 'text-chat-text'}`}>{block.text}</pre>;
            if (block.type === 'image' && block.data)
              return <div key={i} className="px-3 py-2"><img src={`data:${block.mime || 'image/png'};base64,${block.data}`} alt="Output" className="max-w-full rounded" /></div>;
            if (block.type === 'html' && block.content)
              return <div key={i} className="px-3 py-2 overflow-x-auto text-xs" dangerouslySetInnerHTML={{ __html: sanitizeHtml(block.content) }} />;
            if (block.type === 'error')
              return <div key={i} className="px-3 py-2 text-xs text-red-400"><span className="font-semibold">{block.ename}:</span> {block.evalue}</div>;
            return null;
          })}
        </div>
      )}
    </div>
  );
}
