import { useState, useRef, useEffect, useCallback } from 'react';
import {
  XMarkIcon, ArrowsPointingOutIcon, DevicePhoneMobileIcon,
  DeviceTabletIcon, ComputerDesktopIcon, CodeBracketIcon,
  EyeIcon, ArrowDownTrayIcon, ArrowUturnLeftIcon, ArrowUturnRightIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { buildSrcdoc, type ArtifactType } from '../utils/artifactRenderer';
import { highlightCode } from '../utils/codeHighlighter';

type ViewMode = 'preview' | 'code' | 'split';
type DeviceSize = 'desktop' | 'tablet' | 'mobile';

const DEVICE_WIDTHS: Record<DeviceSize, string> = {
  desktop: '100%',
  tablet: '768px',
  mobile: '375px',
};

interface ArtifactVersion {
  code: string;
  type: ArtifactType;
  timestamp: number;
}

interface ArtifactsPanelProps {
  code: string;
  type: ArtifactType;
  onClose: () => void;
}

interface ConsoleEntry {
  level: string;
  args: string[];
  timestamp: number;
}

export function ArtifactsPanel({ code, type, onClose }: ArtifactsPanelProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('preview');
  const [device, setDevice] = useState<DeviceSize>('desktop');
  const [errors, setErrors] = useState<Array<{ msg: string; line?: number }>>([]);
  const [consoleEntries, setConsoleEntries] = useState<ConsoleEntry[]>([]);
  const [showConsole, setShowConsole] = useState(false);
  const [codeHtml, setCodeHtml] = useState('');
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Version history
  const [versions, setVersions] = useState<ArtifactVersion[]>([{ code, type, timestamp: Date.now() }]);
  const [versionIndex, setVersionIndex] = useState(0);

  // Update when new code arrives
  useEffect(() => {
    setVersions((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.code === code && last.type === type) return prev;
      const newVersions = [...prev, { code, type, timestamp: Date.now() }].slice(-10);
      setVersionIndex(newVersions.length - 1);
      return newVersions;
    });
  }, [code, type]);

  const currentVersion = versions[versionIndex] || { code, type };

  // Syntax highlight for code view
  useEffect(() => {
    const lang = currentVersion.type === 'react' ? 'tsx' : currentVersion.type === 'mermaid' ? 'markdown' : currentVersion.type;
    const isDark = !document.documentElement.classList.contains('light');
    highlightCode(currentVersion.code, lang, isDark ? 'dark' : 'light')
      .then(setCodeHtml)
      .catch(() => {});
  }, [currentVersion.code, currentVersion.type]);

  // Listen for iframe messages
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      // Only accept messages from our sandbox iframe
      if (e.source !== iframeRef.current?.contentWindow) return;
      if (!e.data?.type) return;
      if (e.data.type === 'artifact-error') {
        setErrors((prev) => [...prev.slice(-19), { msg: e.data.msg, line: e.data.line }]);
        setShowConsole(true);
      }
      if (e.data.type === 'console') {
        setConsoleEntries((prev) => [...prev.slice(-49), { level: e.data.level, args: e.data.args, timestamp: e.data.timestamp }]);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, []);

  // Build srcdoc
  const srcdoc = buildSrcdoc(currentVersion.type, currentVersion.code);

  const handleUndo = useCallback(() => {
    if (versionIndex > 0) setVersionIndex(versionIndex - 1);
  }, [versionIndex]);

  const handleRedo = useCallback(() => {
    if (versionIndex < versions.length - 1) setVersionIndex(versionIndex + 1);
  }, [versionIndex, versions.length]);

  const handleDownload = useCallback(() => {
    const ext = currentVersion.type === 'svg' ? 'svg' : currentVersion.type === 'markdown' ? 'md' : 'html';
    const content = currentVersion.type === 'html' || currentVersion.type === 'react' ? srcdoc : currentVersion.code;
    const blob = new Blob([content], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `artifact.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentVersion, srcdoc]);

  const handleOpenNewTab = useCallback(() => {
    const win = window.open('', '_blank');
    if (win) { win.document.write(srcdoc); win.document.close(); }
  }, [srcdoc]);

  const showPreview = viewMode === 'preview' || viewMode === 'split';
  const showCode = viewMode === 'code' || viewMode === 'split';

  return (
    <div className="flex flex-col h-full border-l border-chat-border bg-surface-0" style={{ minWidth: 320 }}>
      {/* Toolbar */}
      <div className="flex items-center gap-1 px-3 py-2 border-b border-chat-border flex-shrink-0 flex-wrap">
        {/* Type badge */}
        <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 mr-2">
          {currentVersion.type}
        </span>

        {/* View modes */}
        <div className="flex rounded-md border border-chat-border overflow-hidden">
          {(['preview', 'code', 'split'] as ViewMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setViewMode(m)}
              className={`px-2 py-1 text-[10px] capitalize transition-colors ${viewMode === m ? 'bg-chat-accent text-white' : 'text-chat-text-secondary hover:text-chat-text'}`}
            >
              {m === 'preview' ? <EyeIcon className="w-3.5 h-3.5 inline" /> : m === 'code' ? <CodeBracketIcon className="w-3.5 h-3.5 inline" /> : 'Split'}
            </button>
          ))}
        </div>

        {/* Device preview */}
        <div className="flex rounded-md border border-chat-border overflow-hidden ml-1">
          {([['desktop', ComputerDesktopIcon], ['tablet', DeviceTabletIcon], ['mobile', DevicePhoneMobileIcon]] as [DeviceSize, any][]).map(([d, Icon]) => (
            <button
              key={d}
              onClick={() => setDevice(d)}
              className={`p-1 transition-colors ${device === d ? 'bg-chat-accent text-white' : 'text-chat-text-secondary hover:text-chat-text'}`}
            >
              <Icon className="w-3.5 h-3.5" />
            </button>
          ))}
        </div>

        <div className="flex-1" />

        {/* Undo/Redo */}
        <button onClick={handleUndo} disabled={versionIndex === 0} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30" title="Undo">
          <ArrowUturnLeftIcon className="w-3.5 h-3.5" />
        </button>
        <button onClick={handleRedo} disabled={versionIndex >= versions.length - 1} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30" title="Redo">
          <ArrowUturnRightIcon className="w-3.5 h-3.5" />
        </button>

        {/* Actions */}
        <button onClick={handleDownload} className="p-1 text-chat-text-secondary hover:text-chat-text" title="Download">
          <ArrowDownTrayIcon className="w-3.5 h-3.5" />
        </button>
        <button onClick={handleOpenNewTab} className="p-1 text-chat-text-secondary hover:text-chat-text" title="Open in new tab">
          <ArrowsPointingOutIcon className="w-3.5 h-3.5" />
        </button>

        {/* Console toggle */}
        <button
          onClick={() => setShowConsole(!showConsole)}
          className={`p-1 relative ${showConsole ? 'text-chat-accent' : 'text-chat-text-secondary hover:text-chat-text'}`}
          title="Console"
        >
          <ExclamationTriangleIcon className="w-3.5 h-3.5" />
          {errors.length > 0 && (
            <span className="absolute -top-0.5 -right-0.5 w-3 h-3 rounded-full bg-red-500 text-[8px] text-white flex items-center justify-center">
              {errors.length}
            </span>
          )}
        </button>

        {/* Close */}
        <button onClick={onClose} className="p-1 text-chat-text-secondary hover:text-chat-text ml-1" title="Close">
          <XMarkIcon className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className={`flex-1 overflow-hidden flex ${viewMode === 'split' ? 'flex-row' : 'flex-col'}`}>
        {/* Preview iframe */}
        {showPreview && (
          <div className={`${viewMode === 'split' ? 'w-1/2 border-r border-chat-border' : 'flex-1'} overflow-auto flex justify-center`}>
            <div style={{ width: DEVICE_WIDTHS[device], maxWidth: '100%', height: '100%' }} className="transition-all duration-300">
              <iframe
                ref={iframeRef}
                srcDoc={srcdoc}
                sandbox="allow-scripts"
                className="w-full h-full border-none"
                title="Artifact preview"
              />
            </div>
          </div>
        )}

        {/* Code view */}
        {showCode && (
          <div className={`${viewMode === 'split' ? 'w-1/2' : 'flex-1'} overflow-auto`}>
            {codeHtml ? (
              <div
                dangerouslySetInnerHTML={{ __html: codeHtml }}
                className="shiki-block p-4 text-sm [&_pre]:!bg-transparent [&_pre]:!m-0 [&_pre]:!p-0 [&_code]:!bg-transparent"
              />
            ) : (
              <pre className="p-4 text-sm font-mono text-chat-text whitespace-pre-wrap">{currentVersion.code}</pre>
            )}
          </div>
        )}
      </div>

      {/* Console panel */}
      {showConsole && (
        <div className="border-t border-chat-border max-h-40 overflow-y-auto bg-surface-1">
          <div className="flex items-center justify-between px-3 py-1 border-b border-chat-border">
            <span className="text-[10px] font-semibold text-chat-text-secondary">Console</span>
            <button onClick={() => { setErrors([]); setConsoleEntries([]); }} className="text-[10px] text-chat-text-secondary hover:text-chat-text">Clear</button>
          </div>
          <div className="px-3 py-1 space-y-0.5">
            {errors.map((e, i) => (
              <div key={`e${i}`} className="text-[10px] text-red-400 font-mono">
                {e.line ? `[line ${e.line}] ` : ''}{e.msg}
              </div>
            ))}
            {consoleEntries.map((c, i) => (
              <div key={`c${i}`} className={`text-[10px] font-mono ${c.level === 'error' ? 'text-red-400' : c.level === 'warn' ? 'text-yellow-400' : 'text-chat-text-secondary'}`}>
                [{c.level}] {c.args.join(' ')}
              </div>
            ))}
            {errors.length === 0 && consoleEntries.length === 0 && (
              <div className="text-[10px] text-chat-text-secondary/50 py-2">No messages</div>
            )}
          </div>
        </div>
      )}

      {/* Version info */}
      <div className="flex items-center justify-between px-3 py-1 border-t border-chat-border text-[10px] text-chat-text-secondary/50 flex-shrink-0">
        <span>Version {versionIndex + 1} of {versions.length}</span>
        <span>{new Date(currentVersion.timestamp).toLocaleTimeString()}</span>
      </div>
    </div>
  );
}
