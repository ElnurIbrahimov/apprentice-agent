import React, { Suspense, useEffect, useRef, useState } from 'react';
import { useStore } from './store';
import Header from './components/Header';
import Rail from './components/Rail';
import ErrorBoundary from './components/ErrorBoundary';
import ChatPanel from './panels/ChatPanel';
import CommandPalette from './components/CommandPalette';
import ext from './ext';
import type { PanelId } from './types';

// Lazy-load every non-chat panel so the initial sidebar boot stays lean.
const SearchPanel = React.lazy(() => import('./panels/SearchPanel'));
const TranslatePanel = React.lazy(() => import('./panels/TranslatePanel'));
const WritePanel = React.lazy(() => import('./panels/WritePanel'));
const GrammarPanel = React.lazy(() => import('./panels/GrammarPanel'));
const WisebasePanel = React.lazy(() => import('./panels/WisebasePanel'));
const AskPanel = React.lazy(() => import('./panels/AskPanel'));
const SummaryPanel = React.lazy(() => import('./panels/SummaryPanel'));
const ToolsPanel = React.lazy(() => import('./panels/ToolsPanel'));
const PdfPanel = React.lazy(() => import('./panels/PdfPanel'));
const VoicePanel = React.lazy(() => import('./panels/VoicePanel'));
const RecordPanel = React.lazy(() => import('./panels/RecordPanel'));
const OcrPanel = React.lazy(() => import('./panels/OcrPanel'));
const YoutubePanel = React.lazy(() => import('./panels/YoutubePanel'));
const ResearchPanel = React.lazy(() => import('./panels/ResearchPanel'));
const MathPanel = React.lazy(() => import('./panels/MathPanel'));
const CodePanel = React.lazy(() => import('./panels/CodePanel'));
const ArtifactsPanel = React.lazy(() => import('./panels/ArtifactsPanel'));
const WebCreatorPanel = React.lazy(() => import('./panels/WebCreatorPanel'));
const ImagePanel = React.lazy(() => import('./panels/ImagePanel'));
const ComparePanel = React.lazy(() => import('./panels/ComparePanel'));
const CapturePanel = React.lazy(() => import('./panels/CapturePanel'));
const AgentPanel = React.lazy(() => import('./panels/AgentPanel'));
const SlidesPanel = React.lazy(() => import('./panels/SlidesPanel'));
const ModelsPanel = React.lazy(() => import('./panels/ModelsPanel'));
const SettingsPanel = React.lazy(() => import('./panels/SettingsPanel'));

function LazyFallback() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--mu)', fontSize: '12px' }}>
      Loading...
    </div>
  );
}

const PANEL_ENTRIES: { id: PanelId; Component: React.ComponentType }[] = [
  { id: 'chat', Component: ChatPanel },
  { id: 'search', Component: SearchPanel },
  { id: 'translate', Component: TranslatePanel },
  { id: 'write', Component: WritePanel },
  { id: 'grammar', Component: GrammarPanel },
  { id: 'wisebase', Component: WisebasePanel },
  { id: 'ask', Component: AskPanel },
  { id: 'summary', Component: SummaryPanel },
  { id: 'tools', Component: ToolsPanel },
  { id: 'pdf', Component: PdfPanel },
  { id: 'voice', Component: VoicePanel },
  { id: 'record', Component: RecordPanel },
  { id: 'ocr', Component: OcrPanel },
  { id: 'youtube', Component: YoutubePanel },
  { id: 'research', Component: ResearchPanel },
  { id: 'math', Component: MathPanel },
  { id: 'code', Component: CodePanel },
  { id: 'artifacts', Component: ArtifactsPanel },
  { id: 'webcreator', Component: WebCreatorPanel },
  { id: 'image', Component: ImagePanel },
  { id: 'compare', Component: ComparePanel },
  { id: 'capture', Component: CapturePanel },
  { id: 'agent', Component: AgentPanel },
  { id: 'slides', Component: SlidesPanel },
  { id: 'models', Component: ModelsPanel },
  { id: 'settings', Component: SettingsPanel },
];

export default function App() {
  const { activePanel, setPanel, setPendingCtx } = useStore();
  const [visiblePanel, setVisiblePanel] = useState<PanelId>(activePanel);
  const [transitioning, setTransitioning] = useState(false);
  const [slideDirection, setSlideDirection] = useState<'left' | 'right'>('right');
  const prevPanelRef = useRef<PanelId>(activePanel);
  const scrollPositions = useRef<Map<string, number>>(new Map());

  // Panel index for directional transitions
  const panelIndex = (id: PanelId) => PANEL_ENTRIES.findIndex(e => e.id === id);

  // Cross-fade transition when activePanel changes
  useEffect(() => {
    if (activePanel === prevPanelRef.current) return;

    // Determine slide direction: deeper panels slide in from right
    const oldIdx = panelIndex(prevPanelRef.current);
    const newIdx = panelIndex(activePanel);
    setSlideDirection(newIdx > oldIdx ? 'right' : 'left');

    // Save scroll position of outgoing panel
    const outgoing = prevPanelRef.current;
    const outEl = document.getElementById(`panel-${outgoing}`);
    if (outEl) {
      const scrollable = outEl.querySelector('.panel-scroll-root') || outEl;
      scrollPositions.current.set(outgoing, scrollable.scrollTop);
    }

    setTransitioning(true);

    // Short delay for out-fade, then swap
    const timer = setTimeout(() => {
      setVisiblePanel(activePanel);
      prevPanelRef.current = activePanel;
      // Restore scroll position of incoming panel
      requestAnimationFrame(() => {
        const inEl = document.getElementById(`panel-${activePanel}`);
        if (inEl) {
          const scrollable = inEl.querySelector('.panel-scroll-root') || inEl;
          const saved = scrollPositions.current.get(activePanel);
          if (saved != null) scrollable.scrollTop = saved;
        }
        setTransitioning(false);
      });
    }, 200); // matches slide-out duration

    return () => clearTimeout(timer);
  }, [activePanel]);

  // Listen for background messages
  useEffect(() => {
    const handler = (msg: any) => {
      if (msg.type === 'PREFILL_TEXT') {
        setPendingCtx({
          text: msg.text,
          action: msg.action || 'ask',
          url: msg.url || '',
          title: msg.title || '',
        });
        setPanel('ask');
      }
      if (msg.type === 'YT_TAB_DETECTED') {
        (window as any).__ytAutoUrl = msg.url;
        (window as any).__ytAutoTitle = msg.title;
        window.dispatchEvent(new CustomEvent('yt-detected', { detail: msg }));
      }
      if (msg.type === 'YT_SUBTITLES') {
        window.dispatchEvent(new CustomEvent('yt-subtitles', { detail: msg }));
      }
      if (msg.type === 'YT_METADATA') {
        window.dispatchEvent(new CustomEvent('yt-metadata', { detail: msg }));
      }
      if (msg.type === 'PDF_TAB_DETECTED') {
        window.dispatchEvent(new CustomEvent('pdf-detected', { detail: msg }));
      }
      if (msg.type === 'OCR_RESULT') {
        window.dispatchEvent(new CustomEvent('ocr-result', { detail: msg }));
      }
      if (msg.type === 'SWITCH_PANEL' && msg.panel) {
        setPanel(msg.panel);
      }
      if (msg.type === 'COMPONENT_CAPTURED') {
        setPanel('capture');
        // Forward to CapturePanel via custom event
        window.dispatchEvent(new CustomEvent('component-captured', { detail: msg }));
      }
      if (msg.type === 'CAPTURE_MODE_EXITED') {
        window.dispatchEvent(new CustomEvent('capture-mode-exited'));
      }
      if (msg.type === 'IMAGE_EDIT_LOAD' && msg.dataUrl) {
        setPanel('image');
        // Dispatch to ImagePanel's EditTab via custom event
        window.dispatchEvent(new CustomEvent('image-edit-load', { detail: { dataUrl: msg.dataUrl } }));
      }
    };

    ext?.runtime?.onMessage?.addListener(handler);

    // Check for pending image data URL (from content script hover toolbar)
    ext?.storage?.local?.get(['pendingImageDataUrl'], (data: any) => {
      if (data?.pendingImageDataUrl) {
        setPanel('image');
        window.dispatchEvent(new CustomEvent('image-edit-load', { detail: { dataUrl: data.pendingImageDataUrl } }));
        ext?.storage?.local?.remove(['pendingImageDataUrl']);
      }
    });

    return () => ext?.runtime?.onMessage?.removeListener(handler);
  }, [setPanel, setPendingCtx]);

  return (
    <>
    <div className="grain" aria-hidden="true" />
    <div className="flex h-full relative z-10">
      {/* Main content area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        <Header />
        <div className={`flex-1 relative panel-container ${transitioning ? 'panel-transitioning' : ''}`}>
          {/* ChatPanel always mounted — keeps WebSocket alive */}
          <div
            id="panel-chat"
            className={`panel-wrapper ${
              transitioning
                ? visiblePanel === 'chat'
                  ? slideDirection === 'right' ? 'panel-slide-out-left' : 'panel-slide-out-right'
                  : activePanel === 'chat'
                    ? slideDirection === 'right' ? 'panel-slide-in-right' : 'panel-slide-in-left'
                    : 'panel-hidden'
                : activePanel === 'chat' ? 'panel-visible' : 'panel-hidden'
            }`}
          >
            <ErrorBoundary panelName="chat">
              <ChatPanel />
            </ErrorBoundary>
          </div>
          {/* Other panels: only mount active + previous (for exit animation) */}
          {PANEL_ENTRIES.filter(({ id }) => id !== 'chat').map(({ id, Component }) => {
            const isVisible = id === visiblePanel;
            const isActive = id === activePanel;
            // Only mount if this panel is active or transitioning out
            if (!isActive && !isVisible) return null;
            let cls = 'panel-hidden';
            if (transitioning) {
              if (isVisible) cls = slideDirection === 'right' ? 'panel-slide-out-left' : 'panel-slide-out-right';
              else if (isActive) cls = slideDirection === 'right' ? 'panel-slide-in-right' : 'panel-slide-in-left';
            } else {
              if (isActive) cls = 'panel-visible';
            }
            return (
              <div
                key={id}
                id={`panel-${id}`}
                className={`panel-wrapper ${cls}`}
              >
                <ErrorBoundary panelName={id}>
                  <Suspense fallback={<LazyFallback />}>
                    <Component />
                  </Suspense>
                </ErrorBoundary>
              </div>
            );
          })}
        </div>
      </div>
      {/* Right rail */}
      <Rail />
    </div>
    <CommandPalette />
    </>
  );
}
