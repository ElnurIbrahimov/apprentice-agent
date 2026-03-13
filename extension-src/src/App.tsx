import React, { useEffect } from 'react';
import { useStore } from './store';
import Header from './components/Header';
import Rail from './components/Rail';
import ChatPanel from './panels/ChatPanel';
import SearchPanel from './panels/SearchPanel';
import TranslatePanel from './panels/TranslatePanel';
import WritePanel from './panels/WritePanel';
import GrammarPanel from './panels/GrammarPanel';
import WisebasePanel from './panels/WisebasePanel';
import AskPanel from './panels/AskPanel';
import SummaryPanel from './panels/SummaryPanel';
import ToolsPanel from './panels/ToolsPanel';
import PdfPanel from './panels/PdfPanel';
import VoicePanel from './panels/VoicePanel';
import OcrPanel from './panels/OcrPanel';
import YoutubePanel from './panels/YoutubePanel';
import ResearchPanel from './panels/ResearchPanel';
import MathPanel from './panels/MathPanel';
import ArtifactsPanel from './panels/ArtifactsPanel';
import ImagePanel from './panels/ImagePanel';
import ComparePanel from './panels/ComparePanel';
import AgentPanel from './panels/AgentPanel';
import ModelsPanel from './panels/ModelsPanel';
import ext from './ext';

export default function App() {
  const { activePanel, setPanel, setPendingCtx } = useStore();

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
        // Dispatch to YoutubePanel
        window.dispatchEvent(new CustomEvent('yt-detected', { detail: msg }));
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
    };

    ext?.runtime?.onMessage?.addListener(handler);
    return () => ext?.runtime?.onMessage?.removeListener(handler);
  }, [setPanel, setPendingCtx]);

  const panels: Record<string, React.ReactNode> = {
    chat: <ChatPanel />,
    search: <SearchPanel />,
    translate: <TranslatePanel />,
    write: <WritePanel />,
    grammar: <GrammarPanel />,
    wisebase: <WisebasePanel />,
    ask: <AskPanel />,
    summary: <SummaryPanel />,
    tools: <ToolsPanel />,
    pdf: <PdfPanel />,
    voice: <VoicePanel />,
    ocr: <OcrPanel />,
    youtube: <YoutubePanel />,
    research: <ResearchPanel />,
    math: <MathPanel />,
    artifacts: <ArtifactsPanel />,
    image: <ImagePanel />,
    compare: <ComparePanel />,
    agent: <AgentPanel />,
    models: <ModelsPanel />,
  };

  return (
    <>
    <div className="grain" aria-hidden="true" />
    <div className="flex h-full relative z-10">
      {/* Main content area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        <Header />
        <div className="flex-1 relative overflow-hidden">
          {Object.entries(panels).map(([id, panel]) => (
            <div
              key={id}
              className={`absolute inset-0 flex flex-col ${
                activePanel === id ? 'opacity-100 animate-[panelIn_.18s_ease_forwards]' : 'opacity-0 pointer-events-none'
              }`}
              style={{ display: activePanel === id ? 'flex' : 'none' }}
            >
              {panel}
            </div>
          ))}
        </div>
      </div>
      {/* Right rail */}
      <Rail />
    </div>
    </>
  );
}
