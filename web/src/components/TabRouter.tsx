import { useCallback } from 'react';
import { lazy, Suspense } from 'react';
import type { TabId } from '../types';
import type { ToolId } from './ToolLauncher';
import { ToolLauncher, ToolSubNav } from './ToolLauncher';
import { haptic } from '../utils/haptics';

// Eager imports
import { ConsciousnessPanel } from './ConsciousnessPanel';
import { InsightsFeed } from './InsightsFeed';
import { WorldModelPanel } from './WorldModelPanel';
import { MorningBriefingCard } from './MorningBriefingCard';
import { DreamInsightsPanel } from './DreamInsightsPanel';
import { EvolutionTracker } from './EvolutionTracker';

// Lazy imports
const ThoughtStream = lazy(() => import('./ThoughtStream').then(m => ({ default: m.ThoughtStream })));
const AuraPanel = lazy(() => import('./AuraPanel').then(m => ({ default: m.AuraPanel })));
const NeuroDreamPanel = lazy(() => import('./NeuroDreamPanel').then(m => ({ default: m.NeuroDreamPanel })));
const ToolsPanel = lazy(() => import('./ToolsPanel').then(m => ({ default: m.ToolsPanel })));
const AMEMPanel = lazy(() => import('./AMEMPanel').then(m => ({ default: m.AMEMPanel })));
const ReasoningTreePanel = lazy(() => import('./ReasoningTreePanel').then(m => ({ default: m.ReasoningTreePanel })));
const ActivityTimeline = lazy(() => import('./ActivityTimeline').then(m => ({ default: m.ActivityTimeline })));
const MemoryTimeline = lazy(() => import('./MemoryTimeline').then(m => ({ default: m.MemoryTimeline })));
const KnowledgeGraphExplorer = lazy(() => import('./KnowledgeGraphExplorer').then(m => ({ default: m.KnowledgeGraphExplorer })));
const TaskQueuePanel = lazy(() => import('./TaskQueuePanel').then(m => ({ default: m.TaskQueuePanel })));
const SettingsPage = lazy(() => import('./SettingsPage').then(m => ({ default: m.SettingsPage })));
const CodeInterpreter = lazy(() => import('./CodeInterpreter').then(m => ({ default: m.CodeInterpreter })));
const WebCreator = lazy(() => import('./WebCreator').then(m => ({ default: m.WebCreator })));
const ImageGenPanel = lazy(() => import('./ImageGenPanel').then(m => ({ default: m.ImageGenPanel })));
const SearchPanel = lazy(() => import('./SearchPanel').then(m => ({ default: m.SearchPanel })));
const ResearchPanel = lazy(() => import('./ResearchPanel').then(m => ({ default: m.ResearchPanel })));
const WritePanel = lazy(() => import('./WritePanel').then(m => ({ default: m.WritePanel })));
const TranslatePanel = lazy(() => import('./TranslatePanel').then(m => ({ default: m.TranslatePanel })));
const SummaryPanel = lazy(() => import('./SummaryPanel').then(m => ({ default: m.SummaryPanel })));
const GrammarPanel = lazy(() => import('./GrammarPanel').then(m => ({ default: m.GrammarPanel })));
const MathPanel = lazy(() => import('./MathPanel').then(m => ({ default: m.MathPanel })));
const AskPanel = lazy(() => import('./AskPanel').then(m => ({ default: m.AskPanel })));
const AgentPanel = lazy(() => import('./AgentPanel').then(m => ({ default: m.AgentPanel })));
const ComparePanel = lazy(() => import('./ComparePanel').then(m => ({ default: m.ComparePanel })));
const ModelsPanel = lazy(() => import('./ModelsPanel').then(m => ({ default: m.ModelsPanel })));
const PdfPanel = lazy(() => import('./PdfPanel').then(m => ({ default: m.PdfPanel })));
const VoicePanel = lazy(() => import('./VoicePanel').then(m => ({ default: m.VoicePanel })));
const OcrPanel = lazy(() => import('./OcrPanel').then(m => ({ default: m.OcrPanel })));
const YoutubePanel = lazy(() => import('./YoutubePanel').then(m => ({ default: m.YoutubePanel })));
const CapturePanel = lazy(() => import('./CapturePanel').then(m => ({ default: m.CapturePanel })));
const WisebasePanel = lazy(() => import('./WisebasePanel').then(m => ({ default: m.WisebasePanel })));
const SlidesPanel = lazy(() => import('./SlidesPanel').then(m => ({ default: m.SlidesPanel })));
const RecordPanel = lazy(() => import('./RecordPanel').then(m => ({ default: m.RecordPanel })));
const AppCreator = lazy(() => import('./AppCreator').then(m => ({ default: m.AppCreator })));
const GameCreator = lazy(() => import('./GameCreator').then(m => ({ default: m.GameCreator })));
const DashboardCreator = lazy(() => import('./DashboardCreator').then(m => ({ default: m.DashboardCreator })));

export type CreateSubTab = 'code' | 'webcreator' | 'app' | 'game' | 'dashboard' | 'slides' | 'image';
export type InsightsSubTab = 'monitor' | 'feed' | 'world' | 'briefing' | 'dreams' | 'evolution' | 'activity' | 'memory' | 'knowledge' | 'queue' | 'advanced';
export type ToolsSubTab = 'launcher' | 'system' | ToolId;

function TabSkeleton() {
  return (
    <div className="flex-1 flex items-center justify-center opacity-30">
      <div className="w-5 h-5 rounded-full border-2 border-chat-accent/40 border-t-chat-accent animate-spin" />
    </div>
  );
}

function SubTabBar({ tabs, active, onChange }: { tabs: { id: string; label: string }[]; active: string; onChange: (id: string) => void }) {
  return (
    <div className="flex items-center gap-1.5 px-3 lg:px-4 py-2 border-b border-chat-border overflow-x-auto scrollbar-hide snap-x snap-mandatory">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => { haptic(8); onChange(tab.id); }}
          className={`flex-shrink-0 snap-start px-3.5 py-2 rounded-xl text-xs sm:text-sm font-medium transition-all duration-200 ${
            active === tab.id
              ? 'bg-chat-accent text-white shadow-sm'
              : 'text-chat-text-secondary hover:text-chat-text active:scale-95 hover:bg-chat-assistant'
          }`}
          style={{ minHeight: 36 }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

interface TabRouterProps {
  activeTab: TabId;
  createSubTab: CreateSubTab;
  setCreateSubTab: (v: CreateSubTab) => void;
  insightsSubTab: InsightsSubTab;
  setInsightsSubTab: (v: InsightsSubTab) => void;
  toolsSubTab: ToolsSubTab;
  setToolsSubTab: (v: ToolsSubTab) => void;
}

export default function TabRouter({ activeTab, createSubTab, setCreateSubTab, insightsSubTab, setInsightsSubTab, toolsSubTab, setToolsSubTab }: TabRouterProps) {
  const renderTabContent = useCallback(() => {
    switch (activeTab) {
      case 'create':
        return (
          <div className="h-full flex flex-col">
            <SubTabBar
              tabs={[
                { id: 'code', label: 'Code' },
                { id: 'webcreator', label: 'Website' },
                { id: 'app', label: 'App' },
                { id: 'game', label: 'Game' },
                { id: 'dashboard', label: 'Dashboard' },
                { id: 'slides', label: 'Slides' },
                { id: 'image', label: 'Image' },
              ]}
              active={createSubTab}
              onChange={(id) => setCreateSubTab(id as CreateSubTab)}
            />
            <div className="flex-1 overflow-hidden">
              {createSubTab === 'code' && <CodeInterpreter />}
              {createSubTab === 'webcreator' && <WebCreator />}
              {createSubTab === 'app' && <AppCreator />}
              {createSubTab === 'game' && <GameCreator />}
              {createSubTab === 'dashboard' && <DashboardCreator />}
              {createSubTab === 'slides' && <SlidesPanel />}
              {createSubTab === 'image' && <ImageGenPanel />}
            </div>
          </div>
        );

      case 'tools': {
        const showLauncher = toolsSubTab === 'launcher';
        return (
          <div className="h-full flex flex-col">
            {!showLauncher && (
              <ToolSubNav
                activeId={toolsSubTab === 'system' ? 'tools' : toolsSubTab as ToolId}
                onSelect={(id) => setToolsSubTab(id === 'tools' ? 'system' : id as ToolsSubTab)}
                onBack={() => setToolsSubTab('launcher')}
              />
            )}
            <div className="flex-1 overflow-hidden tab-panel-scroll">
              {showLauncher && (
                <ToolLauncher
                  onSelect={(id) =>
                    setToolsSubTab(id === 'tools' ? 'system' : id as ToolsSubTab)
                  }
                />
              )}
              {toolsSubTab === 'system' && (
                <div className="h-full overflow-y-auto p-4 tab-panel-scroll">
                  <h2 className="text-xl font-semibold text-chat-text mb-4">Tools &amp; Systems</h2>
                  <ToolsPanel />
                </div>
              )}
              {toolsSubTab === 'ask'       && <AskPanel />}
              {toolsSubTab === 'search'    && <SearchPanel />}
              {toolsSubTab === 'research'  && <ResearchPanel />}
              {toolsSubTab === 'agent'     && <AgentPanel />}
              {toolsSubTab === 'compare'   && <ComparePanel />}
              {toolsSubTab === 'write'     && <WritePanel />}
              {toolsSubTab === 'translate' && <TranslatePanel />}
              {toolsSubTab === 'summary'   && <SummaryPanel />}
              {toolsSubTab === 'grammar'   && <GrammarPanel />}
              {toolsSubTab === 'math'      && <MathPanel />}
              {toolsSubTab === 'pdf'       && <PdfPanel />}
              {toolsSubTab === 'ocr'       && <OcrPanel />}
              {toolsSubTab === 'capture'   && <CapturePanel />}
              {toolsSubTab === 'youtube'   && <YoutubePanel />}
              {toolsSubTab === 'voice'     && <VoicePanel />}
              {toolsSubTab === 'record'    && <RecordPanel />}
              {toolsSubTab === 'slides'    && <SlidesPanel />}
              {toolsSubTab === 'wisebase'  && <WisebasePanel />}
              {toolsSubTab === 'models'    && <ModelsPanel />}
            </div>
          </div>
        );
      }

      case 'insights':
        return (
          <div className="h-full flex flex-col">
            <SubTabBar
              tabs={[
                { id: 'monitor', label: 'Mind' },
                { id: 'feed', label: 'Insights' },
                { id: 'briefing', label: 'Briefing' },
                { id: 'dreams', label: 'Dreams' },
                { id: 'evolution', label: 'Evolution' },
                { id: 'world', label: 'World' },
                { id: 'activity', label: 'Activity' },
                { id: 'memory', label: 'Memory' },
                { id: 'knowledge', label: 'Graph' },
                { id: 'queue', label: 'Queue' },
                { id: 'advanced', label: 'Advanced' },
              ]}
              active={insightsSubTab}
              onChange={(id) => setInsightsSubTab(id as InsightsSubTab)}
            />
            <div className="flex-1 overflow-hidden">
              {insightsSubTab === 'monitor' && <ConsciousnessPanel />}
              {insightsSubTab === 'feed' && <InsightsFeed />}
              {insightsSubTab === 'briefing' && (
                <div className="h-full overflow-y-auto p-3 sm:p-4 space-y-4 tab-panel-scroll">
                  <MorningBriefingCard />
                </div>
              )}
              {insightsSubTab === 'dreams' && <DreamInsightsPanel />}
              {insightsSubTab === 'evolution' && <EvolutionTracker />}
              {insightsSubTab === 'world' && <WorldModelPanel />}
              {insightsSubTab === 'activity' && <ActivityTimeline />}
              {insightsSubTab === 'memory' && <MemoryTimeline />}
              {insightsSubTab === 'knowledge' && <KnowledgeGraphExplorer />}
              {insightsSubTab === 'queue' && <TaskQueuePanel />}
              {insightsSubTab === 'advanced' && (
                <div className="h-full overflow-y-auto p-4 space-y-4 tab-panel-scroll">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <ThoughtStream />
                    <AuraPanel />
                    <ReasoningTreePanel />
                    <NeuroDreamPanel />
                    <AMEMPanel />
                  </div>
                </div>
              )}
            </div>
          </div>
        );

      case 'settings':
        return (
          <div className="h-full overflow-hidden">
            <SettingsPage />
          </div>
        );

      default:
        return null;
    }
  }, [activeTab, createSubTab, insightsSubTab, toolsSubTab, setCreateSubTab, setInsightsSubTab, setToolsSubTab]);

  return (
    <Suspense fallback={<TabSkeleton />}>
      {renderTabContent()}
    </Suspense>
  );
}
