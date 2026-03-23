import { useState, useEffect, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import {
  MagnifyingGlassIcon,
  GlobeAltIcon,
  LightBulbIcon,
  SparklesIcon,
  ChevronDownIcon,
  ListBulletIcon,
} from '@heroicons/react/24/outline';
import type { ResearchProgressStep, ResearchStage } from '../types';

const STAGE_CONFIG: Record<ResearchStage, { label: string; icon: typeof MagnifyingGlassIcon; color: string }> = {
  plan: { label: 'Planning Research', icon: ListBulletIcon, color: 'rgba(139,92,246,0.8)' },
  search: { label: 'Searching', icon: MagnifyingGlassIcon, color: 'rgba(59,130,246,0.8)' },
  source: { label: 'Reading Sources', icon: GlobeAltIcon, color: 'rgba(16,185,129,0.8)' },
  finding: { label: 'Analyzing', icon: LightBulbIcon, color: 'rgba(245,158,11,0.8)' },
  synthesis: { label: 'Synthesizing', icon: SparklesIcon, color: 'rgba(236,72,153,0.8)' },
};

function StepItem({ step }: { step: ResearchProgressStep }) {
  const config = STAGE_CONFIG[step.stage];

  if (step.stage === 'plan' && step.data.subtopics) {
    const topics = step.data.subtopics as string[];
    return (
      <div className="flex items-start gap-2 animate-fade-in">
        <div className="flex-shrink-0 mt-0.5 w-4 h-4 rounded" style={{ color: config.color }}>
          <config.icon className="w-4 h-4" />
        </div>
        <div className="text-xs text-white/60">
          <span className="text-white/80">Subtopics:</span>{' '}
          {topics.map((t, i) => (
            <span key={i}>
              <span className="text-white/70">{t}</span>
              {i < topics.length - 1 && <span className="text-white/30"> / </span>}
            </span>
          ))}
        </div>
      </div>
    );
  }

  if (step.stage === 'search') {
    const query = step.data.query as string;
    const stepNum = step.data.step as number;
    const total = step.data.total as number;
    return (
      <div className="flex items-center gap-2 animate-fade-in">
        <div className="flex-shrink-0 w-4 h-4" style={{ color: config.color }}>
          <config.icon className="w-4 h-4" />
        </div>
        <span className="text-xs text-white/50">
          [{stepNum}/{total}]
        </span>
        <span className="text-xs text-white/70 truncate">{query}</span>
      </div>
    );
  }

  if (step.stage === 'source') {
    const url = step.data.url as string;
    const title = step.data.title as string;
    return (
      <div className="flex items-center gap-2 animate-fade-in">
        <div className="flex-shrink-0 w-4 h-4" style={{ color: config.color }}>
          <config.icon className="w-4 h-4" />
        </div>
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-purple-400 hover:text-purple-300 truncate underline decoration-purple-400/30 hover:decoration-purple-300/50 transition-colors"
          title={url}
        >
          {title || url}
        </a>
      </div>
    );
  }

  if (step.stage === 'finding') {
    const text = step.data.text as string;
    return (
      <div className="flex items-start gap-2 animate-fade-in">
        <div className="flex-shrink-0 mt-0.5 w-4 h-4" style={{ color: config.color }}>
          <config.icon className="w-4 h-4" />
        </div>
        <span className="text-xs text-white/70 leading-relaxed line-clamp-2">{text}</span>
      </div>
    );
  }

  if (step.stage === 'synthesis') {
    return (
      <div className="flex items-center gap-2 animate-fade-in">
        <div className="flex-shrink-0 w-4 h-4" style={{ color: config.color }}>
          <config.icon className="w-4 h-4 animate-spin" style={{ animationDuration: '3s' }} />
        </div>
        <span className="text-xs text-white/60">Composing response...</span>
      </div>
    );
  }

  return null;
}

function PulsingDot({ color }: { color: string }) {
  return (
    <span className="relative flex h-2 w-2">
      <span
        className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-60"
        style={{ backgroundColor: color }}
      />
      <span
        className="relative inline-flex rounded-full h-2 w-2"
        style={{ backgroundColor: color }}
      />
    </span>
  );
}

export function ResearchProgress() {
  const researchProgress = useChatStore((s) => s.researchProgress);
  const [collapsed, setCollapsed] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const prevStepCount = useRef(0);

  // Auto-scroll the step list when new steps arrive
  useEffect(() => {
    if (!researchProgress) return;
    const count = researchProgress.steps.length;
    if (count > prevStepCount.current && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
    prevStepCount.current = count;
  }, [researchProgress?.steps.length]);

  // Fade out when research completes
  useEffect(() => {
    if (researchProgress && !researchProgress.active) {
      const timer = setTimeout(() => setFadeOut(true), 1500);
      return () => clearTimeout(timer);
    }
    setFadeOut(false);
  }, [researchProgress?.active]);

  if (!researchProgress || researchProgress.steps.length === 0) return null;

  const stageConfig = STAGE_CONFIG[researchProgress.stage];
  const Icon = stageConfig.icon;
  const isActive = researchProgress.active;

  // Count sources and findings
  const sourceCount = researchProgress.steps.filter((s) => s.stage === 'source').length;
  const findingCount = researchProgress.steps.filter((s) => s.stage === 'finding').length;

  return (
    <div
      className="px-4 md:px-8 py-3 transition-all duration-500"
      style={{ opacity: fadeOut ? 0 : 1 }}
    >
      <div
        className="max-w-3xl mx-auto overflow-hidden transition-all duration-300"
        style={{
          background: 'rgba(15, 15, 20, 0.6)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: 12,
          backdropFilter: 'blur(16px)',
          boxShadow: '0 4px 24px rgba(0,0,0,0.3)',
        }}
      >
        {/* Header */}
        <button
          onClick={() => setCollapsed((p) => !p)}
          className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/[0.02] transition-colors"
        >
          {/* Stage icon */}
          <div
            className="flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center"
            style={{
              background: `${stageConfig.color.replace('0.8', '0.15')}`,
              border: `1px solid ${stageConfig.color.replace('0.8', '0.25')}`,
            }}
          >
            <Icon className="w-3.5 h-3.5" style={{ color: stageConfig.color }} />
          </div>

          {/* Label + stats */}
          <div className="flex-1 flex items-center gap-2 min-w-0">
            <span className="text-sm font-medium text-white/90">
              {stageConfig.label}
            </span>
            {isActive && <PulsingDot color={stageConfig.color} />}

            {/* Compact stats */}
            <div className="flex items-center gap-2 text-xs text-white/40 ml-auto mr-2">
              {sourceCount > 0 && (
                <span>{sourceCount} source{sourceCount !== 1 ? 's' : ''}</span>
              )}
              {findingCount > 0 && (
                <span>{findingCount} finding{findingCount !== 1 ? 's' : ''}</span>
              )}
            </div>
          </div>

          {/* Collapse toggle */}
          <ChevronDownIcon
            className="w-4 h-4 text-white/40 transition-transform duration-200 flex-shrink-0"
            style={{ transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)' }}
          />
        </button>

        {/* Steps list */}
        {!collapsed && (
          <div
            ref={listRef}
            className="px-4 pb-3 space-y-2 overflow-y-auto transition-all duration-200"
            style={{ maxHeight: 200 }}
          >
            {/* Thin separator */}
            <div className="border-t border-white/[0.05] mb-1" />

            {researchProgress.steps.map((step, i) => (
              <StepItem key={i} step={step} />
            ))}

            {/* Active progress bar */}
            {isActive && (
              <div
                className="h-0.5 rounded-full mt-2 overflow-hidden"
                style={{ background: 'rgba(255,255,255,0.05)' }}
              >
                <div
                  className="h-full rounded-full"
                  style={{
                    background: `linear-gradient(90deg, ${stageConfig.color}, transparent)`,
                    animation: 'researchProgress 2s ease-in-out infinite',
                  }}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Inline keyframes */}
      <style>{`
        @keyframes researchProgress {
          0% { width: 0%; margin-left: 0%; }
          50% { width: 60%; margin-left: 20%; }
          100% { width: 0%; margin-left: 100%; }
        }
        .animate-fade-in {
          animation: fadeIn 0.3s ease forwards;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
}
