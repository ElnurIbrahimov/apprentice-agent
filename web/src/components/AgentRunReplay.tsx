/**
 * Inline replay of a past agent run — loads the conversation's messages,
 * parses out the original plan, and renders plan + transcript + a
 * chronological tool timeline. Stays inside the Agent tab so the user
 * never has to switch to Chat just to look at history.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ArrowLeftIcon, ArrowDownTrayIcon, ClockIcon,
  ArrowUturnLeftIcon, BookmarkSquareIcon,
} from '@heroicons/react/24/outline';

import { AgentPlanCard } from './AgentPlanCard';
import { ToolTrace } from './ToolTrace';
import {
  fetchConversationMessages, parseRun, type ParsedRun,
} from '../utils/agentRunParse';
import { downloadRunMarkdown } from '../utils/agentRunExport';
import { saveRecipe } from '../utils/agentRecipes';
import { toast } from './Toast';
import type { AgentPlan } from '../utils/agentPlan';

interface Props {
  conversationId: string;
  conversationTitle: string;
  onBack: () => void;
  onCloneAsNewRun: (goal: string, plan: AgentPlan | null) => void;
}

export function AgentRunReplay({ conversationId, conversationTitle, onBack, onCloneAsNewRun }: Props) {
  const [run, setRun] = useState<ParsedRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchConversationMessages(conversationId)
      .then((msgs) => {
        if (cancelled) return;
        setRun(parseRun(conversationId, conversationTitle, msgs));
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message || 'Failed to load');
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [conversationId, conversationTitle]);

  const tools = useMemo(() => run?.toolTraces ?? [], [run]);

  const assistantMessages = useMemo(
    () => (run?.messages ?? []).filter((m) => m.role === 'assistant'),
    [run],
  );

  const handleExport = useCallback(() => {
    if (!run) return;
    downloadRunMarkdown({
      goal: run.goal,
      plan: run.plan,
      messages: run.messages,
      elapsedSeconds: Math.floor(run.elapsedMs / 1000),
      tokensEstimated: Math.ceil(run.totalChars / 4),
      plannerModel: null,
      executorModel: null,
      completedAt: run.messages[run.messages.length - 1]?.timestamp ?? Date.now(),
    });
  }, [run]);

  const handleSaveRecipe = useCallback(() => {
    if (!run?.plan) return;
    const name = window.prompt('Recipe name?', run.goal.slice(0, 60));
    if (!name) return;
    saveRecipe(name, run.plan);
    toast.success('Recipe saved', 'Find it on the Recipes sub-tab.');
  }, [run]);

  const handleClone = useCallback(() => {
    if (!run) return;
    onCloneAsNewRun(run.goal, run.plan);
  }, [run, onCloneAsNewRun]);

  return (
    <div
      className="rounded-2xl p-4 space-y-4"
      style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
    >
      <header className="flex items-center gap-2 flex-wrap">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
          style={{ border: '1px solid var(--border-default)' }}
        >
          <ArrowLeftIcon className="w-3.5 h-3.5" />
          Back
        </button>
        <div className="min-w-0 flex-1">
          <div className="text-xs uppercase tracking-wide text-chat-text-secondary">Replay</div>
          <div className="text-sm font-medium text-chat-text truncate">
            {run?.goal ?? conversationTitle.replace(/^Agent:\s*/, '')}
          </div>
        </div>
        {run && (
          <>
            <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded text-[10px] text-chat-text-secondary tabular-nums" style={{ background: 'var(--surface-2)' }}>
              <ClockIcon className="w-3 h-3" />
              {formatMs(run.elapsedMs)} · {tools.length} tool call{tools.length !== 1 ? 's' : ''}
            </div>
            <button
              type="button"
              onClick={handleExport}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
              style={{ border: '1px solid var(--border-default)' }}
              title="Export as Markdown"
            >
              <ArrowDownTrayIcon className="w-3.5 h-3.5" />
              Export
            </button>
            {run.plan && (
              <button
                type="button"
                onClick={handleSaveRecipe}
                className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
                style={{ border: '1px solid var(--border-default)' }}
                title="Save this plan as a reusable recipe"
              >
                <BookmarkSquareIcon className="w-3.5 h-3.5" />
                Save as recipe
              </button>
            )}
            <button
              type="button"
              onClick={handleClone}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-white transition-colors"
              style={{
                background: 'var(--chat-accent)',
                boxShadow: '0 2px 10px rgba(124,58,237,0.3)',
              }}
              title="Start a new run with this goal/plan"
            >
              <ArrowUturnLeftIcon className="w-3.5 h-3.5" />
              Clone as new run
            </button>
          </>
        )}
      </header>

      {loading && (
        <div className="text-sm text-chat-text-secondary py-8 text-center">
          <div className="inline-flex items-center gap-2">
            <span className="w-3 h-3 rounded-full border-2 border-chat-accent/40 border-t-chat-accent animate-spin" />
            Loading run…
          </div>
        </div>
      )}

      {error && (
        <div
          className="rounded-lg p-3 text-sm"
          style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}
        >
          {error}
        </div>
      )}

      {!loading && !error && run && (
        <>
          {/* Tool timeline — chronological strip of tool calls */}
          {tools.length > 0 && (
            <ToolTimeline traces={tools} startMs={run.messages[0]?.timestamp ?? 0} />
          )}

          {/* Plan card (disabled / read-only) */}
          {run.plan ? (
            <AgentPlanCard
              plan={run.plan}
              onChange={() => {}}
              onApprove={() => {}}
              onDiscard={onBack}
              disabled
            />
          ) : (
            <div className="text-xs text-chat-text-secondary italic">
              No structured plan was recorded for this run (it may have been a chat-style agent call).
            </div>
          )}

          {/* Transcript */}
          {assistantMessages.length > 0 && (
            <div
              className="rounded-xl p-4 space-y-3"
              style={{ background: 'var(--surface-0)', border: '1px solid var(--border-subtle)' }}
            >
              <div className="text-xs uppercase tracking-wide text-chat-text-secondary">Transcript</div>
              {assistantMessages.map((m) => (
                <div key={m.id} className="space-y-2">
                  {m.toolTrace && m.toolTrace.length > 0 && (
                    <ToolTrace traces={m.toolTrace} />
                  )}
                  <div
                    className="prose prose-invert max-w-none text-sm whitespace-pre-wrap"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {m.content}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Chronological horizontal strip of tool calls. Click a pip to scroll its event into view. */
function ToolTimeline({
  traces, startMs,
}: { traces: { tool: string; event: string; timestamp: number; elapsed_ms?: number }[]; startMs: number }) {
  const total = Math.max(1, (traces[traces.length - 1]?.timestamp ?? 0) - startMs);
  return (
    <div className="space-y-1">
      <div className="text-[10px] uppercase tracking-wide text-chat-text-secondary">Tool timeline</div>
      <div
        className="relative h-7 rounded"
        style={{ background: 'var(--surface-0)', border: '1px solid var(--border-subtle)' }}
      >
        {traces.map((t, i) => {
          const left = Math.max(0, Math.min(99, ((t.timestamp - startMs) / total) * 100));
          const color = t.event === 'error' ? '#f87171' : t.event === 'start' ? '#fbbf24' : '#4ade80';
          return (
            <div
              key={i}
              title={`${t.tool} — ${t.event}${typeof t.elapsed_ms === 'number' ? ` (${(t.elapsed_ms / 1000).toFixed(1)}s)` : ''}`}
              className="absolute top-1/2 -translate-y-1/2 rounded-full"
              style={{
                left: `${left}%`,
                width: 8,
                height: 8,
                background: color,
                boxShadow: `0 0 6px ${color}88`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

function formatMs(total: number): string {
  const s = Math.floor(total / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${(s % 60).toString().padStart(2, '0')}s`;
}
