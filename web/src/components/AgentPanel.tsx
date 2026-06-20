/**
 * AgentPanel — real multi-step agent surface.
 *
 * Flow:
 *   1. Goal input            → user describes a task
 *   2. Plan phase            → POST /api/generate/raw (fast, no tools) with the
 *                              planner prompt; parse JSON plan; render editable cards.
 *   3. Approve               → user edits / approves the plan.
 *   4. Execute phase         → send via WebSocket with action_mode='agent'; the
 *                              backend agent_service executes step-by-step calling
 *                              real tools. Live tool_trace events render here.
 *   5. Approval checkpoints  → when the server pauses for a sensitive action,
 *                              render inline Allow/Deny prompt via useHandApproval.
 *   6. Completion            → final summary with cost + elapsed + artifacts count.
 *
 * Each run becomes a "Agent: <goal>" conversation in the sidebar — free history.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  RocketLaunchIcon, StopIcon, CpuChipIcon, ChevronDownIcon,
  ArrowPathIcon, ClockIcon, SparklesIcon, XCircleIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline';

import { useChatStore } from '../store/chatStore';
import { useWebSocket } from '../hooks/useWebSocket';
import { useHandApproval } from '../hooks/useHandApproval';
import {
  type AgentPlan,
  type PlanStep,
  parsePlan,
  PLANNER_SYSTEM_PROMPT,
  buildExecutorMessage,
  updateStep,
} from '../utils/agentPlan';
import { estimateTokens } from '../utils/tokenEstimate';

import { AgentPlanCard } from './AgentPlanCard';
import { AgentApprovalInline } from './AgentApprovalInline';
import { ToolTrace } from './ToolTrace';
import { toast } from './Toast';
import HandsDashboard from './HandsDashboard';
import { AgentRunReplay } from './AgentRunReplay';
import {
  listRecipes, saveRecipe, deleteRecipe,
  type AgentRecipe,
} from '../utils/agentRecipes';
import { BookmarkSquareIcon, TrashIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { downloadRunMarkdown } from '../utils/agentRunExport';
import { apiFetch } from '../utils/apiFetch';

type Phase = 'idle' | 'planning' | 'reviewing' | 'executing' | 'done' | 'error';
type SubTab = 'run' | 'recipes' | 'history' | 'hands';

const PLANNER_MODEL_KEY = 'aura-agent-planner-model';
const EXECUTOR_MODEL_KEY = 'aura-agent-executor-model';

function readStoredModel(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}

function writeStoredModel(key: string, value: string | null) {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch { /* private mode */ }
}

export function AgentPanel() {
  const [goal, setGoal] = useState('');
  const [phase, setPhase] = useState<Phase>('idle');
  const [plan, setPlan] = useState<AgentPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [plannerModel, setPlannerModel] = useState<string | null>(() => readStoredModel(PLANNER_MODEL_KEY));
  const [executorModel, setExecutorModel] = useState<string | null>(() => readStoredModel(EXECUTOR_MODEL_KEY));
  const [openModelMenu, setOpenModelMenu] = useState<null | 'planner' | 'executor'>(null);
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const plannerAbort = useRef<AbortController | null>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const executorStartIndexRef = useRef<number>(0);

  const [recipes, setRecipes] = useState<AgentRecipe[]>(() => listRecipes());
  const reloadRecipes = useCallback(() => setRecipes(listRecipes()), []);
  const [subTab, setSubTab] = useState<SubTab>('run');
  const [historyFilter, setHistoryFilter] = useState<string>('');
  const [critique, setCritique] = useState<string>('');
  const [critiquing, setCritiquing] = useState(false);
  const critiqueAbort = useRef<AbortController | null>(null);
  const [replay, setReplay] = useState<{ id: string; title: string } | null>(null);

  // Persist planner/executor model choices.
  useEffect(() => writeStoredModel(PLANNER_MODEL_KEY, plannerModel), [plannerModel]);
  useEffect(() => writeStoredModel(EXECUTOR_MODEL_KEY, executorModel), [executorModel]);

  // Force Run sub-tab when the user is in the middle of anything.
  useEffect(() => {
    if (phase !== 'idle') setSubTab('run');
  }, [phase]);

  const { sendMessage, stopGeneration, connectionStatus } = useWebSocket();
  const {
    messages,
    availableModels,
    conversations,
    currentConversationId,
    setCurrentConversationId,
    isLoading,
    setAvailableModels,
  } = useChatStore();

  const approval = useHandApproval();

  // Fetch models once for the dropdown.
  useEffect(() => {
    if (availableModels.length > 0) return;
    apiFetch('/api/models')
      .then((r) => r.json())
      .then((data) => {
        const all = [
          ...(data.chatgpt_models || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models || []),
          ...(data.local_models || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, [availableModels.length, setAvailableModels]);

  // Close model menu on outside click.
  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setOpenModelMenu(null);
      }
    };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  // Elapsed-time ticker during active phases.
  useEffect(() => {
    if (phase !== 'planning' && phase !== 'executing') return;
    const start = runStartedAt ?? Date.now();
    if (!runStartedAt) setRunStartedAt(start);
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 500);
    return () => clearInterval(id);
  }, [phase, runStartedAt]);

  // When the executor finishes (isLoading false while we expected output), mark done.
  const prevIsLoading = useRef(isLoading);
  useEffect(() => {
    if (phase === 'executing' && prevIsLoading.current && !isLoading) {
      setPhase('done');
    }
    prevIsLoading.current = isLoading;
  }, [isLoading, phase]);

  // Infer per-step status from assistant output that mentions "Step <id>:" markers.
  useEffect(() => {
    if (phase !== 'executing' || !plan) return;
    const runMessages = messages.slice(executorStartIndexRef.current);
    const assistantText = runMessages
      .filter((m) => m.role === 'assistant')
      .map((m) => m.content)
      .join('\n');
    let current: AgentPlan = plan;
    let mutated = false;
    for (let i = 0; i < plan.steps.length; i++) {
      const s = plan.steps[i];
      const prev = s.status ?? 'pending';
      const mentioned = assistantText.includes(`Step ${s.id}`) || assistantText.includes(s.title);
      const nextStepMentioned = i + 1 < plan.steps.length
        ? assistantText.includes(`Step ${plan.steps[i + 1].id}`) || assistantText.includes(plan.steps[i + 1].title)
        : false;
      let status: PlanStep['status'] = prev;
      if (!mentioned) status = 'pending';
      else if (nextStepMentioned && !isLoading) status = 'done';
      else if (nextStepMentioned) status = 'done';
      else status = isLoading ? 'running' : 'done';
      if (status !== prev) {
        current = updateStep(current, s.id, { status });
        mutated = true;
      }
    }
    if (mutated) setPlan(current);
  }, [messages, phase, plan, isLoading]);

  const executorMessages = useMemo(
    () => messages.slice(executorStartIndexRef.current),
    [messages],
  );

  const lastAssistant = useMemo(
    () => [...executorMessages].reverse().find((m) => m.role === 'assistant'),
    [executorMessages],
  );

  const tokenEstimate = useMemo(() => estimateTokens(executorMessages), [executorMessages]);

  const canPlan = goal.trim().length > 0 && phase === 'idle' && connectionStatus === 'connected';

  /* ── Plan phase ── */
  const handlePlan = useCallback(async () => {
    if (!canPlan) return;
    setError(null);
    setPlan(null);
    setPhase('planning');
    setRunStartedAt(Date.now());
    setElapsed(0);

    // Create a new conversation so this run has its own thread in the sidebar.
    try {
      const convRes = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ title: `Agent: ${goal.slice(0, 60)}` }),
      });
      if (convRes.ok) {
        const data = await convRes.json();
        if (data?.id) setCurrentConversationId(data.id);
      }
    } catch {
      // Non-fatal: backend may create conversation implicitly on first message.
    }

    // Run the planner through /api/generate/raw (fast path, no tools).
    const controller = new AbortController();
    plannerAbort.current = controller;
    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          message: goal,
          system_prompt: PLANNER_SYSTEM_PROMPT,
          history: [],
          ...(plannerModel ? { model: plannerModel } : {}),
        }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`Planner HTTP ${res.status}`);

      let fullText = '';
      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const text = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
                if (text) fullText += text;
              } catch {
                fullText += data;
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullText += line;
            }
          }
        }
      } else {
        fullText = await res.text();
      }

      const parsed = parsePlan(fullText);
      if (!parsed) {
        setError('Planner returned invalid JSON. Try again or simplify the goal.');
        setPhase('error');
        return;
      }
      setPlan(parsed);
      setPhase('reviewing');
    } catch (e: unknown) {
      if ((e as { name?: string })?.name === 'AbortError') {
        setPhase('idle');
        return;
      }
      setError((e as Error)?.message || 'Planning failed');
      setPhase('error');
    } finally {
      plannerAbort.current = null;
    }
  }, [canPlan, goal, plannerModel, setCurrentConversationId]);

  /* ── Execute phase ── */
  const handleExecute = useCallback(() => {
    if (!plan) return;
    setPhase('executing');
    setRunStartedAt(Date.now());
    setElapsed(0);
    executorStartIndexRef.current = useChatStore.getState().messages.length;
    const msg = buildExecutorMessage(plan);
    sendMessage(msg, undefined, executorModel, 'agent');
  }, [plan, executorModel, sendMessage]);

  /* ── Stop / discard ── */
  const handleStop = useCallback(() => {
    if (phase === 'planning') {
      plannerAbort.current?.abort();
      setPhase('idle');
      return;
    }
    if (phase === 'executing') {
      stopGeneration();
      setPhase('done');
    }
  }, [phase, stopGeneration]);

  const handleReset = useCallback(() => {
    setPhase('idle');
    setPlan(null);
    setError(null);
    setGoal('');
    setRunStartedAt(null);
    setElapsed(0);
    executorStartIndexRef.current = 0;
  }, []);

  /* ── History: past agent conversations ── */
  const pastRuns = useMemo(
    () => conversations.filter((c) => (c.title || '').startsWith('Agent: ')).slice(0, 20),
    [conversations],
  );

  // Similar past runs — fuzzy match current goal against past run titles.
  const similarRuns = useMemo(() => {
    const q = goal.trim().toLowerCase();
    if (q.length < 4 || pastRuns.length === 0) return [];
    const qTokens = q.split(/\s+/).filter((t) => t.length > 2);
    if (qTokens.length === 0) return [];
    const scored = pastRuns.map((c) => {
      const title = (c.title || '').toLowerCase().replace(/^agent:\s*/, '');
      let score = 0;
      for (const t of qTokens) {
        if (title.includes(t)) score += t.length >= 5 ? 2 : 1;
      }
      return { conv: c, score };
    });
    return scored
      .filter((x) => x.score >= 2)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map((x) => x.conv);
  }, [goal, pastRuns]);

  const reopenRun = useCallback((id: string, title: string) => {
    setReplay({ id, title });
    setSubTab('history');
  }, []);

  const closeReplay = useCallback(() => setReplay(null), []);

  const cloneReplayAsNewRun = useCallback((goal: string, plan: AgentPlan | null) => {
    setReplay(null);
    setSubTab('run');
    setGoal(goal);
    if (plan) {
      setPlan({
        goal: plan.goal,
        steps: plan.steps.map((s) => ({ ...s, status: 'pending' as const })),
      });
      setPhase('reviewing');
      toast.info('Plan cloned', 'Review, edit, and approve to re-run.');
    } else {
      setPhase('idle');
    }
  }, []);

  /* ── Recipes ── */
  const loadRecipe = useCallback((recipe: AgentRecipe) => {
    setGoal(recipe.plan.goal);
    setPlan({
      goal: recipe.plan.goal,
      steps: recipe.plan.steps.map((s) => ({ ...s, status: 'pending' as const })),
    });
    setError(null);
    setPhase('reviewing');
    toast.info('Recipe loaded', 'Review, edit, and approve to execute.');
  }, []);

  const saveCurrentAsRecipe = useCallback(() => {
    if (!plan) return;
    const defaultName = plan.goal.length > 60 ? plan.goal.slice(0, 60) + '…' : plan.goal;
    const name = window.prompt('Recipe name?', defaultName);
    if (!name) return;
    saveRecipe(name, plan);
    reloadRecipes();
    toast.success('Recipe saved', 'Find it on the Agent start screen.');
  }, [plan, reloadRecipes]);

  const removeRecipe = useCallback((id: string) => {
    deleteRecipe(id);
    reloadRecipes();
  }, [reloadRecipes]);

  /* ── Self-critique: ask a fresh LLM call to review the completed run ── */
  const runCritique = useCallback(async () => {
    if (!plan || critiquing) return;
    setCritique('');
    setCritiquing(true);
    const controller = new AbortController();
    critiqueAbort.current = controller;

    const transcript = executorMessages
      .filter((m) => m.role === 'assistant')
      .map((m) => m.content)
      .join('\n\n');

    const systemPrompt = `You are a strict but fair reviewer of an AI agent's work.
Evaluate:
1. Did it achieve the goal?
2. Did each planned step actually happen, and was it done well?
3. Are there factual, logical, or reasoning errors?
4. What should be improved on a next pass?

Output a short markdown critique with these sections: **Verdict** (one sentence),
**Strengths** (1-3 bullets), **Issues** (1-3 bullets, specific), **Next step** (one concrete suggestion).
Be direct. No pleasantries.`;

    const userMessage = [
      `GOAL:\n${plan.goal}`,
      '',
      `PLAN:\n${plan.steps.map((s, i) => `${i + 1}. ${s.title} — ${s.rationale}`).join('\n')}`,
      '',
      `AGENT OUTPUT:\n${transcript || '(no output captured)'}`,
    ].join('\n');

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          message: userMessage,
          system_prompt: systemPrompt,
          history: [],
          ...(plannerModel ? { model: plannerModel } : {}),
        }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`Critique HTTP ${res.status}`);

      let full = '';
      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const text = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
                if (text) { full += text; setCritique(full); }
              } catch { full += data; setCritique(full); }
            } else if (line.trim() && !line.startsWith(':')) {
              full += line;
              setCritique(full);
            }
          }
        }
      } else {
        full = await res.text();
        setCritique(full);
      }
    } catch (e: unknown) {
      if ((e as { name?: string })?.name !== 'AbortError') {
        toast.error('Critique failed', (e as Error)?.message ?? 'Unknown error');
      }
    } finally {
      setCritiquing(false);
      critiqueAbort.current = null;
    }
  }, [plan, critiquing, executorMessages, plannerModel]);

  // Reset critique state when starting a new run.
  useEffect(() => {
    if (phase === 'idle' || phase === 'planning' || phase === 'reviewing') {
      setCritique('');
      setCritiquing(false);
      critiqueAbort.current?.abort();
    }
  }, [phase]);

  const exportRun = useCallback(() => {
    if (!plan) return;
    downloadRunMarkdown({
      goal,
      plan,
      messages: executorMessages,
      elapsedSeconds: elapsed,
      tokensEstimated: tokenEstimate,
      plannerModel,
      executorModel,
      completedAt: Date.now(),
    });
  }, [plan, goal, executorMessages, elapsed, tokenEstimate, plannerModel, executorModel]);

  // Keyboard shortcuts — only fire when focus is not in an input/textarea.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const active = document.activeElement as HTMLElement | null;
      const inInput = !!active
        && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable);
      if (inInput) return;
      const k = e.key.toLowerCase();
      if (k === 'p' && phase === 'idle' && goal.trim()) { e.preventDefault(); handlePlan(); }
      else if (k === 'e' && phase === 'reviewing') { e.preventDefault(); handleExecute(); }
      else if (k === 's' && (phase === 'planning' || phase === 'executing')) { e.preventDefault(); handleStop(); }
      else if (k === 'n' && (phase === 'done' || phase === 'error')) { e.preventDefault(); handleReset(); }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [phase, goal, handlePlan, handleExecute, handleStop, handleReset]);

  const isBusy = phase === 'planning' || phase === 'executing';

  return (
    <div className="h-full overflow-y-auto tab-panel-scroll p-4 sm:p-6">
      <div className="max-w-4xl mx-auto space-y-4">
        <header className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(124,58,237,0.18)', border: '1px solid rgba(124,58,237,0.35)' }}
            >
              <SparklesIcon className="w-4 h-4" style={{ color: '#c4b5fd' }} />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-chat-text tracking-tight">Agent</h1>
              <p className="text-xs text-chat-text-secondary">
                Plan → approve → execute. Real tools, visible trace.
              </p>
            </div>
          </div>

          <div className="flex-1" />

          {/* Model pickers — planner (drafts plan, fast) + executor (runs plan, capable) */}
          <div className="flex items-center gap-1.5" ref={modelMenuRef}>
            <ModelPickerPill
              label="Plan"
              value={plannerModel}
              options={availableModels}
              open={openModelMenu === 'planner'}
              onToggle={() => setOpenModelMenu((v) => (v === 'planner' ? null : 'planner'))}
              onPick={(m) => { setPlannerModel(m); setOpenModelMenu(null); }}
              disabled={isBusy}
            />
            <ModelPickerPill
              label="Run"
              value={executorModel}
              options={availableModels}
              open={openModelMenu === 'executor'}
              onToggle={() => setOpenModelMenu((v) => (v === 'executor' ? null : 'executor'))}
              onPick={(m) => { setExecutorModel(m); setOpenModelMenu(null); }}
              disabled={isBusy}
            />
          </div>

          {/* Live budget widget */}
          {isBusy && (
            <div
              className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs tabular-nums"
              style={{ background: 'var(--surface-2)', border: '1px solid var(--border-default)', color: 'var(--text-secondary)' }}
            >
              <ClockIcon className="w-3.5 h-3.5" />
              {formatSeconds(elapsed)}
              {phase === 'executing' && (
                <>
                  <span className="opacity-40">·</span>
                  <span>{tokenEstimate.toLocaleString()} tok</span>
                </>
              )}
            </div>
          )}
        </header>

        {/* Sub-tab bar: disabled when a run is active so nav is obvious */}
        <div
          role="tablist"
          className="flex items-center gap-1 p-1 rounded-xl"
          style={{ background: 'var(--surface-2)', border: '1px solid var(--border-default)' }}
        >
          <SubTabButton label="Run" active={subTab === 'run'} onClick={() => setSubTab('run')} disabled={isBusy} />
          <SubTabButton label={`Recipes${recipes.length ? ` · ${recipes.length}` : ''}`} active={subTab === 'recipes'} onClick={() => setSubTab('recipes')} disabled={isBusy} />
          <SubTabButton label={`History${pastRuns.length ? ` · ${pastRuns.length}` : ''}`} active={subTab === 'history'} onClick={() => setSubTab('history')} disabled={isBusy} />
          <SubTabButton label="Hands" active={subTab === 'hands'} onClick={() => setSubTab('hands')} disabled={isBusy} />
        </div>

        {/* ── Sub-tab: Recipes (full view) ── */}
        {subTab === 'recipes' && (
          <div
            className="rounded-2xl p-4"
            style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
          >
            <div className="flex items-center gap-2 mb-3">
              <BookmarkSquareIcon className="w-4 h-4 text-chat-text-secondary" />
              <span className="text-sm font-medium text-chat-text">Saved recipes</span>
              <span className="text-xs text-chat-text-secondary">{recipes.length} total</span>
            </div>
            {recipes.length === 0 ? (
              <div className="text-sm text-chat-text-secondary py-8 text-center">
                No recipes yet. Start a run, approve a plan, and save it as a recipe to reuse.
              </div>
            ) : (
              <ul className="space-y-2">
                {recipes.map((r) => (
                  <li
                    key={r.id}
                    className="group p-3 rounded-xl flex items-start gap-3 transition-colors hover:bg-white/[0.03]"
                    style={{ border: '1px solid var(--border-subtle)' }}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-chat-text font-medium truncate">{r.name}</div>
                      <div className="text-xs text-chat-text-secondary mt-0.5 truncate">{r.plan.goal}</div>
                      <div className="mt-2 flex flex-wrap gap-1">
                        {r.plan.steps.slice(0, 4).map((s, i) => (
                          <span
                            key={i}
                            className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                            style={{ background: 'var(--surface-2)', color: 'var(--text-secondary)' }}
                          >
                            {s.title}
                          </span>
                        ))}
                        {r.plan.steps.length > 4 && (
                          <span className="text-[10px] text-chat-text-secondary opacity-70 self-center">
                            +{r.plan.steps.length - 4} more
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-chat-text-secondary opacity-60 mt-2">
                        {r.plan.steps.length} steps · saved {new Date(r.createdAt).toLocaleDateString()}
                      </div>
                    </div>
                    <div className="flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        type="button"
                        onClick={() => loadRecipe(r)}
                        className="px-2.5 py-1 rounded text-xs text-white transition-all"
                        style={{ background: 'var(--chat-accent)' }}
                      >
                        Use
                      </button>
                      <button
                        type="button"
                        onClick={() => removeRecipe(r.id)}
                        className="px-2 py-1 rounded text-xs text-chat-text-secondary hover:text-red-300 transition-colors"
                        title="Delete"
                      >
                        <TrashIcon className="w-3.5 h-3.5 inline" />
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* ── Sub-tab: History (full view) or inline replay ── */}
        {subTab === 'history' && replay && (
          <AgentRunReplay
            conversationId={replay.id}
            conversationTitle={replay.title}
            onBack={closeReplay}
            onCloneAsNewRun={cloneReplayAsNewRun}
          />
        )}
        {subTab === 'history' && !replay && (
          <div
            className="rounded-2xl p-4"
            style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
          >
            <div className="flex items-center gap-2 mb-3">
              <ClockIcon className="w-4 h-4 text-chat-text-secondary" />
              <span className="text-sm font-medium text-chat-text">Past runs</span>
              <span className="text-xs text-chat-text-secondary">{pastRuns.length} total</span>
              <div className="flex-1" />
              <input
                value={historyFilter}
                onChange={(e) => setHistoryFilter(e.target.value)}
                placeholder="Filter by goal…"
                className="px-2 py-1 text-xs rounded bg-transparent outline-none text-chat-text placeholder-chat-text-secondary/60"
                style={{ border: '1px solid var(--border-default)', width: 180 }}
              />
            </div>
            {pastRuns.length === 0 ? (
              <div className="text-sm text-chat-text-secondary py-8 text-center">
                No past runs yet.
              </div>
            ) : (
              <ul className="space-y-1">
                {pastRuns
                  .filter((c) => !historyFilter || c.title.toLowerCase().includes(historyFilter.toLowerCase()))
                  .map((c) => (
                    <li key={c.id}>
                      <button
                        type="button"
                        onClick={() => reopenRun(c.id, c.title)}
                        className="w-full text-left p-2.5 rounded-lg hover:bg-white/[0.03] transition-colors flex items-center gap-3"
                        style={{ border: '1px solid var(--border-subtle)' }}
                      >
                        <span
                          className={`w-2 h-2 rounded-full flex-shrink-0 ${c.id === currentConversationId ? 'bg-purple-400' : 'bg-white/20'}`}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm text-chat-text truncate">{c.title.replace(/^Agent:\s*/, '')}</div>
                          <div className="text-[10px] text-chat-text-secondary mt-0.5">
                            {c.message_count} messages · {new Date((c.updated_at || c.created_at) * 1000).toLocaleString()}
                          </div>
                        </div>
                      </button>
                    </li>
                  ))}
              </ul>
            )}
          </div>
        )}

        {/* ── Sub-tab: Hands (long-running autonomous agents) ── */}
        {subTab === 'hands' && (
          <div
            className="rounded-2xl p-4"
            style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
          >
            <HandsDashboard />
          </div>
        )}

        {/* ── Sub-tab: Run (existing flow) ── */}
        {subTab === 'run' && phase === 'idle' && (
          <div className="space-y-3">
            <div
              className="rounded-2xl p-4"
              style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
            >
              <label className="block text-xs font-medium text-chat-text-secondary mb-2">Goal</label>
              <textarea
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="Describe what you want Aura to accomplish — e.g. 'research the top 3 Python web frameworks and compare their ergonomics, ecosystem, and performance'"
                rows={3}
                className="w-full bg-transparent outline-none resize-none text-sm text-chat-text"
                style={{ lineHeight: 1.5 }}
                disabled={isBusy}
              />
              {similarRuns.length > 0 && (
                <div className="mt-3 pt-3 border-t border-white/[0.06] space-y-1.5">
                  <div className="text-[10px] uppercase tracking-wide text-chat-text-secondary">
                    You ran something similar
                  </div>
                  {similarRuns.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() => reopenRun(c.id, c.title)}
                      className="w-full text-left px-2 py-1.5 rounded-md hover:bg-white/[0.04] text-xs transition-colors flex items-center gap-2"
                      style={{ border: '1px solid var(--border-subtle)' }}
                    >
                      <ClockIcon className="w-3 h-3 text-chat-text-secondary flex-shrink-0" />
                      <span className="flex-1 truncate text-chat-text">{c.title.replace(/^Agent:\s*/, '')}</span>
                      <span className="text-[10px] text-chat-text-secondary flex-shrink-0">replay</span>
                    </button>
                  ))}
                </div>
              )}

              <div className="flex items-center justify-between mt-3">
                <span className="text-[10px] text-chat-text-secondary opacity-60 flex items-center gap-2">
                  {connectionStatus === 'connected' ? 'Connected' : 'Waiting for connection…'}
                  <span className="opacity-60">· Shortcuts: <kbd className="font-mono">P</kbd>lan · <kbd className="font-mono">E</kbd>xecute · <kbd className="font-mono">S</kbd>top · <kbd className="font-mono">N</kbd>ew</span>
                </span>
                <button
                  type="button"
                  onClick={handlePlan}
                  disabled={!canPlan}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium text-white transition-all disabled:opacity-50"
                  style={{
                    background: canPlan ? 'var(--chat-accent)' : 'var(--surface-3)',
                    boxShadow: canPlan ? '0 4px 14px rgba(124,58,237,0.4)' : 'none',
                  }}
                >
                  <RocketLaunchIcon className="w-4 h-4" />
                  Plan task
                </button>
              </div>
            </div>

            {(recipes.length > 0 || pastRuns.length > 0) && (
              <div className="flex items-center gap-3 text-xs text-chat-text-secondary">
                <span>Jump to:</span>
                {recipes.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setSubTab('recipes')}
                    className="text-chat-accent hover:underline"
                  >
                    {recipes.length} saved recipe{recipes.length === 1 ? '' : 's'}
                  </button>
                )}
                {pastRuns.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setSubTab('history')}
                    className="text-chat-accent hover:underline"
                  >
                    {pastRuns.length} past run{pastRuns.length === 1 ? '' : 's'}
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Phase: planning (loading) ── */}
        {phase === 'planning' && (
          <div
            className="rounded-2xl p-6 text-center"
            style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
          >
            <div className="inline-flex items-center gap-2 text-sm text-chat-text-secondary">
              <span className="w-3 h-3 rounded-full border-2 border-chat-accent/40 border-t-chat-accent animate-spin" />
              Drafting a plan for your goal…
            </div>
            <div className="text-xs text-chat-text-secondary opacity-60 mt-2">{goal}</div>
            <button
              type="button"
              onClick={handleStop}
              className="mt-4 text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
            >
              Cancel
            </button>
          </div>
        )}

        {/* ── Phase: reviewing ── */}
        {phase === 'reviewing' && plan && (
          <div className="space-y-2">
            <div
              className="rounded-2xl p-4"
              style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
            >
              <AgentPlanCard
                plan={plan}
                onChange={setPlan}
                onApprove={handleExecute}
                onDiscard={handleReset}
              />
            </div>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={saveCurrentAsRecipe}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
                style={{ border: '1px solid var(--border-default)' }}
              >
                <BookmarkSquareIcon className="w-3.5 h-3.5" />
                Save as recipe
              </button>
            </div>
          </div>
        )}

        {/* ── Phase: executing / done ── */}
        {(phase === 'executing' || phase === 'done') && plan && (
          <div className="space-y-3">
            <div
              className="rounded-2xl p-4"
              style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
            >
              <div className="flex items-center justify-between mb-3">
                <div>
                  <div className="text-xs uppercase tracking-wide text-chat-text-secondary">
                    {phase === 'executing' ? 'Running' : 'Completed'}
                  </div>
                  <div className="text-sm text-chat-text mt-0.5 font-medium">{plan.goal}</div>
                </div>
                <div className="flex items-center gap-2">
                  {phase === 'executing' && (
                    <button
                      type="button"
                      onClick={handleStop}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs"
                      style={{ background: 'rgba(239,68,68,0.15)', color: '#fca5a5', border: '1px solid rgba(239,68,68,0.35)' }}
                    >
                      <StopIcon className="w-3.5 h-3.5" />
                      Stop
                    </button>
                  )}
                  {phase === 'done' && (
                    <>
                      <button
                        type="button"
                        onClick={exportRun}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
                        style={{ border: '1px solid var(--border-default)' }}
                        title="Export run as Markdown"
                      >
                        <ArrowDownTrayIcon className="w-3.5 h-3.5" />
                        Export
                      </button>
                      <button
                        type="button"
                        onClick={handleReset}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
                        style={{ border: '1px solid var(--border-default)' }}
                      >
                        <ArrowPathIcon className="w-3.5 h-3.5" />
                        New run
                      </button>
                    </>
                  )}
                </div>
              </div>

              <AgentPlanCard
                plan={plan}
                onChange={() => { /* locked during execution */ }}
                onApprove={() => {}}
                onDiscard={handleReset}
                disabled
              />
            </div>

            {/* Approval prompt */}
            {approval.pending && (
              <AgentApprovalInline
                pending={approval.pending}
                resolving={approval.resolving}
                onAllow={() => approval.resolve(true)}
                onDeny={() => approval.resolve(false)}
                onDismiss={approval.dismiss}
              />
            )}

            {/* Live tool trace + assistant output */}
            {lastAssistant && (
              <div
                className="rounded-2xl p-4"
                style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}
              >
                {lastAssistant.toolTrace && lastAssistant.toolTrace.length > 0 && (
                  <ToolTrace traces={lastAssistant.toolTrace} isStreaming={lastAssistant.isStreaming} />
                )}
                <div
                  className="prose prose-invert max-w-none text-sm whitespace-pre-wrap"
                  style={{ color: 'var(--text-primary)' }}
                >
                  {lastAssistant.content}
                </div>
              </div>
            )}

            {phase === 'done' && (
              <>
                <div
                  className="rounded-2xl p-4 text-sm"
                  style={{
                    background: 'rgba(34,197,94,0.08)',
                    border: '1px solid rgba(34,197,94,0.25)',
                    color: '#bbf7d0',
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-semibold uppercase tracking-wide">Run complete</span>
                    <span className="ml-auto text-xs opacity-80">
                      {formatSeconds(elapsed)} · {tokenEstimate.toLocaleString()} tok
                    </span>
                  </div>
                  <div className="text-xs opacity-80 mb-3">
                    Full transcript saved in the sidebar under "{`Agent: ${goal.slice(0, 60)}`}".
                  </div>
                  {!critique && !critiquing && (
                    <button
                      type="button"
                      onClick={runCritique}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors"
                      style={{
                        background: 'rgba(34,197,94,0.18)',
                        color: '#86efac',
                        border: '1px solid rgba(34,197,94,0.35)',
                      }}
                    >
                      <MagnifyingGlassIcon className="w-3.5 h-3.5" />
                      Review this work
                    </button>
                  )}
                </div>

                {(critiquing || critique) && (
                  <div
                    className="rounded-2xl p-4"
                    style={{
                      background: 'var(--surface-1)',
                      border: '1px solid rgba(99,102,241,0.35)',
                    }}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <MagnifyingGlassIcon className="w-4 h-4" style={{ color: '#a5b4fc' }} />
                      <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: '#c7d2fe' }}>
                        Self-critique
                      </span>
                      {critiquing && (
                        <span className="inline-flex items-center gap-1 text-[10px] text-chat-text-secondary">
                          <span className="w-2 h-2 rounded-full border-2 border-indigo-400/40 border-t-indigo-400 animate-spin" />
                          reviewing…
                        </span>
                      )}
                      <div className="flex-1" />
                      {!critiquing && critique && (
                        <button
                          type="button"
                          onClick={runCritique}
                          className="text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
                          title="Re-run critique"
                        >
                          <ArrowPathIcon className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                    <div
                      className="prose prose-invert max-w-none text-sm whitespace-pre-wrap"
                      style={{ color: 'var(--text-primary)' }}
                    >
                      {critique || '…'}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ── Phase: error ── */}
        {phase === 'error' && (
          <div
            className="rounded-2xl p-4 flex items-start gap-3"
            style={{
              background: 'rgba(239,68,68,0.08)',
              border: '1px solid rgba(239,68,68,0.3)',
            }}
          >
            <XCircleIcon className="w-5 h-5 flex-shrink-0" style={{ color: '#fca5a5' }} />
            <div className="flex-1">
              <div className="text-sm font-medium" style={{ color: '#fecaca' }}>Run failed</div>
              <div className="text-xs mt-1" style={{ color: '#fca5a5' }}>{error}</div>
              <button
                type="button"
                onClick={handleReset}
                className="mt-3 inline-flex items-center gap-1 text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
              >
                <ArrowPathIcon className="w-3.5 h-3.5" />
                Start over
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SubTabButton({ label, active, onClick, disabled }: {
  label: string; active: boolean; onClick: () => void; disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      disabled={disabled}
      className="flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
      style={{
        background: active ? 'var(--surface-0)' : 'transparent',
        color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
        border: active ? '1px solid var(--border-default)' : '1px solid transparent',
        boxShadow: active ? '0 2px 6px rgba(0,0,0,0.2)' : 'none',
      }}
    >
      {label}
    </button>
  );
}

function formatSeconds(total: number): string {
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}m ${s.toString().padStart(2, '0')}s`;
}

interface ModelPickerPillProps {
  label: string;
  value: string | null;
  options: string[];
  open: boolean;
  onToggle: () => void;
  onPick: (model: string | null) => void;
  disabled?: boolean;
}

function ModelPickerPill({ label, value, options, open, onToggle, onPick, disabled }: ModelPickerPillProps) {
  return (
    <div className="relative">
      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors disabled:opacity-50"
        style={{ background: 'var(--surface-2)', border: '1px solid var(--border-default)', color: 'var(--text-secondary)' }}
        title={`${label} model`}
      >
        <CpuChipIcon className="w-3.5 h-3.5" />
        <span className="uppercase tracking-wide opacity-70 text-[10px]">{label}</span>
        <span className="font-mono">{value?.split('/').pop() ?? 'Auto'}</span>
        <ChevronDownIcon className="w-3 h-3 opacity-60" />
      </button>
      {open && (
        <div
          className="absolute right-0 top-full mt-1 z-30 rounded-xl overflow-hidden max-h-80 overflow-y-auto"
          style={{
            minWidth: 240,
            background: 'var(--surface-1)',
            border: '1px solid var(--border-default)',
            boxShadow: '0 12px 36px rgba(0,0,0,0.35)',
          }}
        >
          <button
            className="w-full px-3 py-2 text-left text-xs hover:bg-white/5 text-chat-text"
            onClick={() => onPick(null)}
          >
            🤖 Auto (recommended)
          </button>
          <div className="border-t border-white/5" />
          {options.map((m) => (
            <button
              key={m}
              className="w-full px-3 py-1.5 text-left text-[11px] hover:bg-white/5 text-chat-text-secondary font-mono"
              onClick={() => onPick(m)}
            >
              {m}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
