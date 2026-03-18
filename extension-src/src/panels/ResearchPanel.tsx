import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP } from '../api';
import { md } from '../markdown';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Source {
  index: number;
  url: string;
  title?: string;
  domain?: string;
  snippet?: string;
  relevance?: number;
}

type StepId =
  | 'understanding'
  | 'planning'
  | 'searching'
  | 'synthesizing'
  | 'finalizing';

interface PipelineStep {
  id: StepId;
  label: string;
  sublabel?: string;
  status: 'pending' | 'active' | 'done';
}

interface SubQuestion {
  text: string;
  done: boolean;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const DEEP_STEPS: PipelineStep[] = [
  { id: 'understanding', label: 'Understanding query', sublabel: 'Analyzing the question', status: 'pending' },
  { id: 'planning', label: 'Planning research', sublabel: 'Generating sub-questions', status: 'pending' },
  { id: 'searching', label: 'Searching sources', sublabel: 'Analyzing sources', status: 'pending' },
  { id: 'synthesizing', label: 'Synthesizing findings', sublabel: 'Generating the report', status: 'pending' },
  { id: 'finalizing', label: 'Finalizing report', sublabel: 'Adding citations & formatting', status: 'pending' },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function formatElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  if (m === 0) return `${s}s`;
  return `${m}m ${s.toString().padStart(2, '0')}s`;
}

function downloadFile(content: string, filename: string) {
  const blob = new Blob([content], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ------------------------------------------------------------------ */
/*  Step indicator component                                           */
/* ------------------------------------------------------------------ */

function StepIndicator({ step, isLast }: { step: PipelineStep; isLast: boolean }) {
  const dotSize = 18;

  return (
    <div style={{ display: 'flex', gap: 10, minHeight: isLast ? dotSize : 42 }}>
      {/* Vertical track */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: dotSize, flexShrink: 0 }}>
        {/* Dot */}
        <div
          style={{
            width: dotSize,
            height: dotSize,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            background:
              step.status === 'done'
                ? 'var(--gr)'
                : step.status === 'active'
                ? 'var(--p)'
                : 'var(--s3)',
            border:
              step.status === 'done'
                ? '2px solid var(--gr)'
                : step.status === 'active'
                ? '2px solid var(--p)'
                : '2px solid var(--b2)',
            boxShadow:
              step.status === 'active'
                ? '0 0 12px rgba(124, 58, 237, 0.4), 0 0 4px rgba(124, 58, 237, 0.2)'
                : step.status === 'done'
                ? '0 0 8px rgba(16, 185, 129, 0.3)'
                : 'none',
          }}
        >
          {step.status === 'done' ? (
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <path d="M2 5.5L4 7.5L8 3" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : step.status === 'active' ? (
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'white',
                animation: 'dotPulse 1.2s ease-in-out infinite',
              }}
            />
          ) : (
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'var(--di)',
              }}
            />
          )}
        </div>
        {/* Connector line */}
        {!isLast && (
          <div
            style={{
              flex: 1,
              width: 2,
              minHeight: 16,
              background:
                step.status === 'done'
                  ? 'var(--gr)'
                  : 'var(--b1)',
              transition: 'background 0.4s ease',
              borderRadius: 1,
            }}
          />
        )}
      </div>

      {/* Label */}
      <div style={{ paddingTop: 1, flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontSize: '12px',
            fontWeight: step.status === 'active' ? 600 : 500,
            color:
              step.status === 'done'
                ? 'var(--gr)'
                : step.status === 'active'
                ? 'var(--tx)'
                : 'var(--mu)',
            transition: 'color 0.3s ease',
            lineHeight: 1.2,
          }}
        >
          {step.label}
          {step.status === 'active' && (
            <span style={{ marginLeft: 6, fontSize: '10px', color: 'var(--mu)', fontWeight: 400 }}>
              {step.sublabel}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function ResearchPanel() {
  const { getModel } = useStore();

  // Mode: quick = old behavior, deep = new pipeline
  const [mode, setMode] = useState<'quick' | 'deep'>('quick');

  // Shared
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [resultHtml, setResultHtml] = useState('');
  const [resultRaw, setResultRaw] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Deep-specific
  const [steps, setSteps] = useState<PipelineStep[]>([]);
  const [subQuestions, setSubQuestions] = useState<SubQuestion[]>([]);
  const [sourceCount, setSourceCount] = useState(0);
  const [sourcesTarget, setSourcesTarget] = useState(50);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const [copyLabel, setCopyLabel] = useState('Copy Report');

  // Elapsed timer
  useEffect(() => {
    if (!startTime || !loading) return;
    const iv = setInterval(() => setElapsed(Date.now() - startTime), 1000);
    return () => clearInterval(iv);
  }, [startTime, loading]);

  // Auto-scroll during deep research
  useEffect(() => {
    if (loading && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [steps, resultHtml, sourceCount, loading]);

  /* ---- Step advancement helper ---- */
  const advanceStep = useCallback((stepId: StepId) => {
    setSteps(prev => {
      const idx = prev.findIndex(s => s.id === stepId);
      if (idx === -1) return prev;
      return prev.map((s, i) => {
        if (i < idx) return { ...s, status: 'done' as const };
        if (i === idx) return { ...s, status: 'active' as const };
        return s;
      });
    });
  }, []);

  const completeStep = useCallback((stepId: StepId) => {
    setSteps(prev =>
      prev.map(s => (s.id === stepId ? { ...s, status: 'done' as const } : s))
    );
  }, []);

  /* ---- Cancel ---- */
  const cancelResearch = useCallback(() => {
    abortRef.current?.abort();
    setLoading(false);
    setStatus(mode === 'deep' ? 'Research cancelled' : '');
  }, [mode]);

  /* ---- Quick research (original behavior) ---- */
  const doQuickResearch = async () => {
    const query = inputRef.current?.value.trim();
    if (!query) return;
    setLoading(true);
    setStatus('Searching the web...');
    setResultHtml('');
    setResultRaw('');
    setSources([]);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const resp = await fetch(`${HTTP}/api/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, depth: 'quick', model: getModel('research') }),
        signal: ctrl.signal,
      });

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setStatus('Error: ' + ((d as any).detail || resp.statusText));
        return;
      }

      const reader = resp.body!.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop()!;
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const ev = JSON.parse(line);
            if (ev.status && ev.status !== 'done') setStatus(ev.message || ev.status);
            if (ev.status === 'done') {
              setStatus(`Done - ${ev.sources?.length || 0} sources`);
              const report = ev.report || '';
              setResultRaw(report);
              setResultHtml(md(report));
              setSources(ev.sources || []);
            }
          } catch {}
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus('Error: ' + (err.message || 'Request failed'));
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  /* ---- Deep research ---- */
  const doDeepResearch = async () => {
    const query = inputRef.current?.value.trim();
    if (!query) return;

    // Reset state
    setLoading(true);
    setStatus('');
    setResultHtml('');
    setResultRaw('');
    setSources([]);
    setSubQuestions([]);
    setSourceCount(0);
    setSourcesTarget(50);
    setSteps(DEEP_STEPS.map(s => ({ ...s, status: 'pending' as const })));
    setCopyLabel('Copy Report');
    setSourcesExpanded(false);

    const now = Date.now();
    setStartTime(now);
    setElapsed(0);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    let rawReport = '';
    let collectedSources: Source[] = [];

    try {
      // Try deep endpoint first
      let resp: Response;
      let isDeepEndpoint = true;

      try {
        resp = await fetch(`${HTTP}/api/research/deep`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, depth: 'deep', model: getModel('research') }),
          signal: ctrl.signal,
        });
      } catch {
        // Endpoint doesn't exist, fall back
        isDeepEndpoint = false;
        resp = await fetch(`${HTTP}/api/research`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, depth: 'deep', model: getModel('research') }),
          signal: ctrl.signal,
        });
      }

      // If deep endpoint returned 404/405, fall back
      if (isDeepEndpoint && (resp.status === 404 || resp.status === 405)) {
        isDeepEndpoint = false;
        resp = await fetch(`${HTTP}/api/research`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, depth: 'deep', model: getModel('research') }),
          signal: ctrl.signal,
        });
      }

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setStatus('Error: ' + ((d as any).detail || resp.statusText));
        setSteps(prev => prev.map(s => ({ ...s, status: 'pending' as const })));
        return;
      }

      // Start step 1 immediately
      advanceStep('understanding');

      const reader = resp.body!.getReader();
      const dec = new TextDecoder();
      let buf = '';

      // If using fallback endpoint, simulate the pipeline progression
      if (!isDeepEndpoint) {
        setStatus('Using standard research endpoint (deep endpoint not available)');
        // Simulate step progression with timers
        setTimeout(() => { advanceStep('understanding'); completeStep('understanding'); advanceStep('planning'); }, 800);
        setTimeout(() => { completeStep('planning'); advanceStep('searching'); }, 2500);
        setTimeout(() => { completeStep('searching'); advanceStep('synthesizing'); }, 5000);
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop()!;

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const ev = JSON.parse(line);

            // Handle deep research message types
            if (ev.type === 'research_step') {
              const stepMap: Record<string, StepId> = {
                understanding: 'understanding',
                planning: 'planning',
                searching: 'searching',
                synthesizing: 'synthesizing',
                finalizing: 'finalizing',
              };
              const sid = stepMap[ev.step] || ev.step;
              if (sid) {
                // Complete previous steps and activate current
                advanceStep(sid as StepId);
                if (ev.progress === 1 || ev.progress === 'done') {
                  completeStep(sid as StepId);
                }
              }
              // Handle sub-questions from planning step
              if (ev.sub_questions) {
                setSubQuestions(ev.sub_questions.map((q: string) => ({ text: q, done: false })));
              }
              if (ev.sources_target) {
                setSourcesTarget(ev.sources_target);
              }
            } else if (ev.type === 'research_source') {
              const src: Source = {
                index: ev.index ?? (collectedSources.length + 1),
                url: ev.url || '',
                title: ev.title || '',
                domain: ev.domain || '',
                snippet: ev.snippet || '',
                relevance: ev.relevance,
              };
              collectedSources = [...collectedSources, src];
              setSources([...collectedSources]);
              setSourceCount(collectedSources.length);
              // Mark corresponding sub-question done if provided
              if (ev.sub_question_index !== undefined) {
                setSubQuestions(prev =>
                  prev.map((sq, i) =>
                    i === ev.sub_question_index ? { ...sq, done: true } : sq
                  )
                );
              }
            } else if (ev.type === 'research_chunk') {
              rawReport += ev.content || '';
              setResultRaw(rawReport);
              setResultHtml(md(rawReport));
            } else if (ev.type === 'research_done') {
              // Final payload
              const finalReport = ev.report || rawReport;
              rawReport = finalReport;
              setResultRaw(finalReport);
              setResultHtml(md(finalReport));
              if (ev.sources) {
                collectedSources = ev.sources;
                setSources(ev.sources);
                setSourceCount(ev.sources.length);
              }
              // Mark all steps done
              setSteps(prev => prev.map(s => ({ ...s, status: 'done' as const })));
            } else if (ev.status) {
              // Fallback endpoint messages
              if (ev.status === 'done') {
                const report = ev.report || rawReport;
                rawReport = report;
                setResultRaw(report);
                setResultHtml(md(report));
                if (ev.sources) {
                  collectedSources = ev.sources;
                  setSources(ev.sources);
                  setSourceCount(ev.sources.length);
                }
                // Mark all done
                setSteps(prev => prev.map(s => ({ ...s, status: 'done' as const })));
              } else {
                setStatus(ev.message || ev.status);
                // Infer step from status messages
                const msg = (ev.message || ev.status || '').toLowerCase();
                if (msg.includes('search') || msg.includes('crawl') || msg.includes('fetch')) {
                  advanceStep('searching');
                } else if (msg.includes('synth') || msg.includes('generat') || msg.includes('writ')) {
                  completeStep('searching');
                  advanceStep('synthesizing');
                } else if (msg.includes('final') || msg.includes('format') || msg.includes('cit')) {
                  completeStep('synthesizing');
                  advanceStep('finalizing');
                }
                // Update source count from status if available
                if (ev.sources_count) {
                  setSourceCount(ev.sources_count);
                }
              }
            }
          } catch {}
        }
      }

      // Ensure all steps are complete when stream ends
      setSteps(prev => prev.map(s => ({ ...s, status: 'done' as const })));
      if (!resultHtml && rawReport) {
        setResultHtml(md(rawReport));
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus('Error: ' + (err.message || 'Request failed'));
        setSteps(prev => prev.map(s => s.status === 'active' ? { ...s, status: 'pending' as const } : s));
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  /* ---- Dispatch ---- */
  const doResearch = () => {
    if (mode === 'deep') doDeepResearch();
    else doQuickResearch();
  };

  /* ---- Copy report ---- */
  const copyReport = () => {
    if (!resultRaw) return;
    navigator.clipboard.writeText(resultRaw).then(() => {
      setCopyLabel('Copied!');
      setTimeout(() => setCopyLabel('Copy Report'), 1500);
    });
  };

  /* ---- Export .md ---- */
  const exportMarkdown = () => {
    if (!resultRaw) return;
    const query = inputRef.current?.value.trim() || 'research';
    const safeName = query.slice(0, 40).replace(/[^a-zA-Z0-9 ]/g, '').replace(/\s+/g, '_');
    downloadFile(resultRaw, `${safeName}_report.md`);
  };

  /* ---- Auto-resize textarea ---- */
  const handleInput = () => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 96) + 'px';
  };

  /* ---- Check if we have results to show ---- */
  const hasResults = !!resultHtml;
  const isDeepMode = mode === 'deep';
  const showPipeline = isDeepMode && (loading || steps.some(s => s.status !== 'pending'));

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ============ Input area ============ */}
      <div className="flex flex-col gap-0 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        {/* Textarea */}
        <div style={{ padding: '10px 12px 0' }}>
          <textarea
            ref={inputRef}
            placeholder={isDeepMode ? 'Enter a research topic for deep analysis...' : 'Research topic...'}
            autoFocus
            rows={1}
            onInput={handleInput}
            onKeyDown={e => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                if (!loading) doResearch();
              }
            }}
            style={{
              width: '100%',
              background: 'transparent',
              border: 'none',
              color: 'var(--tx)',
              fontSize: '13px',
              padding: 0,
              outline: 'none',
              fontFamily: 'inherit',
              resize: 'none',
              lineHeight: 1.5,
              minHeight: 22,
              maxHeight: 96,
              overflowY: 'auto',
            }}
          />
        </div>

        {/* Controls row */}
        <div className="flex items-center gap-2 px-3 py-2">
          {/* Mode toggle */}
          <div
            style={{
              display: 'flex',
              background: 'var(--s2)',
              borderRadius: 'var(--r-pill)',
              border: '1px solid var(--b1)',
              padding: 2,
              gap: 1,
            }}
          >
            {(['quick', 'deep'] as const).map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                disabled={loading}
                style={{
                  padding: '4px 12px',
                  background: mode === m
                    ? m === 'deep' ? 'var(--p)' : 'var(--pg2)'
                    : 'transparent',
                  border: 'none',
                  borderRadius: 'var(--r-pill)',
                  color: mode === m
                    ? m === 'deep' ? '#fff' : 'var(--pl)'
                    : 'var(--mu)',
                  fontSize: '11px',
                  fontWeight: mode === m ? 600 : 500,
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontFamily: 'inherit',
                  transition: 'all 0.2s ease',
                  letterSpacing: m === 'deep' ? '0.03em' : undefined,
                }}
              >
                {m === 'quick' ? 'Quick' : 'Deep Research'}
              </button>
            ))}
          </div>

          <div className="flex-1" />
          <ModelPill featureKey="research" />

          {loading ? (
            <button
              onClick={cancelResearch}
              style={{
                background: 'var(--rd)',
                border: 'none',
                borderRadius: 'var(--r-md)',
                color: 'white',
                padding: '5px 12px',
                cursor: 'pointer',
                fontSize: '11px',
                fontFamily: 'inherit',
                fontWeight: 500,
                transition: 'all 0.15s ease',
              }}
            >
              Cancel
            </button>
          ) : (
            <button
              onClick={doResearch}
              disabled={loading}
              style={{
                background: isDeepMode ? 'var(--p)' : 'var(--p)',
                border: 'none',
                borderRadius: 'var(--r-md)',
                color: 'white',
                padding: '5px 14px',
                cursor: 'pointer',
                fontSize: '11px',
                fontFamily: 'inherit',
                fontWeight: 600,
                boxShadow: isDeepMode ? '0 2px 10px rgba(124, 58, 237, 0.35)' : undefined,
                transition: 'all 0.15s ease',
              }}
            >
              {isDeepMode ? 'Start Deep Research' : 'Research'}
            </button>
          )}
        </div>
      </div>

      {/* ============ Results area ============ */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
        {/* ---- Deep research pipeline ---- */}
        {showPipeline && (
          <div
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-lg)',
              padding: '14px 16px',
              animation: 'panelFadeIn 0.3s ease',
            }}
          >
            {/* Header: timer */}
            <div className="flex items-center justify-between" style={{ marginBottom: 12 }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--tx)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                {steps.every(s => s.status === 'done') ? 'Research Complete' : 'Deep Research Pipeline'}
              </div>
              {startTime && (
                <div
                  style={{
                    fontSize: '10.5px',
                    color: 'var(--mu)',
                    fontVariantNumeric: 'tabular-nums',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                  }}
                >
                  {loading && (
                    <span
                      style={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: 'var(--p)',
                        display: 'inline-block',
                        animation: 'dotPulse 1.2s ease-in-out infinite',
                      }}
                    />
                  )}
                  {formatElapsed(loading ? elapsed : elapsed)}
                </div>
              )}
            </div>

            {/* Steps */}
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {steps.map((step, i) => (
                <div key={step.id}>
                  <StepIndicator step={step} isLast={i === steps.length - 1} />

                  {/* Sub-questions checklist (shown under planning step) */}
                  {step.id === 'planning' && subQuestions.length > 0 && (step.status === 'active' || step.status === 'done') && (
                    <div style={{ marginLeft: 28, marginBottom: 6, marginTop: 2 }}>
                      {subQuestions.map((sq, qi) => (
                        <div
                          key={qi}
                          style={{
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: 6,
                            fontSize: '11px',
                            color: sq.done ? 'var(--gr)' : 'var(--mu)',
                            lineHeight: 1.4,
                            padding: '2px 0',
                            transition: 'color 0.3s ease',
                          }}
                        >
                          <span style={{ flexShrink: 0, width: 14, textAlign: 'center' }}>
                            {sq.done ? (
                              <svg width="11" height="11" viewBox="0 0 10 10" fill="none" style={{ verticalAlign: 'middle' }}>
                                <path d="M2 5.5L4 7.5L8 3" stroke="var(--gr)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                              </svg>
                            ) : (
                              <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', border: '1.5px solid var(--b2)', verticalAlign: 'middle' }} />
                            )}
                          </span>
                          {sq.text}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Source counter (shown under searching step) */}
                  {step.id === 'searching' && (step.status === 'active' || step.status === 'done') && sourceCount > 0 && (
                    <div
                      style={{
                        marginLeft: 28,
                        marginBottom: 6,
                        marginTop: 2,
                        fontSize: '11px',
                        color: 'var(--pl)',
                        fontVariantNumeric: 'tabular-nums',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                        <circle cx="6.5" cy="6.5" r="4.5" stroke="var(--pl)" strokeWidth="1.5" />
                        <path d="M10 10L14 14" stroke="var(--pl)" strokeWidth="1.5" strokeLinecap="round" />
                      </svg>
                      Analyzed {sourceCount}/{sourcesTarget}+ sources
                      {step.status === 'active' && (
                        <span style={{ animation: 'dotPulse 1.2s ease-in-out infinite', width: 4, height: 4, borderRadius: '50%', background: 'var(--pl)', display: 'inline-block' }} />
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ---- Status line (quick mode or errors) ---- */}
        {status && (
          <div style={{ color: status.startsWith('Error') ? 'var(--rd)' : 'var(--mu)', fontSize: '11.5px', display: 'flex', alignItems: 'center', gap: 6 }}>
            {loading && !status.startsWith('Error') && (
              <div className="aura-thinking" style={{ padding: 0 }}>
                <span style={{ width: 4, height: 4 }} />
                <span style={{ width: 5, height: 5 }} />
                <span style={{ width: 4, height: 4 }} />
              </div>
            )}
            {status}
          </div>
        )}

        {/* ---- Report action buttons ---- */}
        {hasResults && !loading && (
          <div className="flex items-center gap-2" style={{ animation: 'fadeIn 0.3s ease' }}>
            <button
              onClick={copyReport}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '5px 12px',
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                color: 'var(--mu)',
                fontSize: '11px',
                fontWeight: 500,
                cursor: 'pointer',
                fontFamily: 'inherit',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={e => { (e.target as HTMLElement).style.borderColor = 'var(--b2)'; (e.target as HTMLElement).style.color = 'var(--tx)'; }}
              onMouseLeave={e => { (e.target as HTMLElement).style.borderColor = 'var(--b1)'; (e.target as HTMLElement).style.color = 'var(--mu)'; }}
            >
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <rect x="5" y="1" width="9" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
                <path d="M3 5V13.5C3 14.3 3.7 15 4.5 15H11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
              {copyLabel}
            </button>
            <button
              onClick={exportMarkdown}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '5px 12px',
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                color: 'var(--mu)',
                fontSize: '11px',
                fontWeight: 500,
                cursor: 'pointer',
                fontFamily: 'inherit',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={e => { (e.target as HTMLElement).style.borderColor = 'var(--b2)'; (e.target as HTMLElement).style.color = 'var(--tx)'; }}
              onMouseLeave={e => { (e.target as HTMLElement).style.borderColor = 'var(--b1)'; (e.target as HTMLElement).style.color = 'var(--mu)'; }}
            >
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <path d="M2 11V14H14V11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M8 2V10M8 10L5 7M8 10L11 7" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Export .md
            </button>
            {sources.length > 0 && (
              <span style={{ fontSize: '10.5px', color: 'var(--mu)', marginLeft: 4 }}>
                {sources.length} source{sources.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
        )}

        {/* ---- Report content ---- */}
        {resultHtml && (
          <div
            className="md-body"
            style={{
              fontSize: '12.5px',
              lineHeight: 1.65,
              animation: loading ? undefined : 'fadeIn 0.3s ease',
            }}
            dangerouslySetInnerHTML={{ __html: resultHtml }}
          />
        )}

        {/* Streaming cursor during report generation */}
        {loading && isDeepMode && resultHtml && (
          <span className="streaming-cursor" />
        )}

        {/* ---- Sources list ---- */}
        {sources.length > 0 && !loading && (
          <div className="flex flex-col gap-2" style={{ animation: 'fadeIn 0.3s ease' }}>
            {/* Expandable header */}
            <button
              onClick={() => setSourcesExpanded(!sourcesExpanded)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '4px 0',
                fontFamily: 'inherit',
              }}
            >
              <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)' }}>
                Sources ({sources.length})
              </div>
              <svg
                width="10"
                height="10"
                viewBox="0 0 10 10"
                fill="none"
                style={{
                  transform: sourcesExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s ease',
                }}
              >
                <path d="M2 4L5 7L8 4" stroke="var(--mu)" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>

            {/* Source items */}
            {(sourcesExpanded ? sources : sources.slice(0, 5)).map((s: Source, idx: number) => (
              <div
                key={s.index ?? idx}
                style={{
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  padding: '8px 10px',
                  transition: 'border-color 0.15s ease',
                }}
              >
                <div className="flex items-start gap-2">
                  <span
                    style={{
                      fontSize: '9px',
                      fontWeight: 700,
                      color: 'var(--p)',
                      background: 'var(--pg)',
                      borderRadius: 4,
                      padding: '1px 5px',
                      flexShrink: 0,
                      marginTop: 1,
                    }}
                  >
                    {s.index ?? idx + 1}
                  </span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <a
                      href={s.url}
                      target="_blank"
                      rel="noopener"
                      style={{
                        color: 'var(--pl)',
                        fontSize: '11.5px',
                        fontWeight: 500,
                        textDecoration: 'none',
                        display: 'block',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {s.title || s.domain || s.url}
                    </a>
                    {s.url && (
                      <div
                        style={{
                          fontSize: '10px',
                          color: 'var(--di)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          marginTop: 1,
                        }}
                      >
                        {s.url}
                      </div>
                    )}
                    {s.snippet && (
                      <div style={{ fontSize: '11px', color: 'var(--mu)', marginTop: 3, lineHeight: 1.4 }}>
                        {s.snippet}
                      </div>
                    )}
                  </div>
                  {s.relevance !== undefined && (
                    <span
                      style={{
                        fontSize: '9px',
                        color: 'var(--gr)',
                        background: 'rgba(16, 185, 129, 0.1)',
                        borderRadius: 4,
                        padding: '1px 5px',
                        flexShrink: 0,
                        fontWeight: 600,
                      }}
                    >
                      {Math.round(s.relevance * 100)}%
                    </span>
                  )}
                </div>
              </div>
            ))}

            {!sourcesExpanded && sources.length > 5 && (
              <button
                onClick={() => setSourcesExpanded(true)}
                style={{
                  background: 'none',
                  border: '1px dashed var(--b1)',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--mu)',
                  fontSize: '11px',
                  padding: '6px',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  transition: 'all 0.15s ease',
                }}
              >
                Show {sources.length - 5} more sources
              </button>
            )}
          </div>
        )}

        {/* ---- Empty state ---- */}
        {!loading && !status && !resultHtml && !showPipeline && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 12, paddingTop: 40 }}>
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: '50%',
                background: 'var(--pg)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px solid var(--b1)',
              }}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <circle cx="11" cy="11" r="7" stroke="var(--pl)" strokeWidth="1.8" />
                <path d="M16.5 16.5L21 21" stroke="var(--pl)" strokeWidth="1.8" strokeLinecap="round" />
                <path d="M9 11H13M11 9V13" stroke="var(--pl)" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: 'var(--tx)', fontSize: '13px', fontWeight: 600, marginBottom: 4 }}>
                {isDeepMode ? 'Deep Research' : 'Quick Research'}
              </div>
              <div style={{ color: 'var(--mu)', fontSize: '11.5px', lineHeight: 1.5, maxWidth: 240 }}>
                {isDeepMode
                  ? 'Analyzes 50+ sources over several minutes to produce a comprehensive cited report.'
                  : 'Enter a topic for fast multi-source research with citations.'}
              </div>
            </div>
            <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ fontSize: '9px', padding: '2px 5px', background: 'var(--s2)', borderRadius: 4, border: '1px solid var(--b1)', fontWeight: 500 }}>
                Ctrl+Enter
              </span>
              to start
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
