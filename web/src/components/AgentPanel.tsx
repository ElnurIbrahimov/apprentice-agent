import { useState, useRef, useEffect, useCallback } from 'react';
import { StopIcon, ClipboardDocumentIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/outline';

/* ── Types ── */
type StepStatus = 'pending' | 'running' | 'done';

interface Step {
  number: number;
  title: string;
  content: string;
  status: StepStatus;
}

const MAX_STEPS_OPTIONS = [3, 5, 10] as const;

const SYSTEM_PROMPT = `You are an autonomous AI agent. Break the user's goal into clear numbered steps, then execute each step. Format your output as:

## Step 1: [step title]
[step content]

## Step 2: [step title]
[step content]

...

## Final Result
[compiled result]

Be thorough and complete each step before moving to the next.`;

/* ── Helpers ── */
function parseSteps(text: string): Step[] {
  const steps: Step[] = [];
  // Match ## Step N: title or ## Final Result
  const stepRegex = /^##\s+(?:Step\s+(\d+):\s*(.+)|Final Result)\s*$/gim;
  const matches: { index: number; number: number; title: string }[] = [];

  let m: RegExpExecArray | null;
  while ((m = stepRegex.exec(text)) !== null) {
    if (m[1] !== undefined) {
      matches.push({ index: m.index, number: parseInt(m[1], 10), title: m[2].trim() });
    } else {
      matches.push({ index: m.index, number: -1, title: 'Final Result' });
    }
  }

  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index;
    const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
    // Extract content after the heading line
    const block = text.slice(start, end);
    const newlinePos = block.indexOf('\n');
    const content = newlinePos >= 0 ? block.slice(newlinePos + 1).trim() : '';
    steps.push({
      number: matches[i].number,
      title: matches[i].title,
      content,
      status: 'done',
    });
  }
  return steps;
}

function buildFullResult(steps: Step[]): string {
  return steps
    .map((s) =>
      s.number === -1
        ? `## Final Result\n${s.content}`
        : `## Step ${s.number}: ${s.title}\n${s.content}`
    )
    .join('\n\n');
}

/* ── Sub-components ── */
function StepCard({ step }: { step: Step }) {
  const statusColor =
    step.status === 'running'
      ? 'var(--chat-accent)'
      : step.status === 'done'
      ? '#22c55e'
      : 'var(--chat-border)';

  const label =
    step.status === 'running' ? 'Running' : step.status === 'done' ? 'Done' : 'Pending';

  return (
    <div
      style={{
        border: `1px solid ${step.status === 'running' ? 'var(--chat-accent)' : 'var(--chat-border)'}`,
        borderRadius: 10,
        padding: '12px 14px',
        background: 'var(--surface-1, #1a1a2e)',
        transition: 'border-color 0.2s',
      }}
    >
      {/* Step header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: step.content ? 8 : 0 }}>
        {/* Status dot / number badge */}
        <div
          style={{
            width: 22,
            height: 22,
            borderRadius: '50%',
            background: statusColor,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            fontSize: 10,
            fontWeight: 700,
            color: '#fff',
          }}
        >
          {step.number === -1 ? '★' : step.number}
        </div>
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: 'var(--chat-text)',
            flex: 1,
          }}
        >
          {step.title}
        </span>
        <span
          style={{
            fontSize: 10,
            color: statusColor,
            fontWeight: 500,
            flexShrink: 0,
          }}
        >
          {label}
          {step.status === 'running' && (
            <span
              style={{
                display: 'inline-block',
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'var(--chat-accent)',
                marginLeft: 5,
                verticalAlign: 'middle',
                animation: 'agent-pulse 1s ease-in-out infinite',
              }}
            />
          )}
        </span>
      </div>

      {/* Step content */}
      {step.content && (
        <div
          style={{
            fontSize: 12,
            color: 'var(--chat-text-secondary, #9ca3af)',
            lineHeight: 1.6,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            paddingLeft: 30,
          }}
        >
          {step.content}
          {step.status === 'running' && (
            <span
              style={{
                display: 'inline-block',
                width: 6,
                height: 12,
                background: 'var(--chat-accent)',
                marginLeft: 2,
                verticalAlign: 'text-bottom',
                animation: 'agent-blink 0.8s step-end infinite',
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}

/* ── Main Component ── */
export function AgentPanel() {
  const [goal, setGoal] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [steps, setSteps] = useState<Step[]>([]);
  const [rawOutput, setRawOutput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [maxSteps, setMaxSteps] = useState<3 | 5 | 10>(5);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  // Fetch models
  useEffect(() => {
    fetch('/api/models')
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
  }, []);

  // Close model menu on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Abort on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  // Auto-scroll as steps appear
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [steps, rawOutput]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsRunning(false);
  }, []);

  const handleCopy = useCallback(() => {
    const text = buildFullResult(steps);
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [steps]);

  const handleRun = useCallback(async () => {
    if (!goal.trim() || isRunning) return;

    setSteps([]);
    setRawOutput('');
    setError(null);
    setIsRunning(true);

    const controller = new AbortController();
    abortRef.current = controller;

    const systemCtx = `${SYSTEM_PROMPT}\n\nLimit yourself to a maximum of ${maxSteps} steps (not counting the Final Result).`;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: goal,
          system_prompt: systemCtx,
          history: [],
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullText = '';

      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const text =
                  parsed.choices?.[0]?.delta?.content ||
                  parsed.content ||
                  parsed.chunk ||
                  '';
                if (text) {
                  fullText += text;
                  setRawOutput(fullText);
                  setSteps(parseStepsLive(fullText));
                }
              } catch {
                fullText += data;
                setRawOutput(fullText);
                setSteps(parseStepsLive(fullText));
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullText += line;
              setRawOutput(fullText);
              setSteps(parseStepsLive(fullText));
            }
          }
        }
      } else {
        const text = await res.text();
        fullText = text;
        setRawOutput(fullText);
        setSteps(parseSteps(fullText));
      }

      // Final parse — mark all done
      setSteps(parseSteps(fullText));
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Unknown error');
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  }, [goal, isRunning, maxSteps, selectedModel]);

  const hasResult = steps.length > 0;
  const finalStep = steps.find((s) => s.number === -1);

  return (
    <>
      <style>{`
        @keyframes agent-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        @keyframes agent-blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
      `}</style>

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflow: 'hidden',
          background: 'var(--surface-0)',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '12px 16px',
            borderBottom: '1px solid var(--chat-border)',
            flexShrink: 0,
          }}
        >
          <h2 style={{ fontSize: 13, fontWeight: 600, color: 'var(--chat-text)', margin: 0 }}>
            Agent Runner
          </h2>
          <p style={{ fontSize: 10, color: 'var(--chat-text-secondary, #9ca3af)', margin: '2px 0 0' }}>
            Describe a goal and the agent will break it into steps and execute them
          </p>
        </div>

        {/* Goal input area */}
        <div
          style={{
            padding: '12px 16px',
            borderBottom: '1px solid var(--chat-border)',
            flexShrink: 0,
          }}
        >
          <textarea
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                handleRun();
              }
            }}
            placeholder="Describe a goal... (e.g. Research competitors and create a comparison table)"
            disabled={isRunning}
            rows={3}
            style={{
              width: '100%',
              padding: '10px 12px',
              borderRadius: 8,
              background: 'var(--surface-1, #1a1a2e)',
              border: '1px solid var(--chat-border)',
              color: 'var(--chat-text)',
              fontSize: 13,
              resize: 'none',
              outline: 'none',
              boxSizing: 'border-box',
              fontFamily: 'inherit',
              lineHeight: 1.5,
            }}
          />

          {/* Controls row */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginTop: 8,
              flexWrap: 'wrap',
            }}
          >
            {/* Max steps */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ fontSize: 10, color: 'var(--chat-text-secondary, #9ca3af)' }}>Max steps:</span>
              <div
                style={{
                  display: 'flex',
                  borderRadius: 6,
                  border: '1px solid var(--chat-border)',
                  overflow: 'hidden',
                }}
              >
                {MAX_STEPS_OPTIONS.map((n) => (
                  <button
                    key={n}
                    onClick={() => setMaxSteps(n)}
                    style={{
                      padding: '3px 8px',
                      fontSize: 10,
                      fontWeight: 500,
                      border: 'none',
                      cursor: 'pointer',
                      background: maxSteps === n ? 'var(--chat-accent)' : 'transparent',
                      color: maxSteps === n ? '#fff' : 'var(--chat-text-secondary, #9ca3af)',
                      transition: 'background 0.15s, color 0.15s',
                    }}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>

            {/* Model selector */}
            <div ref={modelMenuRef} style={{ position: 'relative' }}>
              <button
                onClick={() => setShowModelMenu((p) => !p)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 10,
                  color: 'var(--chat-text-secondary, #9ca3af)',
                  background: 'var(--border-subtle, #2a2a3e)',
                  border: 'none',
                  borderRadius: 6,
                  padding: '3px 8px',
                  cursor: 'pointer',
                  maxWidth: 160,
                }}
              >
                <span
                  style={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    maxWidth: 120,
                  }}
                >
                  {selectedModel ? selectedModel.split('/').pop() : 'Auto'}
                </span>
                <svg
                  style={{ width: 10, height: 10, flexShrink: 0, opacity: 0.5 }}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showModelMenu && availableModels.length > 0 && (
                <div
                  style={{
                    position: 'absolute',
                    top: 28,
                    left: 0,
                    width: 220,
                    maxHeight: 240,
                    background: 'var(--surface-1, #1a1a2e)',
                    border: '1px solid var(--chat-border)',
                    borderRadius: 10,
                    overflow: 'hidden',
                    zIndex: 50,
                  }}
                >
                  <div style={{ maxHeight: 240, overflowY: 'auto', padding: 4 }}>
                    <button
                      onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                      style={{
                        width: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: 'none',
                        cursor: 'pointer',
                        fontSize: 11,
                        textAlign: 'left',
                        background: !selectedModel ? 'var(--surface-3, #2a2a3e)' : 'transparent',
                        color: !selectedModel ? 'var(--chat-text)' : 'var(--chat-text-secondary, #9ca3af)',
                      }}
                    >
                      Auto (recommended)
                    </button>
                    {availableModels.map((m) => (
                      <button
                        key={m}
                        onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                        style={{
                          width: '100%',
                          display: 'flex',
                          alignItems: 'center',
                          padding: '6px 10px',
                          borderRadius: 6,
                          border: 'none',
                          cursor: 'pointer',
                          fontSize: 11,
                          textAlign: 'left',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          background: selectedModel === m ? 'var(--surface-3, #2a2a3e)' : 'transparent',
                          color: selectedModel === m ? 'var(--chat-text)' : 'var(--chat-text-secondary, #9ca3af)',
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div style={{ flex: 1 }} />

            {/* Copy button */}
            {hasResult && (
              <button
                onClick={handleCopy}
                title="Copy full result"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  padding: '5px 10px',
                  borderRadius: 7,
                  border: '1px solid var(--chat-border)',
                  background: 'transparent',
                  color: copied ? '#22c55e' : 'var(--chat-text-secondary, #9ca3af)',
                  fontSize: 11,
                  cursor: 'pointer',
                  transition: 'color 0.2s',
                }}
              >
                {copied ? (
                  <ClipboardDocumentCheckIcon style={{ width: 13, height: 13 }} />
                ) : (
                  <ClipboardDocumentIcon style={{ width: 13, height: 13 }} />
                )}
                {copied ? 'Copied' : 'Copy'}
              </button>
            )}

            {/* Run / Stop button */}
            <button
              onClick={isRunning ? handleStop : handleRun}
              disabled={!isRunning && !goal.trim()}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '6px 14px',
                borderRadius: 8,
                border: 'none',
                background: isRunning ? '#ef4444' : 'var(--chat-accent)',
                color: '#fff',
                fontSize: 12,
                fontWeight: 600,
                cursor: !isRunning && !goal.trim() ? 'not-allowed' : 'pointer',
                opacity: !isRunning && !goal.trim() ? 0.4 : 1,
                transition: 'background 0.15s, opacity 0.15s',
              }}
            >
              {isRunning ? (
                <>
                  <StopIcon style={{ width: 13, height: 13 }} />
                  Stop
                </>
              ) : (
                <>
                  <svg style={{ width: 13, height: 13 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Run Agent
                </>
              )}
            </button>
          </div>
        </div>

        {/* Steps output */}
        <div
          ref={scrollRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '14px 16px',
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
          }}
        >
          {/* Empty state */}
          {!hasResult && !isRunning && !error && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: 'var(--chat-text-secondary, #9ca3af)',
                textAlign: 'center',
                gap: 8,
              }}
            >
              <svg style={{ width: 36, height: 36, opacity: 0.3 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <p style={{ fontSize: 13, margin: 0 }}>Describe a goal and click Run Agent</p>
              <p style={{ fontSize: 11, margin: 0, opacity: 0.7 }}>
                The agent will plan steps and execute them one by one
              </p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div
              style={{
                padding: '10px 14px',
                borderRadius: 8,
                background: 'rgba(239,68,68,0.1)',
                border: '1px solid rgba(239,68,68,0.3)',
                color: '#ef4444',
                fontSize: 12,
              }}
            >
              Error: {error}
            </div>
          )}

          {/* Step cards */}
          {steps.map((step, i) => (
            <StepCard key={`${step.number}-${i}`} step={step} />
          ))}

          {/* Running shimmer if no steps parsed yet */}
          {isRunning && steps.length === 0 && (
            <div
              style={{
                padding: '12px 14px',
                borderRadius: 10,
                border: '1px solid var(--chat-accent)',
                background: 'var(--surface-1, #1a1a2e)',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 12,
                color: 'var(--chat-accent)',
              }}
            >
              <span
                style={{
                  display: 'inline-block',
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: 'var(--chat-accent)',
                  animation: 'agent-pulse 1s ease-in-out infinite',
                }}
              />
              Planning steps...
            </div>
          )}

          {/* Highlight final result box */}
          {finalStep && !isRunning && (
            <div
              style={{
                padding: '10px 14px',
                borderRadius: 8,
                background: 'rgba(34,197,94,0.07)',
                border: '1px solid rgba(34,197,94,0.25)',
                fontSize: 11,
                color: '#22c55e',
                marginTop: 4,
              }}
            >
              Agent completed — {steps.filter((s) => s.number !== -1).length} steps executed.
            </div>
          )}
        </div>
      </div>
    </>
  );
}

/* ── Live parse: marks last incomplete step as "running" ── */
function parseStepsLive(text: string): Step[] {
  const steps = parseSteps(text);
  if (steps.length === 0) return steps;

  // The last step in an in-progress stream is still being written
  const last = steps[steps.length - 1];
  return steps.map((s, i) =>
    i === steps.length - 1
      ? { ...last, status: 'running' as StepStatus }
      : { ...s, status: 'done' as StepStatus }
  );
}
