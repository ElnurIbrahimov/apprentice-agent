import React, { useEffect, useRef, useState } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, getAuthHeaders } from '../api';
import { sendMsg } from '../ext';

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

export default function AgentPanel() {
  const { getModel } = useStore();
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const taskRef = useRef<HTMLTextAreaElement>(null);
  const runningRef = useRef(false);
  const logRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Cleanup: stop agent loop and abort fetch on unmount
  useEffect(() => {
    return () => {
      runningRef.current = false;
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const addLog = (text: string) => {
    setLog(prev => {
      const next = [...prev, text];
      setTimeout(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, 0);
      return next;
    });
  };

  const runAgentLoop = async (task: string) => {
    const history: any[] = [];
    let stepCount = 0;

    while (runningRef.current && stepCount < 15) {
      stepCount++;
      setStep(stepCount);

      const dom = await sendMsg({ type: 'AGENT_DOM' });
      if (!dom?.ok) {
        addLog('⚠ Could not read page DOM.');
        break;
      }

      const domStr = dom.dom
        .map((e: any) => `[${e.index}] ${e.type} "${e.text}" → ${e.selector}`)
        .join('\n');

      const prompt =
        `Task: "${task}"\nURL: ${dom.url}\nTitle: ${dom.title}\n` +
        `History: ${JSON.stringify(history.slice(-5))}\n\n` +
        `Interactive elements on page:\n${domStr.slice(0, 3000)}\n\n` +
        `Respond ONLY with valid JSON (no markdown, no explanation):\n` +
        `{"action":"click"|"type"|"scroll"|"navigate"|"done","selector":"","text":"","url":"","amount":300,"description":""}`;

      try {
        if (abortRef.current) abortRef.current.abort();
        const ctrl = new AbortController();
        abortRef.current = ctrl;
        const action = await fetch(`${HTTP}/api/agent/action`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
          body: JSON.stringify({ prompt, model: getModel('agent') }),
          signal: ctrl.signal,
        }).then(r => r.json());

        addLog(`Step ${stepCount}: [${action.action}] ${action.description || ''}`);

        if (action.action === 'done') {
          addLog('✓ Task complete.');
          runningRef.current = false;
          break;
        }

        if (action.action === 'navigate') {
          await sendMsg({ type: 'AGENT_NAV', url: action.url });
          await sleep(2500);
        } else {
          const result = await sendMsg({ type: 'AGENT_EXEC', action });
          if (!result?.ok) addLog(`  ⚠ ${result?.error || 'Action failed'}`);
          await sleep(600);
        }

        history.push({ step: stepCount, ...action });
      } catch (err: any) {
        addLog('⚠ Error: ' + err.message);
        runningRef.current = false;
        break;
      }
    }

    if (stepCount >= 15 && runningRef.current) addLog('⚠ Max steps (15) reached.');
    runningRef.current = false;
    setRunning(false);
  };

  const start = () => {
    const task = taskRef.current?.value.trim();
    if (!task) return;
    setLog([]);
    setStep(0);
    runningRef.current = true;
    setRunning(true);
    addLog(`▶ Starting: "${task.slice(0, 60)}"`);
    runAgentLoop(task);
  };

  const stop = () => {
    runningRef.current = false;
    setRunning(false);
    addLog('■ Stopped by user.');
  };

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
          Task
        </div>
        <textarea
          ref={taskRef}
          placeholder="Describe what the agent should do on the current page…"
          style={{
            width: '100%',
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12.5px',
            padding: '8px 10px',
            resize: 'none',
            height: 70,
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ModelPill featureKey="agent" />
          {running && (
            <span style={{ fontSize: '11px', color: 'var(--mu)' }}>Step {step} / 15</span>
          )}
        </div>
        <div className="flex gap-2">
          {running ? (
            <button
              onClick={stop}
              style={{
                background: 'rgba(239,68,68,0.15)',
                border: '1px solid var(--rd)',
                borderRadius: 'var(--r-md)',
                color: 'var(--rd)',
                padding: '7px 16px',
                cursor: 'pointer',
                fontSize: '12px',
                fontFamily: 'inherit',
              }}
            >
              ■ Stop
            </button>
          ) : (
            <button
              onClick={start}
              style={{
                background: 'var(--p)',
                border: 'none',
                borderRadius: 'var(--r-md)',
                color: 'white',
                padding: '7px 16px',
                cursor: 'pointer',
                fontSize: '12px',
                fontFamily: 'inherit',
              }}
            >
              ▶ Start
            </button>
          )}
        </div>
      </div>

      {/* Agent log */}
      <div
        ref={logRef}
        className="flex-1 overflow-y-auto"
        style={{
          background: 'var(--cb)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-md)',
          padding: '10px',
          fontFamily: 'monospace',
          fontSize: '11.5px',
          lineHeight: 1.7,
        }}
      >
        {log.length === 0 ? (
          <span style={{ color: 'var(--di)' }}>Agent log will appear here…</span>
        ) : (
          log.map((entry, i) => (
            <div
              key={i}
              style={{
                color: entry.startsWith('✓') ? 'var(--gr)' : entry.startsWith('⚠') ? 'var(--rd)' : entry.startsWith('■') ? 'var(--mu)' : 'var(--tx)',
              }}
            >
              {entry}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
