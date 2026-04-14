import React, { useRef } from 'react';
import { Bot } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import AgentStepRow from '../components/AgentStepRow';

/**
 * Legacy isolated agent view. The primary agent surface is now the Chat panel
 * (type `/agent <task>` in the input bar). This panel stays as a dedicated
 * log view for users who prefer running agent tasks outside the conversation.
 * It delegates all lifecycle to the shared store actions.
 */
export default function AgentPanel() {
  const agentRunning = useStore(s => s.agentRunning);
  const messages = useStore(s => s.messages);
  const runAgentTask = useStore(s => s.runAgentTask);
  const stopAgentTask = useStore(s => s.stopAgentTask);
  const setPanel = useStore(s => s.setPanel);
  const taskRef = useRef<HTMLTextAreaElement>(null);

  // Find the most recent agent message — that's the one we're showing steps for
  const latestAgentMessage = [...messages].reverse().find(m => m.agentSteps !== undefined);

  const start = () => {
    const task = taskRef.current?.value.trim();
    if (!task) return;
    if (taskRef.current) taskRef.current.value = '';
    runAgentTask(task);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
      <div>
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Bot size={12} /> Browser Agent
        </div>
        <textarea
          ref={taskRef}
          placeholder="Describe what the agent should do on the current page…"
          disabled={agentRunning}
          style={{
            width: '100%',
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: 12.5,
            padding: '8px 10px',
            resize: 'none',
            height: 70,
            outline: 'none',
            fontFamily: 'inherit',
            opacity: agentRunning ? 0.6 : 1,
          }}
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ModelPill featureKey="agent" />
          {agentRunning && latestAgentMessage?.agentSteps && (
            <span style={{ fontSize: 11, color: 'var(--mu)' }}>
              Step {latestAgentMessage.agentSteps.length} / 15
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setPanel('chat')}
            title="Open chat view"
            style={{
              background: 'transparent',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              color: 'var(--mu)',
              padding: '7px 12px',
              cursor: 'pointer',
              fontSize: 11.5,
              fontFamily: 'inherit',
            }}
          >
            Open in Chat
          </button>
          {agentRunning ? (
            <button
              onClick={stopAgentTask}
              style={{
                background: 'rgba(239,68,68,0.15)',
                border: '1px solid var(--rd)',
                borderRadius: 'var(--r-md)',
                color: 'var(--rd)',
                padding: '7px 16px',
                cursor: 'pointer',
                fontSize: 12,
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
                fontSize: 12,
                fontFamily: 'inherit',
              }}
            >
              ▶ Start
            </button>
          )}
        </div>
      </div>

      {/* Step log */}
      <div
        className="flex-1 overflow-y-auto"
        style={{
          background: 'var(--cb)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-md)',
          padding: 10,
        }}
      >
        {latestAgentMessage?.agentTask && (
          <div style={{ fontSize: 11.5, color: 'var(--pl)', marginBottom: 8, fontWeight: 500 }}>
            {latestAgentMessage.agentTask}
          </div>
        )}
        {latestAgentMessage?.agentSteps && latestAgentMessage.agentSteps.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {latestAgentMessage.agentSteps.map((s, i) => <AgentStepRow key={i} step={s} />)}
          </div>
        ) : (
          <span style={{ color: 'var(--di)', fontSize: 11.5 }}>Agent log will appear here…</span>
        )}
        {latestAgentMessage?.agentDone && latestAgentMessage?.text && (
          <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--mu)', fontStyle: 'italic' }}>
            {latestAgentMessage.text}
          </div>
        )}
      </div>
    </div>
  );
}
