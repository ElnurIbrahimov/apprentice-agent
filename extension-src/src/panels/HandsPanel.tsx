import React, { useEffect, useMemo, useState } from 'react';
import { Plus, FileText, Pause, Play, RefreshCw, Hand as HandIcon } from 'lucide-react';
import { useStore } from '../store';
import ApprovalCard from '../components/hands/ApprovalCard';
import HandStatusCard from '../components/hands/HandStatusCard';
import HandsLiveTrace from '../components/hands/HandsLiveTrace';
import CreateHandModal from '../components/hands/CreateHandModal';
import TemplatePickerModal from '../components/hands/TemplatePickerModal';

export default function HandsPanel() {
  const hands = useStore(s => s.hands);
  const approvals = useStore(s => s.handApprovals);
  const templates = useStore(s => s.handTemplates);
  const liveTrace = useStore(s => s.handLiveTrace);
  const history = useStore(s => s.handHistory);
  const loaded = useStore(s => s.handsLoaded);
  const error = useStore(s => s.handsError);
  const handsError = useStore(s => s.handsError);
  const handsLoaded = useStore(s => s.handsLoaded);
  const handsPollingActive = useStore(s => s.handsPollingActive);
  const handsPollingInterval = useStore(s => s.handsPollingInterval);
  const handsLastLoaded = useStore(s => s.handsLastLoaded);
  const wsReady = useStore(s => s.wsReady);
  const backendStatus = useStore(s => s.backendStatus);

  const loadHands = useStore(s => s.loadHands);
  const loadApprovals = useStore(s => s.loadHandApprovals);
  const loadTemplates = useStore(s => s.loadHandTemplates);
  const loadHistory = useStore(s => s.loadHandHistory);
  const runHand = useStore(s => s.runHand);
  const pauseHand = useStore(s => s.pauseHand);
  const activateHand = useStore(s => s.activateHand);
  const deactivateHand = useStore(s => s.deactivateHand);
  const deleteHand = useStore(s => s.deleteHand);
  const approveHand = useStore(s => s.approveHand);
  const createHand = useStore(s => s.createHand);
  const createHandFromTemplate = useStore(s => s.createHandFromTemplate);

  const [createOpen, setCreateOpen] = useState(false);
  const [templatesOpen, setTemplatesOpen] = useState(false);

  useEffect(() => {
    loadHands();
    loadApprovals();
    loadTemplates();
    loadHistory(30);
  }, [loadHands, loadApprovals, loadTemplates, loadHistory]);

  // Polling lifecycle - start interval on mount
  useEffect(() => {
    const wsReady = useStore.getState().wsReady;
    const backendStatus = useStore.getState().backendStatus;
    const interval = (wsReady && backendStatus === 'online') ? 15000 : 60000;
    useStore.getState().setHandsPollingInterval(interval);
    useStore.getState().setHandsPollingActive(true);
    useStore.getState().loadHands();
    useStore.getState().loadHandApprovals();
    useStore.getState().loadHandHistory();

    const id = setInterval(() => {
      useStore.getState().loadHands();
      useStore.getState().loadHandApprovals();
      useStore.getState().loadHandHistory();
    }, interval);

    return () => {
      clearInterval(id);
      useStore.getState().setHandsPollingActive(false);
    };
  }, []);

  // React to wsReady/backendStatus changes and restart the interval if it changes
  useEffect(() => {
    const unsubscribe = useStore.subscribe(
      (state) => [state.wsReady, state.backendStatus] as const,
      ([wsReady, backendStatus], prev) => {
        if (wsReady === prev[0] && backendStatus === prev[1]) return;
        const isActive = useStore.getState().handsPollingActive;
        if (!isActive) return;
        const interval = (wsReady && backendStatus === 'online') ? 15000 : 60000;
        useStore.getState().setHandsPollingInterval(interval);
      }
    );
    return unsubscribe;
  }, []);

  const { activeCount, successRate } = useMemo(() => {
    const active = hands.filter(h => h.state === 'active' || h.state === 'running' || h.state === 'cooldown').length;
    const completed = history.length;
    const successes = history.filter(h => (h.action_data as any)?.success === true).length;
    const rate = completed > 0 ? Math.round((successes / completed) * 100) : 100;
    return { activeCount: active, successRate: rate };
  }, [hands, history]);

  const anyPaused = hands.some(h => h.state === 'paused');
  const anyActive = hands.some(h => h.state === 'active' || h.state === 'running');

  const pauseAll = async () => {
    for (const h of hands) {
      if (h.state === 'active' || h.state === 'running') {
        await pauseHand(h.name);
      }
    }
  };

  const resumeAll = async () => {
    for (const h of hands) {
      if (h.state === 'paused') {
        await activateHand(h.name);
      }
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="panel-scroll-root" style={{ flex: 1, overflowY: 'auto', padding: 12 }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <HandIcon size={14} style={{ color: 'var(--pl)' }} />
          <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
            Hands — Mission Control
          </span>
          <button
            onClick={() => { loadHands(); loadApprovals(); loadHistory(30); }}
            title="Refresh"
            style={{
              marginLeft: 'auto',
              background: 'transparent',
              border: 'none',
              color: 'var(--mu)',
              cursor: 'pointer',
              padding: 2,
            }}
          >
            <RefreshCw size={12} />
          </button>
        </div>

        {/* Inline error banner */}
        {handsError && (
          <div style={{
            margin: '0 0 12px 0',
            padding: '8px 12px',
            background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: 6,
            color: '#ef4444',
            fontSize: 12,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <span>⚠ {handsError}</span>
            <button
              onClick={() => loadHands()}
              style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: 12, padding: '2px 6px' }}
            >
              Retry
            </button>
          </div>
        )}

        {/* Connection status indicator */}
        {handsLoaded && !handsError && handsPollingActive && (
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', textAlign: 'center', padding: '2px 0', marginBottom: 8 }}>
            {handsPollingInterval >= 60000 ? '◌ Polling (degraded)' : '◎ Polling'}
            {handsLastLoaded ? ` · Last updated ${new Date(handsLastLoaded).toLocaleTimeString()}` : ''}
          </div>
        )}

        {/* Sticky approvals */}
        {approvals.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
            {approvals.map(a => (
              <ApprovalCard
                key={a.request_id}
                approval={a}
                onApprove={(name, rid) => approveHand(name, rid, true)}
                onDeny={(name, rid) => approveHand(name, rid, false)}
              />
            ))}
          </div>
        )}

        {/* Stats pill */}
        <div
          style={{
            display: 'flex',
            gap: 0,
            padding: '6px 10px',
            borderRadius: 8,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            marginBottom: 10,
            fontSize: 11,
          }}
        >
          <span style={{ flex: 1, color: 'var(--mu)' }}>
            Total <span style={{ color: 'var(--tx)', fontWeight: 600 }}>{hands.length}</span>
          </span>
          <span style={{ flex: 1, color: 'var(--mu)' }}>
            Active <span style={{ color: 'var(--gr, #22c55e)', fontWeight: 600 }}>{activeCount}</span>
          </span>
          <span style={{ flex: 1, color: 'var(--mu)' }}>
            Success <span style={{ color: successRate >= 80 ? 'var(--gr, #22c55e)' : successRate >= 60 ? 'var(--yl, #f59e0b)' : 'var(--rd, #ef4444)', fontWeight: 600 }}>{successRate}%</span>
          </span>
        </div>

        {/* Action bar */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
          <button
            onClick={() => setCreateOpen(true)}
            style={actionBtn('var(--p)', 'white')}
          >
            <Plus size={11} /> New Hand
          </button>
          <button
            onClick={() => setTemplatesOpen(true)}
            style={actionBtn('transparent', 'var(--tx)')}
          >
            <FileText size={11} /> Templates
          </button>
          {anyActive && (
            <button onClick={pauseAll} style={actionBtn('transparent', 'var(--yl, #f59e0b)')}>
              <Pause size={11} /> Pause all
            </button>
          )}
          {anyPaused && (
            <button onClick={resumeAll} style={actionBtn('transparent', 'var(--gr, #22c55e)')}>
              <Play size={11} /> Resume all
            </button>
          )}
        </div>

        {/* Status list */}
        {!loaded && (
          <div style={{ fontSize: 11, color: 'var(--mu)', textAlign: 'center', padding: 20 }}>
            Loading hands…
          </div>
        )}
        {loaded && error && (
          <div style={{ fontSize: 11, color: 'var(--rd)', padding: 10, borderRadius: 6, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', marginBottom: 10 }}>
            {error}
          </div>
        )}
        {loaded && hands.length === 0 && !error && (
          <div style={{ fontSize: 11.5, color: 'var(--mu)', padding: 20, textAlign: 'center', lineHeight: 1.5 }}>
            No hands yet. Create one from scratch or pick a template to get started.
          </div>
        )}
        {hands.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {hands.map(h => (
              <HandStatusCard
                key={h.name}
                hand={h}
                handsError={handsError}
                onRun={(n) => runHand(n).catch(() => {})}
                onPause={(n) => pauseHand(n).catch(() => {})}
                onActivate={(n) => activateHand(n).catch(() => {})}
                onDeactivate={(n) => deactivateHand(n).catch(() => {})}
                onDelete={(n) => { if (confirm(`Delete hand "${n}"?`)) deleteHand(n).catch(() => {}); }}
              />
            ))}
          </div>
        )}

        {/* Live trace */}
        <div style={{ marginTop: 12 }}>
          <HandsLiveTrace traces={liveTrace} />
        </div>
      </div>

      {/* Modals */}
      <CreateHandModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onSubmit={createHand}
      />
      <TemplatePickerModal
        open={templatesOpen}
        templates={templates}
        onClose={() => setTemplatesOpen(false)}
        onSubmit={createHandFromTemplate}
      />
    </div>
  );
}

function actionBtn(bg: string, color: string): React.CSSProperties {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    background: bg,
    border: bg === 'transparent' ? '1px solid var(--b1)' : 'none',
    borderRadius: 6,
    color,
    padding: '6px 10px',
    cursor: 'pointer',
    fontSize: 11,
    fontFamily: 'inherit',
  };
}
