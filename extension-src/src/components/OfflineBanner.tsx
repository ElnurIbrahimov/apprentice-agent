import React from 'react';
import { WifiOff, Globe } from 'lucide-react';
import { useStore } from '../store';

/**
 * Slim banner shown when backend is offline.
 * Optional `hint` prop for panel-specific messages (e.g., "Pyodide available for code").
 */
export default function OfflineBanner({ hint }: { hint?: string }) {
  const backendStatus = useStore(s => s.backendStatus);
  if (backendStatus === 'online') return null;

  const isConnecting = backendStatus === 'connecting';

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '6px 12px', fontSize: '11px', flexShrink: 0,
      borderBottom: '1px solid var(--b1)',
      background: isConnecting ? 'rgba(245,158,11,0.06)' : 'rgba(239,68,68,0.06)',
    }}>
      <WifiOff size={12} style={{ color: isConnecting ? '#f59e0b' : '#ef4444', flexShrink: 0 }} />
      <span style={{ color: isConnecting ? '#f59e0b' : '#ef4444', flex: 1 }}>
        {isConnecting ? 'Connecting to server...' : 'Server offline'}
      </span>
      {hint && !isConnecting && (
        <span style={{ color: 'var(--mu)', fontSize: '10px', display: 'flex', alignItems: 'center', gap: 3 }}>
          <Globe size={10} /> {hint}
        </span>
      )}
    </div>
  );
}
