/**
 * DeployDashboard — lists all deployments with status, actions.
 * Shows in a modal overlay from WebCreatorPanel.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  ExternalLink, Copy, Trash2, Check, RefreshCw, Globe, X,
} from 'lucide-react';
import {
  loadDeployments, removeDeployment, deleteShare, listShares,
  type Deployment, type ShareInfo,
} from '../utils/deployUtils';

interface DeployDashboardProps {
  open: boolean;
  onClose: () => void;
}

export default function DeployDashboard({ open, onClose }: DeployDashboardProps) {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [shares, setShares] = useState<ShareInfo[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [deps, shareList] = await Promise.all([loadDeployments(), listShares()]);
      setDeployments(deps);
      setShares(shareList);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (open) refresh();
  }, [open, refresh]);

  const copyUrl = async (url: string, id: string) => {
    try {
      await navigator.clipboard.writeText(url);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch { /* clipboard unavailable */ }
  };

  const handleDelete = async (dep: Deployment) => {
    if (dep.platform === 'aura') {
      await deleteShare(dep.id);
    }
    await removeDeployment(dep.id);
    refresh();
  };

  if (!open) return null;

  // Merge shares into deployments (shares not yet tracked as deployments)
  const shareIds = new Set(deployments.filter(d => d.platform === 'aura').map(d => d.id));
  const untracked = shares.filter(s => !shareIds.has(s.id));

  const allItems: Deployment[] = [
    ...deployments,
    ...untracked.map(s => ({
      id: s.id,
      projectName: s.project_name,
      platform: 'aura' as const,
      url: s.url,
      status: (s.expires_at > Date.now() / 1000 ? 'live' : 'expired') as Deployment['status'],
      createdAt: s.created_at * 1000,
      expiresAt: s.expires_at * 1000,
    })),
  ];

  return (
    <div style={{
      position: 'absolute', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{
        flex: 1, margin: '12px', background: 'var(--bg)',
        borderRadius: 'var(--r-lg)', border: '1px solid var(--b1)',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '12px 14px', borderBottom: '1px solid var(--b1)',
        }}>
          <Globe size={15} style={{ color: 'var(--pl)' }} />
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)', flex: 1 }}>
            Deployments
          </span>
          <button
            onClick={refresh}
            disabled={loading}
            style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', padding: 2, display: 'flex' }}
          >
            <RefreshCw size={13} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          </button>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', padding: 2, display: 'flex' }}
          >
            <X size={15} />
          </button>
        </div>

        {/* List */}
        <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
          {allItems.length === 0 ? (
            <div style={{ padding: '24px 12px', textAlign: 'center', color: 'var(--mu)', fontSize: '12px' }}>
              No deployments yet. Use the Share or Deploy button to create one.
            </div>
          ) : (
            allItems.map(dep => (
              <div
                key={dep.id}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '10px 12px', marginBottom: 6,
                  background: 'var(--s2)', borderRadius: 'var(--r-md)',
                  border: '1px solid var(--b1)',
                }}
              >
                <div style={{
                  width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                  background: dep.status === 'live' ? '#3fb950'
                    : dep.status === 'building' ? '#d29922'
                    : dep.status === 'expired' ? '#484f58'
                    : '#f85149',
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {dep.projectName}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--mu)', marginTop: 2 }}>
                    {dep.platform === 'aura' ? 'Aura Share' : 'GitHub Pages'}
                    {dep.expiresAt && ` · Expires ${new Date(dep.expiresAt).toLocaleDateString()}`}
                  </div>
                </div>
                <button
                  onClick={() => copyUrl(dep.url, dep.id)}
                  style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', padding: 2, display: 'flex' }}
                  title="Copy URL"
                >
                  {copiedId === dep.id ? <Check size={13} style={{ color: '#3fb950' }} /> : <Copy size={13} />}
                </button>
                <a
                  href={dep.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: 'var(--mu)', display: 'flex', padding: 2 }}
                  title="Open"
                >
                  <ExternalLink size={13} />
                </a>
                <button
                  onClick={() => handleDelete(dep)}
                  style={{ background: 'none', border: 'none', color: '#f85149', cursor: 'pointer', padding: 2, display: 'flex' }}
                  title="Delete"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
