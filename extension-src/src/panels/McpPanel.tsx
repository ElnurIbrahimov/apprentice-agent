import React, { useEffect, useState } from 'react';
import { Plug, Plus, X, CheckCircle2, AlertCircle, Trash2, RefreshCw, Copy } from 'lucide-react';
import { useStore } from '../store';
import { HTTP } from '../api';
import type { McpServerCreate } from '../types';

export default function McpPanel() {
  const servers = useStore(s => s.mcpServers);
  const loaded = useStore(s => s.mcpLoaded);
  const error = useStore(s => s.mcpError);
  const loadServers = useStore(s => s.loadMcpServers);
  const addServer = useStore(s => s.addMcpServer);
  const removeServer = useStore(s => s.removeMcpServer);
  const setEnabled = useStore(s => s.setMcpServerEnabled);
  const testServer = useStore(s => s.testMcpServer);

  const [addOpen, setAddOpen] = useState(false);

  useEffect(() => {
    loadServers();
  }, [loadServers]);

  const handleCopySelfUrl = async () => {
    try {
      await navigator.clipboard.writeText(`${HTTP}/mcp`);
    } catch { /* noop */ }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="panel-scroll-root" style={{ flex: 1, overflowY: 'auto', padding: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Plug size={14} style={{ color: 'var(--pl)' }} />
          <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
            Connections — MCP Hub
          </span>
          <button
            onClick={() => loadServers()}
            title="Refresh"
            style={{ marginLeft: 'auto', background: 'transparent', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}
          >
            <RefreshCw size={12} />
          </button>
        </div>

        {/* Aura self-server info card */}
        <div
          style={{
            borderRadius: 10,
            background: 'linear-gradient(135deg, rgba(124,58,237,0.12), rgba(124,58,237,0.04))',
            border: '1px solid rgba(124,58,237,0.35)',
            padding: '10px 12px',
            marginBottom: 12,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <CheckCircle2 size={12} style={{ color: 'var(--pl)' }} />
            <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--tx)' }}>
              Aura (self)
            </span>
          </div>
          <div style={{ fontSize: 10.5, color: 'var(--mu)', marginBottom: 6, lineHeight: 1.5 }}>
            Aura exposes its own browser-agent + memory tools as an MCP server.
            Connect Claude Code, Cursor, or ChatGPT desktop to drive your real
            Chrome through Aura's backend.
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <code
              style={{
                flex: 1,
                fontSize: 10.5,
                color: 'var(--pl)',
                background: 'var(--s2)',
                padding: '3px 6px',
                borderRadius: 4,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {`${HTTP}/mcp`}
            </code>
            <button
              onClick={handleCopySelfUrl}
              title="Copy endpoint URL"
              style={{
                background: 'transparent',
                border: '1px solid var(--b1)',
                borderRadius: 4,
                color: 'var(--mu)',
                padding: '3px 6px',
                cursor: 'pointer',
              }}
            >
              <Copy size={10} />
            </button>
          </div>
        </div>

        {/* Outbound server list */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
          <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--mu)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
            Outbound servers
          </span>
          <button
            onClick={() => setAddOpen(true)}
            style={{
              marginLeft: 'auto',
              background: 'var(--p)',
              border: 'none',
              borderRadius: 6,
              color: 'white',
              padding: '5px 10px',
              fontSize: 11,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              fontFamily: 'inherit',
            }}
          >
            <Plus size={11} /> Add
          </button>
        </div>

        {!loaded && (
          <div style={{ fontSize: 11, color: 'var(--mu)', textAlign: 'center', padding: 16 }}>
            Loading…
          </div>
        )}
        {loaded && error && (
          <div style={{ fontSize: 11, color: 'var(--rd)', padding: 10, borderRadius: 6, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', marginBottom: 10 }}>
            {error}
          </div>
        )}
        {loaded && !error && servers.length === 0 && (
          <div style={{ fontSize: 11, color: 'var(--mu)', padding: 16, textAlign: 'center', lineHeight: 1.5 }}>
            No outbound MCP servers configured. Add GitHub, Notion, Gmail, or a local
            filesystem server to give Aura more tools.
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {servers.map(srv => (
            <div
              key={srv.name}
              style={{
                borderRadius: 10,
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                padding: '10px 12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: srv.connected ? 'var(--gr, #22c55e)' : srv.enabled ? 'var(--yl, #f59e0b)' : 'var(--di, #6b7280)',
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--tx)', flex: 1 }}>
                  {srv.name}
                </span>
                <span style={{ fontSize: 10, color: 'var(--mu)' }}>{srv.transport}</span>
              </div>
              <div style={{ fontSize: 10.5, color: 'var(--mu)', marginBottom: 6 }}>
                {srv.tool_count > 0 ? `${srv.tool_count} tools` : 'no tools cached'}
                {srv.error && <span style={{ color: 'var(--rd)', marginLeft: 6 }}>· {srv.error}</span>}
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <button
                  onClick={() => testServer(srv.name)}
                  style={tinyBtn('var(--pl)')}
                >
                  Test
                </button>
                <button
                  onClick={() => setEnabled(srv.name, !srv.enabled)}
                  style={tinyBtn(srv.enabled ? 'var(--yl, #f59e0b)' : 'var(--gr, #22c55e)')}
                >
                  {srv.enabled ? 'Disable' : 'Enable'}
                </button>
                <button
                  onClick={() => {
                    if (confirm(`Remove MCP server "${srv.name}"?`)) removeServer(srv.name).catch(() => {});
                  }}
                  style={tinyBtn('var(--rd)')}
                >
                  <Trash2 size={9} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <AddServerModal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onSubmit={addServer}
      />
    </div>
  );
}

function tinyBtn(color: string): React.CSSProperties {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 3,
    background: 'transparent',
    border: '1px solid var(--b1)',
    borderRadius: 5,
    color,
    padding: '3px 7px',
    cursor: 'pointer',
    fontSize: 10.5,
    fontFamily: 'inherit',
  };
}

interface AddProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (server: McpServerCreate) => Promise<void>;
}

function AddServerModal({ open, onClose, onSubmit }: AddProps) {
  const [name, setName] = useState('');
  const [transport, setTransport] = useState<'stdio' | 'http'>('http');
  const [url, setUrl] = useState('');
  const [command, setCommand] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const handleSubmit = async () => {
    if (!/^[a-zA-Z0-9_-]{1,40}$/.test(name)) {
      setError('Name must be 1-40 chars, letters/digits/_/- only');
      return;
    }
    if (transport === 'http' && !url.trim()) {
      setError('HTTP transport requires a URL');
      return;
    }
    if (transport === 'stdio' && !command.trim()) {
      setError('Stdio transport requires a command');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const payload: McpServerCreate = {
        name: name.trim(),
        transport,
        enabled: true,
      };
      if (transport === 'http') payload.url = url.trim();
      if (transport === 'stdio') payload.command = command.trim().split(/\s+/);
      await onSubmit(payload);
      setName(''); setUrl(''); setCommand('');
      onClose();
    } catch (err: any) {
      setError(err?.message || 'Failed to add server');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.5)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          maxWidth: 360,
          background: 'var(--s1)',
          border: '1px solid var(--b1)',
          borderRadius: 12,
          padding: 14,
          boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
          <Plug size={14} style={{ color: 'var(--pl)' }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>Add MCP server</span>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
            <X size={14} />
          </button>
        </div>

        <label style={{ display: 'block', fontSize: 10.5, color: 'var(--mu)', marginBottom: 4 }}>Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="github"
          style={fieldStyle}
        />

        <label style={{ display: 'block', fontSize: 10.5, color: 'var(--mu)', margin: '10px 0 4px' }}>Transport</label>
        <div style={{ display: 'flex', gap: 6 }}>
          <button
            onClick={() => setTransport('http')}
            style={segBtn(transport === 'http')}
          >
            HTTP
          </button>
          <button
            onClick={() => setTransport('stdio')}
            style={segBtn(transport === 'stdio')}
          >
            Stdio
          </button>
        </div>

        {transport === 'http' ? (
          <>
            <label style={{ display: 'block', fontSize: 10.5, color: 'var(--mu)', margin: '10px 0 4px' }}>
              URL
            </label>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://mcp.notion.com/mcp"
              style={fieldStyle}
            />
          </>
        ) : (
          <>
            <label style={{ display: 'block', fontSize: 10.5, color: 'var(--mu)', margin: '10px 0 4px' }}>
              Command
            </label>
            <input
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="npx @modelcontextprotocol/server-filesystem /root/notes"
              style={{ ...fieldStyle, fontFamily: 'monospace', fontSize: 11 }}
            />
          </>
        )}

        {error && <div style={{ fontSize: 11, color: 'var(--rd)', marginTop: 8 }}>{error}</div>}

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 12 }}>
          <button
            onClick={onClose}
            disabled={submitting}
            style={{
              background: 'transparent',
              border: '1px solid var(--b1)',
              borderRadius: 6,
              color: 'var(--mu)',
              padding: '6px 12px',
              cursor: 'pointer',
              fontSize: 11.5,
              fontFamily: 'inherit',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting}
            style={{
              background: 'var(--p)',
              border: 'none',
              borderRadius: 6,
              color: 'white',
              padding: '6px 14px',
              cursor: submitting ? 'wait' : 'pointer',
              fontSize: 11.5,
              fontFamily: 'inherit',
            }}
          >
            {submitting ? 'Adding…' : 'Add'}
          </button>
        </div>
      </div>
    </div>
  );
}

const fieldStyle: React.CSSProperties = {
  width: '100%',
  background: 'var(--s2)',
  border: '1px solid var(--b1)',
  borderRadius: 6,
  color: 'var(--tx)',
  fontSize: 12,
  padding: '7px 10px',
  outline: 'none',
  fontFamily: 'inherit',
};

function segBtn(active: boolean): React.CSSProperties {
  return {
    flex: 1,
    background: active ? 'var(--p)' : 'transparent',
    border: `1px solid ${active ? 'var(--p)' : 'var(--b1)'}`,
    borderRadius: 6,
    color: active ? 'white' : 'var(--mu)',
    padding: '6px 10px',
    cursor: 'pointer',
    fontSize: 11.5,
    fontFamily: 'inherit',
  };
}
