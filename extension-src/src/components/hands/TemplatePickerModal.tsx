import React, { useState } from 'react';
import { X, FileText } from 'lucide-react';
import type { HandTemplate } from '../../types';

interface Props {
  open: boolean;
  templates: HandTemplate[];
  onClose: () => void;
  onSubmit: (templateName: string, variables: Record<string, string>) => Promise<void>;
}

function parseVariables(raw: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const pair of raw.split(/[\n,]/)) {
    const [k, ...rest] = pair.split('=');
    const key = k?.trim();
    const value = rest.join('=').trim();
    if (key && value) out[key] = value;
  }
  return out;
}

export default function TemplatePickerModal({ open, templates, onClose, onSubmit }: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const [variables, setVariables] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const handleSubmit = async () => {
    if (!selected) { setError('Pick a template first.'); return; }
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit(selected, parseVariables(variables));
      setSelected(null);
      setVariables('');
      onClose();
    } catch (err: any) {
      setError(err?.message || 'Failed to create from template');
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
          maxHeight: '80vh',
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--s1)',
          border: '1px solid var(--b1)',
          borderRadius: 12,
          padding: 14,
          boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
          <FileText size={14} style={{ color: 'var(--pl)' }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>Hand templates</span>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
            <X size={14} />
          </button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 10 }}>
          {templates.length === 0 ? (
            <div style={{ fontSize: 11, color: 'var(--mu)', padding: 12, textAlign: 'center' }}>
              No templates available.
            </div>
          ) : templates.map(t => (
            <button
              key={t.name}
              onClick={() => setSelected(t.name)}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                padding: '8px 10px',
                borderRadius: 8,
                background: selected === t.name ? 'rgba(124,58,237,0.15)' : 'var(--s2)',
                border: `1px solid ${selected === t.name ? 'var(--pl)' : 'var(--b1)'}`,
                cursor: 'pointer',
                textAlign: 'left',
                fontFamily: 'inherit',
              }}
            >
              <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--tx)' }}>{t.name}</span>
              <span style={{ fontSize: 10.5, color: 'var(--mu)' }}>{t.description}</span>
              {t.trigger_on_drive && (
                <span style={{ fontSize: 10, color: 'var(--pl)', marginTop: 2 }}>
                  drive: {t.trigger_on_drive}
                </span>
              )}
            </button>
          ))}
        </div>
        {selected && (
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 10.5, color: 'var(--mu)', marginBottom: 4 }}>
              Variables (optional) — <span style={{ fontFamily: 'monospace' }}>key=value</span>, one per line
            </div>
            <textarea
              value={variables}
              onChange={(e) => setVariables(e.target.value)}
              placeholder="topic=AI safety"
              style={{
                width: '100%',
                minHeight: 50,
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 6,
                color: 'var(--tx)',
                fontSize: 11.5,
                padding: 6,
                outline: 'none',
                fontFamily: 'monospace',
                resize: 'vertical',
              }}
            />
          </div>
        )}
        {error && <div style={{ fontSize: 11, color: 'var(--rd)', marginBottom: 6 }}>{error}</div>}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
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
            disabled={submitting || !selected}
            style={{
              background: 'var(--p)',
              border: 'none',
              borderRadius: 6,
              color: 'white',
              padding: '6px 14px',
              cursor: (submitting || !selected) ? 'not-allowed' : 'pointer',
              fontSize: 11.5,
              fontFamily: 'inherit',
              opacity: (submitting || !selected) ? 0.6 : 1,
            }}
          >
            {submitting ? 'Creating…' : 'Use template'}
          </button>
        </div>
      </div>
    </div>
  );
}
