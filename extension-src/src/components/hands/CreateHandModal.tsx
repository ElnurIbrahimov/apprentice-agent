import React, { useState } from 'react';
import { X, Hand as HandIcon } from 'lucide-react';

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (description: string) => Promise<void>;
}

export default function CreateHandModal({ open, onClose, onSubmit }: Props) {
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const handleSubmit = async () => {
    const text = description.trim();
    if (text.length < 5) {
      setError('Describe what the hand should do (at least 5 characters).');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit(text);
      setDescription('');
      onClose();
    } catch (err: any) {
      setError(err?.message || 'Failed to create hand');
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
          <HandIcon size={14} style={{ color: 'var(--pl)' }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>Create custom hand</span>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
            <X size={14} />
          </button>
        </div>
        <div style={{ fontSize: 11, color: 'var(--mu)', marginBottom: 8, lineHeight: 1.45 }}>
          Describe what the hand should do autonomously — Aura will extract a config.
        </div>
        <textarea
          autoFocus
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Monitor Hacker News for AI papers and summarize them daily at 9am."
          style={{
            width: '100%',
            minHeight: 90,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 8,
            color: 'var(--tx)',
            fontSize: 12,
            padding: 8,
            resize: 'vertical',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        {error && (
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--rd)' }}>{error}</div>
        )}
        <div style={{ display: 'flex', gap: 8, marginTop: 10, justifyContent: 'flex-end' }}>
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
            {submitting ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}
