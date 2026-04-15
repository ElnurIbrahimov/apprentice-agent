/**
 * FlashcardsPanel — due cards + quality buttons (SM-2 spaced repetition).
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Layers, RefreshCw, Loader2 } from 'lucide-react';
import { tools } from '../api/client';
import type { Flashcard } from '../api/types';

export default function FlashcardsPanel() {
  const [due, setDue] = useState<number>(0);
  const [card, setCard] = useState<Flashcard | null>(null);
  const [showBack, setShowBack] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [stats, setStats] = useState<{ total_cards: number; due_today: number } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [d, s] = await Promise.all([
        tools.flashcards.due(),
        tools.flashcards.stats(),
      ]);
      setDue(d.due_count);
      setCard(d.next_card || null);
      setShowBack(false);
      setStats({ total_cards: s.total_cards, due_today: s.due_today });
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const answer = useCallback(async (quality: number) => {
    if (!card || busy) return;
    setBusy(true);
    try {
      await tools.flashcards.answer(card.id, quality);
    } catch { /* silent */ }
    setBusy(false);
    load();
  }, [card, busy, load]);

  return (
    <div className="panel-scroll-root" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Layers size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>Flashcards</span>
        <button onClick={load} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
          <RefreshCw size={12} />
        </button>
      </div>

      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <MiniStat label="Total" value={stats.total_cards} />
          <MiniStat label="Due today" value={stats.due_today} />
        </div>
      )}

      {loading && <div style={{ fontSize: 11, color: 'var(--mu)' }}>Loading…</div>}

      {!loading && !card && due === 0 && (
        <div style={{ fontSize: 11, color: 'var(--mu)', textAlign: 'center', padding: 20 }}>
          No cards due. 🎉
        </div>
      )}

      {card && (
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 12, padding: 20, minHeight: 160, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ fontSize: 9, color: 'var(--mu)', textTransform: 'uppercase', marginBottom: 6 }}>
            {showBack ? 'Answer' : 'Question'} · {due} due
          </div>
          <div style={{ fontSize: 14, color: 'var(--tx)', lineHeight: 1.5, whiteSpace: 'pre-wrap', flex: 1 }}>
            {showBack ? card.back : card.front}
          </div>
          <div style={{ marginTop: 16 }}>
            {!showBack ? (
              <button
                onClick={() => setShowBack(true)}
                style={{ width: '100%', padding: '10px', background: 'var(--p)', border: 'none', borderRadius: 8, color: '#fff', fontSize: 12, fontWeight: 600, cursor: 'pointer' }}
              >
                Show answer
              </button>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 4 }}>
                {[0, 1, 2, 3, 4, 5].map((q) => (
                  <button
                    key={q}
                    onClick={() => answer(q)}
                    disabled={busy}
                    style={{
                      padding: '8px 0',
                      background: ['#ef4444', '#f87171', '#fb923c', '#fbbf24', '#84cc16', '#10b981'][q],
                      border: 'none',
                      borderRadius: 6,
                      color: '#fff',
                      fontSize: 14,
                      fontWeight: 700,
                      cursor: busy ? 'not-allowed' : 'pointer',
                      opacity: busy ? 0.5 : 1,
                    }}
                    title={['Again', 'Hard', 'Difficult', 'Good', 'Easy', 'Perfect'][q]}
                  >
                    {busy ? <Loader2 size={11} className="spin" /> : q}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: number }) {
  return (
    <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 8, padding: '10px 12px', textAlign: 'center' }}>
      <div style={{ fontSize: 9, color: 'var(--mu)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--tx)' }}>{value}</div>
    </div>
  );
}
