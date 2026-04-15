/**
 * EmailPanel — read-only inbox preview from /api/email/inbox.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Mail, RefreshCw } from 'lucide-react';
import { tools } from '../api/client';
import type { EmailMessage } from '../api/types';

export default function EmailPanel() {
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [provider, setProvider] = useState<string>('');
  const [account, setAccount] = useState<string>('');
  const [emails, setEmails] = useState<EmailMessage[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const s = await tools.email.status();
      // Backend can return {success: false, error: "…not loaded"} when the
      // email tool isn't installed. Treat either unset `configured` or an
      // explicit `success: false` as "not configured" for display.
      const raw = s as any;
      if (raw?.success === false) {
        setConfigured(false);
        setProvider('');
        setAccount('');
      } else {
        setConfigured(!!s.configured);
        setProvider(s.provider || '');
        setAccount(s.account || '');
        if (s.configured) {
          const r = await tools.email.inbox(30);
          setEmails(r.emails ?? []);
        }
      }
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="panel-scroll-root" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Mail size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>
          Email
          {account && <span style={{ color: 'var(--mu)', fontWeight: 400, marginLeft: 6, fontSize: 10 }}>{account}</span>}
        </span>
        <button onClick={load} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
          <RefreshCw size={12} />
        </button>
      </div>

      {loading && <div style={{ fontSize: 11, color: 'var(--mu)' }}>Loading…</div>}

      {configured === false && (
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 14, textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: 'var(--tx)', marginBottom: 6 }}>Email not configured</div>
          <div style={{ fontSize: 10, color: 'var(--mu)' }}>Set up email provider in backend config.</div>
        </div>
      )}

      {configured && emails.length === 0 && !loading && (
        <div style={{ fontSize: 11, color: 'var(--mu)', textAlign: 'center', padding: 20 }}>
          Inbox empty.
        </div>
      )}

      {emails.map((e, i) => (
        <div key={i} style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 8, padding: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
              {e.from}
            </span>
            <span style={{ fontSize: 9, color: 'var(--mu)', flexShrink: 0, marginLeft: 8 }}>
              {new Date(e.timestamp * 1000).toLocaleDateString()}
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--tx)', marginBottom: 3, fontWeight: 500 }}>{e.subject}</div>
          <div style={{ fontSize: 10, color: 'var(--mu)', lineHeight: 1.4, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
            {e.snippet}
          </div>
        </div>
      ))}
    </div>
  );
}
