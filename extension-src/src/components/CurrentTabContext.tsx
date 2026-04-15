/**
 * CurrentTabContext — dismissable card that shows the active tab's URL/title
 * when pageContextEnabled is on, with a toggle to include page context in the
 * next outgoing chat message.
 *
 * This surfaces the otherwise-invisible pageContext state that ChatPanel
 * already loads. If `pageContextEnabled` is off, the card renders nothing.
 */

import React, { useMemo } from 'react';
import { Globe, Eye, EyeOff } from 'lucide-react';
import { useStore } from '../store';

export default function CurrentTabContext() {
  const { pageContextEnabled, pageContext, setPageContextEnabled } = useStore();

  const display = useMemo(() => {
    if (!pageContext) return null;
    const title = pageContext.title || pageContext.url || 'Current page';
    let domain = '';
    try {
      domain = pageContext.url ? new URL(pageContext.url).hostname : '';
    } catch { /* silent */ }
    return { title, domain };
  }, [pageContext]);

  if (!pageContextEnabled && !pageContext) return null;

  return (
    <div
      style={{
        margin: '0 10px',
        padding: '6px 10px',
        background: 'var(--s2)',
        border: '1px solid var(--b1)',
        borderRadius: 8,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: 10,
        color: 'var(--mu)',
      }}
    >
      <Globe size={11} style={{ color: 'var(--pl)', flexShrink: 0 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {display?.title || 'No page loaded'}
        </div>
        {display?.domain && (
          <div style={{ fontSize: 9, color: 'var(--mu)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {display.domain}
          </div>
        )}
      </div>
      <button
        onClick={() => setPageContextEnabled(!pageContextEnabled)}
        title={pageContextEnabled ? 'Disable page context' : 'Enable page context'}
        aria-label="Toggle page context"
        style={{
          background: pageContextEnabled ? 'rgba(124, 58, 237, 0.2)' : 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 6,
          color: pageContextEnabled ? 'var(--pl)' : 'var(--mu)',
          cursor: 'pointer',
          padding: '3px 8px',
          fontSize: 10,
          display: 'flex',
          alignItems: 'center',
          gap: 3,
          flexShrink: 0,
        }}
      >
        {pageContextEnabled ? <Eye size={10} /> : <EyeOff size={10} />}
        {pageContextEnabled ? 'On' : 'Off'}
      </button>
    </div>
  );
}
