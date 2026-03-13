import React from 'react';
import type { Message } from '../types';
import { md } from '../markdown';

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex mb-3 justify-end">
        <div style={{
          background: '#fff',
          color: '#000',
          padding: '9px 16px',
          borderRadius: '18px 18px 3px 18px',
          fontSize: '12.5px',
          fontWeight: 500,
          maxWidth: '82%',
          lineHeight: 1.55,
          boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-2.5 mb-4">
      <div
        className="flex-shrink-0 flex items-center justify-center"
        style={{
          width: 28, height: 28, borderRadius: 7, marginTop: 2,
          background: 'linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.02) 100%)',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          fontSize: '11px', color: 'white', fontWeight: 600,
        }}
      >
        A
      </div>
      <div className="flex-1 min-w-0">
        <div
          className="md-body"
          style={{ fontSize: '12.5px', lineHeight: 1.65, color: 'var(--tx)' }}
          dangerouslySetInnerHTML={{ __html: md(message.text) }}
        />
        <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 4 }}>
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  );
}
