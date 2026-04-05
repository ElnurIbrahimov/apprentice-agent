import { useState, useRef, useEffect } from 'react';
import { ChevronDownIcon, ArrowRightIcon } from '@heroicons/react/24/outline';
import { useChatStore } from '../store/chatStore';
import type { ToolId } from './ToolLauncher';

const SEND_TO_OPTIONS: Array<{ id: ToolId; label: string; icon: string }> = [
  { id: 'summary',   label: 'Summary',   icon: '📋' },
  { id: 'translate', label: 'Translate', icon: '🌐' },
  { id: 'write',     label: 'Write',     icon: '✏️' },
  { id: 'grammar',   label: 'Grammar',   icon: '📝' },
  { id: 'research',  label: 'Research',  icon: '🔬' },
];

interface SendToMenuProps {
  content: string;
  sourceToolId: ToolId;
}

export function SendToMenu({ content, sourceToolId }: SendToMenuProps) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  if (!content || content.trim().length < 20) return null;

  const options = SEND_TO_OPTIONS.filter(o => o.id !== sourceToolId);

  return (
    <div ref={menuRef} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text px-2 py-1 rounded-md transition-colors"
        style={{ background: 'var(--border-subtle)' }}
        title="Send output to another tool"
      >
        <ArrowRightIcon className="w-3 h-3" />
        Send to
        <ChevronDownIcon className="w-2.5 h-2.5 opacity-50" />
      </button>
      {open && (
        <div className="absolute bottom-full right-0 mb-1 w-40 rounded-lg border border-chat-border shadow-xl py-1 z-50" style={{ background: 'var(--surface-1)' }}>
          {options.map(opt => (
            <button
              key={opt.id}
              onClick={() => {
                useChatStore.getState().setToolPrefill({ toolId: opt.id, query: content });
                document.dispatchEvent(new CustomEvent('aura:tool-open', { detail: { toolId: opt.id } }));
                setOpen(false);
              }}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-chat-text-secondary hover:text-chat-text hover:bg-white/5 text-left transition-colors"
            >
              <span>{opt.icon}</span>
              <span>{opt.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}