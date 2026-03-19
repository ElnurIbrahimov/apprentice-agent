import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { useStore } from '../store';
import type { PanelId } from '../types';
import {
  MessageSquare, Search, Languages, PenLine, CheckSquare,
  Database, Zap, FileText, Wrench, File, Mic,
  Radio, Camera, Youtube, FlaskConical, Calculator,
  Terminal, Code2, Image, BarChart2, Crosshair,
  Presentation, Globe, Bot, Cpu, Settings,
  Command,
} from 'lucide-react';

interface PanelEntry {
  id: PanelId;
  label: string;
  icon: React.ReactNode;
}

const ICON_SIZE = 18;

const PANELS: PanelEntry[] = [
  { id: 'chat', label: 'Chat', icon: <MessageSquare size={ICON_SIZE} /> },
  { id: 'search', label: 'Search', icon: <Search size={ICON_SIZE} /> },
  { id: 'translate', label: 'Translate', icon: <Languages size={ICON_SIZE} /> },
  { id: 'write', label: 'Write', icon: <PenLine size={ICON_SIZE} /> },
  { id: 'grammar', label: 'Grammar', icon: <CheckSquare size={ICON_SIZE} /> },
  { id: 'wisebase', label: 'Wisebase', icon: <Database size={ICON_SIZE} /> },
  { id: 'ask', label: 'Ask', icon: <Zap size={ICON_SIZE} /> },
  { id: 'summary', label: 'Summary', icon: <FileText size={ICON_SIZE} /> },
  { id: 'tools', label: 'Tools', icon: <Wrench size={ICON_SIZE} /> },
  { id: 'pdf', label: 'PDF', icon: <File size={ICON_SIZE} /> },
  { id: 'voice', label: 'Voice', icon: <Mic size={ICON_SIZE} /> },
  { id: 'record', label: 'Record', icon: <Radio size={ICON_SIZE} /> },
  { id: 'ocr', label: 'OCR', icon: <Camera size={ICON_SIZE} /> },
  { id: 'youtube', label: 'YouTube', icon: <Youtube size={ICON_SIZE} /> },
  { id: 'research', label: 'Research', icon: <FlaskConical size={ICON_SIZE} /> },
  { id: 'math', label: 'Math', icon: <Calculator size={ICON_SIZE} /> },
  { id: 'code', label: 'Code', icon: <Terminal size={ICON_SIZE} /> },
  { id: 'artifacts', label: 'Artifacts', icon: <Code2 size={ICON_SIZE} /> },
  { id: 'image', label: 'Image', icon: <Image size={ICON_SIZE} /> },
  { id: 'compare', label: 'Compare', icon: <BarChart2 size={ICON_SIZE} /> },
  { id: 'capture', label: 'Capture', icon: <Crosshair size={ICON_SIZE} /> },
  { id: 'slides', label: 'Slides', icon: <Presentation size={ICON_SIZE} /> },
  { id: 'webcreator', label: 'Web Creator', icon: <Globe size={ICON_SIZE} /> },
  { id: 'agent', label: 'Agent', icon: <Bot size={ICON_SIZE} /> },
  { id: 'models', label: 'Models', icon: <Cpu size={ICON_SIZE} /> },
  { id: 'settings', label: 'Settings', icon: <Settings size={ICON_SIZE} /> },
];

function fuzzyMatch(query: string, text: string): boolean {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  let qi = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) qi++;
  }
  return qi === q.length;
}

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const setPanel = useStore((s) => s.setPanel);

  // Filter panels by fuzzy search
  const filtered = useMemo(() => {
    if (!query.trim()) return PANELS;
    return PANELS.filter((p) => fuzzyMatch(query, p.label));
  }, [query]);

  // Clamp activeIdx when filtered list changes
  useEffect(() => {
    setActiveIdx(0);
  }, [filtered.length, query]);

  // Scroll active item into view
  useEffect(() => {
    if (!listRef.current) return;
    const active = listRef.current.children[activeIdx] as HTMLElement | undefined;
    active?.scrollIntoView({ block: 'nearest' });
  }, [activeIdx]);

  const close = useCallback(() => {
    setOpen(false);
    setQuery('');
    setActiveIdx(0);
  }, []);

  const selectPanel = useCallback(
    (id: PanelId) => {
      setPanel(id);
      close();
    },
    [setPanel, close],
  );

  // Global Ctrl+K listener
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        e.stopPropagation();
        setOpen((prev) => {
          if (prev) {
            // If already open, close and reset
            setQuery('');
            setActiveIdx(0);
            return false;
          }
          return true;
        });
      }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, []);

  // Auto-focus input when opened
  useEffect(() => {
    if (open) {
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Keyboard navigation inside the palette
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        close();
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIdx((i) => (i + 1 < filtered.length ? i + 1 : 0));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIdx((i) => (i - 1 >= 0 ? i - 1 : filtered.length - 1));
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        if (filtered[activeIdx]) selectPanel(filtered[activeIdx].id);
        return;
      }
    },
    [filtered, activeIdx, selectPanel, close],
  );

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '18vh',
        background: 'rgba(0, 0, 0, 0.45)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        animation: 'cp-fade-in 0.15s ease-out',
      }}
      onClick={close}
    >
      <style>{`
        @keyframes cp-fade-in {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes cp-scale-in {
          from { opacity: 0; transform: scale(0.98) translateY(-4px); }
          to   { opacity: 1; transform: scale(1) translateY(0); }
        }
      `}</style>

      <div
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
        style={{
          width: '100%',
          maxWidth: 400,
          background: 'var(--glass)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-lg)',
          boxShadow: 'var(--sh-lg)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          animation: 'cp-scale-in 0.18s ease-out',
        }}
      >
        {/* Search input */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '12px 14px',
            borderBottom: '1px solid var(--b1)',
          }}
        >
          <Command size={16} style={{ color: 'var(--mu)', flexShrink: 0 }} />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search panels..."
            spellCheck={false}
            autoComplete="off"
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: 'var(--tx)',
              fontSize: 14,
              fontFamily: 'inherit',
              caretColor: 'var(--pl)',
            }}
          />
          <kbd
            style={{
              fontSize: 11,
              color: 'var(--mu)',
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-sm)',
              padding: '2px 6px',
              lineHeight: 1.4,
              flexShrink: 0,
            }}
          >
            ESC
          </kbd>
        </div>

        {/* Results list */}
        <div
          ref={listRef}
          style={{
            maxHeight: 320,
            overflowY: 'auto',
            padding: '6px',
          }}
        >
          {filtered.length === 0 && (
            <div
              style={{
                padding: '20px 14px',
                textAlign: 'center',
                color: 'var(--mu)',
                fontSize: 13,
              }}
            >
              No panels found
            </div>
          )}
          {filtered.map((panel, idx) => {
            const isActive = idx === activeIdx;
            return (
              <div
                key={panel.id}
                onClick={() => selectPanel(panel.id)}
                onMouseEnter={() => setActiveIdx(idx)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '8px 10px',
                  borderRadius: 'var(--r-md)',
                  cursor: 'pointer',
                  color: isActive ? 'var(--tx)' : 'var(--mu)',
                  background: isActive ? 'var(--s2)' : 'transparent',
                  transition: 'background 0.1s, color 0.1s',
                }}
              >
                <span style={{ flexShrink: 0, display: 'flex', color: isActive ? 'var(--pl)' : 'var(--mu)' }}>
                  {panel.icon}
                </span>
                <span style={{ fontSize: 13, fontWeight: isActive ? 500 : 400 }}>
                  {panel.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
