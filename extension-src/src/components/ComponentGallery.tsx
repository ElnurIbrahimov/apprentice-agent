/**
 * ComponentGallery — browse, search, preview, and insert saved UI components.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  X, Search, Copy, Check, Trash2, Plus, Tag, FileCode2,
} from 'lucide-react';
import {
  componentLibrary,
  CATEGORIES,
  type UIComponent,
  type ComponentCategory,
} from '../utils/componentLibrary';

interface ComponentGalleryProps {
  open: boolean;
  onClose: () => void;
  onInsert?: (component: UIComponent) => void;
}

export default function ComponentGallery({ open, onClose, onInsert }: ComponentGalleryProps) {
  const [components, setComponents] = useState<UIComponent[]>([]);
  const [query, setQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<ComponentCategory | ''>('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const list = categoryFilter
      ? await componentLibrary.list(categoryFilter)
      : query
        ? await componentLibrary.search(query)
        : await componentLibrary.list();
    setComponents(list);
  }, [query, categoryFilter]);

  useEffect(() => {
    if (open) refresh();
  }, [open, refresh]);

  const handleDelete = async (id: string) => {
    await componentLibrary.delete(id);
    refresh();
  };

  const handleInsert = async (comp: UIComponent) => {
    await componentLibrary.incrementUsage(comp.id);
    onInsert?.(comp);
    onClose();
  };

  const copyHtml = async (html: string, id: string) => {
    try {
      await navigator.clipboard.writeText(html);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch { /* ok */ }
  };

  if (!open) return null;

  const previewComp = previewId ? components.find(c => c.id === previewId) : null;

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
          <FileCode2 size={15} style={{ color: 'var(--pl)' }} />
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)' }}>
            Component Library
          </span>
          <span style={{ fontSize: '10px', color: 'var(--mu)' }}>({components.length})</span>
          <span style={{ flex: 1 }} />
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', padding: 2, display: 'flex' }}
          >
            <X size={15} />
          </button>
        </div>

        {/* Search + Filter */}
        <div style={{ display: 'flex', gap: 6, padding: '8px 12px', borderBottom: '1px solid var(--b1)' }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 4, flex: 1,
            background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
            padding: '4px 8px',
          }}>
            <Search size={12} style={{ color: 'var(--mu)' }} />
            <input
              value={query}
              onChange={e => { setQuery(e.target.value); setCategoryFilter(''); }}
              placeholder="Search components..."
              style={{
                flex: 1, background: 'none', border: 'none', outline: 'none',
                color: 'var(--tx)', fontSize: '11px', fontFamily: 'inherit',
              }}
            />
          </div>
          <select
            value={categoryFilter}
            onChange={e => { setCategoryFilter(e.target.value as ComponentCategory | ''); setQuery(''); }}
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: 'var(--tx)', fontSize: '10.5px', padding: '4px 6px', fontFamily: 'inherit',
            }}
          >
            <option value="">All</option>
            {CATEGORIES.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
          </select>
        </div>

        {/* Grid */}
        <div style={{ flex: 1, overflow: 'auto', padding: '8px 12px' }}>
          {components.length === 0 ? (
            <div style={{ padding: '24px 12px', textAlign: 'center', color: 'var(--mu)', fontSize: '12px' }}>
              {query ? 'No components match your search.' : 'No saved components yet. Save components from Web Creator or Artifacts.'}
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 8 }}>
              {components.map(comp => (
                <div
                  key={comp.id}
                  onClick={() => setPreviewId(previewId === comp.id ? null : comp.id)}
                  style={{
                    background: previewId === comp.id ? 'var(--pg)' : 'var(--s2)',
                    border: `1px solid ${previewId === comp.id ? 'var(--p)' : 'var(--b1)'}`,
                    borderRadius: 'var(--r-md)', padding: '8px', cursor: 'pointer',
                    display: 'flex', flexDirection: 'column', gap: 6,
                  }}
                >
                  {/* Thumbnail or placeholder */}
                  <div style={{
                    height: 60, borderRadius: 'var(--r-sm)', overflow: 'hidden',
                    background: comp.thumbnail ? `url(${comp.thumbnail}) center/cover` : 'var(--s3)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    {!comp.thumbnail && <FileCode2 size={20} style={{ color: 'var(--mu)', opacity: 0.5 }} />}
                  </div>
                  <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {comp.name}
                  </div>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                    <Tag size={9} style={{ color: 'var(--mu)' }} />
                    <span style={{ fontSize: '9px', color: 'var(--mu)' }}>{comp.category}</span>
                    {comp.usageCount > 0 && (
                      <span style={{ fontSize: '9px', color: 'var(--mu)', marginLeft: 'auto' }}>{comp.usageCount}x</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Preview / Actions */}
        {previewComp && (
          <div style={{
            borderTop: '1px solid var(--b1)', padding: '10px 12px',
            maxHeight: 180, overflow: 'auto',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--tx)' }}>{previewComp.name}</span>
              <span style={{ fontSize: '10px', color: 'var(--mu)' }}>{previewComp.description}</span>
              <span style={{ flex: 1 }} />
              {onInsert && (
                <button
                  onClick={() => handleInsert(previewComp)}
                  style={{
                    background: 'var(--p)', color: '#fff', border: 'none',
                    borderRadius: 'var(--r-sm)', padding: '4px 10px', fontSize: '10px',
                    fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
                    display: 'flex', alignItems: 'center', gap: 4,
                  }}
                >
                  <Plus size={11} /> Insert
                </button>
              )}
              <button
                onClick={() => copyHtml(previewComp.html, previewComp.id)}
                style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                  color: 'var(--mu)', padding: '4px 8px', fontSize: '10px', cursor: 'pointer',
                  fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 4,
                }}
              >
                {copiedId === previewComp.id ? <Check size={11} style={{ color: '#3fb950' }} /> : <Copy size={11} />}
                Copy
              </button>
              <button
                onClick={() => { handleDelete(previewComp.id); setPreviewId(null); }}
                style={{
                  background: 'none', border: 'none', color: '#f85149',
                  cursor: 'pointer', padding: 2, display: 'flex',
                }}
              >
                <Trash2 size={13} />
              </button>
            </div>
            <pre style={{
              background: '#0d1117', border: '1px solid #30363d', borderRadius: 'var(--r-sm)',
              padding: '8px', fontSize: '10px', color: '#e6edf3', overflow: 'auto',
              maxHeight: 100, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {previewComp.html.slice(0, 2000)}{previewComp.html.length > 2000 ? '...' : ''}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
