import React, { useRef, useState, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';
import { useStore } from '../store';
import { HTTP } from '../api';

interface Props {
  featureKey: string;
}

export default function ModelPill({ featureKey }: Props) {
  const { featureModels, setModel, mdlCloudList, mdlLocalList, setMdlLists } = useStore();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  const current = featureModels[featureKey] || null;
  const displayName = current ? current.replace(/:cloud$/, '') : 'Auto';

  const loadModels = async () => {
    if (mdlCloudList.length || mdlLocalList.length) return;
    setLoading(true);
    try {
      const d = await fetch('http://localhost:11434/api/tags').then(r => r.json());
      const all: string[] = (d.models || []).map((m: any) => m.name);
      setMdlLists(all.filter(n => n.includes(':cloud')), all.filter(n => !n.includes(':cloud')));
    } catch {
      try {
        const d = await fetch(`${HTTP}/api/models/available`).then(r => r.json());
        setMdlLists((d.cloud || []).map((m: any) => m.name), (d.local || []).map((m: any) => m.name));
      } catch {
        // leave empty
      }
    }
    setLoading(false);
  };

  const toggle = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (open) { setOpen(false); return; }
    await loadModels();
    setOpen(true);
  };

  // Position dropdown + click-outside
  useEffect(() => {
    if (!open || !btnRef.current || !dropRef.current) return;
    const pRect = btnRef.current.getBoundingClientRect();
    const drop = dropRef.current;
    drop.style.right = Math.max(4, window.innerWidth - pRect.right) + 'px';
    drop.style.left = 'auto';
    drop.style.top = (pRect.bottom + 5) + 'px';
    drop.style.bottom = 'auto';
    // Flip upward if needed
    const dropRect = drop.getBoundingClientRect();
    if (window.innerHeight - pRect.bottom < dropRect.height + 16) {
      drop.style.top = 'auto';
      drop.style.bottom = (window.innerHeight - pRect.top + 5) + 'px';
    }

    const outside = (e: MouseEvent) => {
      if (!dropRef.current?.contains(e.target as Node) && !btnRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', outside);
    return () => document.removeEventListener('mousedown', outside);
  }, [open]);

  const pick = (model: string | null) => {
    setModel(featureKey, model);
    setOpen(false);
  };

  return (
    <span style={{ position: 'relative', display: 'inline-block', verticalAlign: 'middle' }}>
      <button
        ref={btnRef}
        onClick={toggle}
        className="flex items-center gap-1 px-2 py-1 transition-all duration-150"
        style={{
          background: 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-pill)',
          color: current ? 'var(--pl)' : 'var(--mu)',
          fontSize: '11px',
          cursor: 'pointer',
          fontFamily: 'inherit',
          gap: 4,
        }}
      >
        <span
          className="w-[6px] h-[6px] rounded-full flex-shrink-0"
          style={{ background: current ? 'var(--pl)' : 'var(--di)' }}
        />
        <span style={{ maxWidth: 90, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {displayName}
        </span>
        <ChevronDown size={9} style={{ opacity: 0.55, marginLeft: 1 }} />
      </button>

      {open && (
        <div
          ref={dropRef}
          className="mdl-drop open"
          style={{ position: 'fixed', minWidth: 180, maxWidth: 240 }}
        >
          <div
            className={`mdl-drop-item${!current ? ' on' : ''}`}
            onClick={() => pick(null)}
          >
            ⚡  Auto
          </div>
          {loading && (
            <div style={{ padding: '10px', fontSize: '11px', color: 'var(--mu)', textAlign: 'center' }}>
              Loading…
            </div>
          )}
          {!loading && !mdlCloudList.length && !mdlLocalList.length && (
            <div style={{ padding: '10px', fontSize: '11px', color: 'rgba(160,148,210,.6)', textAlign: 'center' }}>
              No models — is Ollama running?
            </div>
          )}
          {mdlCloudList.length > 0 && (
            <>
              <div className="mdl-drop-sec">☁  Cloud  ({mdlCloudList.length})</div>
              {mdlCloudList.map(m => (
                <div
                  key={m}
                  className={`mdl-drop-item${current === m ? ' on' : ''}`}
                  title={m}
                  onClick={() => pick(m)}
                >
                  {m.replace(/:cloud$/, '')}
                </div>
              ))}
            </>
          )}
          {mdlLocalList.length > 0 && (
            <>
              <div className="mdl-drop-sec">🖥  Local  ({mdlLocalList.length})</div>
              {mdlLocalList.map(m => (
                <div
                  key={m}
                  className={`mdl-drop-item${current === m ? ' on' : ''}`}
                  title={m}
                  onClick={() => pick(m)}
                >
                  {m}
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </span>
  );
}
