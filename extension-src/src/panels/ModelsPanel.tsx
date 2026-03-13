import React, { useEffect, useState } from 'react';
import { useStore } from '../store';
import { HTTP } from '../api';
import { FEATURE_DEFS } from '../types';

export default function ModelsPanel() {
  const { featureModels, setModel, mdlCloudList, mdlLocalList, setMdlLists, activePanel } = useStore();
  const [chatgptModels, setChatgptModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const loadModels = async () => {
    const ollamaLoaded = mdlCloudList.length > 0 || mdlLocalList.length > 0;
    setLoading(true);

    // Fetch Ollama models (skip if already cached)
    if (!ollamaLoaded) {
      try {
        const d = await fetch('http://localhost:11434/api/tags').then(r => r.json());
        const all: string[] = (d.models || []).map((m: any) => m.name);
        setMdlLists(all.filter(n => n.includes(':cloud')), all.filter(n => !n.includes(':cloud')));
      } catch {
        try {
          const d = await fetch(`${HTTP}/api/models/available`).then(r => r.json());
          setMdlLists((d.cloud || []).map((m: any) => m.name), (d.local || []).map((m: any) => m.name));
        } catch {}
      }
    }

    // Always fetch ChatGPT models (separate from Ollama cache)
    if (chatgptModels.length === 0) {
      try {
        const authStatus = await fetch(`${HTTP}/api/auth/chatgpt/status`).then(r => r.json());
        if (authStatus.authenticated) {
          const modelsResp = await fetch(`${HTTP}/api/models/available`).then(r => r.json());
          setChatgptModels((modelsResp.chatgpt || []).map((m: any) => m.name || m));
        }
      } catch {}
    }

    setLoading(false);
  };

  useEffect(() => {
    if (activePanel === 'models') loadModels();
  }, [activePanel]);

  const allModels = [...chatgptModels, ...mdlCloudList, ...mdlLocalList];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--tx)', marginBottom: 4 }}>
          Per-Feature Models
        </div>
        <div style={{ fontSize: '11px', color: 'var(--mu)' }}>
          Assign a specific model to each feature. Leave as Auto to let the backend decide.
        </div>
      </div>

      {loading && (
        <div className="flex justify-center mt-8">
          <div className="dots"><span /><span /><span /></div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
        {FEATURE_DEFS.map(def => {
          const current = featureModels[def.key] || '';
          return (
            <div
              key={def.key}
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                padding: '10px 12px',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
              }}
            >
              <span style={{ fontSize: '16px', flexShrink: 0 }}>{def.icon}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '12.5px', fontWeight: 500 }}>{def.label}</div>
                <div style={{ fontSize: '11px', color: 'var(--mu)' }}>{def.desc}</div>
              </div>
              <select
                value={current}
                onChange={e => setModel(def.key, e.target.value || null)}
                style={{
                  background: 'var(--s3)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-sm)',
                  color: current ? 'var(--pl)' : 'var(--mu)',
                  fontSize: '11px',
                  padding: '4px 6px',
                  maxWidth: 130,
                  fontFamily: 'inherit',
                  cursor: 'pointer',
                }}
              >
                <option value="">⚡ Auto</option>
                {chatgptModels.length > 0 && (
                  <optgroup label="🟢 ChatGPT">
                    {chatgptModels.map(m => (
                      <option key={m} value={m}>{m.replace(/^chatgpt:/, '')}</option>
                    ))}
                  </optgroup>
                )}
                {mdlCloudList.length > 0 && (
                  <optgroup label="☁ Cloud">
                    {mdlCloudList.map(m => (
                      <option key={m} value={m}>{m.replace(/:cloud$/, '')}</option>
                    ))}
                  </optgroup>
                )}
                {mdlLocalList.length > 0 && (
                  <optgroup label="🖥 Local">
                    {mdlLocalList.map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </optgroup>
                )}
              </select>
            </div>
          );
        })}

        {!loading && allModels.length === 0 && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 24 }}>
            No models found — is Ollama running?
          </div>
        )}
      </div>
    </div>
  );
}
