import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useStore } from '../store';
import { User, FileText, Sparkles, RotateCcw, Save, Check, Server, Key, ChevronDown, ChevronRight, Zap, Eye, EyeOff } from 'lucide-react';
import { getBackendUrl, setBackendUrl, setApiKey, API_KEY } from '../api';
import { connectWS, fetchStatus, resetWsRetry } from '../ws';

// Direct API provider definitions
const API_PROVIDERS = [
  { name: 'anthropic', display: 'Anthropic', placeholder: 'sk-ant-...' },
  { name: 'openai', display: 'OpenAI', placeholder: 'sk-...' },
  { name: 'gemini', display: 'Google Gemini', placeholder: 'AIza...' },
  { name: 'grok', display: 'xAI (Grok)', placeholder: 'xai-...' },
  { name: 'perplexity', display: 'Perplexity', placeholder: 'pplx-...' },
  { name: 'deepseek', display: 'DeepSeek', placeholder: 'sk-...' },
  { name: 'minimax', display: 'MiniMax', placeholder: 'eyJ...' },
  { name: 'qwen', display: 'Qwen (Alibaba)', placeholder: 'sk-...' },
  { name: 'kimi', display: 'Kimi (Moonshot)', placeholder: 'sk-...' },
  { name: 'glm', display: 'GLM (Zhipu)', placeholder: '' },
] as const;

const STYLE_PRESETS: { label: string; icon: string; instructions: string }[] = [
  {
    label: 'Concise',
    icon: '',
    instructions:
      'Be concise and to the point. Use short sentences. Avoid filler words, unnecessary explanations, and over-qualifying. Get straight to the answer.',
  },
  {
    label: 'Detailed',
    icon: '',
    instructions:
      'Provide thorough, detailed responses. Include context, examples, and edge cases. Explain the reasoning behind your answers. Be comprehensive but organized.',
  },
  {
    label: 'Technical',
    icon: '',
    instructions:
      'Respond in a precise, technical manner. Use proper terminology. Include code examples, specs, and technical details. Assume I have engineering knowledge.',
  },
  {
    label: 'Casual',
    icon: '',
    instructions:
      'Keep it casual and conversational. Use simple language, be friendly, and feel free to use humor. Explain things like you would to a friend.',
  },
  {
    label: 'Creative',
    icon: '',
    instructions:
      'Be creative and expressive. Use vivid language, analogies, and metaphors. Think outside the box. Offer unique perspectives and imaginative solutions.',
  },
];

const MAX_CHARS = 2000;

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  fontSize: 13,
  borderRadius: 8,
  border: '1px solid var(--b2)',
  background: 'var(--glass)',
  color: 'var(--fg)',
  outline: 'none',
  fontFamily: 'inherit',
  transition: 'border-color 0.2s',
};

export default function SettingsPanel() {
  const { customInstructions, userName, setCustomInstructions, setUserName } = useStore();

  const [localName, setLocalName] = useState(userName);
  const [localInstructions, setLocalInstructions] = useState(customInstructions);
  const [saved, setSaved] = useState(false);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Backend connection state
  const [localBackendUrl, setLocalBackendUrl] = useState('');
  const [localApiKey, setLocalApiKey] = useState('');
  const [connSaved, setConnSaved] = useState(false);
  const [connTesting, setConnTesting] = useState(false);
  const [connStatus, setConnStatus] = useState<'idle' | 'ok' | 'fail'>('idle');
  const connTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load backend URL and API key on mount
  useEffect(() => {
    const current = getBackendUrl();
    setLocalBackendUrl(current);
    setLocalApiKey(API_KEY || '');
  }, []);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (savedTimerRef.current) clearTimeout(savedTimerRef.current);
      if (connTimerRef.current) clearTimeout(connTimerRef.current);
    };
  }, []);

  // Sync from store when it loads from storage (async)
  useEffect(() => {
    setLocalName(userName);
  }, [userName]);

  useEffect(() => {
    setLocalInstructions(customInstructions);
  }, [customInstructions]);

  // Detect which preset matches (if any)
  useEffect(() => {
    const match = STYLE_PRESETS.find((p) => p.instructions === localInstructions);
    setActivePreset(match ? match.label : null);
  }, [localInstructions]);

  const handleSave = () => {
    setUserName(localName.trim());
    setCustomInstructions(localInstructions.trim());
    setSaved(true);
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current);
    savedTimerRef.current = setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setLocalName('');
    setLocalInstructions('');
    setUserName('');
    setCustomInstructions('');
    setActivePreset(null);
  };

  const handlePreset = (preset: (typeof STYLE_PRESETS)[number]) => {
    if (activePreset === preset.label) {
      // Toggle off
      setLocalInstructions('');
      setActivePreset(null);
    } else {
      setLocalInstructions(preset.instructions);
      setActivePreset(preset.label);
    }
  };

  const handleTestConnection = async () => {
    setConnTesting(true);
    setConnStatus('idle');
    const testUrl = localBackendUrl.trim().replace(/\/+$/, '') || 'http://localhost:8000';
    try {
      const headers: Record<string, string> = {};
      if (localApiKey.trim()) headers['X-API-Key'] = localApiKey.trim();
      const r = await fetch(`${testUrl}/api/status`, {
        signal: AbortSignal.timeout(8000),
        headers,
      });
      if (r.ok) {
        setConnStatus('ok');
      } else {
        setConnStatus('fail');
      }
    } catch {
      setConnStatus('fail');
    } finally {
      setConnTesting(false);
      if (connTimerRef.current) clearTimeout(connTimerRef.current);
      connTimerRef.current = setTimeout(() => setConnStatus('idle'), 4000);
    }
  };

  const handleSaveConnection = () => {
    setBackendUrl(localBackendUrl.trim());
    setApiKey(localApiKey.trim());
    setConnSaved(true);
    if (connTimerRef.current) clearTimeout(connTimerRef.current);
    connTimerRef.current = setTimeout(() => setConnSaved(false), 2000);

    // Close existing WebSocket and reconnect with new URL
    const { ws } = useStore.getState();
    if (ws) {
      ws.close();
    }
    // Reset retry counters so reconnect starts fresh for the new server
    resetWsRetry();
    useStore.getState().setBackendStatus('connecting');
    // Small delay for close to propagate, then reconnect
    setTimeout(() => {
      fetchStatus();
      connectWS();
    }, 500);
  };

  // ChatGPT token state
  const [gptRefresh, setGptRefresh] = useState('');
  const [gptAccountId, setGptAccountId] = useState('');
  const [gptSaving, setGptSaving] = useState(false);
  const [gptStatus, setGptStatus] = useState<'idle' | 'ok' | 'fail'>('idle');
  const [gptMsg, setGptMsg] = useState('');
  const [gptAuthenticated, setGptAuthenticated] = useState<boolean | null>(null);
  const gptTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Check ChatGPT auth status on mount
  useEffect(() => {
    const checkGpt = async () => {
      try {
        const base = getBackendUrl().replace(/\/+$/, '');
        const headers: Record<string, string> = {};
        if (API_KEY) headers['X-API-Key'] = API_KEY;
        const r = await fetch(`${base}/api/auth/chatgpt/status`, { headers, signal: AbortSignal.timeout(5000) });
        if (r.ok) {
          const data = await r.json();
          setGptAuthenticated(data.authenticated);
        }
      } catch { /* ignore */ }
    };
    checkGpt();
    return () => { if (gptTimerRef.current) clearTimeout(gptTimerRef.current); };
  }, []);

  const handleSaveGptToken = async () => {
    if (!gptRefresh.trim()) return;
    setGptSaving(true);
    setGptStatus('idle');
    setGptMsg('');
    try {
      const base = getBackendUrl().replace(/\/+$/, '');
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (API_KEY) headers['X-API-Key'] = API_KEY;
      const r = await fetch(`${base}/api/auth/chatgpt/set-token`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ refresh: gptRefresh.trim(), account_id: gptAccountId.trim() }),
        signal: AbortSignal.timeout(30000),
      });
      const data = await r.json();
      if (data.success) {
        setGptStatus('ok');
        setGptMsg(data.message || 'Token saved');
        setGptAuthenticated(true);
        setGptRefresh('');
        setGptAccountId('');
      } else {
        setGptStatus('fail');
        setGptMsg(data.error || 'Failed to save token');
      }
    } catch (e: any) {
      setGptStatus('fail');
      setGptMsg(e.message || 'Request failed');
    } finally {
      setGptSaving(false);
      if (gptTimerRef.current) clearTimeout(gptTimerRef.current);
      gptTimerRef.current = setTimeout(() => { setGptStatus('idle'); setGptMsg(''); }, 6000);
    }
  };

  // API Providers state
  const [providerKeys, setProviderKeys] = useState<Record<string, string>>({});
  const [providerStatus, setProviderStatus] = useState<Record<string, boolean>>({});
  const [providerExpanded, setProviderExpanded] = useState<string | null>(null);
  const [providerShowKey, setProviderShowKey] = useState<Record<string, boolean>>({});
  const [providerSaving, setProviderSaving] = useState<string | null>(null);
  const [providerMsg, setProviderMsg] = useState<Record<string, { ok: boolean; text: string }>>({});

  // Load provider status on mount
  useEffect(() => {
    const fetchProviders = async () => {
      try {
        const base = getBackendUrl().replace(/\/+$/, '');
        const headers: Record<string, string> = {};
        if (API_KEY) headers['X-API-Key'] = API_KEY;
        const r = await fetch(`${base}/api/providers`, { headers, signal: AbortSignal.timeout(5000) });
        if (r.ok) {
          const data = await r.json();
          const statusMap: Record<string, boolean> = {};
          for (const p of data.providers || []) {
            statusMap[p.name] = p.configured;
          }
          setProviderStatus(statusMap);
        }
      } catch { /* ignore */ }
    };
    fetchProviders();
  }, []);

  const handleSaveProviderKey = useCallback(async (name: string) => {
    const key = providerKeys[name]?.trim();
    if (!key) return;
    setProviderSaving(name);
    setProviderMsg((prev) => ({ ...prev, [name]: undefined as any }));
    try {
      const base = getBackendUrl().replace(/\/+$/, '');
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (API_KEY) headers['X-API-Key'] = API_KEY;
      const r = await fetch(`${base}/api/providers/${name}/key`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ key }),
        signal: AbortSignal.timeout(10000),
      });
      const data = await r.json();
      if (data.ok) {
        setProviderStatus((prev) => ({ ...prev, [name]: true }));
        setProviderKeys((prev) => ({ ...prev, [name]: '' }));
        setProviderMsg((prev) => ({ ...prev, [name]: { ok: true, text: 'Key saved' } }));
        // Reload models to pick up newly available provider models
        useStore.getState().loadModels();
      } else {
        setProviderMsg((prev) => ({ ...prev, [name]: { ok: false, text: 'Failed to save' } }));
      }
    } catch (e: any) {
      setProviderMsg((prev) => ({ ...prev, [name]: { ok: false, text: e.message || 'Request failed' } }));
    } finally {
      setProviderSaving(null);
      setTimeout(() => setProviderMsg((prev) => ({ ...prev, [name]: undefined as any })), 4000);
    }
  }, [providerKeys]);

  const handleRemoveProviderKey = useCallback(async (name: string) => {
    try {
      const base = getBackendUrl().replace(/\/+$/, '');
      const headers: Record<string, string> = {};
      if (API_KEY) headers['X-API-Key'] = API_KEY;
      await fetch(`${base}/api/providers/${name}/key`, { method: 'DELETE', headers, signal: AbortSignal.timeout(10000) });
      setProviderStatus((prev) => ({ ...prev, [name]: false }));
      setProviderMsg((prev) => ({ ...prev, [name]: { ok: true, text: 'Key removed' } }));
      setTimeout(() => setProviderMsg((prev) => ({ ...prev, [name]: undefined as any })), 4000);
    } catch { /* ignore */ }
  }, []);

  const charCount = localInstructions.length;
  const isDirty = localName !== userName || localInstructions !== customInstructions;

  const isRemote = !!(localBackendUrl.trim() && !localBackendUrl.includes('localhost') && !localBackendUrl.includes('127.0.0.1'));

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto px-4 py-4 panel-scroll-root" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* Section: Backend Connection */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <Server size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              BACKEND CONNECTION
            </h3>
          </div>
          <label
            htmlFor="settings-backend-url"
            style={{ display: 'block', fontSize: 11, color: 'var(--fg2)', marginBottom: 6 }}
          >
            Server URL
          </label>
          <input
            id="settings-backend-url"
            type="url"
            value={localBackendUrl}
            onChange={(e) => setLocalBackendUrl(e.target.value)}
            placeholder="http://localhost:8000 (default)"
            maxLength={200}
            style={inputStyle}
            onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
            onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
          />
          <p style={{ margin: '6px 0 0', fontSize: 10, color: 'var(--fg3)' }}>
            Leave empty for localhost. For remote servers, use your full URL (e.g. https://aura.example.com).
          </p>

          {/* API Key field */}
          <label
            htmlFor="settings-api-key"
            style={{ display: 'block', fontSize: 11, color: 'var(--fg2)', marginBottom: 6, marginTop: 12 }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Key size={12} />
              API Key
            </span>
          </label>
          <input
            id="settings-api-key"
            type="password"
            value={localApiKey}
            onChange={(e) => setLocalApiKey(e.target.value)}
            placeholder={isRemote ? 'Required for remote servers' : 'Not needed for localhost'}
            maxLength={200}
            style={inputStyle}
            onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
            onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
          />
          <p style={{ margin: '6px 0 0', fontSize: 10, color: isRemote && !localApiKey.trim() ? 'var(--err, #f87171)' : 'var(--fg3)' }}>
            {isRemote && !localApiKey.trim()
              ? 'Warning: Remote server without API key is insecure.'
              : 'Matches AURA_API_KEY on the server. Required when AURA_API_AUTH_ENABLED=true.'}
          </p>

          {/* Connection action buttons */}
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <button
              onClick={handleTestConnection}
              disabled={connTesting}
              style={{
                padding: '7px 14px',
                fontSize: 12,
                fontWeight: 500,
                borderRadius: 8,
                border: '1px solid var(--b2)',
                background: connStatus === 'ok' ? 'rgba(52, 211, 153, 0.15)' : connStatus === 'fail' ? 'rgba(248, 113, 113, 0.15)' : 'var(--glass)',
                color: connStatus === 'ok' ? '#34d399' : connStatus === 'fail' ? '#f87171' : 'var(--fg)',
                cursor: connTesting ? 'wait' : 'pointer',
                fontFamily: 'inherit',
                transition: 'all 0.2s',
                outline: 'none',
              }}
            >
              {connTesting ? 'Testing...' : connStatus === 'ok' ? 'Connected' : connStatus === 'fail' ? 'Failed' : 'Test Connection'}
            </button>
            <button
              onClick={handleSaveConnection}
              style={{
                padding: '7px 14px',
                fontSize: 12,
                fontWeight: 600,
                borderRadius: 8,
                border: 'none',
                background: connSaved
                  ? 'rgba(52, 211, 153, 0.2)'
                  : 'linear-gradient(135deg, var(--p), var(--pl))',
                color: connSaved ? '#34d399' : 'white',
                cursor: 'pointer',
                fontFamily: 'inherit',
                transition: 'all 0.25s',
              }}
            >
              {connSaved ? 'Saved & Reconnecting' : 'Save & Reconnect'}
            </button>
          </div>
        </section>

        {/* Section: ChatGPT Token */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <Key size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              CHATGPT TOKEN
            </h3>
            {gptAuthenticated === true && (
              <span style={{ fontSize: 10, color: '#34d399', fontWeight: 500, marginLeft: 'auto' }}>Authenticated</span>
            )}
            {gptAuthenticated === false && (
              <span style={{ fontSize: 10, color: '#f87171', fontWeight: 500, marginLeft: 'auto' }}>Not set</span>
            )}
          </div>
          <p style={{ margin: '0 0 10px', fontSize: 11, color: 'var(--fg2)' }}>
            Set your ChatGPT refresh token to enable chatgpt: models on the server.
          </p>
          <label
            htmlFor="settings-gpt-refresh"
            style={{ display: 'block', fontSize: 11, color: 'var(--fg2)', marginBottom: 6 }}
          >
            Refresh Token
          </label>
          <textarea
            id="settings-gpt-refresh"
            value={gptRefresh}
            onChange={(e) => setGptRefresh(e.target.value)}
            placeholder="Paste your ChatGPT refresh token here (starts with rt_...)"
            rows={3}
            style={{
              ...inputStyle,
              fontSize: 11,
              fontFamily: 'monospace',
              resize: 'vertical',
              minHeight: 60,
              wordBreak: 'break-all' as const,
            }}
            onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
            onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
          />
          <label
            htmlFor="settings-gpt-account"
            style={{ display: 'block', fontSize: 11, color: 'var(--fg2)', marginBottom: 6, marginTop: 10 }}
          >
            Account ID (optional)
          </label>
          <input
            id="settings-gpt-account"
            type="text"
            value={gptAccountId}
            onChange={(e) => setGptAccountId(e.target.value)}
            placeholder="e.g. 92dff04b-90a9-4dfa-..."
            maxLength={100}
            style={{ ...inputStyle, fontSize: 12, fontFamily: 'monospace' }}
            onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
            onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
          />
          <p style={{ margin: '6px 0 0', fontSize: 10, color: 'var(--fg3)' }}>
            The token is sent to the server and saved there. It never leaves the server after that.
          </p>

          {gptMsg && (
            <p style={{ margin: '8px 0 0', fontSize: 11, color: gptStatus === 'ok' ? '#34d399' : '#f87171' }}>
              {gptMsg}
            </p>
          )}

          <div style={{ marginTop: 12 }}>
            <button
              onClick={handleSaveGptToken}
              disabled={gptSaving || !gptRefresh.trim()}
              style={{
                padding: '7px 18px',
                fontSize: 12,
                fontWeight: 600,
                borderRadius: 8,
                border: 'none',
                background: gptStatus === 'ok'
                  ? 'rgba(52, 211, 153, 0.2)'
                  : gptRefresh.trim()
                    ? 'linear-gradient(135deg, var(--p), var(--pl))'
                    : 'var(--glass)',
                color: gptStatus === 'ok' ? '#34d399' : gptRefresh.trim() ? 'white' : 'var(--fg3)',
                cursor: gptSaving ? 'wait' : gptRefresh.trim() ? 'pointer' : 'default',
                fontFamily: 'inherit',
                transition: 'all 0.25s',
                opacity: !gptRefresh.trim() && gptStatus === 'idle' ? 0.5 : 1,
              }}
            >
              {gptSaving ? 'Saving...' : gptStatus === 'ok' ? 'Token Saved' : 'Save Token to Server'}
            </button>
          </div>
        </section>

        {/* Section: API Providers */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <Zap size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              API PROVIDERS
            </h3>
            <span style={{ fontSize: 10, color: 'var(--fg3)', marginLeft: 'auto' }}>
              {Object.values(providerStatus).filter(Boolean).length} configured
            </span>
          </div>
          <p style={{ margin: '0 0 10px', fontSize: 11, color: 'var(--fg2)' }}>
            Add API keys for direct model access. Models appear in the picker once a key is set.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {API_PROVIDERS.map((prov) => {
              const isExpanded = providerExpanded === prov.name;
              const isConfigured = providerStatus[prov.name] || false;
              const isSaving = providerSaving === prov.name;
              const msg = providerMsg[prov.name];
              const showKey = providerShowKey[prov.name] || false;
              return (
                <div key={prov.name} style={{
                  border: '1px solid var(--b2)',
                  borderRadius: 8,
                  background: 'var(--glass)',
                  overflow: 'hidden',
                }}>
                  <button
                    onClick={() => setProviderExpanded(isExpanded ? null : prov.name)}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '8px 12px',
                      fontSize: 12,
                      fontWeight: 500,
                      border: 'none',
                      background: 'transparent',
                      color: 'var(--fg)',
                      cursor: 'pointer',
                      fontFamily: 'inherit',
                      textAlign: 'left',
                    }}
                  >
                    {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    <span style={{ flex: 1 }}>{prov.display}</span>
                    <span style={{
                      fontSize: 9,
                      padding: '2px 6px',
                      borderRadius: 10,
                      background: isConfigured ? 'rgba(52, 211, 153, 0.15)' : 'rgba(148, 163, 184, 0.1)',
                      color: isConfigured ? '#34d399' : 'var(--fg3)',
                      fontWeight: 600,
                    }}>
                      {isConfigured ? 'Active' : 'Not set'}
                    </span>
                  </button>
                  {isExpanded && (
                    <div style={{ padding: '0 12px 10px' }}>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <div style={{ position: 'relative', flex: 1 }}>
                          <input
                            type={showKey ? 'text' : 'password'}
                            value={providerKeys[prov.name] || ''}
                            onChange={(e) => setProviderKeys((prev) => ({ ...prev, [prov.name]: e.target.value }))}
                            placeholder={prov.placeholder || 'Paste API key'}
                            maxLength={500}
                            style={{ ...inputStyle, fontSize: 11, fontFamily: 'monospace', paddingRight: 32 }}
                            onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
                            onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
                          />
                          <button
                            onClick={() => setProviderShowKey((prev) => ({ ...prev, [prov.name]: !showKey }))}
                            style={{
                              position: 'absolute', right: 6, top: '50%', transform: 'translateY(-50%)',
                              background: 'none', border: 'none', cursor: 'pointer', padding: 2,
                              color: 'var(--fg3)',
                            }}
                          >
                            {showKey ? <EyeOff size={13} /> : <Eye size={13} />}
                          </button>
                        </div>
                        <button
                          onClick={() => handleSaveProviderKey(prov.name)}
                          disabled={isSaving || !(providerKeys[prov.name]?.trim())}
                          style={{
                            padding: '7px 12px',
                            fontSize: 11,
                            fontWeight: 600,
                            borderRadius: 6,
                            border: 'none',
                            background: providerKeys[prov.name]?.trim()
                              ? 'linear-gradient(135deg, var(--p), var(--pl))'
                              : 'var(--glass)',
                            color: providerKeys[prov.name]?.trim() ? 'white' : 'var(--fg3)',
                            cursor: isSaving ? 'wait' : providerKeys[prov.name]?.trim() ? 'pointer' : 'default',
                            fontFamily: 'inherit',
                            whiteSpace: 'nowrap',
                            opacity: !providerKeys[prov.name]?.trim() ? 0.5 : 1,
                          }}
                        >
                          {isSaving ? '...' : 'Save'}
                        </button>
                      </div>
                      {isConfigured && (
                        <button
                          onClick={() => handleRemoveProviderKey(prov.name)}
                          style={{
                            marginTop: 6,
                            padding: '4px 10px',
                            fontSize: 10,
                            border: '1px solid rgba(248, 113, 113, 0.3)',
                            borderRadius: 5,
                            background: 'transparent',
                            color: '#f87171',
                            cursor: 'pointer',
                            fontFamily: 'inherit',
                          }}
                        >
                          Remove key
                        </button>
                      )}
                      {msg && (
                        <p style={{ margin: '6px 0 0', fontSize: 10, color: msg.ok ? '#34d399' : '#f87171' }}>
                          {msg.text}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* Section 1: Your Profile */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <User size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              YOUR PROFILE
            </h3>
          </div>
          <label
            htmlFor="settings-name"
            style={{ display: 'block', fontSize: 11, color: 'var(--fg2)', marginBottom: 6 }}
          >
            Name
          </label>
          <input
            id="settings-name"
            type="text"
            value={localName}
            onChange={(e) => setLocalName(e.target.value)}
            placeholder="Your name"
            maxLength={100}
            style={inputStyle}
            onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
            onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
          />
          <p style={{ margin: '6px 0 0', fontSize: 10, color: 'var(--fg3)' }}>
            AURA will use your name to personalize responses.
          </p>
        </section>

        {/* Section 2: Custom Instructions */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <FileText size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              CUSTOM INSTRUCTIONS
            </h3>
          </div>
          <div style={{ position: 'relative' }}>
            <textarea
              value={localInstructions}
              onChange={(e) => {
                if (e.target.value.length <= MAX_CHARS) {
                  setLocalInstructions(e.target.value);
                }
              }}
              placeholder="Tell AURA how to respond. Example: 'Always be concise. Use bullet points. Explain like I'm an expert.'"
              rows={6}
              style={{
                width: '100%',
                padding: '10px 12px',
                paddingBottom: 28,
                fontSize: 12.5,
                lineHeight: 1.5,
                borderRadius: 8,
                border: '1px solid var(--b2)',
                background: 'var(--glass)',
                color: 'var(--fg)',
                outline: 'none',
                resize: 'vertical',
                fontFamily: 'inherit',
                minHeight: 120,
                transition: 'border-color 0.2s',
              }}
              onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
              onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
            />
            <span
              style={{
                position: 'absolute',
                bottom: 8,
                right: 12,
                fontSize: 10,
                color: charCount > MAX_CHARS * 0.9 ? 'var(--err, #f87171)' : 'var(--fg3)',
                pointerEvents: 'none',
              }}
            >
              {charCount}/{MAX_CHARS}
            </span>
          </div>
          <p style={{ margin: '6px 0 0', fontSize: 10, color: 'var(--fg3)' }}>
            These instructions are prepended to every message you send.
          </p>
        </section>

        {/* Section 3: Response Style Presets */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <Sparkles size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              RESPONSE STYLE
            </h3>
          </div>
          <p style={{ margin: '0 0 10px', fontSize: 11, color: 'var(--fg2)' }}>
            Quick presets — click to fill the instructions above.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {STYLE_PRESETS.map((preset) => {
              const isActive = activePreset === preset.label;
              return (
                <button
                  key={preset.label}
                  onClick={() => handlePreset(preset)}
                  style={{
                    padding: '6px 14px',
                    fontSize: 12,
                    fontWeight: 500,
                    borderRadius: 20,
                    border: isActive ? '1px solid var(--pl)' : '1px solid var(--b2)',
                    background: isActive ? 'rgba(124, 58, 237, 0.15)' : 'var(--glass)',
                    color: isActive ? 'var(--pl)' : 'var(--fg)',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    transition: 'all 0.2s',
                    outline: 'none',
                  }}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </section>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: 10, paddingTop: 4, paddingBottom: 16 }}>
          <button
            onClick={handleSave}
            disabled={!isDirty && !saved}
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
              padding: '10px 0',
              fontSize: 13,
              fontWeight: 600,
              borderRadius: 10,
              border: 'none',
              background: saved
                ? 'rgba(52, 211, 153, 0.2)'
                : isDirty
                  ? 'linear-gradient(135deg, var(--p), var(--pl))'
                  : 'var(--glass)',
              color: saved ? '#34d399' : isDirty ? 'white' : 'var(--fg3)',
              cursor: isDirty ? 'pointer' : 'default',
              fontFamily: 'inherit',
              transition: 'all 0.25s',
              opacity: !isDirty && !saved ? 0.5 : 1,
            }}
          >
            {saved ? <Check size={15} /> : <Save size={15} />}
            {saved ? 'Saved' : 'Save'}
          </button>
          <button
            onClick={handleReset}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
              padding: '10px 18px',
              fontSize: 13,
              fontWeight: 500,
              borderRadius: 10,
              border: '1px solid var(--b2)',
              background: 'transparent',
              color: 'var(--fg2)',
              cursor: 'pointer',
              fontFamily: 'inherit',
              transition: 'all 0.2s',
            }}
          >
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
