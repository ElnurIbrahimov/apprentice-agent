import { useState, useEffect, useCallback } from 'react';
import {
  SunIcon, MoonIcon, ComputerDesktopIcon,
  EyeIcon, EyeSlashIcon, CheckIcon, XMarkIcon,
  KeyIcon, TrashIcon,
} from '@heroicons/react/24/outline';
import { useSettingsStore } from '../store/settingsStore';
import { toast } from './Toast';

// ─── Provider Definitions ────────────────────────────────────────────────────

interface ProviderDef {
  name: string;
  display: string;
  placeholder: string;
  description: string;
}

const TEXT_PROVIDERS: ProviderDef[] = [
  { name: 'anthropic', display: 'Anthropic', placeholder: 'sk-ant-...', description: 'Claude Opus, Sonnet, Haiku' },
  { name: 'openai', display: 'OpenAI', placeholder: 'sk-...', description: 'GPT-5.4, o3, DALL-E, Whisper' },
  { name: 'gemini', display: 'Google Gemini', placeholder: 'AIza...', description: 'Gemini 2.5 Pro/Flash, Imagen' },
  { name: 'grok', display: 'xAI (Grok)', placeholder: 'xai-...', description: 'Grok-3, Grok-3 Mini' },
  { name: 'mistral', display: 'Mistral AI', placeholder: 'Bearer...', description: 'Mistral Large, Codestral' },
  { name: 'cohere', display: 'Cohere', placeholder: 'Bearer...', description: 'Command R+, Embed, Rerank' },
  { name: 'perplexity', display: 'Perplexity', placeholder: 'pplx-...', description: 'Sonar Pro, online search' },
  { name: 'deepseek', display: 'DeepSeek', placeholder: 'sk-...', description: 'DeepSeek V3, Coder' },
  { name: 'minimax', display: 'MiniMax', placeholder: 'eyJ...', description: 'MiniMax-M2.7, 1M context' },
  { name: 'qwen', display: 'Qwen (Alibaba)', placeholder: 'sk-...', description: 'Qwen 3.5, Qwen-Coder' },
  { name: 'kimi', display: 'Kimi (Moonshot)', placeholder: 'sk-...', description: 'Kimi K2.5, 256K context' },
  { name: 'glm', display: 'GLM (Zhipu)', placeholder: 'Bearer...', description: 'GLM-5, CogView' },
  { name: 'groq', display: 'Groq', placeholder: 'gsk_...', description: 'Ultra-fast LPU inference' },
  { name: 'together', display: 'Together AI', placeholder: 'Bearer...', description: '100+ open models hosted' },
  { name: 'fireworks', display: 'Fireworks AI', placeholder: 'Bearer...', description: 'Fast open model inference' },
  { name: 'openrouter', display: 'OpenRouter', placeholder: 'sk-or-...', description: 'Routes to 200+ models' },
];

const IMAGE_PROVIDERS: ProviderDef[] = [
  { name: 'stability', display: 'Stability AI', placeholder: 'sk-...', description: 'Stable Diffusion 3, SDXL' },
  { name: 'falai', display: 'Fal.ai', placeholder: 'Bearer...', description: 'Fast Flux, SDXL inference' },
  { name: 'leonardo', display: 'Leonardo AI', placeholder: 'Bearer...', description: 'Leonardo Phoenix, fine-tuning' },
  { name: 'replicate', display: 'Replicate', placeholder: 'r8_...', description: 'Run any model via API' },
  { name: 'ideogram', display: 'Ideogram', placeholder: 'Bearer...', description: 'Strong text-in-image' },
  { name: 'recraft', display: 'Recraft AI', placeholder: 'Bearer...', description: 'Vector and raster image gen' },
];

const VIDEO_PROVIDERS: ProviderDef[] = [
  { name: 'runway', display: 'Runway', placeholder: 'Bearer...', description: 'Gen-3 Alpha video generation' },
  { name: 'luma', display: 'Luma AI', placeholder: 'Bearer...', description: 'Dream Machine, text-to-video' },
  { name: 'kling', display: 'Kling', placeholder: 'Bearer...', description: 'High-quality video gen' },
  { name: 'heygen', display: 'HeyGen', placeholder: 'Bearer...', description: 'AI avatar videos' },
  { name: 'did', display: 'D-ID', placeholder: 'Bearer...', description: 'Talking avatar videos' },
];

const AUDIO_PROVIDERS: ProviderDef[] = [
  { name: 'elevenlabs', display: 'ElevenLabs', placeholder: 'xi-...', description: 'TTS, voice cloning, STS' },
  { name: 'playht', display: 'PlayHT', placeholder: 'Bearer...', description: 'TTS, voice cloning' },
  { name: 'deepgram', display: 'Deepgram', placeholder: 'Bearer...', description: 'STT (best-in-class), TTS' },
  { name: 'assemblyai', display: 'AssemblyAI', placeholder: 'Bearer...', description: 'STT, audio intelligence' },
  { name: 'cartesia', display: 'Cartesia', placeholder: 'Bearer...', description: 'Ultra-low-latency TTS (Sonic)' },
];

const SEARCH_PROVIDERS: ProviderDef[] = [
  { name: 'tavily', display: 'Tavily', placeholder: 'tvly-...', description: 'AI-optimized search' },
  { name: 'brave_search', display: 'Brave Search', placeholder: 'Bearer...', description: 'Web search API' },
  { name: 'exa', display: 'Exa', placeholder: 'Bearer...', description: 'Semantic neural search' },
  { name: 'serper', display: 'Serper', placeholder: 'Bearer...', description: 'Google search results' },
  { name: 'firecrawl', display: 'Firecrawl', placeholder: 'fc-...', description: 'Web scraping for RAG' },
  { name: 'jina', display: 'Jina AI', placeholder: 'Bearer...', description: 'URL-to-text, embeddings' },
];

const PROVIDER_CATEGORIES = [
  { id: 'text', label: 'Text & Chat', icon: '💬', color: 'purple', providers: TEXT_PROVIDERS },
  { id: 'image', label: 'Image Generation', icon: '🎨', color: 'pink', providers: IMAGE_PROVIDERS },
  { id: 'video', label: 'Video Generation', icon: '🎬', color: 'amber', providers: VIDEO_PROVIDERS },
  { id: 'audio', label: 'Audio & Voice', icon: '🎙️', color: 'cyan', providers: AUDIO_PROVIDERS },
  { id: 'search', label: 'Search & RAG', icon: '🔍', color: 'emerald', providers: SEARCH_PROVIDERS },
];

// ─── Appearance Helpers ──────────────────────────────────────────────────────

const THEMES = [
  { value: 'dark' as const, label: 'Dark', icon: MoonIcon },
  { value: 'light' as const, label: 'Light', icon: SunIcon },
  { value: 'system' as const, label: 'System', icon: ComputerDesktopIcon },
];

const FONT_SIZES = [
  { value: 'small' as const, label: 'Small' },
  { value: 'medium' as const, label: 'Medium' },
  { value: 'large' as const, label: 'Large' },
];

// ─── Subcomponents ───────────────────────────────────────────────────────────

function ProviderRow({ provider, status, onSave, onRemove }: {
  provider: ProviderDef;
  status: boolean;
  onSave: (key: string) => Promise<void>;
  onRemove: () => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [keyInput, setKeyInput] = useState('');
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);

  const handleSave = async () => {
    if (!keyInput.trim()) return;
    setSaving(true);
    await onSave(keyInput.trim());
    setSaving(false);
    setEditing(false);
    setKeyInput('');
  };

  return (
    <div className="group flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/[0.03] transition-all duration-200">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm text-chat-text font-medium">{provider.display}</span>
          {status ? (
            <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Active
            </span>
          ) : (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-white/5 text-chat-text-secondary/50 font-medium">
              Not configured
            </span>
          )}
        </div>
        <p className="text-[11px] text-chat-text-secondary/60 mt-0.5">{provider.description}</p>
      </div>

      {editing ? (
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <div className="relative">
            <input
              type={showKey ? 'text' : 'password'}
              value={keyInput}
              onChange={e => setKeyInput(e.target.value)}
              placeholder={provider.placeholder}
              className="w-48 px-2.5 py-1.5 text-xs bg-black/30 border border-white/10 rounded-lg text-chat-text placeholder:text-chat-text-secondary/30 outline-none focus:border-purple-500/50 transition-colors font-mono"
              autoFocus
              onKeyDown={e => e.key === 'Enter' && handleSave()}
            />
            <button
              onClick={() => setShowKey(!showKey)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-chat-text-secondary/40 hover:text-chat-text-secondary"
            >
              {showKey ? <EyeSlashIcon className="w-3.5 h-3.5" /> : <EyeIcon className="w-3.5 h-3.5" />}
            </button>
          </div>
          <button
            onClick={handleSave}
            disabled={saving || !keyInput.trim()}
            className="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 disabled:opacity-30 transition-colors"
          >
            <CheckIcon className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => { setEditing(false); setKeyInput(''); }}
            className="p-1.5 rounded-lg bg-white/5 text-chat-text-secondary hover:bg-white/10 transition-colors"
          >
            <XMarkIcon className="w-3.5 h-3.5" />
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-1.5 flex-shrink-0 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => setEditing(true)}
            className="flex items-center gap-1 px-2.5 py-1.5 text-[11px] rounded-lg bg-white/5 text-chat-text-secondary hover:bg-white/10 hover:text-chat-text transition-colors font-medium"
          >
            <KeyIcon className="w-3 h-3" />
            {status ? 'Change' : 'Set Key'}
          </button>
          {status && (
            <button
              onClick={onRemove}
              className="p-1.5 rounded-lg text-red-400/60 hover:bg-red-500/10 hover:text-red-400 transition-colors"
            >
              <TrashIcon className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function ProviderCategory({ category, providerStatus, onSave, onRemove }: {
  category: typeof PROVIDER_CATEGORIES[0];
  providerStatus: Record<string, boolean>;
  onSave: (provider: string, key: string) => Promise<void>;
  onRemove: (provider: string) => Promise<void>;
}) {
  const [expanded, setExpanded] = useState(true);
  const activeCount = category.providers.filter(p => providerStatus[p.name]).length;

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/[0.02] transition-colors"
      >
        <span className="text-lg">{category.icon}</span>
        <span className="text-sm font-semibold text-chat-text flex-1 text-left">{category.label}</span>
        {activeCount > 0 && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 font-bold">
            {activeCount}/{category.providers.length}
          </span>
        )}
        <svg
          className={`w-4 h-4 text-chat-text-secondary/50 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {expanded && (
        <div className="px-1 pb-2 space-y-0.5">
          {category.providers.map(p => (
            <ProviderRow
              key={p.name}
              provider={p}
              status={!!providerStatus[p.name]}
              onSave={(key) => onSave(p.name, key)}
              onRemove={() => onRemove(p.name)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function PersonalitySection() {
  const [traits, setTraits] = useState({ openness: 0.8, conscientiousness: 0.7, extraversion: 0.5, agreeableness: 0.75, neuroticism: 0.25 });
  const [available, setAvailable] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetch('/api/alma/personality').then(r => r.json()).then(data => {
      if (data.available) {
        setAvailable(true);
        setTraits(data.traits);
      }
    }).catch(() => {});
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      const resp = await fetch('/api/alma/personality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(traits),
      });
      if (resp.ok) toast.success('Saved', 'Personality updated');
      else toast.error('Error', 'Could not save');
    } catch { toast.error('Error', 'Network error'); }
    setSaving(false);
  };

  if (!available) return (
    <p className="text-xs text-chat-text-secondary/50 italic px-1">ALMA emotional system not available</p>
  );

  const sliders: { key: keyof typeof traits; label: string; emoji: string }[] = [
    { key: 'openness', label: 'Openness', emoji: '🎨' },
    { key: 'conscientiousness', label: 'Conscientiousness', emoji: '📋' },
    { key: 'extraversion', label: 'Extraversion', emoji: '🗣️' },
    { key: 'agreeableness', label: 'Agreeableness', emoji: '🤝' },
    { key: 'neuroticism', label: 'Emotional Sensitivity', emoji: '💭' },
  ];

  return (
    <div className="space-y-4">
      {sliders.map(s => (
        <div key={s.key}>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs text-chat-text font-medium">{s.emoji} {s.label}</span>
            <span className="text-[10px] text-chat-text-secondary font-mono tabular-nums">{(traits[s.key] * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range" min={0} max={1} step={0.01}
            value={traits[s.key]}
            onChange={e => setTraits(prev => ({ ...prev, [s.key]: parseFloat(e.target.value) }))}
            className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer accent-purple-500"
          />
        </div>
      ))}
      <div className="flex gap-2">
        <button
          onClick={save}
          disabled={saving}
          className="flex-1 px-4 py-2 text-xs font-medium bg-purple-500/15 hover:bg-purple-500/25 text-purple-300 rounded-xl transition-colors border border-purple-500/20"
        >
          {saving ? 'Saving...' : 'Save Personality'}
        </button>
        <button
          onClick={async () => {
            try {
              const resp = await fetch('/api/alma/personality/reset', { method: 'POST' });
              if (resp.ok) {
                const data = await resp.json();
                if (data.traits) setTraits(data.traits);
                toast.success('Reset', 'Personality reset to defaults');
              }
            } catch { toast.error('Error', 'Could not reset'); }
          }}
          className="px-4 py-2 text-xs font-medium bg-white/5 hover:bg-white/10 text-chat-text-secondary rounded-xl transition-colors border border-white/10"
        >
          Reset
        </button>
      </div>
    </div>
  );
}

// ─── Main Settings Page ──────────────────────────────────────────────────────

export function SettingsPage() {
  const { settings, updateSettings, resetSettings } = useSettingsStore();
  const [providerStatus, setProviderStatus] = useState<Record<string, boolean>>({});
  const [activeSection, setActiveSection] = useState('providers');

  // PWA install prompt
  const [installPrompt, setInstallPrompt] = useState<any>(null);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    if (window.matchMedia('(display-mode: standalone)').matches) {
      setIsInstalled(true);
      return;
    }
    const handler = (e: Event) => {
      e.preventDefault();
      setInstallPrompt(e);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const handleInstall = async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    const result = await installPrompt.userChoice;
    if (result.outcome === 'accepted') {
      setIsInstalled(true);
      toast.success('App installed!');
    }
    setInstallPrompt(null);
  };

  // Fetch provider status
  useEffect(() => {
    fetch('/api/providers').then(r => r.json()).then(data => {
      const status: Record<string, boolean> = {};
      const list = Array.isArray(data) ? data : (data.providers || []);
      list.forEach((p: any) => { status[p.name] = p.configured; });
      setProviderStatus(status);
    }).catch(() => {});
  }, []);

  const saveProviderKey = useCallback(async (provider: string, key: string) => {
    try {
      const resp = await fetch(`/api/providers/${provider}/key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key }),
      });
      if (resp.ok) {
        setProviderStatus(prev => ({ ...prev, [provider]: true }));
        toast.success('API key saved', `${provider} is now active`);
      } else toast.error('Failed', 'Could not save API key');
    } catch { toast.error('Error', 'Network error'); }
  }, []);

  const removeProviderKey = useCallback(async (provider: string) => {
    try {
      const resp = await fetch(`/api/providers/${provider}/key`, { method: 'DELETE' });
      if (resp.ok) {
        setProviderStatus(prev => ({ ...prev, [provider]: false }));
        toast.success('Removed', `${provider} key removed`);
      } else {
        toast.error('Failed', 'Could not remove key');
      }
    } catch {
      toast.error('Error', 'Network error');
    }
  }, []);

  const totalActive = Object.values(providerStatus).filter(Boolean).length;

  const sections = [
    { id: 'providers', label: 'AI Providers', badge: totalActive > 0 ? `${totalActive}` : undefined },
    { id: 'appearance', label: 'Appearance' },
    { id: 'behavior', label: 'Behavior' },
    { id: 'personality', label: 'Personality' },
    { id: 'about', label: 'About' },
  ];

  return (
    <div className="h-full flex flex-col sm:flex-row">
      {/* Sidebar navigation — horizontal on mobile, vertical on desktop */}
      <nav className="sm:w-48 flex-shrink-0 border-b sm:border-b-0 sm:border-r border-white/[0.06] p-2 sm:p-4 flex sm:flex-col gap-1 overflow-x-auto sm:overflow-visible">
        <h2 className="text-lg font-bold text-chat-text mb-4 px-2 hidden sm:block">Settings</h2>
        {sections.map(s => (
          <button
            key={s.id}
            onClick={() => setActiveSection(s.id)}
            className={`sm:w-full whitespace-nowrap flex items-center justify-between px-3 py-2 rounded-xl text-sm font-medium transition-all duration-150 ${
              activeSection === s.id
                ? 'bg-purple-500/15 text-purple-300 shadow-sm shadow-purple-500/5'
                : 'text-chat-text-secondary hover:text-chat-text hover:bg-white/[0.03]'
            }`}
          >
            {s.label}
            {s.badge && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 font-bold">
                {s.badge}
              </span>
            )}
          </button>
        ))}
      </nav>

      {/* Content area */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto space-y-6">

          {/* AI Providers */}
          {activeSection === 'providers' && (
            <>
              <div className="mb-6">
                <h3 className="text-xl font-bold text-chat-text mb-1">AI Providers</h3>
                <p className="text-sm text-chat-text-secondary/60">
                  Add API keys for direct model access. Models appear in the picker once configured.
                </p>
                {totalActive > 0 && (
                  <p className="text-xs text-emerald-400/70 mt-1 font-medium">
                    {totalActive} provider{totalActive > 1 ? 's' : ''} configured
                  </p>
                )}
              </div>
              <div className="space-y-4">
                {PROVIDER_CATEGORIES.map(cat => (
                  <ProviderCategory
                    key={cat.id}
                    category={cat}
                    providerStatus={providerStatus}
                    onSave={saveProviderKey}
                    onRemove={removeProviderKey}
                  />
                ))}
              </div>
            </>
          )}

          {/* Appearance */}
          {activeSection === 'appearance' && (
            <>
              <div className="mb-6">
                <h3 className="text-xl font-bold text-chat-text mb-1">Appearance</h3>
                <p className="text-sm text-chat-text-secondary/60">Customize the look and feel.</p>
              </div>

              {/* Theme */}
              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-4">
                <h4 className="text-sm font-semibold text-chat-text mb-3">Theme</h4>
                <div className="flex gap-2">
                  {THEMES.map(t => (
                    <button
                      key={t.value}
                      onClick={() => updateSettings({ theme: t.value })}
                      className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                        settings.theme === t.value
                          ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30 shadow-sm shadow-purple-500/10'
                          : 'bg-white/[0.03] text-chat-text-secondary hover:bg-white/[0.06] border border-transparent'
                      }`}
                    >
                      <t.icon className="w-4 h-4" />
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Font Size */}
              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-4">
                <h4 className="text-sm font-semibold text-chat-text mb-3">Font Size</h4>
                <div className="flex gap-2">
                  {FONT_SIZES.map(f => (
                    <button
                      key={f.value}
                      onClick={() => updateSettings({ fontSize: f.value })}
                      className={`flex-1 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                        settings.fontSize === f.value
                          ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                          : 'bg-white/[0.03] text-chat-text-secondary hover:bg-white/[0.06] border border-transparent'
                      }`}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Install App */}
              {!isInstalled && (
                <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-4">
                  <h4 className="text-sm font-semibold text-chat-text mb-2">Install App</h4>
                  <p className="text-xs text-chat-text-secondary mb-3">
                    Install AURA as an app for faster access and offline support.
                  </p>
                  {installPrompt ? (
                    <button
                      onClick={handleInstall}
                      className="px-4 py-2 rounded-xl text-sm font-medium text-white bg-purple-500/20 hover:bg-purple-500/30 border border-purple-500/30 transition-colors"
                    >
                      Install AURA
                    </button>
                  ) : (
                    <p className="text-xs text-chat-text-tertiary">
                      Open in a supported browser to install
                    </p>
                  )}
                </div>
              )}
            </>
          )}

          {/* Behavior */}
          {activeSection === 'behavior' && (
            <>
              <div className="mb-6">
                <h3 className="text-xl font-bold text-chat-text mb-1">Behavior</h3>
                <p className="text-sm text-chat-text-secondary/60">Control how AURA responds and displays.</p>
              </div>

              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] divide-y divide-white/[0.04]">
                {[
                  { key: 'showThinking' as const, label: 'Show Thinking Process', desc: "Display AURA's reasoning steps" },
                  { key: 'autoScroll' as const, label: 'Auto Scroll', desc: 'Scroll to new messages automatically' },
                  { key: 'soundEnabled' as const, label: 'Sound Effects', desc: 'Play sounds for notifications' },
                ].map(toggle => (
                  <div key={toggle.key} className="flex items-center justify-between px-4 py-3.5">
                    <div>
                      <p className="text-sm text-chat-text font-medium">{toggle.label}</p>
                      <p className="text-[11px] text-chat-text-secondary/50 mt-0.5">{toggle.desc}</p>
                    </div>
                    <button
                      onClick={() => updateSettings({ [toggle.key]: !settings[toggle.key] })}
                      className={`relative w-10 h-5.5 rounded-full transition-colors duration-200 flex-shrink-0 ${
                        settings[toggle.key] ? 'bg-purple-500' : 'bg-white/10'
                      }`}
                      style={{ width: 40, height: 22 }}
                    >
                      <span
                        className="absolute top-0.5 left-0.5 w-4.5 h-4.5 rounded-full bg-white shadow-sm transition-transform duration-200"
                        style={{
                          width: 18, height: 18, top: 2, left: 2,
                          transform: settings[toggle.key] ? 'translateX(18px)' : 'translateX(0)',
                        }}
                      />
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Personality */}
          {activeSection === 'personality' && (
            <>
              <div className="mb-6">
                <h3 className="text-xl font-bold text-chat-text mb-1">AURA Personality (OCEAN)</h3>
                <p className="text-sm text-chat-text-secondary/60">Tune AURA's emotional personality traits.</p>
              </div>
              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5">
                <PersonalitySection />
              </div>
            </>
          )}

          {/* About */}
          {activeSection === 'about' && (
            <>
              <div className="mb-6">
                <h3 className="text-xl font-bold text-chat-text mb-1">About</h3>
              </div>
              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5 space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center text-lg">🧠</div>
                  <div>
                    <p className="text-sm text-chat-text font-bold">AURA</p>
                    <p className="text-[11px] text-chat-text-secondary">Autonomous Universal Reasoning Agent</p>
                  </div>
                </div>
                <div className="text-xs text-chat-text-secondary/60 space-y-1 pl-[52px]">
                  <p>Version 4.3.0 with Multi-Agent System</p>
                  <p>Built with cognitive architecture, emotional intelligence, and multi-surface sync.</p>
                </div>
                <div className="pt-2 pl-[52px]">
                  <button
                    onClick={resetSettings}
                    className="px-3 py-1.5 text-[11px] font-medium text-red-400/70 bg-red-500/10 rounded-lg hover:bg-red-500/15 transition-colors"
                  >
                    Reset All Settings
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
