import { useState, useRef, useCallback, useEffect } from 'react';
import {
  SparklesIcon as SparklesOutline,
  ArrowDownTrayIcon, ClipboardDocumentIcon, CheckIcon,
  TrashIcon, MagnifyingGlassIcon, ChevronDownIcon, ChevronUpIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import { SparklesIcon } from '@heroicons/react/24/solid';
import { copyImage } from '../utils/clipboard';

/* ── Types ── */
interface GeneratedImage {
  id: string;
  prompt: string;
  negativePrompt: string;
  imageB64: string;
  timestamp: number;
  steps: number;
}

const STYLE_PRESETS = [
  'None', 'Photographic', 'Digital Art', 'Anime', 'Cinematic',
  'Fantasy', 'Pixel Art', 'Watercolor', 'Oil Painting',
];

function imgSrc(b64: string): string {
  // SVG base64 starts with PHN2Zy (decoded: <svg)
  if (b64.startsWith('PHN2Zy') || b64.startsWith('PD94bW')) return `data:image/svg+xml;base64,${b64}`;
  return `data:image/png;base64,${b64}`;
}

const STORAGE_KEY = 'aura-image-history';
const MAX_HISTORY = 50;
const MAX_STORAGE_BYTES = 4 * 1024 * 1024; // 4MB limit (leave 1MB headroom in 5MB quota)

function loadHistory(): GeneratedImage[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch { return []; }
}

function saveHistory(images: GeneratedImage[]) {
  let list = images.slice(0, MAX_HISTORY);
  // Evict oldest images until under size limit
  while (list.length > 0) {
    const json = JSON.stringify(list);
    if (json.length * 2 <= MAX_STORAGE_BYTES) { // *2 for UTF-16 storage
      try {
        localStorage.setItem(STORAGE_KEY, json);
        return;
      } catch {
        // Quota exceeded — evict more
        list = list.slice(0, -1);
      }
    } else {
      list = list.slice(0, -1);
    }
  }
  // All evicted — save empty
  try { localStorage.setItem(STORAGE_KEY, '[]'); } catch {}
}

const SVG_SYSTEM_PROMPT = `You are an expert SVG artist. Generate a beautiful, detailed SVG image based on the user's description.

Rules:
- Output ONLY valid SVG code starting with <svg>
- Use viewBox="0 0 800 600" for landscape or "0 0 600 800" for portrait
- Use gradients, patterns, filters for depth and realism
- Use modern SVG features: clipPath, mask, filter effects
- Include rich colors and subtle details
- NO markdown, NO explanation, ONLY the SVG code
- Make it visually impressive and artistic`;

/* ── Main Component ── */
export function ImageGenPanel() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [showNegative, setShowNegative] = useState(false);
  const [steps, setSteps] = useState(20);
  const [style, setStyle] = useState('None');
  const [isGenerating, setIsGenerating] = useState(false);
  const [genMode, setGenMode] = useState<'auto' | 'svg'>('auto');
  const [error, setError] = useState<string | null>(null);
  const [currentImage, setCurrentImage] = useState<GeneratedImage | null>(null);
  const [history, setHistory] = useState<GeneratedImage[]>(() => loadHistory());
  const [copied, setCopied] = useState(false);
  const [historySearch, setHistorySearch] = useState('');
  const [showHistory, setShowHistory] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const modelMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    return () => { if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current); };
  }, []);

  // Fetch models
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all = [...(data.chatgpt_models || []), ...(data.direct_api_models || []), ...(data.cloud_models || []), ...(data.local_models || [])];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, []);

  // Close model menu
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) setShowModelMenu(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const generateSvg = useCallback(async (fullPrompt: string): Promise<string> => {
    const res = await fetch('/api/generate/raw', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: fullPrompt,
        system_prompt: SVG_SYSTEM_PROMPT,
        ...(selectedModel ? { model: selectedModel } : {}),
      }),
    });
    if (!res.ok) throw new Error(`SVG generation failed (${res.status})`);

    let fullResponse = '';
    if (res.body) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              const text = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
              if (text) fullResponse += text;
            } catch { fullResponse += data; }
          } else if (line.trim() && !line.startsWith(':')) {
            fullResponse += line;
          }
        }
      }
    } else {
      fullResponse = await res.text();
    }

    // Extract SVG
    let svg = fullResponse.trim();
    const fenceMatch = svg.match(/```(?:svg|xml)?\s*\n([\s\S]*?)```/);
    if (fenceMatch) svg = fenceMatch[1].trim();
    if (!svg.includes('<svg')) throw new Error('No valid SVG generated');

    // Convert SVG to base64 data URL
    const b64 = btoa(unescape(encodeURIComponent(svg)));
    return b64;
  }, [selectedModel]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setError(null);

    const fullPrompt = style !== 'None'
      ? `${prompt}, ${style.toLowerCase()} style`
      : prompt;

    try {
      let imageB64: string;

      if (genMode === 'svg') {
        // Direct SVG generation via LLM
        imageB64 = await generateSvg(fullPrompt);
      } else {
        // Try ComfyUI first, fallback to SVG
        try {
          const res = await fetch('/api/image/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              prompt: fullPrompt,
              negative_prompt: negativePrompt || undefined,
              steps,
            }),
          });

          if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            // If ComfyUI not running, auto-fallback to SVG
            if (data?.detail?.error?.includes?.('ComfyUI') || data?.error?.includes?.('ComfyUI')) {
              setGenMode('svg');
              imageB64 = await generateSvg(fullPrompt);
            } else {
              throw new Error(data.error || data.detail || `Generation failed (${res.status})`);
            }
          } else {
            const data = await res.json();
            if (!data.image_b64) throw new Error('No image returned');
            imageB64 = data.image_b64;
          }
        } catch (e: any) {
          // Network error or ComfyUI not running — fallback to SVG
          if (e.message?.includes('ComfyUI') || e.message?.includes('Failed to fetch') || e.message?.includes('NetworkError')) {
            setGenMode('svg');
            imageB64 = await generateSvg(fullPrompt);
          } else {
            throw e;
          }
        }
      }

      const img: GeneratedImage = {
        id: `img-${Date.now()}`,
        prompt: fullPrompt,
        negativePrompt,
        imageB64: imageB64,
        timestamp: Date.now(),
        steps,
      };

      setCurrentImage(img);
      setHistory((prev) => {
        const next = [img, ...prev].slice(0, MAX_HISTORY);
        saveHistory(next);
        return next;
      });
    } catch (e: any) {
      setError(e.message || 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  }, [prompt, negativePrompt, steps, style, isGenerating, genMode, generateSvg]);

  const handleDownload = useCallback(() => {
    if (!currentImage) return;
    const a = document.createElement('a');
    a.href = imgSrc(currentImage.imageB64);
    const isSvg = currentImage.imageB64.startsWith('PHN2Zy') || currentImage.imageB64.startsWith('PD94bW');
    a.download = `aura-image-${currentImage.id}.${isSvg ? 'svg' : 'png'}`;
    a.click();
  }, [currentImage]);

  const handleCopy = useCallback(async () => {
    if (!currentImage) return;
    const ok = await copyImage(
      imgSrc(currentImage.imageB64),
      currentImage.prompt,
    );
    if (ok) {
      setCopied(true);
      clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 1500);
    }
  }, [currentImage]);

  const handleDeleteImage = useCallback((id: string) => {
    setHistory((prev) => {
      const next = prev.filter((img) => img.id !== id);
      saveHistory(next);
      return next;
    });
    if (currentImage?.id === id) setCurrentImage(null);
  }, [currentImage]);

  const handleClearHistory = useCallback(() => {
    setHistory([]);
    saveHistory([]);
    setCurrentImage(null);
  }, []);

  const filteredHistory = historySearch
    ? history.filter((img) => img.prompt.toLowerCase().includes(historySearch.toLowerCase()))
    : history;

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">
      {/* Left: Controls — full width on mobile */}
      <div className="flex flex-col md:w-[360px] md:min-w-[280px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[50vh] max-md:overflow-y-auto bg-surface-0">
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text flex items-center gap-2">
            <SparklesIcon className="w-4 h-4 text-purple-400" />
            Image Generation
          </h2>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Prompt */}
          <div>
            <label className="text-xs font-medium text-chat-text-secondary mb-1 block">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); handleGenerate(); } }}
              placeholder="Describe the image you want..."
              className="w-full p-3 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
              rows={3}
              disabled={isGenerating}
            />
          </div>

          {/* Negative prompt (collapsible) */}
          <div>
            <button
              onClick={() => setShowNegative(!showNegative)}
              className="flex items-center gap-1 text-xs text-chat-text-secondary hover:text-chat-text"
            >
              {showNegative ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />}
              Negative prompt
            </button>
            {showNegative && (
              <textarea
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                placeholder="What to avoid..."
                className="w-full mt-1 p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-xs resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
                rows={2}
              />
            )}
          </div>

          {/* Style */}
          <div>
            <label className="text-xs font-medium text-chat-text-secondary mb-1 block">Style</label>
            <div className="flex flex-wrap gap-1.5">
              {STYLE_PRESETS.map((s) => (
                <button
                  key={s}
                  onClick={() => setStyle(s)}
                  className={`px-2.5 py-1 rounded-full text-[10px] border transition-colors ${
                    style === s
                      ? 'bg-chat-accent text-white border-chat-accent'
                      : 'border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-purple-500/30'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* Steps */}
          <div>
            <label className="text-xs font-medium text-chat-text-secondary mb-1 flex items-center justify-between">
              <span>Steps</span>
              <span className="text-chat-text">{steps}</span>
            </label>
            <input
              type="range"
              min={5}
              max={50}
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
              className="w-full accent-purple-500"
            />
          </div>

          {/* Mode + Model selector */}
          <div className="flex items-center gap-2">
            <div className="flex rounded-md border border-chat-border overflow-hidden">
              <button onClick={() => setGenMode('auto')} className={`px-2.5 py-1 text-[10px] transition-colors ${genMode === 'auto' ? 'bg-chat-accent text-white' : 'text-chat-text-secondary'}`}>Auto</button>
              <button onClick={() => setGenMode('svg')} className={`px-2.5 py-1 text-[10px] transition-colors ${genMode === 'svg' ? 'bg-chat-accent text-white' : 'text-chat-text-secondary'}`}>SVG (AI)</button>
            </div>
            <div ref={modelMenuRef} className="relative">
              <button type="button" onClick={() => setShowModelMenu(p => !p)} className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text px-2 py-1 rounded-md" style={{ background: 'var(--border-subtle)' }}>
                <span className="max-w-[120px] truncate">{selectedModel ? selectedModel.split('/').pop() : 'Auto'}</span>
                <ChevronDownIcon className="w-2.5 h-2.5 opacity-50" />
              </button>
              {showModelMenu && availableModels.length > 0 && (
                <div style={{ position: 'absolute', bottom: 28, left: 0, width: 220, maxHeight: 260, background: 'var(--surface-1)', border: '1px solid var(--border-default)', borderRadius: 10, overflow: 'hidden', zIndex: 50 }}>
                  <div style={{ maxHeight: 260, overflowY: 'auto', padding: 4 }}>
                    <button onClick={() => { setSelectedModel(null); setShowModelMenu(false); }} className="w-full px-2.5 py-1.5 rounded-lg text-xs text-left" style={{ color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)', background: !selectedModel ? 'var(--surface-3)' : 'transparent' }}>Auto</button>
                    {availableModels.map(m => (
                      <button key={m} onClick={() => { setSelectedModel(m); setShowModelMenu(false); }} className="w-full px-2.5 py-1.5 rounded-lg text-xs text-left truncate" style={{ color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)', background: selectedModel === m ? 'var(--surface-3)' : 'transparent' }}>{m}</button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={!prompt.trim() || isGenerating}
            className="w-full py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-all flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <><ArrowPathIcon className="w-4 h-4 animate-spin" />Generating{genMode === 'svg' ? ' SVG' : ''}...</>
            ) : (
              <><SparklesOutline className="w-4 h-4" />Generate{genMode === 'svg' ? ' SVG' : ''}</>
            )}
          </button>

          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 rounded-lg p-2.5">{error}</div>
          )}

          {/* Prompt history */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="flex items-center gap-1 text-xs font-medium text-chat-text-secondary hover:text-chat-text"
              >
                {showHistory ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />}
                History ({history.length})
              </button>
              {history.length > 0 && (
                <button onClick={handleClearHistory} className="text-[10px] text-red-400 hover:text-red-300">Clear all</button>
              )}
            </div>
            {showHistory && history.length > 0 && (
              <>
                <div className="relative mb-2">
                  <input
                    type="text"
                    value={historySearch}
                    onChange={(e) => setHistorySearch(e.target.value)}
                    placeholder="Search..."
                    className="w-full px-3 py-1.5 pl-7 text-xs bg-surface-1 border border-chat-border rounded-lg text-chat-text placeholder-chat-text-secondary/50 outline-none focus:border-chat-accent"
                  />
                  <MagnifyingGlassIcon className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-chat-text-secondary/50" />
                </div>
                <div className="grid grid-cols-3 gap-1.5 max-h-[200px] overflow-y-auto">
                  {filteredHistory.map((img) => (
                    <div
                      key={img.id}
                      className={`relative group cursor-pointer rounded-md overflow-hidden border transition-colors ${
                        currentImage?.id === img.id ? 'border-purple-500' : 'border-chat-border hover:border-purple-500/30'
                      }`}
                      onClick={() => setCurrentImage(img)}
                    >
                      <img src={imgSrc(img.imageB64)} alt={img.prompt} className="w-full aspect-square object-cover" />
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDeleteImage(img.id); }}
                        className="absolute top-0.5 right-0.5 p-0.5 rounded bg-black/60 text-white opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <TrashIcon className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Right: Image display */}
      <div className="flex-1 flex flex-col items-center justify-center min-w-0 p-6 bg-surface-1">
        {currentImage ? (
          <div className="flex flex-col items-center gap-4 max-w-lg w-full">
            <img
              src={imgSrc(currentImage.imageB64)}
              alt={currentImage.prompt}
              className="w-full rounded-xl shadow-lg"
            />
            <div className="flex items-center gap-2">
              <button onClick={handleDownload} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-chat-border text-xs text-chat-text-secondary hover:text-chat-text hover:border-purple-500/30 transition-colors">
                <ArrowDownTrayIcon className="w-3.5 h-3.5" />Download
              </button>
              <button onClick={handleCopy} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-chat-border text-xs text-chat-text-secondary hover:text-chat-text hover:border-purple-500/30 transition-colors">
                {copied ? <><CheckIcon className="w-3.5 h-3.5 text-green-400" />Copied</> : <><ClipboardDocumentIcon className="w-3.5 h-3.5" />Copy</>}
              </button>
              <button
                onClick={() => { setPrompt(currentImage.prompt); }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-chat-border text-xs text-chat-text-secondary hover:text-chat-text hover:border-purple-500/30 transition-colors"
              >
                <ArrowPathIcon className="w-3.5 h-3.5" />Variations
              </button>
            </div>
            <div className="text-xs text-chat-text-secondary text-center max-w-md">
              <p className="truncate">{currentImage.prompt}</p>
              <p className="text-[10px] mt-0.5">{currentImage.steps} steps &middot; {new Date(currentImage.timestamp).toLocaleTimeString()}</p>
            </div>
          </div>
        ) : (
          <div className="text-center text-chat-text-secondary">
            <div className="text-4xl mb-3">🎨</div>
            <p className="text-sm">Generated images will appear here</p>
            <p className="text-[10px] mt-1">Enter a prompt and click Generate</p>
          </div>
        )}
      </div>
    </div>
  );
}
