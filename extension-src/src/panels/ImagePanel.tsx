import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Download, Upload, Scissors, ArrowUpCircle, Type, Eye, Loader2, ChevronLeft, ChevronRight } from 'lucide-react';
import { HTTP } from '../api';

// ── Generate Tab Constants ──────────────────────────────────────────────────

const STYLES = [
  { label: 'None', value: '' },
  { label: 'Photorealistic', value: 'photorealistic, 8k, detailed' },
  { label: 'Oil Painting', value: 'oil painting style, artistic' },
  { label: 'Anime', value: 'anime style, vibrant colors' },
  { label: 'Sketch', value: 'pencil sketch, detailed line art' },
  { label: 'Watercolor', value: 'watercolor painting, soft colors' },
];

// ── Edit Tool Definitions ───────────────────────────────────────────────────

interface EditTool {
  id: string;
  label: string;
  icon: React.ReactNode;
  endpoint: string;
  description: string;
}

const EDIT_TOOLS: EditTool[] = [
  {
    id: 'remove-bg',
    label: 'Remove BG',
    icon: <Scissors size={14} />,
    endpoint: '/api/image/remove-bg',
    description: 'Remove the background from this image',
  },
  {
    id: 'upscale',
    label: 'Upscale 2x',
    icon: <ArrowUpCircle size={14} />,
    endpoint: '/api/image/upscale',
    description: 'Upscale the image to 2x resolution',
  },
  {
    id: 'remove-text',
    label: 'Remove Text',
    icon: <Type size={14} />,
    endpoint: '/api/image/remove-text',
    description: 'Remove text overlays from this image',
  },
  {
    id: 'describe',
    label: 'Describe',
    icon: <Eye size={14} />,
    endpoint: '/api/chat',
    description: 'Get an AI description of this image',
  },
];

// ── Spinner ─────────────────────────────────────────────────────────────────

function Spinner({ size = 16 }: { size?: number }) {
  return (
    <Loader2
      size={size}
      style={{ animation: 'spin 1s linear infinite' }}
    />
  );
}

// ── Before/After Slider ─────────────────────────────────────────────────────

function BeforeAfterSlider({
  before,
  after,
}: {
  before: string;
  after: string;
}) {
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updatePosition = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setPosition((x / rect.width) * 100);
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    dragging.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    updatePosition(e.clientX);
  }, [updatePosition]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    updatePosition(e.clientX);
  }, [updatePosition]);

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div
      ref={containerRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{
        position: 'relative',
        width: '100%',
        borderRadius: 'var(--r-md)',
        border: '1px solid var(--b1)',
        overflow: 'hidden',
        cursor: 'ew-resize',
        userSelect: 'none',
        touchAction: 'none',
      }}
    >
      {/* After image (full) */}
      <img
        src={after}
        alt="After"
        style={{ width: '100%', display: 'block' }}
      />

      {/* Before image (clipped) */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          clipPath: `inset(0 ${100 - position}% 0 0)`,
        }}
      >
        <img
          src={before}
          alt="Before"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
      </div>

      {/* Slider line + handle */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: `${position}%`,
          transform: 'translateX(-50%)',
          width: 2,
          height: '100%',
          background: 'rgba(124, 58, 237, 0.8)',
          pointerEvents: 'none',
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: `${position}%`,
          transform: 'translate(-50%, -50%)',
          width: 28,
          height: 28,
          borderRadius: '50%',
          background: 'rgba(10, 8, 24, 0.85)',
          border: '2px solid rgba(124, 58, 237, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          pointerEvents: 'none',
          boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
        }}
      >
        <ChevronLeft size={10} style={{ color: '#fff', marginRight: -2 }} />
        <ChevronRight size={10} style={{ color: '#fff', marginLeft: -2 }} />
      </div>

      {/* Labels */}
      <div style={{
        position: 'absolute', top: 6, left: 6,
        background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
        padding: '2px 6px', borderRadius: 4, fontSize: 9,
        color: '#fff', fontWeight: 600, letterSpacing: '0.05em',
        textTransform: 'uppercase', pointerEvents: 'none',
      }}>
        Before
      </div>
      <div style={{
        position: 'absolute', top: 6, right: 6,
        background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
        padding: '2px 6px', borderRadius: 4, fontSize: 9,
        color: '#fff', fontWeight: 600, letterSpacing: '0.05em',
        textTransform: 'uppercase', pointerEvents: 'none',
      }}>
        After
      </div>
    </div>
  );
}

// ── Drop Zone ───────────────────────────────────────────────────────────────

function DropZone({
  onImage,
}: {
  onImage: (b64: string, dataUrl: string) => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      const b64 = dataUrl.split(',')[1];
      onImage(b64, dataUrl);
    };
    reader.readAsDataURL(file);
  }, [onImage]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    // Reset input so same file can be re-selected
    if (inputRef.current) inputRef.current.value = '';
  }, [processFile]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragOver ? 'var(--p)' : 'var(--b1)'}`,
        borderRadius: 'var(--r-md)',
        padding: '28px 16px',
        textAlign: 'center',
        cursor: 'pointer',
        background: dragOver ? 'rgba(124, 58, 237, 0.06)' : 'var(--s2)',
        transition: 'all 0.2s ease',
      }}
    >
      <Upload size={24} style={{ color: 'var(--mu)', margin: '0 auto 8px' }} />
      <div style={{ fontSize: '12px', color: 'var(--mu)', fontWeight: 500 }}>
        Drop an image here or click to upload
      </div>
      <div style={{ fontSize: '10px', color: 'var(--mu)', opacity: 0.6, marginTop: 4 }}>
        PNG, JPG, WebP
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={onFileChange}
        style={{ display: 'none' }}
      />
    </div>
  );
}

// ── Edit Tab ────────────────────────────────────────────────────────────────

function EditTab() {
  const [sourceB64, setSourceB64] = useState('');
  const [sourceDataUrl, setSourceDataUrl] = useState('');
  const [resultDataUrl, setResultDataUrl] = useState('');
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [status, setStatus] = useState('');
  const [description, setDescription] = useState('');
  const [showCompare, setShowCompare] = useState(false);

  // Listen for IMAGE_EDIT_LOAD events from content script (hover toolbar)
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.dataUrl) {
        setSourceDataUrl(detail.dataUrl);
        setSourceB64(detail.dataUrl.split(',')[1] || '');
        setResultDataUrl('');
        setDescription('');
        setStatus('');
        setShowCompare(false);
      }
    };
    window.addEventListener('image-edit-load', handler);
    return () => window.removeEventListener('image-edit-load', handler);
  }, []);

  // Also listen for message-based image loads (from background script)
  useEffect(() => {
    const handler = (msg: any) => {
      if (msg.type === 'IMAGE_EDIT_LOAD' && msg.dataUrl) {
        setSourceDataUrl(msg.dataUrl);
        setSourceB64(msg.dataUrl.split(',')[1] || '');
        setResultDataUrl('');
        setDescription('');
        setStatus('');
        setShowCompare(false);
      }
    };
    chrome?.runtime?.onMessage?.addListener(handler);
    return () => chrome?.runtime?.onMessage?.removeListener(handler);
  }, []);

  const onImage = useCallback((b64: string, dataUrl: string) => {
    setSourceB64(b64);
    setSourceDataUrl(dataUrl);
    setResultDataUrl('');
    setDescription('');
    setStatus('');
    setShowCompare(false);
  }, []);

  const runTool = useCallback(async (tool: EditTool) => {
    if (!sourceB64) return;
    setActiveTool(tool.id);
    setStatus('');
    setResultDataUrl('');
    setDescription('');
    setShowCompare(false);

    try {
      if (tool.id === 'describe') {
        // Send to chat endpoint with image attachment
        const res = await fetch(`${HTTP}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: 'Describe this image in detail. What do you see?',
            conversation_id: '__image_describe__',
            stream: false,
            images: [sourceB64],
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error((errData as any).detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        setDescription(data.response || data.message || 'No description returned.');
        setStatus('');
      } else {
        // Image editing endpoints
        const body: any = { image_b64: sourceB64 };
        if (tool.id === 'upscale') body.scale = 2;

        const res = await fetch(`${HTTP}${tool.endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });

        if (res.status === 503 || res.status === 502) {
          setStatus('Start AURA backend to use image editing');
          return;
        }

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error((errData as any).detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        if (data.image_b64) {
          setResultDataUrl('data:image/png;base64,' + data.image_b64);
          setShowCompare(true);
          setStatus('');
        } else {
          setStatus('No result returned from server');
        }
      }
    } catch (err: any) {
      if (
        err.message?.includes('Failed to fetch') ||
        err.message?.includes('NetworkError') ||
        err.message?.includes('ERR_CONNECTION_REFUSED')
      ) {
        setStatus('Start AURA backend to use image editing');
      } else {
        setStatus(err.message || 'Processing failed');
      }
    } finally {
      setActiveTool(null);
    }
  }, [sourceB64]);

  const downloadResult = useCallback(() => {
    if (!resultDataUrl) return;
    const a = document.createElement('a');
    a.href = resultDataUrl;
    a.download = 'aura-edited.png';
    a.click();
  }, [resultDataUrl]);

  const clearImage = useCallback(() => {
    setSourceB64('');
    setSourceDataUrl('');
    setResultDataUrl('');
    setDescription('');
    setStatus('');
    setShowCompare(false);
  }, []);

  return (
    <div className="flex flex-col gap-3">
      {!sourceDataUrl ? (
        <DropZone onImage={onImage} />
      ) : (
        <>
          {/* Source image preview */}
          <div style={{ position: 'relative' }}>
            {showCompare && resultDataUrl ? (
              <BeforeAfterSlider before={sourceDataUrl} after={resultDataUrl} />
            ) : (
              <img
                src={resultDataUrl || sourceDataUrl}
                alt="Source"
                style={{
                  width: '100%',
                  borderRadius: 'var(--r-md)',
                  border: '1px solid var(--b1)',
                }}
              />
            )}
            <button
              onClick={clearImage}
              title="Remove image"
              style={{
                position: 'absolute',
                top: 6,
                right: showCompare ? 'auto' : 6,
                left: showCompare ? '50%' : 'auto',
                transform: showCompare ? 'translateX(-50%)' : 'none',
                width: 22,
                height: 22,
                borderRadius: '50%',
                background: 'rgba(0,0,0,0.6)',
                backdropFilter: 'blur(4px)',
                border: '1px solid rgba(255,255,255,0.1)',
                color: '#fff',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 12,
                lineHeight: 1,
                padding: 0,
              }}
            >
              x
            </button>
          </div>

          {/* Tool buttons */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 6,
            }}
          >
            {EDIT_TOOLS.map((tool) => {
              const isRunning = activeTool === tool.id;
              const isDisabled = activeTool !== null;
              return (
                <button
                  key={tool.id}
                  onClick={() => runTool(tool)}
                  disabled={isDisabled}
                  title={tool.description}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 6,
                    padding: '8px 10px',
                    background: isRunning ? 'var(--pg2)' : 'var(--s2)',
                    border: `1px solid ${isRunning ? 'var(--p)' : 'var(--b1)'}`,
                    borderRadius: 'var(--r-md)',
                    color: isRunning ? 'var(--pl)' : isDisabled ? 'var(--mu)' : 'var(--tx)',
                    fontSize: '11.5px',
                    fontWeight: 500,
                    cursor: isDisabled ? 'not-allowed' : 'pointer',
                    fontFamily: 'inherit',
                    opacity: isDisabled && !isRunning ? 0.5 : 1,
                    transition: 'all 0.15s ease',
                  }}
                >
                  {isRunning ? <Spinner size={14} /> : tool.icon}
                  {tool.label}
                </button>
              );
            })}
          </div>

          {/* Status / error */}
          {status && (
            <div
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                padding: 10,
                fontSize: '11.5px',
                color: 'var(--mu)',
                textAlign: 'center',
              }}
            >
              {status}
            </div>
          )}

          {/* Description result */}
          {description && (
            <div
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                padding: 10,
                fontSize: '12px',
                color: 'var(--tx)',
                lineHeight: 1.5,
                maxHeight: 200,
                overflowY: 'auto',
              }}
            >
              {description}
            </div>
          )}

          {/* Download result */}
          {resultDataUrl && (
            <button
              onClick={downloadResult}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
                background: 'var(--p)',
                border: 'none',
                borderRadius: 'var(--r-md)',
                color: 'white',
                padding: '8px',
                cursor: 'pointer',
                fontSize: '12px',
                fontFamily: 'inherit',
                fontWeight: 500,
              }}
            >
              <Download size={13} /> Download Result
            </button>
          )}
        </>
      )}
    </div>
  );
}

// ── Generate Tab (preserved as-is) ─────────────────────────────────────────

function GenerateTab() {
  const [style, setStyle] = useState('');
  const [status, setStatus] = useState('');
  const [imgSrc, setImgSrc] = useState('');
  const [loading, setLoading] = useState(false);
  const [comfyNote, setComfyNote] = useState(false);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const negRef = useRef<HTMLInputElement>(null);

  const generate = async () => {
    const prompt = promptRef.current?.value.trim();
    if (!prompt) return;
    const neg = negRef.current?.value.trim() || '';
    const fullPrompt = style ? `${prompt}, ${style}` : prompt;

    setLoading(true);
    setStatus('Generating... (15-60s)');
    setImgSrc('');
    setComfyNote(false);

    try {
      const res = await fetch(`${HTTP}/api/image/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: fullPrompt, negative_prompt: neg, steps: 20 }),
      });

      if (res.status === 503) {
        setStatus('');
        setComfyNote(true);
        return;
      }

      const data = await res.json();
      if (data.image_b64) {
        setImgSrc('data:image/png;base64,' + data.image_b64);
        setStatus('Done');
      } else {
        setStatus((data.detail || 'Generation failed'));
      }
    } catch (err: any) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const download = () => {
    if (!imgSrc) return;
    const a = document.createElement('a');
    a.href = imgSrc;
    a.download = 'aura-image.png';
    a.click();
  };

  return (
    <div className="flex flex-col gap-3">
      {/* Style selector */}
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
          Style
        </div>
        <div className="flex flex-wrap gap-1.5">
          {STYLES.map(s => (
            <button
              key={s.value}
              onClick={() => setStyle(s.value)}
              style={{
                padding: '4px 10px',
                background: style === s.value ? 'var(--pg2)' : 'var(--s2)',
                border: `1px solid ${style === s.value ? 'var(--p)' : 'var(--b1)'}`,
                borderRadius: 'var(--r-pill)',
                color: style === s.value ? 'var(--pl)' : 'var(--mu)',
                fontSize: '11px',
                cursor: 'pointer',
                fontFamily: 'inherit',
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <textarea
        ref={promptRef}
        placeholder="Describe the image..."
        onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); } }}
        style={{
          background: 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-md)',
          color: 'var(--tx)',
          fontSize: '12.5px',
          padding: '8px 10px',
          resize: 'none',
          height: 70,
          outline: 'none',
          fontFamily: 'inherit',
        }}
      />

      <input
        ref={negRef}
        placeholder="Negative prompt (optional)..."
        style={{
          background: 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-md)',
          color: 'var(--tx)',
          fontSize: '12px',
          padding: '7px 10px',
          outline: 'none',
          fontFamily: 'inherit',
        }}
      />

      <button
        onClick={generate}
        disabled={loading}
        style={{
          background: loading ? 'var(--s3)' : 'var(--p)',
          border: 'none',
          borderRadius: 'var(--r-md)',
          color: 'white',
          padding: '9px',
          cursor: loading ? 'not-allowed' : 'pointer',
          fontSize: '13px',
          fontFamily: 'inherit',
          fontWeight: 500,
        }}
      >
        {loading ? '...' : 'Generate'}
      </button>

      {status && (
        <div style={{ color: status === 'Done' ? 'var(--gr)' : 'var(--mu)', fontSize: '12px', textAlign: 'center' }}>
          {status}
        </div>
      )}

      {comfyNote && (
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)', padding: 12, fontSize: '12px', color: 'var(--mu)' }}>
          ComfyUI is not running. Start ComfyUI to enable image generation.
        </div>
      )}

      {imgSrc && (
        <div className="flex flex-col gap-2">
          <img
            src={imgSrc}
            alt="Generated"
            style={{ width: '100%', borderRadius: 'var(--r-md)', border: '1px solid var(--b1)' }}
          />
          <button
            onClick={download}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              color: 'var(--tx)',
              padding: '7px',
              cursor: 'pointer',
              fontSize: '12px',
              fontFamily: 'inherit',
            }}
          >
            <Download size={13} /> Download
          </button>
        </div>
      )}
    </div>
  );
}

// ── Main Panel ──────────────────────────────────────────────────────────────

type Tab = 'generate' | 'edit';

export default function ImagePanel() {
  const [tab, setTab] = useState<Tab>('generate');

  // Auto-switch to edit tab when image is loaded from content script
  useEffect(() => {
    const handler = () => setTab('edit');
    window.addEventListener('image-edit-load', handler);
    return () => window.removeEventListener('image-edit-load', handler);
  }, []);

  useEffect(() => {
    const handler = (msg: any) => {
      if (msg.type === 'IMAGE_EDIT_LOAD') setTab('edit');
    };
    chrome?.runtime?.onMessage?.addListener(handler);
    return () => chrome?.runtime?.onMessage?.removeListener(handler);
  }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Tab bar */}
      <div
        style={{
          display: 'flex',
          borderBottom: '1px solid var(--b1)',
          flexShrink: 0,
        }}
      >
        {(['generate', 'edit'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              flex: 1,
              padding: '9px 0',
              background: 'transparent',
              border: 'none',
              borderBottom: tab === t ? '2px solid var(--p)' : '2px solid transparent',
              color: tab === t ? 'var(--tx)' : 'var(--mu)',
              fontSize: '12.5px',
              fontWeight: tab === t ? 600 : 500,
              fontFamily: 'inherit',
              cursor: 'pointer',
              textTransform: 'capitalize',
              transition: 'all 0.15s ease',
              letterSpacing: '0.02em',
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div
        className="panel-scroll-root"
        style={{ flex: 1, overflowY: 'auto', padding: 12 }}
      >
        {tab === 'generate' ? <GenerateTab /> : <EditTab />}
      </div>

      {/* Spinner keyframe (injected once) */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
