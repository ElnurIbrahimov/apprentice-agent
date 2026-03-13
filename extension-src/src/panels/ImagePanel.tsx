import React, { useState, useRef } from 'react';
import { Download } from 'lucide-react';
import { HTTP } from '../api';

const STYLES = [
  { label: 'None', value: '' },
  { label: 'Photorealistic', value: 'photorealistic, 8k, detailed' },
  { label: 'Oil Painting', value: 'oil painting style, artistic' },
  { label: 'Anime', value: 'anime style, vibrant colors' },
  { label: 'Sketch', value: 'pencil sketch, detailed line art' },
  { label: 'Watercolor', value: 'watercolor painting, soft colors' },
];

export default function ImagePanel() {
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
    setStatus('Generating… (15–60s)');
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
        setStatus('✓ Generated');
      } else {
        setStatus('⚠ ' + (data.detail || 'Generation failed'));
      }
    } catch (err: any) {
      setStatus('⚠ ' + err.message);
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
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
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
        placeholder="Describe the image…"
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
        placeholder="Negative prompt (optional)…"
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
        {loading ? '…' : '✨ Generate'}
      </button>

      {status && (
        <div style={{ color: status.startsWith('⚠') ? 'var(--rd)' : status.startsWith('✓') ? 'var(--gr)' : 'var(--mu)', fontSize: '12px', textAlign: 'center' }}>
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
