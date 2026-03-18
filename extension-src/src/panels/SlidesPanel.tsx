import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Presentation, ChevronLeft, ChevronRight, Copy, Download,
  Plus, Trash2, RefreshCw, Edit3, Check, X, FileText,
} from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { sendMsg } from '../ext';

/* ─── Types ─── */
interface Slide {
  title: string;
  bullets: string[];
  notes: string;
}

type SlideStyle = 'professional' | 'casual' | 'academic' | 'creative';
type SlideCount = 5 | 10 | 15;

/* ─── Style themes for slide rendering ─── */
const STYLE_THEMES: Record<SlideStyle, { bg: string; accent: string; text: string; bullet: string }> = {
  professional: {
    bg: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
    accent: '#60a5fa',
    text: '#f0f0f5',
    bullet: '#94a3b8',
  },
  casual: {
    bg: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)',
    accent: '#a78bfa',
    text: '#f8fafc',
    bullet: '#cbd5e1',
  },
  academic: {
    bg: 'linear-gradient(135deg, #1c1917 0%, #292524 100%)',
    accent: '#fbbf24',
    text: '#fafaf9',
    bullet: '#a8a29e',
  },
  creative: {
    bg: 'linear-gradient(135deg, #3b0764 0%, #581c87 40%, #7c3aed 100%)',
    accent: '#f472b6',
    text: '#fdf4ff',
    bullet: '#d8b4fe',
  },
};

/* ─── Build the full HTML slideshow ─── */
function buildSlideshowHtml(slides: Slide[], style: SlideStyle, currentIdx: number): string {
  const theme = STYLE_THEMES[style];

  const slidesHtml = slides.map((s, i) => {
    const bulletsHtml = s.bullets
      .map(b => `<li style="margin-bottom:8px;padding-left:4px;">${escHtml(b)}</li>`)
      .join('\n');

    return `
    <div class="slide" id="slide-${i}" style="
      display:${i === currentIdx ? 'flex' : 'none'};
      flex-direction:column;
      justify-content:center;
      width:100%;height:100%;
      padding:32px 36px;
      box-sizing:border-box;
      animation:fadeIn 0.3s ease;
    ">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:${theme.accent};margin-bottom:12px;font-weight:600;">
        Slide ${i + 1} of ${slides.length}
      </div>
      <h1 style="font-size:24px;font-weight:700;margin:0 0 18px 0;line-height:1.3;color:${theme.text};">
        ${escHtml(s.title)}
      </h1>
      <ul style="font-size:15px;line-height:1.7;list-style:none;padding:0;margin:0;color:${theme.bullet};">
        ${bulletsHtml}
      </ul>
      ${s.notes ? `<div style="margin-top:auto;padding-top:16px;font-size:11px;color:${theme.bullet};opacity:0.6;font-style:italic;">${escHtml(s.notes)}</div>` : ''}
    </div>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; overflow: hidden; }
  body {
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    background: ${theme.bg};
    color: ${theme.text};
  }
  ul li::before {
    content: "\\2022";
    color: ${theme.accent};
    font-weight: bold;
    display: inline-block;
    width: 1em;
    margin-left: -0.2em;
  }
</style>
</head>
<body>
${slidesHtml}
</body>
</html>`;
}

/* ─── Build the full downloadable HTML with navigation built in ─── */
function buildDownloadableHtml(slides: Slide[], style: SlideStyle): string {
  const theme = STYLE_THEMES[style];

  const slidesHtml = slides.map((s, i) => {
    const bulletsHtml = s.bullets
      .map(b => `<li style="margin-bottom:8px;padding-left:4px;">${escHtml(b)}</li>`)
      .join('\n');

    return `
    <div class="slide" data-idx="${i}" style="
      display:${i === 0 ? 'flex' : 'none'};
      flex-direction:column;
      justify-content:center;
      width:100%;height:100%;
      padding:48px 60px;
      box-sizing:border-box;
      position:absolute;top:0;left:0;
      animation:fadeIn 0.35s ease;
    ">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.12em;color:${theme.accent};margin-bottom:16px;font-weight:600;">
        Slide ${i + 1} of ${slides.length}
      </div>
      <h1 style="font-size:32px;font-weight:700;margin:0 0 24px 0;line-height:1.3;color:${theme.text};">
        ${escHtml(s.title)}
      </h1>
      <ul style="font-size:20px;line-height:1.8;list-style:none;padding:0;margin:0;color:${theme.bullet};">
        ${bulletsHtml}
      </ul>
      ${s.notes ? `<div style="margin-top:auto;padding-top:20px;font-size:13px;color:${theme.bullet};opacity:0.5;font-style:italic;">${escHtml(s.notes)}</div>` : ''}
    </div>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Slide Deck</title>
<style>
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; overflow: hidden; }
  body {
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    background: ${theme.bg};
    color: ${theme.text};
    position: relative;
  }
  ul li::before {
    content: "\\2022";
    color: ${theme.accent};
    font-weight: bold;
    display: inline-block;
    width: 1em;
    margin-left: -0.2em;
  }
  .nav-btn {
    position: fixed; bottom: 20px; z-index: 100;
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15);
    color: white; font-size: 18px; cursor: pointer; width: 40px; height: 40px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    backdrop-filter: blur(8px); transition: background 0.2s;
  }
  .nav-btn:hover { background: rgba(255,255,255,0.2); }
  .counter {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); z-index: 100;
    font-size: 13px; color: rgba(255,255,255,0.5); font-weight: 500;
  }
</style>
</head>
<body>
${slidesHtml}
<button class="nav-btn" style="left:20px;" onclick="go(-1)">&larr;</button>
<button class="nav-btn" style="right:20px;" onclick="go(1)">&rarr;</button>
<div class="counter" id="counter">1 / ${slides.length}</div>
<script>
let cur = 0;
const total = ${slides.length};
function go(dir) {
  const next = cur + dir;
  if (next < 0 || next >= total) return;
  document.querySelector('[data-idx="'+cur+'"]').style.display = 'none';
  cur = next;
  const el = document.querySelector('[data-idx="'+cur+'"]');
  el.style.display = 'flex';
  el.style.animation = 'none';
  el.offsetHeight;
  el.style.animation = 'fadeIn 0.35s ease';
  document.getElementById('counter').textContent = (cur+1) + ' / ' + total;
}
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === ' ') go(1);
  if (e.key === 'ArrowLeft') go(-1);
});
</script>
</body>
</html>`;
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ─── System prompt for slide generation ─── */
const SLIDES_SYSTEM_PROMPT = `You are a presentation designer AI. Generate slide deck content as valid JSON only.
Return ONLY a JSON object in this exact format, nothing else:
{"slides":[{"title":"Slide Title","bullets":["Point 1","Point 2","Point 3"],"notes":"Optional speaker notes"},{"title":"...","bullets":["..."],"notes":"..."}]}

Rules:
- Each slide must have a title, 2-5 bullet points, and optional notes
- Keep bullet points concise (under 15 words each)
- First slide should be a title/intro slide
- Last slide should be a summary or conclusion
- Make content informative, well-structured, and engaging
- Return ONLY valid JSON, no markdown fences, no explanation`;

/* ─── Shared button style ─── */
const pillBtn = (active: boolean): React.CSSProperties => ({
  padding: '4px 10px',
  background: active ? 'var(--pg2)' : 'var(--s2)',
  border: `1px solid ${active ? 'var(--p)' : 'var(--b1)'}`,
  borderRadius: 'var(--r-pill)',
  color: active ? 'var(--pl)' : 'var(--mu)',
  fontSize: '11px',
  cursor: 'pointer',
  fontFamily: 'inherit',
  fontWeight: active ? 500 : 400,
  transition: 'all 0.15s ease',
});

const iconBtn: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 4,
  background: 'var(--s2)',
  border: '1px solid var(--b1)',
  borderRadius: 'var(--r-md)',
  color: 'var(--mu)',
  padding: '5px 8px',
  cursor: 'pointer',
  fontSize: '11px',
  fontFamily: 'inherit',
  transition: 'all 0.15s ease',
};

/* ─── Component ─── */
export default function SlidesPanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();

  // Input state
  const [topic, setTopic] = useState('');
  const [slideCount, setSlideCount] = useState<SlideCount>(5);
  const [style, setStyle] = useState<SlideStyle>('professional');

  // Slides state
  const [slides, setSlides] = useState<Slide[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [status, setStatus] = useState('');
  const [copied, setCopied] = useState(false);

  // Edit state
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [editBullets, setEditBullets] = useState('');
  const [editNotes, setEditNotes] = useState('');

  const iframeRef = useRef<HTMLIFrameElement>(null);
  const copiedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup
  useEffect(() => {
    return () => {
      if (copiedTimer.current) clearTimeout(copiedTimer.current);
    };
  }, []);

  // Keyboard navigation
  useEffect(() => {
    if (slides.length === 0 || editing) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') setCurrentIdx(i => Math.min(i + 1, slides.length - 1));
      if (e.key === 'ArrowLeft') setCurrentIdx(i => Math.max(i - 1, 0));
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [slides.length, editing]);

  // Update iframe when slides or currentIdx change
  useEffect(() => {
    if (slides.length === 0 || !iframeRef.current) return;
    const html = buildSlideshowHtml(slides, style, currentIdx);
    iframeRef.current.srcdoc = html;
  }, [slides, currentIdx, style]);

  /* ── Parse JSON from AI response ── */
  const parseSlides = useCallback((raw: string): Slide[] | null => {
    // Try to find JSON in the response
    let json = raw.trim();
    // Strip markdown fences if present
    json = json.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '');
    // Find the JSON object
    const start = json.indexOf('{');
    const end = json.lastIndexOf('}');
    if (start === -1 || end === -1) return null;
    json = json.slice(start, end + 1);
    try {
      const parsed = JSON.parse(json);
      if (parsed.slides && Array.isArray(parsed.slides)) {
        return parsed.slides.map((s: any) => ({
          title: s.title || 'Untitled',
          bullets: Array.isArray(s.bullets) ? s.bullets.map(String) : [],
          notes: s.notes || '',
        }));
      }
    } catch { /* ignore */ }
    return null;
  }, []);

  /* ── Generate slides from topic ── */
  const generate = useCallback(() => {
    if (!topic.trim()) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { setStatus('AURA is offline.'); return; }
    if (activeStream) return;

    setStatus('Generating slides...');
    setSlides([]);
    setCurrentIdx(0);
    setEditing(false);

    const prompt = `${SLIDES_SYSTEM_PROMPT}\n\nCreate a ${slideCount}-slide ${style} presentation about: ${topic.trim()}`;

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setStatus('Writing slides...'),
      onDone: (rawText) => {
        const parsed = parseSlides(rawText);
        if (parsed && parsed.length > 0) {
          setSlides(parsed);
          setStatus('');
        } else {
          setStatus('Failed to parse slides. Try again.');
        }
      },
    });

    ws!.send(JSON.stringify({
      type: 'chat',
      message: prompt,
      model: getModel('slides'),
      conversation_id: null,
    }));
  }, [topic, slideCount, style, wsReady, ws, activeStream, setActiveStream, getModel, parseSlides]);

  /* ── Generate from page content ── */
  const generateFromPage = useCallback(async () => {
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { setStatus('AURA is offline.'); return; }
    if (activeStream) return;

    setStatus('Loading page...');
    const resp = await sendMsg({ type: 'GET_PAGE_CONTENT' });
    if (!resp?.ok || !resp.text) {
      setStatus('Could not read page content.');
      return;
    }

    setSlides([]);
    setCurrentIdx(0);
    setEditing(false);
    setStatus('Generating slides from page...');

    const pageText = resp.text.slice(0, 12000);
    const prompt = `${SLIDES_SYSTEM_PROMPT}\n\nCreate a ${slideCount}-slide ${style} presentation based on this content:\n\nTitle: ${resp.title || 'Untitled'}\n\n${pageText}`;

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setStatus('Writing slides...'),
      onDone: (rawText) => {
        const parsed = parseSlides(rawText);
        if (parsed && parsed.length > 0) {
          setSlides(parsed);
          setStatus('');
        } else {
          setStatus('Failed to parse slides. Try again.');
        }
      },
    });

    ws!.send(JSON.stringify({
      type: 'chat',
      message: prompt,
      model: getModel('slides'),
      conversation_id: null,
    }));
  }, [slideCount, style, wsReady, ws, activeStream, setActiveStream, getModel, parseSlides]);

  /* ── Regenerate a single slide ── */
  const regenerateSlide = useCallback(() => {
    if (!wsReady || ws?.readyState !== WebSocket.OPEN || activeStream) return;
    if (slides.length === 0) return;

    const idx = currentIdx;
    const slideContext = slides.map((s, i) =>
      i === idx ? `[SLIDE ${i + 1} — REGENERATE THIS]` : `Slide ${i + 1}: ${s.title}`
    ).join('\n');

    setStatus(`Regenerating slide ${idx + 1}...`);

    const prompt = `${SLIDES_SYSTEM_PROMPT}\n\nI have a ${slides.length}-slide presentation. Regenerate ONLY slide ${idx + 1} with a fresh take. Return the full deck JSON.\n\nCurrent deck outline:\n${slideContext}\n\nThe overall topic context from slide 1: "${slides[0].title}".\nReturn the complete slide deck as JSON with all ${slides.length} slides, but make slide ${idx + 1} different and better.`;

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => {},
      onDone: (rawText) => {
        const parsed = parseSlides(rawText);
        if (parsed && parsed.length > 0) {
          // If the AI returned the right number, use the new one for the target slide
          if (parsed.length === slides.length) {
            setSlides(parsed);
          } else if (parsed.length >= idx + 1) {
            // Replace just the target slide
            const updated = [...slides];
            updated[idx] = parsed[idx];
            setSlides(updated);
          }
          setStatus('');
        } else {
          setStatus('Failed to regenerate. Try again.');
        }
      },
    });

    ws!.send(JSON.stringify({
      type: 'chat',
      message: prompt,
      model: getModel('slides'),
      conversation_id: null,
    }));
  }, [slides, currentIdx, wsReady, ws, activeStream, setActiveStream, getModel, parseSlides]);

  /* ── Edit slide ── */
  const startEdit = useCallback(() => {
    const slide = slides[currentIdx];
    if (!slide) return;
    setEditTitle(slide.title);
    setEditBullets(slide.bullets.join('\n'));
    setEditNotes(slide.notes);
    setEditing(true);
  }, [slides, currentIdx]);

  const saveEdit = useCallback(() => {
    const updated = [...slides];
    updated[currentIdx] = {
      title: editTitle,
      bullets: editBullets.split('\n').filter(l => l.trim()),
      notes: editNotes,
    };
    setSlides(updated);
    setEditing(false);
  }, [slides, currentIdx, editTitle, editBullets, editNotes]);

  const cancelEdit = useCallback(() => setEditing(false), []);

  /* ── Add / Delete slide ── */
  const addSlide = useCallback(() => {
    const newSlide: Slide = { title: 'New Slide', bullets: ['Point 1', 'Point 2'], notes: '' };
    const updated = [...slides];
    updated.splice(currentIdx + 1, 0, newSlide);
    setSlides(updated);
    setCurrentIdx(currentIdx + 1);
  }, [slides, currentIdx]);

  const deleteSlide = useCallback(() => {
    if (slides.length <= 1) return;
    const updated = slides.filter((_, i) => i !== currentIdx);
    setSlides(updated);
    setCurrentIdx(Math.min(currentIdx, updated.length - 1));
  }, [slides, currentIdx]);

  /* ── Copy / Download ── */
  const copyHtml = useCallback(async () => {
    if (slides.length === 0) return;
    const html = buildDownloadableHtml(slides, style);
    try {
      await navigator.clipboard.writeText(html);
      setCopied(true);
      if (copiedTimer.current) clearTimeout(copiedTimer.current);
      copiedTimer.current = setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  }, [slides, style]);

  const downloadHtml = useCallback(() => {
    if (slides.length === 0) return;
    const html = buildDownloadableHtml(slides, style);
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'slides.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [slides, style]);

  // Stream state for loading indicator
  const stream = useStore(s => s.activeStream);
  const isStreaming = stream && stream !== true && stream.type === 'write';

  /* ── Render ── */
  const hasSlides = slides.length > 0;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center gap-2 flex-shrink-0 px-3 py-2"
        style={{ borderBottom: '1px solid var(--b1)' }}
      >
        <Presentation size={15} style={{ color: 'var(--pl)' }} />
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--tx)' }}>AI Slides</span>
        <div style={{ flex: 1 }} />
        <ModelPill featureKey="slides" />
      </div>

      {/* Content area */}
      <div className="flex-1 flex flex-col gap-2 p-3 overflow-hidden">

        {/* Input section — show when no slides or always show compact */}
        {!hasSlides && (
          <>
            {/* Topic input */}
            <textarea
              value={topic}
              onChange={e => setTopic(e.target.value)}
              placeholder="Create a presentation about..."
              onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); } }}
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                color: 'var(--tx)',
                fontSize: '12.5px',
                padding: '8px 10px',
                resize: 'none',
                height: 72,
                outline: 'none',
                fontFamily: 'inherit',
                flexShrink: 0,
              }}
            />

            {/* Options: Slide count */}
            <div>
              <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 4 }}>
                Slides
              </div>
              <div className="flex gap-1">
                {([5, 10, 15] as SlideCount[]).map(n => (
                  <button key={n} onClick={() => setSlideCount(n)} style={pillBtn(slideCount === n)}>
                    {n}
                  </button>
                ))}
              </div>
            </div>

            {/* Options: Style */}
            <div>
              <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 4 }}>
                Style
              </div>
              <div className="flex flex-wrap gap-1">
                {(['professional', 'casual', 'academic', 'creative'] as SlideStyle[]).map(s => (
                  <button key={s} onClick={() => setStyle(s)} style={pillBtn(style === s)}>
                    {s.charAt(0).toUpperCase() + s.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex gap-2 flex-shrink-0">
              <button
                onClick={generate}
                disabled={!!activeStream || !topic.trim()}
                style={{
                  flex: 1,
                  background: (activeStream || !topic.trim()) ? 'var(--s3)' : 'var(--p)',
                  border: 'none',
                  borderRadius: 'var(--r-md)',
                  color: 'white',
                  padding: '8px 16px',
                  cursor: (activeStream || !topic.trim()) ? 'not-allowed' : 'pointer',
                  fontSize: '12px',
                  fontFamily: 'inherit',
                  fontWeight: 500,
                }}
              >
                {isStreaming ? 'Generating...' : 'Generate Slides'}
              </button>
              <button
                onClick={generateFromPage}
                disabled={!!activeStream}
                title="Generate from current page"
                style={{
                  ...iconBtn,
                  opacity: activeStream ? 0.5 : 1,
                  cursor: activeStream ? 'not-allowed' : 'pointer',
                }}
              >
                <FileText size={13} />
                <span>From Page</span>
              </button>
            </div>
          </>
        )}

        {/* Status */}
        {status && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', padding: '4px 0' }}>
            {isStreaming && <span className="dots" style={{ display: 'inline-flex', marginRight: 6 }}><span /><span /><span /></span>}
            {status}
          </div>
        )}

        {/* Slide viewer */}
        {hasSlides && !editing && (
          <>
            {/* Iframe preview */}
            <div
              style={{
                flex: 1,
                minHeight: 0,
                borderRadius: 'var(--r-md)',
                overflow: 'hidden',
                border: '1px solid var(--b1)',
                background: '#0a0a0f',
                position: 'relative',
              }}
            >
              <iframe
                ref={iframeRef}
                sandbox="allow-scripts"
                style={{
                  width: '100%',
                  height: '100%',
                  border: 'none',
                  display: 'block',
                }}
                title="Slide Preview"
              />
            </div>

            {/* Navigation */}
            <div className="flex items-center justify-center gap-3 flex-shrink-0" style={{ padding: '2px 0' }}>
              <button
                onClick={() => setCurrentIdx(i => Math.max(i - 1, 0))}
                disabled={currentIdx === 0}
                style={{
                  ...iconBtn,
                  opacity: currentIdx === 0 ? 0.3 : 1,
                  cursor: currentIdx === 0 ? 'default' : 'pointer',
                  padding: '5px 6px',
                }}
              >
                <ChevronLeft size={15} />
              </button>

              <span style={{ fontSize: '12px', color: 'var(--tx)', fontWeight: 500, minWidth: 50, textAlign: 'center' }}>
                {currentIdx + 1} / {slides.length}
              </span>

              <button
                onClick={() => setCurrentIdx(i => Math.min(i + 1, slides.length - 1))}
                disabled={currentIdx === slides.length - 1}
                style={{
                  ...iconBtn,
                  opacity: currentIdx === slides.length - 1 ? 0.3 : 1,
                  cursor: currentIdx === slides.length - 1 ? 'default' : 'pointer',
                  padding: '5px 6px',
                }}
              >
                <ChevronRight size={15} />
              </button>
            </div>

            {/* Slide thumbnails */}
            <div
              className="flex gap-1 flex-shrink-0 overflow-x-auto"
              style={{ padding: '2px 0' }}
            >
              {slides.map((s, i) => (
                <button
                  key={i}
                  onClick={() => setCurrentIdx(i)}
                  style={{
                    minWidth: 56,
                    maxWidth: 56,
                    height: 34,
                    borderRadius: 'var(--r-sm)',
                    border: `1.5px solid ${i === currentIdx ? 'var(--p)' : 'var(--b1)'}`,
                    background: i === currentIdx ? 'var(--pg)' : 'var(--s2)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    transition: 'all 0.15s ease',
                  }}
                >
                  <span style={{
                    fontSize: '8px',
                    color: i === currentIdx ? 'var(--pl)' : 'var(--mu)',
                    fontWeight: 600,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    padding: '0 3px',
                    fontFamily: 'inherit',
                  }}>
                    {i + 1}. {s.title}
                  </span>
                </button>
              ))}
            </div>

            {/* Action toolbar */}
            <div className="flex flex-wrap gap-1 flex-shrink-0">
              <button onClick={startEdit} style={iconBtn} title="Edit slide">
                <Edit3 size={12} /> <span>Edit</span>
              </button>
              <button onClick={addSlide} style={iconBtn} title="Add slide after current">
                <Plus size={12} /> <span>Add</span>
              </button>
              <button
                onClick={deleteSlide}
                disabled={slides.length <= 1}
                style={{ ...iconBtn, opacity: slides.length <= 1 ? 0.4 : 1 }}
                title="Delete current slide"
              >
                <Trash2 size={12} /> <span>Del</span>
              </button>
              <button
                onClick={regenerateSlide}
                disabled={!!activeStream}
                style={{ ...iconBtn, opacity: activeStream ? 0.4 : 1 }}
                title="Regenerate current slide"
              >
                <RefreshCw size={12} /> <span>Regen</span>
              </button>
              <div style={{ flex: 1 }} />
              <button onClick={copyHtml} style={iconBtn} title="Copy as HTML">
                {copied ? <Check size={12} style={{ color: '#22c55e' }} /> : <Copy size={12} />}
              </button>
              <button onClick={downloadHtml} style={iconBtn} title="Download HTML">
                <Download size={12} />
              </button>
            </div>

            {/* New deck button */}
            <button
              onClick={() => { setSlides([]); setCurrentIdx(0); setStatus(''); }}
              style={{
                ...iconBtn,
                width: '100%',
                justifyContent: 'center',
                fontSize: '11px',
                color: 'var(--mu)',
              }}
            >
              <Presentation size={12} /> New Deck
            </button>
          </>
        )}

        {/* Edit mode */}
        {hasSlides && editing && (
          <div className="flex flex-col gap-2 flex-1 overflow-y-auto" style={{ minHeight: 0 }}>
            <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--pl)', marginBottom: 2 }}>
              Editing Slide {currentIdx + 1}
            </div>

            <div>
              <label style={{ fontSize: '10px', color: 'var(--mu)', fontWeight: 500 }}>Title</label>
              <input
                value={editTitle}
                onChange={e => setEditTitle(e.target.value)}
                style={{
                  width: '100%',
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--tx)',
                  fontSize: '12.5px',
                  padding: '7px 10px',
                  outline: 'none',
                  fontFamily: 'inherit',
                  marginTop: 3,
                }}
              />
            </div>

            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <label style={{ fontSize: '10px', color: 'var(--mu)', fontWeight: 500 }}>Bullet Points (one per line)</label>
              <textarea
                value={editBullets}
                onChange={e => setEditBullets(e.target.value)}
                style={{
                  flex: 1,
                  width: '100%',
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--tx)',
                  fontSize: '12px',
                  padding: '7px 10px',
                  outline: 'none',
                  fontFamily: 'inherit',
                  resize: 'none',
                  marginTop: 3,
                  minHeight: 80,
                }}
              />
            </div>

            <div>
              <label style={{ fontSize: '10px', color: 'var(--mu)', fontWeight: 500 }}>Speaker Notes (optional)</label>
              <textarea
                value={editNotes}
                onChange={e => setEditNotes(e.target.value)}
                style={{
                  width: '100%',
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--tx)',
                  fontSize: '12px',
                  padding: '7px 10px',
                  outline: 'none',
                  fontFamily: 'inherit',
                  resize: 'none',
                  height: 48,
                  marginTop: 3,
                }}
              />
            </div>

            <div className="flex gap-2 flex-shrink-0">
              <button
                onClick={saveEdit}
                style={{
                  flex: 1,
                  background: 'var(--p)',
                  border: 'none',
                  borderRadius: 'var(--r-md)',
                  color: 'white',
                  padding: '7px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontFamily: 'inherit',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 4,
                }}
              >
                <Check size={13} /> Save
              </button>
              <button
                onClick={cancelEdit}
                style={{
                  ...iconBtn,
                  flex: 1,
                  justifyContent: 'center',
                }}
              >
                <X size={13} /> Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
