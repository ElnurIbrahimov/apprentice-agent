/**
 * Dynamic favicon + tab title state machine.
 *
 * States:
 *   idle      → static AURA mark
 *   thinking  → animated pulsing ring (used while the assistant is streaming)
 *   attention → static mark + red dot (used when work finished while the tab was hidden)
 *
 * The favicon is drawn to a 64×64 offscreen canvas and swapped into
 * <link rel="icon"> via toDataURL. The tab title reflects the same state.
 */

export type FaviconState = 'idle' | 'thinking' | 'attention';

const SIZE = 64;
const ACCENT = '#7c3aed';
const ACCENT_SOFT = '#a78bfa';
const ATTENTION = '#ef4444';
const BG = '#030303';

let canvas: HTMLCanvasElement | null = null;
let linkEl: HTMLLinkElement | null = null;
let rafId: number | null = null;
let currentState: FaviconState = 'idle';
const BASE_TITLE = 'AURA';

function getCanvas(): HTMLCanvasElement {
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.width = SIZE;
    canvas.height = SIZE;
  }
  return canvas;
}

function getLink(): HTMLLinkElement {
  if (linkEl && document.head.contains(linkEl)) return linkEl;
  let el = document.querySelector<HTMLLinkElement>('link[rel~="icon"]');
  if (!el) {
    el = document.createElement('link');
    el.rel = 'icon';
    document.head.appendChild(el);
  }
  el.type = 'image/png';
  linkEl = el;
  return el;
}

/** Draw the AURA "A" mark into the given 2d context at the given rect. */
function drawMark(ctx: CanvasRenderingContext2D) {
  // Rounded background
  ctx.fillStyle = BG;
  const r = 14;
  ctx.beginPath();
  ctx.moveTo(r, 0);
  ctx.lineTo(SIZE - r, 0);
  ctx.quadraticCurveTo(SIZE, 0, SIZE, r);
  ctx.lineTo(SIZE, SIZE - r);
  ctx.quadraticCurveTo(SIZE, SIZE, SIZE - r, SIZE);
  ctx.lineTo(r, SIZE);
  ctx.quadraticCurveTo(0, SIZE, 0, SIZE - r);
  ctx.lineTo(0, r);
  ctx.quadraticCurveTo(0, 0, r, 0);
  ctx.closePath();
  ctx.fill();

  // Gradient "A"
  const g = ctx.createLinearGradient(0, 0, SIZE, SIZE);
  g.addColorStop(0, ACCENT);
  g.addColorStop(1, ACCENT_SOFT);
  ctx.fillStyle = g;
  ctx.font = 'bold 44px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('A', SIZE / 2, SIZE / 2 + 2);
}

function drawAttentionDot(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = ATTENTION;
  ctx.beginPath();
  ctx.arc(SIZE - 12, 12, 10, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = BG;
  ctx.lineWidth = 3;
  ctx.stroke();
}

function drawThinkingRing(ctx: CanvasRenderingContext2D, phase: number) {
  const cx = SIZE / 2;
  const cy = SIZE / 2;
  const radius = SIZE / 2 - 3;
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  // Dim base ring
  ctx.strokeStyle = 'rgba(139, 92, 246, 0.15)';
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  // Moving arc
  ctx.strokeStyle = ACCENT_SOFT;
  const start = phase;
  const end = phase + Math.PI * 0.9;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, start, end);
  ctx.stroke();
}

function render(state: FaviconState, phase = 0) {
  const c = getCanvas();
  const ctx = c.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, SIZE, SIZE);
  drawMark(ctx);
  if (state === 'thinking') drawThinkingRing(ctx, phase);
  if (state === 'attention') drawAttentionDot(ctx);
  try {
    getLink().href = c.toDataURL('image/png');
  } catch {
    // Canvas tainted or OOM — fall back silently; the old favicon persists.
  }
}

function updateTitle(state: FaviconState) {
  switch (state) {
    case 'thinking':
      document.title = `💭 Thinking... ${BASE_TITLE}`;
      break;
    case 'attention':
      document.title = `✓ Done — ${BASE_TITLE}`;
      break;
    default:
      document.title = BASE_TITLE;
  }
}

function stopAnim() {
  if (rafId != null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

function startThinkingAnim() {
  stopAnim();
  const start = performance.now();
  const tick = (t: number) => {
    const phase = ((t - start) / 1000) * Math.PI * 1.4;
    render('thinking', phase);
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}

/** Set the current favicon/title state. Idempotent. */
export function setFaviconState(state: FaviconState) {
  if (state === currentState) return;
  currentState = state;
  stopAnim();
  if (state === 'thinking') {
    startThinkingAnim();
  } else {
    render(state);
  }
  updateTitle(state);
}

export function getFaviconState(): FaviconState {
  return currentState;
}
