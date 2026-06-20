/**
 * OCR Overlay — extracted from page-services.ts
 * Provides the drag-to-select OCR region overlay UI.
 */

export interface OcrOverlayResult {
  ok: boolean;
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  dpr?: number;
}

// ── OCR Overlay ───────────────────────────────────────────────────────────────

export function showOcrOverlay(
  dataUrl: string,
  sendResponse: (result: OcrOverlayResult) => void
): void {
  // Create fullscreen overlay
  const overlay: HTMLDivElement = document.createElement('div');
  Object.assign(overlay.style, {
    position: 'fixed', top: '0', left: '0', width: '100vw', height: '100vh',
    zIndex: '2147483646', cursor: 'crosshair', background: 'rgba(0,0,0,0.4)',
  });

  // Show the screenshot as background for reference
  const img: HTMLImageElement = new Image();
  img.src = dataUrl;
  img.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;opacity:0.7;pointer-events:none;';
  overlay.appendChild(img);

  // Canvas for drawing selection rect
  const canvas: HTMLCanvasElement = document.createElement('canvas');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  Object.assign(canvas.style, {
    position: 'absolute', top: '0', left: '0', width: '100%', height: '100%',
  });
  overlay.appendChild(canvas);
  const ctx: CanvasRenderingContext2D | null = canvas.getContext('2d');

  const hint: HTMLDivElement = document.createElement('div');
  Object.assign(hint.style, {
    position: 'fixed', top: '12px', left: '50%', transform: 'translateX(-50%)',
    background: 'rgba(0,0,0,0.75)', color: '#fff', padding: '6px 14px',
    borderRadius: '6px', fontSize: '13px', pointerEvents: 'none',
  });
  hint.textContent = 'Drag to select region • Press Esc to cancel';
  overlay.appendChild(hint);

  document.body.appendChild(overlay);

  let startX = 0;
  let startY = 0;
  let dragging = false;
  const dpr: number = window.devicePixelRatio || 1;

  function drawRect(x: number, y: number, w: number, h: number): void {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#7c3aed';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = 'rgba(124,58,237,0.12)';
    ctx.fillRect(x, y, w, h);
  }

  overlay.addEventListener('mousedown', (e: MouseEvent) => {
    startX = e.clientX; startY = e.clientY; dragging = true;
  });

  overlay.addEventListener('mousemove', (e: MouseEvent) => {
    if (!dragging) return;
    drawRect(startX, startY, e.clientX - startX, e.clientY - startY);
  });

  // onEsc must be declared before mouseup so we can remove it from both paths
  function onEsc(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      if (document.body.contains(overlay)) document.body.removeChild(overlay);
      document.removeEventListener('keydown', onEsc);
      sendResponse({ ok: false });
    }
  }

  overlay.addEventListener('mouseup', (e: MouseEvent) => {
    dragging = false;
    const x: number = Math.min(startX, e.clientX);
    const y: number = Math.min(startY, e.clientY);
    const w: number = Math.abs(e.clientX - startX);
    const h: number = Math.abs(e.clientY - startY);
    // Clean up Esc listener on mouseup path too, or it leaks and may double-fire
    document.removeEventListener('keydown', onEsc);
    if (document.body.contains(overlay)) document.body.removeChild(overlay);
    if (w < 5 || h < 5) { sendResponse({ ok: false }); return; }
    sendResponse({ ok: true, x, y, w, h, dpr });
  });

  document.addEventListener('keydown', onEsc);

  // Safety cleanup: remove listener if overlay is removed by page navigation or other means
  const ocrCleanupObserver = new MutationObserver(() => {
    if (!document.body.contains(overlay)) {
      document.removeEventListener('keydown', onEsc);
      ocrCleanupObserver.disconnect();
    }
  });
  ocrCleanupObserver.observe(document.body, { childList: true });
}