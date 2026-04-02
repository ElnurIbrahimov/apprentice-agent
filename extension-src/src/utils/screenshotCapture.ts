/**
 * screenshotCapture — captures iframe content as a base64 image
 * for visual AI feedback. Uses html2canvas for accurate rendering.
 */

import html2canvas from 'html2canvas';

interface CaptureOptions {
  maxWidth?: number;
  maxHeight?: number;
  quality?: number;
  /** Target max size in bytes (default 150KB) */
  maxBytes?: number;
}

/**
 * Capture a screenshot of an iframe's content as a base64 JPEG.
 * Returns a data URL (data:image/jpeg;base64,...).
 */
export async function captureIframeScreenshot(
  iframe: HTMLIFrameElement,
  options: CaptureOptions = {},
): Promise<string> {
  const {
    maxWidth = 800,
    maxHeight = 1200,
    quality = 0.7,
    maxBytes = 150_000,
  } = options;

  const doc = iframe.contentDocument;
  if (!doc?.body) {
    throw new Error('Cannot access iframe document');
  }

  // Capture using html2canvas
  const canvas = await html2canvas(doc.body, {
    width: Math.min(iframe.clientWidth || 1024, maxWidth),
    height: Math.min(
      Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight, 400),
      maxHeight,
    ),
    scale: 1,
    useCORS: true,
    logging: false,
    allowTaint: true,
    backgroundColor: getComputedStyle(doc.body).backgroundColor || '#ffffff',
  });

  // Resize if wider than maxWidth
  const resized = resizeCanvas(canvas, maxWidth, maxHeight);

  // Compress to target size
  return compressToTarget(resized, quality, maxBytes);
}

/**
 * Capture using the existing SVG foreignObject approach (faster, less accurate).
 * Fallback when html2canvas fails (e.g., cross-origin issues).
 */
export async function captureIframeFast(
  iframe: HTMLIFrameElement,
  options: CaptureOptions = {},
): Promise<string> {
  const { maxWidth = 800, maxHeight = 600, quality = 0.7 } = options;
  const doc = iframe.contentDocument;
  if (!doc?.documentElement) throw new Error('No iframe document');

  const clone = doc.documentElement.cloneNode(true) as HTMLElement;
  clone.querySelectorAll('script').forEach(n => n.remove());
  clone.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');

  const w = Math.min(doc.documentElement.scrollWidth || 800, maxWidth);
  const h = Math.min(doc.documentElement.scrollHeight || 600, maxHeight);

  const serialized = new XMLSerializer().serializeToString(clone);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><foreignObject width="100%" height="100%">${serialized}</foreignObject></svg>`;

  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('SVG render failed'));
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  });

  const canvas = document.createElement('canvas');
  canvas.width = maxWidth;
  canvas.height = Math.round(maxWidth * (h / w));
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  return canvas.toDataURL('image/jpeg', quality);
}

/**
 * Capture with fallback: try html2canvas first, fall back to SVG method.
 */
export async function captureScreenshot(
  iframe: HTMLIFrameElement,
  options: CaptureOptions = {},
): Promise<string> {
  try {
    return await captureIframeScreenshot(iframe, options);
  } catch {
    return await captureIframeFast(iframe, options);
  }
}

/** Strip the data URI prefix, returning only the base64 payload. */
export function stripDataUri(dataUrl: string): string {
  const comma = dataUrl.indexOf(',');
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

// ── Internal helpers ──

function resizeCanvas(source: HTMLCanvasElement, maxW: number, maxH: number): HTMLCanvasElement {
  if (source.width <= maxW && source.height <= maxH) return source;

  const scale = Math.min(maxW / source.width, maxH / source.height);
  const w = Math.round(source.width * scale);
  const h = Math.round(source.height * scale);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, w, h);
  return canvas;
}

function compressToTarget(canvas: HTMLCanvasElement, startQuality: number, maxBytes: number): string {
  let quality = startQuality;
  let result = canvas.toDataURL('image/jpeg', quality);

  // Iteratively reduce quality if over target
  while (result.length > maxBytes && quality > 0.2) {
    quality -= 0.1;
    result = canvas.toDataURL('image/jpeg', quality);
  }

  // If still too large, resize and try again
  if (result.length > maxBytes) {
    const smaller = resizeCanvas(canvas, Math.round(canvas.width * 0.6), Math.round(canvas.height * 0.6));
    result = smaller.toDataURL('image/jpeg', 0.5);
  }

  return result;
}
