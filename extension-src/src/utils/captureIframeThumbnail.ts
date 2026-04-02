interface IframeThumbnailOptions<TFallback> {
  backgroundColor?: string;
  buildFallback: (fallback: TFallback) => string;
  fallback: TFallback;
  iframe: HTMLIFrameElement | null;
  targetHeight?: number;
  targetWidth?: number;
}

export async function captureIframeThumbnailDataUrl<TFallback>({
  backgroundColor = '#0f172a',
  buildFallback,
  fallback,
  iframe,
  targetHeight = 320,
  targetWidth = 640,
}: IframeThumbnailOptions<TFallback>): Promise<string> {
  if (!iframe?.contentDocument) {
    return buildFallback(fallback);
  }

  try {
    const doc = iframe.contentDocument;
    const clone = doc.documentElement.cloneNode(true) as HTMLElement;
    clone.querySelectorAll('script').forEach((node) => node.remove());
    clone.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');

    const width = Math.max(
      doc.documentElement.scrollWidth,
      doc.body?.scrollWidth || 0,
      iframe.clientWidth || 0,
      320,
    );
    const height = Math.max(
      doc.documentElement.scrollHeight,
      doc.body?.scrollHeight || 0,
      iframe.clientHeight || 0,
      180,
    );

    const serialized = new XMLSerializer().serializeToString(clone);
    const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <foreignObject width="100%" height="100%">${serialized}</foreignObject>
</svg>`;

    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('thumbnail image load failed'));
      img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
    });

    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('canvas unavailable');

    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const scale = Math.min(canvas.width / width, canvas.height / height);
    const drawWidth = width * scale;
    const drawHeight = height * scale;
    const dx = (canvas.width - drawWidth) / 2;
    const dy = (canvas.height - drawHeight) / 2;
    ctx.drawImage(img, dx, dy, drawWidth, drawHeight);

    return canvas.toDataURL('image/png', 0.85);
  } catch {
    return buildFallback(fallback);
  }
}
