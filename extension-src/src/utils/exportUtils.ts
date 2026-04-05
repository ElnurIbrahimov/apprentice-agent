/**
 * Shared export utilities for downloading panel content.
 */

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  // Delay revoke to let the browser queue the download
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

/** Export data as JSON file */
export function exportJSON(data: unknown, filename: string) {
  const json = JSON.stringify(data, null, 2);
  downloadBlob(new Blob([json], { type: 'application/json' }), filename);
}

/** Export HTML string as .html file */
export function exportHTML(html: string, filename: string) {
  downloadBlob(new Blob([html], { type: 'text/html' }), filename);
}

/** Export plain text */
export function exportText(text: string, filename: string) {
  downloadBlob(new Blob([text], { type: 'text/plain' }), filename);
}

/** Format chat messages for export */
export function formatChatExport(messages: Array<{ role: string; text?: string; content?: string; timestamp?: number }>) {
  return messages.map(m => ({
    role: m.role,
    text: m.text || m.content || '',
    time: m.timestamp ? new Date(m.timestamp).toISOString() : undefined,
  }));
}

/** Copy conversation to clipboard as formatted text */
export async function copyConversationToClipboard(messages: Array<{ role: string; text: string }>) {
  const text = messages
    .map(m => `${m.role === 'user' ? 'You' : 'AURA'}:\n${m.text}`)
    .join('\n\n');
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Fallback for older browsers
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }
}

/** Export conversation as PNG image */
export async function exportConversationAsImage(messages: Array<{ role: string; text: string; timestamp: number }>) {
  // Dynamic import to avoid loading html2canvas in every panel
  const { default: html2canvas } = await import('html2canvas');

  const container = document.createElement('div');
  container.style.cssText = 'position:fixed;left:-9999px;top:0;width:700px;padding:24px;background:#0a0a0c;color:#f0eff8;font-family:system-ui,sans-serif;font-size:13px;line-height:1.6';

  // Header
  const hdr = document.createElement('div');
  hdr.style.cssText = 'margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.08)';
  hdr.innerHTML = `<div style="font-size:16px;font-weight:600;margin-bottom:4px">AURA Chat Export</div>
    <div style="font-size:11px;color:#8e8eb0">${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</div>`;
  container.appendChild(hdr);

  // Messages
  for (const m of messages) {
    const el = document.createElement('div');
    el.style.cssText = 'margin-bottom:12px;padding:10px;background:rgba(255,255,255,0.03);border-radius:8px';
    const role = m.role === 'user' ? 'You' : 'AURA';
    const color = m.role === 'user' ? '#10b981' : '#7c3aed';
    el.innerHTML = `<div style="color:${color};font-weight:600;margin-bottom:4px;font-size:12px">${role}</div>
      <div style="white-space:pre-wrap;word-break:break-word">${m.text.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>`;
    container.appendChild(el);
  }

  document.body.appendChild(container);
  try {
    const canvas = await html2canvas(container, { backgroundColor: '#0a0a0c', scale: 2, logging: false });
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `aura-chat-${new Date().toISOString().slice(0, 10)}.png`;
    link.click();
  } finally {
    document.body.removeChild(container);
  }
}
