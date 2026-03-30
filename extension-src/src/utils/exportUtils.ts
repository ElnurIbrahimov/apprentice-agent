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
