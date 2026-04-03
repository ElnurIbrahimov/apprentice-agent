import type { Message } from '../types';

export function exportAsMarkdown(messages: Message[]): string {
  return messages
    .filter((m) => m.role !== 'system')
    .map((m) => {
      const role = m.role === 'user' ? '**You**' : '**Aura**';
      const time = new Date(m.timestamp).toLocaleString();
      return `### ${role} — ${time}\n\n${m.content}\n`;
    })
    .join('\n---\n\n');
}

export function exportAsJSON(messages: Message[]): string {
  return JSON.stringify(
    messages
      .filter((m) => m.role !== 'system')
      .map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        model_used: m.model_used || undefined,
        citations: m.citations?.length ? m.citations : undefined,
        attachments: m.attachments?.map((a) => ({ name: a.filename, type: a.type })),
      })),
    null,
    2
  );
}

export function downloadExport(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  // Delay revoke to ensure browser initiates download (Firefox needs this)
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
