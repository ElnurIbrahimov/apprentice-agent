import type { Message } from '../types';

export function exportJSON(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function formatChatExport(
  messages: Array<Pick<Message, 'role' | 'text' | 'timestamp'>>,
): Array<{ role: string; text: string; timestamp: number }> {
  return messages.map(m => ({
    role: m.role,
    text: m.text || '',
    timestamp: m.timestamp,
  }));
}
