import type { Message } from './types';

/** Strip markdown/HTML to plain text */
function stripMarkdown(text: string): string {
  return text
    // Remove code blocks (``` ... ```)
    .replace(/```[\s\S]*?```/g, (m) => m.replace(/```\w*\n?/g, '').replace(/```/g, ''))
    // Remove inline code
    .replace(/`([^`]+)`/g, '$1')
    // Remove bold/italic
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/_(.+?)_/g, '$1')
    // Remove headers
    .replace(/^#{1,6}\s+/gm, '')
    // Remove links — keep text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Remove images
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    // Remove blockquotes
    .replace(/^>\s+/gm, '')
    // Remove horizontal rules
    .replace(/^[-*_]{3,}\s*$/gm, '')
    .trim();
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleString();
}

function formatDate(ts?: number): string {
  return new Date(ts || Date.now()).toLocaleDateString('en-US', {
    year: 'numeric', month: 'long', day: 'numeric',
  });
}

/** Export full conversation as Markdown */
export function exportAsMarkdown(messages: Message[], modelName?: string): string {
  const date = messages.length ? formatDate(messages[0].timestamp) : formatDate();
  let out = `# AURA Chat Export\nDate: ${date}\n`;
  if (modelName) out += `Model: ${modelName}\n`;
  out += '\n---\n\n';

  for (const msg of messages) {
    const role = msg.role === 'user' ? 'User' : 'AURA';
    const time = formatTimestamp(msg.timestamp);
    out += `## ${role}\n*${time}*\n\n${msg.text}\n\n`;
  }
  return out;
}

/** Export full conversation as JSON */
export function exportAsJSON(messages: Message[], modelName?: string): string {
  const arr = messages.map((m) => ({
    role: m.role === 'user' ? 'user' : 'aura',
    content: m.text,
    timestamp: new Date(m.timestamp).toISOString(),
    ...(m.role === 'ai' && modelName ? { model: modelName } : {}),
    ...(m.thinkingContent ? { thinking: m.thinkingContent } : {}),
  }));
  return JSON.stringify(arr, null, 2);
}

/** Export full conversation as plain text */
export function exportAsText(messages: Message[]): string {
  return messages
    .map((m) => {
      const role = m.role === 'user' ? 'User' : 'AURA';
      const time = formatTimestamp(m.timestamp);
      const text = stripMarkdown(m.text);
      return `[${time}] ${role}:\n${text}`;
    })
    .join('\n\n');
}

/** Trigger a file download in the browser */
export function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Single-message export as Markdown content */
export function messageAsMarkdown(msg: Message): string {
  const role = msg.role === 'user' ? 'User' : 'AURA';
  const time = formatTimestamp(msg.timestamp);
  return `## ${role}\n*${time}*\n\n${msg.text}\n`;
}

/** Single-message export as plain text */
export function messageAsText(msg: Message): string {
  const role = msg.role === 'user' ? 'User' : 'AURA';
  return `${role}:\n${stripMarkdown(msg.text)}`;
}

/** Export full conversation — dispatches download based on format */
export function exportChat(messages: Message[], format: 'markdown' | 'json' | 'text', modelName?: string) {
  const ts = new Date().toISOString().slice(0, 10);
  switch (format) {
    case 'markdown': {
      const content = exportAsMarkdown(messages, modelName);
      downloadFile(content, `aura-chat-${ts}.md`, 'text/markdown;charset=utf-8');
      break;
    }
    case 'json': {
      const content = exportAsJSON(messages, modelName);
      downloadFile(content, `aura-chat-${ts}.json`, 'application/json;charset=utf-8');
      break;
    }
    case 'text': {
      const content = exportAsText(messages);
      downloadFile(content, `aura-chat-${ts}.txt`, 'text/plain;charset=utf-8');
      break;
    }
  }
}
