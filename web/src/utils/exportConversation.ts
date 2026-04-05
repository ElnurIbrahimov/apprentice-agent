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

/** Escape HTML special chars for safe inline injection */
function escHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Very simple Markdown→HTML for export: bold, inline code, links, code fences, paragraphs */
function mdToHtml(md: string): string {
  const lines = md.split('\n');
  const out: string[] = [];
  let inCode = false;
  let codeLang = '';
  let codeBuf: string[] = [];

  for (const rawLine of lines) {
    // Code fence start/end
    const fenceMatch = /^```(\w*)/.exec(rawLine);
    if (fenceMatch && !inCode) {
      inCode = true;
      codeLang = fenceMatch[1] || '';
      codeBuf = [];
      continue;
    }
    if (inCode) {
      if (rawLine.startsWith('```')) {
        const langAttr = codeLang ? ` class="language-${escHtml(codeLang)}"` : '';
        out.push(`<pre><code${langAttr}>${escHtml(codeBuf.join('\n'))}</code></pre>`);
        inCode = false;
        codeLang = '';
        codeBuf = [];
      } else {
        codeBuf.push(rawLine);
      }
      continue;
    }

    let line = rawLine;
    // Headings
    const h3 = /^### (.+)/.exec(line);
    const h2 = /^## (.+)/.exec(line);
    const h1 = /^# (.+)/.exec(line);
    if (h3) { out.push(`<h3>${escHtml(h3[1])}</h3>`); continue; }
    if (h2) { out.push(`<h2>${escHtml(h2[1])}</h2>`); continue; }
    if (h1) { out.push(`<h1>${escHtml(h1[1])}</h1>`); continue; }
    // Horizontal rule
    if (/^---+$/.test(line.trim())) { out.push('<hr>'); continue; }
    // Empty line
    if (!line.trim()) { out.push('<br>'); continue; }

    // Inline: escape first, then apply patterns (so we don't double-escape)
    line = escHtml(line);
    // Bold
    line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    line = line.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Inline code
    line = line.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Links
    line = line.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    out.push(`<p>${line}</p>`);
  }

  return out.join('\n');
}

export function exportAsHTML(messages: Message[], title = 'Aura Conversation'): string {
  const filtered = messages.filter((m) => m.role !== 'system');
  const date = new Date().toLocaleString();

  const bubbles = filtered.map((m) => {
    const isUser = m.role === 'user';
    const name = isUser ? 'You' : 'AURA';
    const time = new Date(m.timestamp).toLocaleTimeString();
    const model = !isUser && m.model_used ? `<span class="model">${escHtml(m.model_used)}</span>` : '';
    const body = isUser
      ? `<p class="user-text">${escHtml(m.content).replace(/\n/g, '<br>')}</p>`
      : mdToHtml(m.content);

    return `
    <div class="msg ${isUser ? 'msg-user' : 'msg-aura'}">
      <div class="msg-header">
        <span class="msg-name">${name}</span>
        ${model}
        <span class="msg-time">${time}</span>
      </div>
      <div class="msg-body">${body}</div>
    </div>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escHtml(title)}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0d0d12;
      color: #e2e2e8;
      min-height: 100vh;
      padding: 24px 16px;
    }
    .container { max-width: 780px; margin: 0 auto; }
    header { margin-bottom: 28px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.08); }
    header h1 { font-size: 1.25rem; font-weight: 600; color: #fff; }
    header p { font-size: 0.75rem; color: #888; margin-top: 4px; }
    .msg { margin-bottom: 20px; }
    .msg-aura { padding: 16px 20px; background: rgba(255,255,255,0.04); border-radius: 12px; border: 1px solid rgba(255,255,255,0.07); }
    .msg-user { display: flex; justify-content: flex-end; }
    .msg-user .msg-header { display: none; }
    .msg-user .msg-body { background: linear-gradient(135deg,#7c3aed,#6d28d9); border-radius: 20px 20px 4px 20px; padding: 10px 18px; max-width: 80%; }
    .msg-user .user-text { color: #fff; font-size: 0.9375rem; line-height: 1.6; white-space: pre-wrap; }
    .msg-header { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
    .msg-name { font-size: 0.8125rem; font-weight: 600; color: #c4b5fd; }
    .model { font-size: 0.6875rem; color: #555; background: rgba(255,255,255,0.05); padding: 1px 6px; border-radius: 999px; }
    .msg-time { font-size: 0.6875rem; color: #555; margin-left: auto; }
    .msg-body p { font-size: 0.9375rem; line-height: 1.7; color: #d1d1db; margin-bottom: 8px; }
    .msg-body p:last-child { margin-bottom: 0; }
    .msg-body h1, .msg-body h2, .msg-body h3 { color: #fff; margin: 14px 0 6px; font-weight: 600; }
    .msg-body h1 { font-size: 1.2rem; }
    .msg-body h2 { font-size: 1.05rem; }
    .msg-body h3 { font-size: 0.95rem; }
    .msg-body strong { color: #e9d5ff; font-weight: 600; }
    .msg-body em { color: #c4b5fd; font-style: italic; }
    .msg-body code { background: rgba(139,92,246,0.15); color: #c4b5fd; padding: 1px 6px; border-radius: 4px; font-family: 'Fira Code', 'Cascadia Code', monospace; font-size: 0.875em; }
    .msg-body pre { background: #1a1a24; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px 16px; overflow-x: auto; margin: 10px 0; }
    .msg-body pre code { background: none; padding: 0; color: #a5b4fc; font-size: 0.875rem; line-height: 1.6; }
    .msg-body a { color: #a78bfa; text-decoration: underline; }
    .msg-body hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 14px 0; }
    .msg-body br { display: block; content: ''; margin: 4px 0; }
    footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.75rem; color: #444; text-align: center; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>${escHtml(title)}</h1>
      <p>Exported ${date} &middot; ${filtered.length} messages</p>
    </header>
    ${bubbles}
    <footer>Generated by AURA</footer>
  </div>
</body>
</html>`;
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
