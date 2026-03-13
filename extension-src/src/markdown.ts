function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function safeUrl(url: string): string {
  // After esc(), &quot; could break out of href="…" in the HTML parser.
  // Re-encode HTML entities that are dangerous inside attribute values.
  return url
    .replace(/&quot;/g, '%22')
    .replace(/&lt;/g, '%3C')
    .replace(/&gt;/g, '%3E')
    .replace(/&#39;/g, '%27')
    .replace(/&amp;/g, '&');
}

function inline(s: string): string {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/_([^_\n]+)_/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(
      /\[(.+?)\]\((https?:\/\/[^)]+)\)/g,
      (_, text, url) => `<a href="${safeUrl(url)}" target="_blank" rel="noopener noreferrer">${text}</a>`
    );
}

export function md(raw: string): string {
  let out = '';
  let listBuf = '';
  let listType: 'ul' | 'ol' | null = null;

  const flush = () => {
    if (!listType) return;
    out += `<${listType}>${listBuf}</${listType}>`;
    listBuf = '';
    listType = null;
  };

  const lines = raw.split('\n');
  let i = 0;

  while (i < lines.length) {
    const l = lines[i];

    if (/^```/.test(l)) {
      flush();
      const lang = l.slice(3).trim() || 'text';
      let code = '';
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) {
        code += lines[i] + '\n';
        i++;
      }
      const cid = 'c' + Math.random().toString(36).slice(2, 8);
      out += `<pre><div class="chdr"><span>${esc(lang)}</span><button class="ccopy" onclick="window.cpCode('${cid}')">Copy</button></div><code id="${cid}">${esc(code.replace(/\n$/, ''))}</code></pre>`;
      i++;
      continue;
    }

    const hm = l.match(/^(#{1,3})\s+(.+)/);
    if (hm) {
      flush();
      out += `<h${hm[1].length}>${inline(hm[2])}</h${hm[1].length}>`;
      i++;
      continue;
    }

    if (/^---+$/.test(l.trim())) {
      flush();
      out += '<hr>';
      i++;
      continue;
    }

    const bq = l.match(/^>\s*(.*)/);
    if (bq) {
      flush();
      out += `<blockquote>${inline(bq[1])}</blockquote>`;
      i++;
      continue;
    }

    const ul = l.match(/^[-*]\s+(.+)/);
    if (ul) {
      if (listType !== 'ul') {
        flush();
        listType = 'ul';
      }
      listBuf += `<li>${inline(ul[1])}</li>`;
      i++;
      continue;
    }

    const ol = l.match(/^\d+\.\s+(.+)/);
    if (ol) {
      if (listType !== 'ol') {
        flush();
        listType = 'ol';
      }
      listBuf += `<li>${inline(ol[1])}</li>`;
      i++;
      continue;
    }

    if (l.trim() === '') {
      flush();
      out += '<p></p>';
      i++;
      continue;
    }

    flush();
    out += `<p>${inline(l)}</p>`;
    i++;
  }

  flush();
  return out.replace(/(<p><\/p>){2,}/g, '<p></p>');
}

// Global copy function for code blocks
(window as any).cpCode = function (id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent || '').then(() => {
    const btn = el.closest('pre')?.querySelector('.ccopy') as HTMLButtonElement | null;
    if (btn) {
      btn.textContent = 'Copied!';
      setTimeout(() => (btn.textContent = 'Copy'), 1500);
    }
  }).catch(e => console.warn('[Copy] Failed:', e));
};
