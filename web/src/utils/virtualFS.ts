/**
 * Virtual File System for multi-file web projects.
 * Uses localStorage for persistence (adapted from extension's chrome.storage version).
 */

export interface VirtualFile {
  path: string;
  content: string;
  language: string;
  createdAt: number;
  updatedAt: number;
}

export interface VirtualProject {
  id: string;
  name: string;
  files: Map<string, VirtualFile>;
  entryPoint: string;
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'aura-web-creator-projects';
const MAX_FILES = 50;
const MAX_FILE_SIZE = 200 * 1024; // 200KB

function detectLanguage(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase() || '';
  const map: Record<string, string> = {
    html: 'html', htm: 'html', css: 'css', js: 'javascript', jsx: 'jsx',
    ts: 'typescript', tsx: 'tsx', json: 'json', md: 'markdown', svg: 'svg',
    py: 'python', yaml: 'yaml', yml: 'yaml',
  };
  return map[ext] || 'plaintext';
}

export class VirtualFS {
  private project: VirtualProject;

  constructor(id?: string, name?: string) {
    this.project = {
      id: id || `proj-${Date.now()}`,
      name: name || 'Untitled Project',
      files: new Map(),
      entryPoint: 'index.html',
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
  }

  get id() { return this.project.id; }
  get name() { return this.project.name; }
  get entryPoint() { return this.project.entryPoint; }
  get files() { return this.project.files; }

  setName(name: string) { this.project.name = name; }

  createFile(path: string, content: string): boolean {
    if (this.project.files.size >= MAX_FILES) return false;
    if (content.length > MAX_FILE_SIZE) return false;
    const now = Date.now();
    this.project.files.set(path, {
      path, content, language: detectLanguage(path),
      createdAt: now, updatedAt: now,
    });
    this.project.updatedAt = now;
    return true;
  }

  readFile(path: string): string | null {
    return this.project.files.get(path)?.content ?? null;
  }

  updateFile(path: string, content: string): boolean {
    const file = this.project.files.get(path);
    if (!file) return this.createFile(path, content);
    if (content.length > MAX_FILE_SIZE) return false;
    file.content = content;
    file.updatedAt = Date.now();
    this.project.updatedAt = Date.now();
    return true;
  }

  deleteFile(path: string): boolean {
    const result = this.project.files.delete(path);
    if (result) this.project.updatedAt = Date.now();
    return result;
  }

  listFiles(): string[] {
    return Array.from(this.project.files.keys()).sort();
  }

  getFile(path: string): VirtualFile | undefined {
    return this.project.files.get(path);
  }

  /** Build a single HTML bundle from the project files for preview. */
  buildBundle(): string {
    const entry = this.readFile(this.project.entryPoint);
    if (!entry) return '<html><body><p>No entry point found</p></body></html>';

    let html = entry;

    // Inline CSS files referenced via <link>
    // Escape </style> inside content to prevent tag breakout
    html = html.replace(/<link\s+[^>]*href=["']([^"']+\.css)["'][^>]*>/g, (_match, href) => {
      const css = this.readFile(href);
      if (!css) return _match;
      const safe = css.replace(/<\/style>/gi, '<\\/style>');
      return `<style>/* ${href} */\n${safe}</style>`;
    });

    // Inline JS files referenced via <script src>
    // Escape </script> inside content to prevent tag breakout
    html = html.replace(/<script\s+[^>]*src=["']([^"']+\.js)["'][^>]*>\s*<\/script>/g, (_match, src) => {
      const js = this.readFile(src);
      if (!js) return _match;
      const safe = js.replace(/<\/script>/gi, '<\\/script>');
      return `<script>/* ${src} */\n${safe}<\/script>`;
    });

    return html;
  }

  /** Save project to localStorage. */
  save(): void {
    const data = {
      ...this.project,
      files: Object.fromEntries(this.project.files),
    };
    try {
      const all = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      all[this.project.id] = data;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    } catch (e) {
      console.warn('[VirtualFS] Save failed:', e);
    }
  }

  /** Load a project from localStorage. */
  static load(id: string): VirtualFS | null {
    try {
      const all = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      const data = all[id];
      if (!data) return null;
      const fs = new VirtualFS(data.id, data.name);
      fs.project.entryPoint = data.entryPoint || 'index.html';
      fs.project.createdAt = data.createdAt;
      fs.project.updatedAt = data.updatedAt;
      for (const [path, file] of Object.entries(data.files)) {
        fs.project.files.set(path, file as VirtualFile);
      }
      return fs;
    } catch {
      return null;
    }
  }

  /** List all saved project IDs and names. */
  static listProjects(): Array<{ id: string; name: string; updatedAt: number }> {
    try {
      const all = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      return Object.values(all).map((p: any) => ({
        id: p.id, name: p.name, updatedAt: p.updatedAt,
      }));
    } catch {
      return [];
    }
  }

  /** Delete a project from localStorage. */
  static deleteProject(id: string): void {
    try {
      const all = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      delete all[id];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    } catch {}
  }
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/** Create a default starter web project. */
export function createDefaultWebProject(name = 'My Website'): VirtualFS {
  const safeName = escapeHtml(name);
  const fs = new VirtualFS(undefined, name);
  fs.createFile('index.html', `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${safeName}</title>
  <link rel="stylesheet" href="styles/main.css">
</head>
<body>
  <header>
    <h1>${safeName}</h1>
    <nav>
      <a href="#">Home</a>
      <a href="#">About</a>
      <a href="#">Contact</a>
    </nav>
  </header>
  <main>
    <p>Welcome to your new website!</p>
  </main>
  <script src="scripts/main.js"><\/script>
</body>
</html>`);

  fs.createFile('styles/main.css', `* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; color: #333; }
header { background: #7c3aed; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
header h1 { font-size: 1.25rem; }
header nav a { color: white; text-decoration: none; margin-left: 1.5rem; opacity: 0.8; }
header nav a:hover { opacity: 1; }
main { max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
`);

  fs.createFile('scripts/main.js', `// Main Script
console.log('Project loaded');
`);

  return fs;
}
