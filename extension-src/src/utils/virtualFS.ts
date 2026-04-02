declare const browser: any;

const ext = typeof chrome !== 'undefined' ? chrome : typeof browser !== 'undefined' ? browser : null;

const PROJECT_STORAGE_PREFIX = 'aura_projects_';
const PROJECT_INDEX_KEY = 'aura_project_index';
const MAX_PROJECT_FILES = 50;
const MAX_FILE_BYTES = 200 * 1024;
const MAX_PROJECT_BYTES = 2 * 1024 * 1024;

export type VirtualFramework = 'static' | 'react' | 'vue' | 'svelte' | 'nextjs' | 'custom';

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
  framework: VirtualFramework;
  createdAt: number;
  updatedAt: number;
}

export interface SerializedVirtualProject {
  id: string;
  name: string;
  files: VirtualFile[];
  directories: string[];
  entryPoint: string;
  framework: VirtualFramework;
  createdAt: number;
  updatedAt: number;
}

interface CreateProjectOptions {
  entryPoint?: string;
  files?: Record<string, string>;
  framework?: VirtualFramework;
  id?: string;
  name?: string;
}

function normalizePath(path: string): string {
  const normalized = path
    .replace(/\\/g, '/')
    .replace(/^\.\//, '')
    .replace(/^\/+/, '')
    .replace(/\/{2,}/g, '/')
    .trim();

  if (!normalized) {
    throw new Error('Path cannot be empty');
  }

  const segments = normalized.split('/');
  if (segments.some((segment) => !segment || segment === '.' || segment === '..')) {
    throw new Error('Invalid path');
  }

  return normalized;
}

function dirname(path: string): string {
  const normalized = normalizePath(path);
  const parts = normalized.split('/');
  parts.pop();
  return parts.join('/');
}

function basename(path: string): string {
  return normalizePath(path).split('/').pop() || path;
}

function resolveRelativePath(fromPath: string, target: string): string {
  if (/^(https?:|data:|blob:|#|mailto:|tel:)/i.test(target)) return target;
  const normalizedTarget = target.replace(/\\/g, '/').trim();
  if (!normalizedTarget || normalizedTarget.startsWith('/')) {
    return normalizedTarget.replace(/^\/+/, '');
  }

  const fromDir = dirname(fromPath);
  const baseSegments = fromDir ? fromDir.split('/') : [];
  for (const segment of normalizedTarget.split('/')) {
    if (!segment || segment === '.') continue;
    if (segment === '..') {
      baseSegments.pop();
      continue;
    }
    baseSegments.push(segment);
  }
  return baseSegments.join('/');
}

function getByteSize(value: string): number {
  return new Blob([value]).size;
}

function inferLanguageFromExtension(path: string): string {
  const normalized = normalizePath(path).toLowerCase();
  if (normalized.endsWith('.html') || normalized.endsWith('.htm')) return 'html';
  if (normalized.endsWith('.css')) return 'css';
  if (normalized.endsWith('.js') || normalized.endsWith('.mjs')) return 'javascript';
  if (normalized.endsWith('.ts')) return 'typescript';
  if (normalized.endsWith('.jsx')) return 'jsx';
  if (normalized.endsWith('.tsx')) return 'tsx';
  if (normalized.endsWith('.json')) return 'json';
  if (normalized.endsWith('.md')) return 'markdown';
  if (normalized.endsWith('.svg')) return 'svg';
  if (normalized.endsWith('.py')) return 'python';
  if (normalized.endsWith('.txt')) return 'text';
  return 'text';
}

export function detectCodeLanguageFromPath(path: string): string {
  return inferLanguageFromExtension(path);
}

function cloneProject(project: VirtualProject): VirtualProject {
  return {
    ...project,
    files: new Map(
      Array.from(project.files.entries()).map(([path, file]) => [path, { ...file }]),
    ),
  };
}

export class VirtualFS {
  private project: VirtualProject;
  private directories: Set<string>;

  private constructor(project: VirtualProject, directories?: Iterable<string>) {
    this.project = cloneProject(project);
    this.directories = new Set(
      Array.from(directories || []).map((dir) => normalizePath(dir)).filter(Boolean),
    );

    for (const filePath of this.project.files.keys()) {
      this.ensureDirectoryChain(dirname(filePath));
    }
  }

  static createProject(options: CreateProjectOptions = {}): VirtualFS {
    const createdAt = Date.now();
    const project: VirtualProject = {
      id: options.id || crypto.randomUUID(),
      name: options.name || 'Untitled project',
      files: new Map(),
      entryPoint: options.entryPoint || 'index.html',
      framework: options.framework || 'static',
      createdAt,
      updatedAt: createdAt,
    };

    const fs = new VirtualFS(project);
    for (const [path, content] of Object.entries(options.files || {})) {
      fs.createFile(path, content);
    }
    if (!fs.project.files.has(project.entryPoint) && project.entryPoint) {
      fs.createFile(project.entryPoint, '');
    }
    return fs;
  }

  static fromJSON(json: string): VirtualFS {
    const parsed = JSON.parse(json) as SerializedVirtualProject;
    return VirtualFS.fromSerializable(parsed);
  }

  static fromSerializable(project: SerializedVirtualProject): VirtualFS {
    return new VirtualFS(
      {
        id: project.id,
        name: project.name,
        files: new Map(
          project.files.map((file) => [normalizePath(file.path), { ...file, path: normalizePath(file.path) }]),
        ),
        entryPoint: normalizePath(project.entryPoint),
        framework: project.framework,
        createdAt: project.createdAt,
        updatedAt: project.updatedAt,
      },
      project.directories,
    );
  }

  private touch(): void {
    this.project.updatedAt = Date.now();
  }

  private ensureDirectoryChain(dirPath: string): void {
    if (!dirPath) return;
    const parts = normalizePath(dirPath).split('/');
    let current = '';
    for (const part of parts) {
      current = current ? `${current}/${part}` : part;
      this.directories.add(current);
    }
  }

  private validateProjectLimits(): void {
    if (this.project.files.size > MAX_PROJECT_FILES) {
      throw new Error(`Projects are limited to ${MAX_PROJECT_FILES} files`);
    }

    let totalBytes = 0;
    for (const file of this.project.files.values()) {
      const fileBytes = getByteSize(file.content);
      if (fileBytes > MAX_FILE_BYTES) {
        throw new Error(`${file.path} exceeds the 200KB per-file limit`);
      }
      totalBytes += fileBytes;
    }

    if (totalBytes > MAX_PROJECT_BYTES) {
      throw new Error('Project exceeds the 2MB storage limit');
    }
  }

  private pruneEmptyDirectories(): void {
    const next = new Set<string>();
    for (const filePath of this.project.files.keys()) {
      let current = dirname(filePath);
      while (current) {
        next.add(current);
        current = dirname(current);
      }
    }
    this.directories = next;
  }

  createFile(path: string, content: string): void {
    const normalized = normalizePath(path);
    const existing = this.project.files.get(normalized);
    const now = Date.now();
    const nextFile: VirtualFile = {
      path: normalized,
      content,
      language: inferLanguageFromExtension(normalized),
      createdAt: existing?.createdAt || now,
      updatedAt: now,
    };

    this.project.files.set(normalized, nextFile);
    this.ensureDirectoryChain(dirname(normalized));
    this.touch();
    this.validateProjectLimits();
  }

  readFile(path: string): string | null {
    return this.project.files.get(normalizePath(path))?.content || null;
  }

  updateFile(path: string, content: string): void {
    const normalized = normalizePath(path);
    if (!this.project.files.has(normalized)) {
      this.createFile(normalized, content);
      return;
    }

    const current = this.project.files.get(normalized)!;
    this.project.files.set(normalized, {
      ...current,
      content,
      updatedAt: Date.now(),
    });
    this.touch();
    this.validateProjectLimits();
  }

  deleteFile(path: string): void {
    this.project.files.delete(normalizePath(path));
    this.touch();
    this.pruneEmptyDirectories();
  }

  deleteDir(dirPath: string): void {
    const normalized = normalizePath(dirPath);
    for (const filePath of this.listFiles()) {
      if (filePath === normalized || filePath.startsWith(`${normalized}/`)) {
        this.project.files.delete(filePath);
      }
    }
    for (const dir of Array.from(this.directories)) {
      if (dir === normalized || dir.startsWith(`${normalized}/`)) {
        this.directories.delete(dir);
      }
    }
    this.touch();
    this.pruneEmptyDirectories();
  }

  renameFile(oldPath: string, newPath: string): void {
    const oldNormalized = normalizePath(oldPath);
    const nextNormalized = normalizePath(newPath);
    const current = this.project.files.get(oldNormalized);
    if (!current) return;

    this.project.files.delete(oldNormalized);
    this.project.files.set(nextNormalized, {
      ...current,
      path: nextNormalized,
      language: inferLanguageFromExtension(nextNormalized),
      updatedAt: Date.now(),
    });

    if (this.project.entryPoint === oldNormalized) {
      this.project.entryPoint = nextNormalized;
    }

    this.ensureDirectoryChain(dirname(nextNormalized));
    this.touch();
    this.pruneEmptyDirectories();
    this.validateProjectLimits();
  }

  listFiles(): string[] {
    return Array.from(this.project.files.keys()).sort((a, b) => a.localeCompare(b));
  }

  listDirectories(): string[] {
    return Array.from(this.directories).sort((a, b) => a.localeCompare(b));
  }

  listDir(dirPath = ''): string[] {
    const normalized = dirPath ? normalizePath(dirPath) : '';
    const prefix = normalized ? `${normalized}/` : '';
    const directChildren = new Set<string>();

    for (const dir of this.directories) {
      if (normalized && !dir.startsWith(prefix)) continue;
      const remainder = normalized ? dir.slice(prefix.length) : dir;
      if (!remainder || remainder.includes('/')) continue;
      directChildren.add(remainder);
    }

    for (const filePath of this.project.files.keys()) {
      if (normalized && !filePath.startsWith(prefix)) continue;
      const remainder = normalized ? filePath.slice(prefix.length) : filePath;
      if (!remainder || remainder.includes('/')) continue;
      directChildren.add(remainder);
    }

    return Array.from(directChildren).sort((a, b) => a.localeCompare(b));
  }

  createDir(dirPath: string): void {
    this.ensureDirectoryChain(dirPath);
    this.touch();
  }

  getProject(): VirtualProject {
    return cloneProject(this.project);
  }

  toSerializable(): SerializedVirtualProject {
    return {
      id: this.project.id,
      name: this.project.name,
      files: Array.from(this.project.files.values()).map((file) => ({ ...file })),
      directories: this.listDirectories(),
      entryPoint: this.project.entryPoint,
      framework: this.project.framework,
      createdAt: this.project.createdAt,
      updatedAt: this.project.updatedAt,
    };
  }

  toJSON(): string {
    return JSON.stringify(this.toSerializable());
  }

  buildBundle(): string {
    const entryPath = this.project.entryPoint;
    const entryHtml = this.readFile(entryPath) || '';
    if (!entryHtml.trim()) return '';

    let bundled = entryHtml;

    bundled = bundled.replace(
      /<link\b([^>]*?)rel=["']stylesheet["']([^>]*?)href=["']([^"']+)["']([^>]*?)>/gi,
      (_, beforeRel: string, between: string, href: string, afterHref: string) => {
        const resolvedPath = resolveRelativePath(entryPath, href);
        const css = this.readFile(resolvedPath);
        if (css == null) {
          return `<link${beforeRel}rel="stylesheet"${between}href="${href}"${afterHref}>`;
        }
        return `<style data-aura-source="${resolvedPath}">\n${css}\n</style>`;
      },
    );

    bundled = bundled.replace(
      /<script\b([^>]*?)src=["']([^"']+)["']([^>]*)>\s*<\/script>/gi,
      (_, beforeSrc: string, src: string, afterSrc: string) => {
        if (/^(https?:|data:|blob:)/i.test(src)) {
          return `<script${beforeSrc}src="${src}"${afterSrc}></script>`;
        }

        const resolvedPath = resolveRelativePath(entryPath, src);
        const js = this.readFile(resolvedPath);
        if (js == null) {
          return `<script${beforeSrc}src="${src}"${afterSrc}></script>`;
        }

        const attrs = `${beforeSrc}${afterSrc}`.replace(/\s+type=["']module["']/i, '');
        return `<script${attrs}>\n${js}\n</script>`;
      },
    );

    return bundled;
  }

  async save(): Promise<void> {
    if (!ext?.storage?.local) return;
    const serialized = this.toJSON();
    const key = `${PROJECT_STORAGE_PREFIX}${this.project.id}`;

    const existingIndex = await new Promise<string[]>((resolve) => {
      ext.storage.local.get([PROJECT_INDEX_KEY], (data: any) => resolve(data?.[PROJECT_INDEX_KEY] || []));
    });

    const nextIndex = Array.from(new Set([...existingIndex, this.project.id]));

    await new Promise<void>((resolve) => {
      ext.storage.local.set(
        {
          [key]: serialized,
          [PROJECT_INDEX_KEY]: nextIndex,
        },
        () => resolve(),
      );
    });
  }

  static async load(id: string): Promise<VirtualFS | null> {
    if (!ext?.storage?.local) return null;
    const key = `${PROJECT_STORAGE_PREFIX}${id}`;

    const raw = await new Promise<string | null>((resolve) => {
      ext.storage.local.get([key], (data: any) => resolve(data?.[key] || null));
    });

    return raw ? VirtualFS.fromJSON(raw) : null;
  }
}

export function createDefaultWebProject(name = 'Untitled project'): VirtualFS {
  return VirtualFS.createProject({
    name,
    framework: 'static',
    entryPoint: 'index.html',
    files: {
      'index.html': `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${name}</title>
    <link rel="stylesheet" href="styles/main.css" />
  </head>
  <body>
    <main class="app-shell">
      <section class="hero">
        <p class="eyebrow">Aura Project Mode</p>
        <h1>Start building a multi-file website.</h1>
        <p>Edit the files in the tree or ask Aura to expand this starter into a complete project.</p>
        <button class="cta" type="button">Ship something bold</button>
      </section>
    </main>
    <script src="scripts/main.js"></script>
  </body>
</html>`,
      'styles/main.css': `:root {
  color-scheme: dark;
  --bg: #09090f;
  --panel: rgba(255, 255, 255, 0.06);
  --text: #f5f7fb;
  --muted: rgba(245, 247, 251, 0.72);
  --accent: #60a5fa;
  --accent-2: #a78bfa;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "Inter", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(96, 165, 250, 0.18), transparent 32%),
    radial-gradient(circle at bottom right, rgba(167, 139, 250, 0.14), transparent 30%),
    var(--bg);
  color: var(--text);
}

.app-shell {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 48px 20px;
}

.hero {
  width: min(720px, 100%);
  padding: 40px;
  border-radius: 28px;
  background: var(--panel);
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(18px);
}

.eyebrow {
  margin: 0 0 12px;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--accent);
  font-size: 0.78rem;
}

h1 {
  margin: 0;
  font-size: clamp(2.6rem, 6vw, 4.6rem);
  line-height: 0.95;
}

.hero p:last-of-type {
  color: var(--muted);
  max-width: 52ch;
  line-height: 1.6;
}

.cta {
  margin-top: 16px;
  padding: 14px 22px;
  border-radius: 999px;
  border: none;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: white;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}`,
      'scripts/main.js': `document.querySelector('.cta')?.addEventListener('click', () => {
  const button = document.querySelector('.cta');
  if (button) {
    button.textContent = 'Aura is building...';
  }
});`,
    },
  });
}

export function getDefaultFileContent(path: string): string {
  const language = inferLanguageFromExtension(path);
  switch (language) {
    case 'html':
      return '<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n    <title>New Page</title>\n  </head>\n  <body>\n  </body>\n</html>\n';
    case 'css':
      return '/* Styles */\n';
    case 'javascript':
    case 'typescript':
    case 'jsx':
    case 'tsx':
      return '// Script\n';
    case 'json':
      return '{\n  \n}\n';
    case 'markdown':
      return '# Notes\n';
    default:
      return '';
  }
}

export function getParentDirectory(path: string): string {
  return dirname(path);
}

export function getBaseName(path: string): string {
  return basename(path);
}
