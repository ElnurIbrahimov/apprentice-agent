# Phase 3: Multi-File Projects + Virtual Filesystem

**Effort:** 3-4 days
**Impact:** Unlocks real project creation instead of single-file toys
**Depends on:** Phase 2 (CodeMirror for editing files)

---

## The Problem

WebCreatorPanel forces everything into a single `<!DOCTYPE html>` document. ArtifactsPanel handles one file at a time. No panel supports multi-file projects — separate HTML, CSS, JS, components, assets.

Every serious competitor (Bolt.new, Lovable, Replit, StackBlitz) supports multi-file projects with a file tree.

---

## 3A. Virtual Filesystem

### Core Data Model

**New file:** `extension-src/src/utils/virtualFS.ts`

```typescript
interface VirtualFile {
  path: string;          // e.g. "src/App.tsx", "styles/main.css"
  content: string;
  language: string;      // auto-detected from extension
  createdAt: number;
  updatedAt: number;
}

interface VirtualProject {
  id: string;
  name: string;
  files: Map<string, VirtualFile>;
  entryPoint: string;    // e.g. "index.html" or "src/App.tsx"
  framework: 'static' | 'react' | 'vue' | 'svelte' | 'nextjs' | 'custom';
  createdAt: number;
  updatedAt: number;
}

class VirtualFS {
  private project: VirtualProject;
  
  // File operations
  createFile(path: string, content: string): void;
  readFile(path: string): string | null;
  updateFile(path: string, content: string): void;
  deleteFile(path: string): void;
  renameFile(oldPath: string, newPath: string): void;
  listFiles(): string[];
  
  // Directory operations
  listDir(dirPath: string): string[];
  createDir(dirPath: string): void;
  
  // Project operations
  getProject(): VirtualProject;
  toJSON(): string;                    // serialize for storage
  static fromJSON(json: string): VirtualFS;  // restore from storage
  
  // Rendering
  buildBundle(): string;  // assemble files into renderable HTML for iframe
  
  // Persistence
  async save(): Promise<void>;         // chrome.storage.local
  static async load(id: string): Promise<VirtualFS | null>;
}
```

### Bundle Builder (for iframe rendering)

For **static/HTML projects**, `buildBundle()` assembles files:
```typescript
buildBundle(): string {
  const indexHtml = this.readFile(this.project.entryPoint) || '';
  
  // Inline all CSS <link> tags
  // Replace <link rel="stylesheet" href="styles/main.css">
  // with <style>{content of styles/main.css}</style>
  
  // Inline all <script src="..."> tags
  // Replace <script src="app.js"></script>
  // with <script>{content of app.js}</script>
  
  // For React projects: build import map from project files
  // Map relative imports to blob URLs
  
  return assembledHtml;
}
```

For **React projects** (pre-WebContainer):
- Use blob URLs for local module resolution
- Build an import map that maps `./Component` to `blob:...`
- External deps still via esm.sh CDN
- This is a bridge solution until Phase 5 (WebContainer)

### Storage Limits
- Max 50 files per project
- Max 200KB per file
- Max 2MB total per project
- Projects stored in `chrome.storage.local` under `aura_projects_{id}`
- Project index stored under `aura_project_index`

---

## 3B. File Tree Component

**New file:** `extension-src/src/components/FileTree.tsx`

```typescript
interface FileTreeProps {
  files: string[];           // list of file paths
  activeFile: string;        // currently open file
  onFileSelect: (path: string) => void;
  onFileCreate: (path: string) => void;
  onFileDelete: (path: string) => void;
  onFileRename: (oldPath: string, newPath: string) => void;
  onFolderCreate: (path: string) => void;
  modifiedFiles?: Set<string>;  // files with unsaved AI changes
}
```

UI Design:
- Tree structure with expand/collapse folders
- Icons per file type (HTML, CSS, JS, TS, JSX, TSX, JSON, SVG, MD, image)
- Right-click context menu: New File, New Folder, Rename, Delete, Duplicate
- "+" button at top for quick new file
- Modified files show a dot indicator
- Active file highlighted
- Drag-and-drop to move files between folders (stretch goal)
- Compact design — this sits in a narrow sidebar within the panel

### Layout Change

Current layout of WebCreator/Artifacts:
```
[toolbar]
[preview or code — full width]
```

New layout with file tree:
```
[toolbar]
[file-tree (180px) | preview/code (remaining)]
```

File tree is collapsible. On narrow screens (<600px panel width), it becomes a dropdown instead of a sidebar.

---

## 3C. AI File Operations

### Updated System Prompt for Multi-File Mode

When multi-file mode is active, the system prompt changes to instruct the AI to output structured file operations instead of a single HTML blob:

```
You are building a multi-file web project. Respond with file operations in this format:

===FILE: path/to/file.ext===
file content here
===END FILE===

===FILE: another/file.ext===
more content
===END FILE===

===DELETE: old-file.ext===

You can create, update, or delete multiple files in one response.
Current project files:
- index.html (entry point)
- styles/main.css
- scripts/app.js
```

### Parsing AI Response

**New utility:** `extension-src/src/utils/parseFileOperations.ts`

```typescript
interface FileOperation {
  type: 'create' | 'update' | 'delete';
  path: string;
  content?: string;
}

function parseFileOperations(response: string): FileOperation[] {
  // Parse ===FILE: ...=== blocks
  // Parse ===DELETE: ...=== blocks
  // Fallback: if no file markers found, treat entire response as update to current active file
}
```

### Streaming Multi-File
During streaming, detect `===FILE: ...===` markers:
- When a new file marker is found, switch the preview to show that file being built
- Show a "files being created" indicator in the toolbar
- After stream completes, apply all file operations to the VirtualFS

---

## 3D. Integration with Existing Panels

### WebCreatorPanel
- Add a toggle: "Single Page" vs "Project" mode
- Single Page mode works exactly as today (backward compatible)
- Project mode enables file tree + multi-file AI operations
- Default new projects start with:
  ```
  index.html
  styles/main.css
  scripts/app.js
  ```
- Preview always renders the entry point (index.html) with inlined deps

### ArtifactsPanel
- Generate mode: stays single-file (artifacts are inherently single-file)
- Live mode: already multi-file (receives WebSocket updates per file) — add file tree to browse live files
- New "Project Artifact" type: uses VirtualFS, enables multi-file generation

### CodePanel
- Each Python "notebook" becomes a project with multiple `.py` files
- Main execution file is configurable
- Import resolution: files in the project are available as Python modules in Pyodide

---

## 3E. Project Templates

Extend the "Quick Start" templates to create multi-file projects:

| Template | Files Created |
|----------|---------------|
| Landing Page | `index.html`, `styles/main.css`, `scripts/main.js`, `assets/` |
| React App | `index.html`, `src/App.jsx`, `src/main.jsx`, `src/styles.css` |
| Dashboard | `index.html`, `styles/dashboard.css`, `scripts/charts.js`, `scripts/data.js` |
| Portfolio | `index.html`, `styles/main.css`, `scripts/gallery.js`, `pages/about.html` |
| API + Frontend | `index.html`, `styles/main.css`, `scripts/api.js`, `scripts/app.js` |

These mirror what `scaffold.py` does on the backend but for the browser-only workflow.

---

## Definition of Done — Phase 3
- [ ] VirtualFS class handles create/read/update/delete/rename for files and folders
- [ ] Projects serialize to/from chrome.storage.local reliably
- [ ] File tree component renders project structure with icons and context menu
- [ ] WebCreatorPanel has "Project" mode with file tree sidebar
- [ ] AI can create/update/delete multiple files in one response
- [ ] Streaming multi-file generation shows files being built in real-time
- [ ] Bundle builder assembles multi-file projects into renderable iframe HTML
- [ ] React projects resolve local imports via blob URLs + esm.sh for externals
- [ ] ArtifactsPanel Live mode shows file tree for WebSocket-synced files
- [ ] 5 multi-file project templates available
- [ ] Backward compatible: single-file mode still works as before
