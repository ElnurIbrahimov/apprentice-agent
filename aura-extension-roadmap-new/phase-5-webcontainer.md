# Phase 5: WebContainer Runtime — Full Node.js in Browser

**Effort:** 4-5 days
**Impact:** Unlocks the entire npm ecosystem, real build tools, server-side code
**Depends on:** Phase 3 (VirtualFS — WebContainer replaces the bundle builder)

---

## What This Changes

Today: React artifacts run via CDN imports (esm.sh) in an iframe. This breaks when:
- A package isn't on esm.sh
- A package needs Node.js APIs
- Imports use complex resolution (barrel exports, package.json exports maps)
- You need a build step (TypeScript compilation, PostCSS, etc.)
- You need a dev server (HMR, API routes)

WebContainers run a full Node.js environment in the browser via WebAssembly. This means:
- Real `npm install` / `pnpm install`
- Real Vite/webpack dev server with HMR
- Real TypeScript compilation
- Real module resolution
- Server-side code (Express, Fastify, Next.js API routes)
- File system operations
- Process spawning

This is what powers StackBlitz and Bolt.new.

---

## 5A. WebContainer Setup

### Installation

```bash
cd D:/Aura/extension-src
npm install @webcontainer/api
```

### Important Constraints
- WebContainers require `SharedArrayBuffer` which needs `Cross-Origin-Isolation` headers
- In a Chrome extension, the sidebar page can set these headers via `manifest.json`
- Only ONE WebContainer instance can run per origin (singleton)
- Initial boot: ~5 seconds. Subsequent: ~1 second (cached)
- Memory: ~100-200MB for a basic Vite project

### WebContainer Manager

**New file:** `extension-src/src/utils/WebContainerManager.ts`

```typescript
import { WebContainer } from '@webcontainer/api';

class WebContainerManager {
  private static instance: WebContainerManager;
  private container: WebContainer | null = null;
  private serverUrl: string | null = null;
  private bootPromise: Promise<void> | null = null;
  
  static getInstance(): WebContainerManager;
  
  // Lifecycle
  async boot(): Promise<void>;          // Boot the WebContainer (idempotent)
  async teardown(): Promise<void>;      // Destroy the container
  isBooted(): boolean;
  
  // File system (syncs with VirtualFS)
  async mountFiles(files: VirtualProject): Promise<void>;
  async writeFile(path: string, content: string): Promise<void>;
  async readFile(path: string): Promise<string>;
  async deleteFile(path: string): Promise<void>;
  
  // Package management
  async installDeps(): Promise<{ exitCode: number; output: string }>;
  
  // Dev server
  async startDevServer(): Promise<string>;  // returns preview URL
  async stopDevServer(): Promise<void>;
  getServerUrl(): string | null;
  
  // Arbitrary commands
  async spawn(cmd: string, args: string[]): Promise<{ exitCode: number; output: string }>;
  
  // Events
  onServerReady(callback: (port: number, url: string) => void): void;
  onError(callback: (error: Error) => void): void;
  onOutput(callback: (data: string) => void): void;
}
```

### Boot Sequence
```
1. WebContainer.boot()
2. Mount project files from VirtualFS
3. If package.json exists:
   a. Run: npm install (or pnpm install)
   b. Wait for install to complete
4. Run: npm run dev (or npx vite)
5. Wait for server-ready event
6. Set iframe src to the dev server URL
```

---

## 5B. Integration with Preview

### Current Flow (iframe srcdoc)
```
Code → buildSrcdoc() → iframe.srcdoc = assembled HTML
```

### New Flow (WebContainer)
```
Code → WebContainerManager.writeFile() → Vite HMR detects change → iframe auto-refreshes
```

The iframe no longer uses `srcdoc`. Instead:
```typescript
// Get the dev server URL from WebContainer
const url = webContainerManager.getServerUrl(); // e.g., "http://localhost:3111"
iframeRef.current.src = url;
```

Vite's HMR handles live updates — when a file changes, the preview updates instantly without full page reload.

### Fallback Strategy
Not all projects need WebContainer. Use it selectively:

| Project Type | Rendering Method |
|---|---|
| Static HTML (no npm deps) | iframe srcdoc (existing, fast) |
| HTML + CDN libraries | iframe srcdoc (existing) |
| React/Vue/Svelte with npm deps | WebContainer (new) |
| Next.js / Express / server-side | WebContainer (new) |
| Python | Pyodide (existing) |

Auto-detect: if `package.json` exists in VirtualFS → use WebContainer. Otherwise → use srcdoc bundler.

---

## 5C. Terminal Output Panel

WebContainer operations (npm install, dev server, build) produce terminal output. Users need to see this.

**New component:** `extension-src/src/components/Terminal.tsx`

```typescript
interface TerminalProps {
  lines: TerminalLine[];
  maxLines?: number;      // default 500
  autoScroll?: boolean;   // default true
}

interface TerminalLine {
  text: string;
  type: 'stdout' | 'stderr' | 'system';  // system = "[Installing dependencies...]"
  timestamp: number;
}
```

UI:
- Dark terminal background with monospace font
- Colored output: stdout white, stderr red, system blue
- Auto-scroll to bottom
- Copy all button
- Collapsible — sits below the preview
- Shows npm install progress, Vite build output, runtime errors

---

## 5D. Package Management UI

When the AI generates code that imports a new package, or the user wants to add one:

1. **Auto-detect imports** — scan code for `import ... from 'package-name'`
2. **Check if installed** — compare against `package.json` dependencies
3. **Auto-install missing** — run `npm install package-name` in WebContainer
4. **UI indicator** — "Installing react-spring..." with progress
5. **Manual add** — search bar to find and add npm packages

### Import Scanner

**New utility:** `extension-src/src/utils/importScanner.ts`

```typescript
function scanImports(code: string): string[] {
  // Match: import ... from 'package'
  // Match: import 'package'
  // Match: require('package')
  // Match: import('package')
  // Exclude relative imports (./  ../)
  // Exclude node builtins
  // Return list of package names
}

function getMissingPackages(imports: string[], packageJson: object): string[] {
  // Compare scanned imports against dependencies + devDependencies
  // Return packages that need to be installed
}
```

---

## 5E. Cross-Origin Isolation for Extensions

WebContainers require `Cross-Origin-Isolation` (COOP + COEP headers). In a Chrome extension:

### Option 1: CSP in manifest.json
```json
{
  "content_security_policy": {
    "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';"
  },
  "cross_origin_embedder_policy": { "value": "require-corp" },
  "cross_origin_opener_policy": { "value": "same-origin" }
}
```

### Option 2: Separate page for WebContainer
If CSP conflicts arise, run the WebContainer in a separate extension page (`webcontainer.html`) that loads in a new tab or popup, with its own strict headers. Communicate with the sidebar via `chrome.runtime.sendMessage`.

### Option 3: Service Worker proxy
Use the extension's service worker to add COOP/COEP headers to the sidebar's responses. This is hacky but works.

Test all three approaches. Option 1 is cleanest if it works.

---

## 5F. Memory and Performance

WebContainers use significant memory (~100-200MB). Considerations:

- **Lazy boot**: Don't boot WebContainer until the user creates/opens a project that needs it
- **Auto-shutdown**: Tear down WebContainer after 10 minutes of inactivity
- **Memory warning**: If the system is under memory pressure, warn before booting
- **Single instance**: Only one WebContainer runs at a time. Switching projects tears down and reboots.
- **Cache**: npm packages are cached in the WebContainer's virtual FS between sessions (if possible)

---

## Definition of Done — Phase 5
- [ ] WebContainer boots successfully in the extension environment
- [ ] Cross-origin isolation configured and working
- [ ] VirtualFS files mount into WebContainer filesystem
- [ ] `npm install` runs and installs packages from package.json
- [ ] Vite dev server starts and serves preview in iframe
- [ ] HMR works — file changes reflect in preview without full reload
- [ ] Terminal panel shows install/build/server output
- [ ] Import scanner auto-detects missing packages and installs them
- [ ] Fallback: static HTML projects still use srcdoc (no WebContainer overhead)
- [ ] Auto-detection: package.json presence triggers WebContainer mode
- [ ] Lazy boot: WebContainer only starts when needed
- [ ] Auto-shutdown after 10 min idle
- [ ] Memory usage stays under 250MB for typical projects
