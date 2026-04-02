/**
 * WebContainerManager — singleton manager for the WebContainer runtime.
 *
 * Provides a full Node.js environment in the browser for npm-based projects.
 * Used by WebCreatorPanel and ArtifactsPanel when a project has a package.json.
 *
 * Only ONE WebContainer can run per origin. Lazy-booted, auto-shutdown after idle.
 */

import { WebContainer, type FileSystemTree, type WebContainerProcess } from '@webcontainer/api';

export type WCStatus = 'idle' | 'booting' | 'ready' | 'installing' | 'running' | 'error' | 'shutdown';

export interface WCEventHandlers {
  onOutput?: (text: string, stream: 'stdout' | 'stderr') => void;
  onServerReady?: (port: number, url: string) => void;
  onStatusChange?: (status: WCStatus) => void;
  onError?: (error: Error) => void;
}

const IDLE_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes
const INSTALL_TIMEOUT_MS = 120_000; // 2 minutes
const BOOT_TIMEOUT_MS = 30_000; // 30 seconds

class WebContainerManager {
  private static _instance: WebContainerManager | null = null;

  private container: WebContainer | null = null;
  private bootPromise: Promise<void> | null = null;
  private serverProcess: WebContainerProcess | null = null;
  private serverUrl: string | null = null;
  private status: WCStatus = 'idle';
  private handlers: WCEventHandlers = {};
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private lastActivity = 0;

  static getInstance(): WebContainerManager {
    if (!WebContainerManager._instance) {
      WebContainerManager._instance = new WebContainerManager();
    }
    return WebContainerManager._instance;
  }

  /** Check if SharedArrayBuffer is available (required for WebContainers) */
  static isSupported(): boolean {
    return typeof SharedArrayBuffer !== 'undefined';
  }

  setHandlers(handlers: WCEventHandlers) {
    this.handlers = handlers;
  }

  getStatus(): WCStatus {
    return this.status;
  }

  getServerUrl(): string | null {
    return this.serverUrl;
  }

  isBooted(): boolean {
    return this.container != null && this.status !== 'idle' && this.status !== 'shutdown';
  }

  private setStatus(s: WCStatus) {
    this.status = s;
    this.handlers.onStatusChange?.(s);
  }

  private touchActivity() {
    this.lastActivity = Date.now();
    this.resetIdleTimer();
  }

  private resetIdleTimer() {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      if (Date.now() - this.lastActivity >= IDLE_TIMEOUT_MS) {
        this.teardown();
      }
    }, IDLE_TIMEOUT_MS);
  }

  /** Boot the WebContainer (idempotent — returns immediately if already booted) */
  async boot(): Promise<void> {
    if (this.container) return;
    if (this.bootPromise) return this.bootPromise;

    if (!WebContainerManager.isSupported()) {
      this.setStatus('error');
      throw new Error('SharedArrayBuffer is not available. WebContainers require Cross-Origin Isolation.');
    }

    this.bootPromise = this._doBoot();
    try {
      await this.bootPromise;
    } finally {
      this.bootPromise = null;
    }
  }

  private async _doBoot(): Promise<void> {
    this.setStatus('booting');
    try {
      const container = await Promise.race([
        WebContainer.boot(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('WebContainer boot timed out')), BOOT_TIMEOUT_MS)
        ),
      ]);
      this.container = container;

      // Listen for server-ready events
      container.on('server-ready', (port: number, url: string) => {
        this.serverUrl = url;
        this.setStatus('running');
        this.handlers.onServerReady?.(port, url);
      });

      container.on('error', (err: { message: string }) => {
        this.handlers.onError?.(new Error(err.message));
      });

      this.setStatus('ready');
      this.touchActivity();
    } catch (err: any) {
      this.setStatus('error');
      this.handlers.onError?.(err);
      throw err;
    }
  }

  /** Mount project files into the WebContainer filesystem */
  async mountFiles(files: Record<string, string>): Promise<void> {
    if (!this.container) throw new Error('WebContainer not booted');
    this.touchActivity();

    const tree: FileSystemTree = {};
    for (const [path, content] of Object.entries(files)) {
      this.setFileInTree(tree, path, content);
    }
    await this.container.mount(tree);
  }

  private setFileInTree(tree: FileSystemTree, path: string, content: string) {
    const parts = path.split('/').filter(Boolean);
    let current: any = tree;
    for (let i = 0; i < parts.length - 1; i++) {
      const dir = parts[i];
      if (!current[dir]) {
        current[dir] = { directory: {} };
      }
      current = current[dir].directory;
    }
    const filename = parts[parts.length - 1];
    current[filename] = { file: { contents: content } };
  }

  /** Write a single file (creates directories as needed) */
  async writeFile(path: string, content: string): Promise<void> {
    if (!this.container) throw new Error('WebContainer not booted');
    this.touchActivity();
    // Ensure parent directories exist
    const parts = path.split('/').filter(Boolean);
    if (parts.length > 1) {
      const dirPath = parts.slice(0, -1).join('/');
      await this.container.fs.mkdir(dirPath, { recursive: true });
    }
    await this.container.fs.writeFile(path, content);
  }

  /** Read a file from the WebContainer filesystem */
  async readFile(path: string): Promise<string> {
    if (!this.container) throw new Error('WebContainer not booted');
    this.touchActivity();
    return await this.container.fs.readFile(path, 'utf-8');
  }

  /** Delete a file */
  async deleteFile(path: string): Promise<void> {
    if (!this.container) throw new Error('WebContainer not booted');
    this.touchActivity();
    await this.container.fs.rm(path);
  }

  /** Run `npm install` (or `pnpm install`) */
  async installDeps(): Promise<{ exitCode: number; output: string }> {
    if (!this.container) throw new Error('WebContainer not booted');
    this.setStatus('installing');
    this.touchActivity();

    const output: string[] = [];
    const emit = (text: string, stream: 'stdout' | 'stderr') => {
      output.push(text);
      this.handlers.onOutput?.(text, stream);
    };

    const process = await this.container.spawn('npm', ['install', '--no-audit', '--no-fund']);

    process.output.pipeTo(new WritableStream({
      write(chunk) { emit(chunk, 'stdout'); },
    }));

    const exitCode = await Promise.race([
      process.exit,
      new Promise<number>((_, reject) =>
        setTimeout(() => { process.kill(); reject(new Error('npm install timed out')); }, INSTALL_TIMEOUT_MS)
      ),
    ]);

    if (exitCode === 0) {
      this.setStatus('ready');
    } else {
      this.setStatus('error');
    }
    this.touchActivity();
    return { exitCode, output: output.join('') };
  }

  /** Start the dev server (e.g., `npx vite` or `npm run dev`) */
  async startDevServer(command?: string, args?: string[]): Promise<string> {
    if (!this.container) throw new Error('WebContainer not booted');
    await this.stopDevServer();
    this.touchActivity();

    const cmd = command || 'npx';
    const cmdArgs = args || ['vite', '--host'];

    this.serverProcess = await this.container.spawn(cmd, cmdArgs);

    this.serverProcess.output.pipeTo(new WritableStream({
      write: (chunk) => {
        this.handlers.onOutput?.(chunk, 'stdout');
      },
    }));

    // Wait for server-ready event (fires via container.on('server-ready'))
    return new Promise<string>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Dev server did not start within 30 seconds'));
      }, 30_000);

      const originalHandler = this.handlers.onServerReady;
      this.handlers.onServerReady = (port, url) => {
        clearTimeout(timeout);
        this.handlers.onServerReady = originalHandler;
        originalHandler?.(port, url);
        resolve(url);
      };
    });
  }

  /** Stop the dev server */
  async stopDevServer(): Promise<void> {
    if (this.serverProcess) {
      this.serverProcess.kill();
      this.serverProcess = null;
      this.serverUrl = null;
    }
  }

  /** Spawn an arbitrary command */
  async spawn(cmd: string, args: string[]): Promise<{ exitCode: number; output: string }> {
    if (!this.container) throw new Error('WebContainer not booted');
    this.touchActivity();

    const output: string[] = [];
    const process = await this.container.spawn(cmd, args);

    process.output.pipeTo(new WritableStream({
      write: (chunk) => {
        output.push(chunk);
        this.handlers.onOutput?.(chunk, 'stdout');
      },
    }));

    const exitCode = await process.exit;
    return { exitCode, output: output.join('') };
  }

  /** Tear down the WebContainer and release resources */
  async teardown(): Promise<void> {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    await this.stopDevServer();
    if (this.container) {
      this.container.teardown();
      this.container = null;
    }
    this.serverUrl = null;
    this.setStatus('shutdown');
  }
}

export default WebContainerManager;
