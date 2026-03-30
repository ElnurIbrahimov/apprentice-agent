/**
 * PyodideExecutor — Manages Pyodide Web Worker lifecycle and provides
 * intelligent routing between browser-side Python (Pyodide) and backend fallback.
 */

import { HTTP, getAuthHeaders } from '../api';

/* ─── Types ─── */

export interface OutputBlock {
  type: 'stdout' | 'stderr' | 'image' | 'html' | 'error' | 'result';
  text?: string;
  mime?: string;
  data?: string;
  content?: string;
  ename?: string;
  evalue?: string;
  traceback?: string;
  repr?: string;
  type_name?: string;
}

export interface VariableInfo {
  name: string;
  type_name: string;
  repr: string;
}

export interface ExecutionCallbacks {
  onOutput: (block: OutputBlock) => void;
  onVariables: (vars: VariableInfo[]) => void;
  onDone: (success: boolean, executionTime: number) => void;
  onStatus?: (msg: string) => void;
}

type PyodideState = 'idle' | 'loading' | 'ready' | 'executing' | 'error';

/* ─── Constants ─── */

// Packages that require the backend (not available in Pyodide)
const BACKEND_ONLY_PACKAGES = new Set([
  'torch', 'tensorflow', 'keras', 'transformers', 'huggingface_hub',
  'requests', 'httpx', 'aiohttp', 'flask', 'fastapi', 'django',
  'subprocess', 'socket', 'multiprocessing', 'threading',
  'psutil', 'docker', 'boto3', 'google',
  'cv2', 'opencv', 'moviepy', 'ffmpeg',
  'pyspark', 'dask', 'ray',
]);

/* ─── Executor ─── */

export class PyodideExecutor {
  private worker: Worker | null = null;
  private state: PyodideState = 'idle';
  private pendingCallbacks: Map<string, ExecutionCallbacks> = new Map();
  private onStateChange?: (state: PyodideState) => void;
  private initPromise: Promise<void> | null = null;
  private _initResolve: (() => void) | null = null;
  private _initTimeout: ReturnType<typeof setTimeout> | null = null;

  constructor(onStateChange?: (state: PyodideState) => void) {
    this.onStateChange = onStateChange;
  }

  get currentState(): PyodideState {
    return this.state;
  }

  get isReady(): boolean {
    return this.state === 'ready';
  }

  private setState(s: PyodideState) {
    this.state = s;
    this.onStateChange?.(s);
  }

  /**
   * Initialize the Pyodide worker. Safe to call multiple times — will only init once.
   */
  async init(): Promise<void> {
    if (this.state === 'ready') return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise<void>((resolve, reject) => {
      this._initResolve = resolve;

      try {
        this.setState('loading');

        // Create worker from the built pyodide-worker.js
        const workerUrl = typeof chrome !== 'undefined' && chrome.runtime?.getURL
          ? chrome.runtime.getURL('pyodide-worker.js')
          : 'pyodide-worker.js';

        this.worker = new Worker(workerUrl);
        this.worker.onmessage = (e) => this.handleMessage(e);
        this.worker.onerror = (e) => {
          console.error('[PyodideExecutor] Worker error:', e);
          this.setState('error');
          reject(new Error('Worker failed to load'));
        };

        // Tell worker to initialize
        this.worker.postMessage({ type: 'init' });

        // Timeout after 60s
        this._initTimeout = setTimeout(() => {
          this._initTimeout = null;
          if (this.state === 'loading') {
            this.setState('error');
            reject(new Error('Pyodide initialization timed out'));
          }
        }, 60000);
      } catch (err) {
        this.setState('error');
        reject(err);
      }
    });

    return this.initPromise;
  }

  /**
   * Handle messages from the Pyodide worker.
   */
  private handleMessage(e: MessageEvent) {
    const msg = e.data;

    switch (msg.type) {
      case 'ready':
        if (this._initTimeout) { clearTimeout(this._initTimeout); this._initTimeout = null; }
        this.setState('ready');
        this._initResolve?.();
        this._initResolve = null;
        break;

      case 'loading':
        // Forward loading status to any active onStatus callback
        for (const cb of this.pendingCallbacks.values()) {
          cb.onStatus?.(msg.stage);
        }
        break;

      case 'output': {
        const cb = this.pendingCallbacks.get(msg.id);
        cb?.onOutput(msg.block);
        break;
      }

      case 'variables': {
        const cb = this.pendingCallbacks.get(msg.id);
        cb?.onVariables(msg.variables);
        break;
      }

      case 'done': {
        const cb = this.pendingCallbacks.get(msg.id);
        cb?.onDone(msg.success, msg.executionTime);
        this.pendingCallbacks.delete(msg.id);
        if (this.pendingCallbacks.size === 0 && this.state === 'executing') this.setState('ready');
        break;
      }

      case 'error': {
        const cb = this.pendingCallbacks.get(msg.id);
        if (cb) {
          cb.onOutput({ type: 'error', ename: 'PyodideError', evalue: msg.message, traceback: msg.message });
          cb.onDone(false, 0);
          this.pendingCallbacks.delete(msg.id);
        }
        if (this.pendingCallbacks.size === 0 && this.state === 'executing') this.setState('ready');
        break;
      }
    }
  }

  /**
   * Check if code can run in Pyodide or needs the backend.
   */
  static needsBackend(code: string): boolean {
    const imports = PyodideExecutor.parseImports(code);
    return imports.some(pkg => BACKEND_ONLY_PACKAGES.has(pkg));
  }

  /**
   * Parse import statements from Python code.
   */
  static parseImports(code: string): string[] {
    const imports = new Set<string>();
    for (const line of code.split('\n')) {
      const trimmed = line.trim();
      const importMatch = trimmed.match(/^import\s+([\w.]+)/);
      if (importMatch) imports.add(importMatch[1].split('.')[0]);
      const fromMatch = trimmed.match(/^from\s+([\w.]+)\s+import/);
      if (fromMatch) imports.add(fromMatch[1].split('.')[0]);
    }
    return Array.from(imports);
  }

  /**
   * Execute Python code — routes to Pyodide or backend automatically.
   */
  async execute(
    code: string,
    callbacks: ExecutionCallbacks,
    opts: { sessionId?: string; forceBackend?: boolean } = {}
  ): Promise<void> {
    // Route to backend if code needs packages Pyodide can't handle
    if (opts.forceBackend || PyodideExecutor.needsBackend(code)) {
      return this.executeOnBackend(code, callbacks, opts.sessionId);
    }

    // Initialize Pyodide on first use (lazy loading)
    if (!this.isReady) {
      callbacks.onStatus?.('Starting Python runtime...');
      try {
        await this.init();
      } catch {
        // Pyodide failed — fall back to backend
        callbacks.onStatus?.('Browser Python unavailable, using server...');
        return this.executeOnBackend(code, callbacks, opts.sessionId);
      }
    }

    // Execute in Pyodide worker
    if (!this.worker) {
      callbacks.onOutput({ type: 'error', ename: 'WorkerError', evalue: 'Worker not initialized', traceback: '' });
      callbacks.onDone(false, 0);
      return;
    }
    const id = crypto.randomUUID();
    this.pendingCallbacks.set(id, callbacks);
    this.setState('executing');
    this.worker.postMessage({ type: 'execute', id, code });
  }

  /**
   * Execute code on the backend as fallback.
   */
  private async executeOnBackend(
    code: string,
    callbacks: ExecutionCallbacks,
    sessionId?: string,
  ): Promise<void> {
    callbacks.onStatus?.('Running on server...');
    const startTime = performance.now();

    try {
      const resp = await fetch(`${HTTP}/api/code/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ code, session_id: sessionId, timeout: 60 }),
      });

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        throw new Error((d as any).detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();

      // Forward output blocks
      if (data.outputs) {
        for (const block of data.outputs) {
          callbacks.onOutput(block);
        }
      }

      // Forward variables
      if (data.variables) {
        callbacks.onVariables(data.variables);
      }

      const executionTime = data.execution_time ?? (performance.now() - startTime) / 1000;
      callbacks.onDone(data.success !== false, executionTime);
    } catch (err: any) {
      callbacks.onOutput({
        type: 'error',
        ename: 'BackendError',
        evalue: err.message || 'Server execution failed',
        traceback: err.message,
      });
      callbacks.onDone(false, (performance.now() - startTime) / 1000);
    }
  }

  /**
   * Reset the Pyodide runtime (clear variables).
   */
  reset() {
    if (this.worker && this.isReady) {
      this.worker.postMessage({ type: 'reset' });
    }
  }

  /**
   * Dispose the worker entirely. Resolves any pending callbacks with an error.
   */
  dispose() {
    // Notify pending callbacks so consumers don't hang
    for (const [id, cb] of this.pendingCallbacks) {
      try { cb.onDone(false, 0); } catch { /* ignore */ }
    }
    this.pendingCallbacks.clear();

    if (this._initTimeout) { clearTimeout(this._initTimeout); this._initTimeout = null; }
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.setState('idle');
    this.initPromise = null;
    this._initResolve = null;
  }
}

/** Singleton instance — shared across component lifecycles */
let _instance: PyodideExecutor | null = null;

export function getPyodideExecutor(onStateChange?: (state: PyodideState) => void): PyodideExecutor {
  if (!_instance) {
    _instance = new PyodideExecutor(onStateChange);
  } else if (onStateChange) {
    // Update callback for new component mount
    (_instance as any).onStateChange = onStateChange;
  }
  return _instance;
}
