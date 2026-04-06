/**
 * Shared Pyodide executor — singleton worker shared between
 * Code Interpreter tab and inline code run buttons.
 */

export type OutputBlock = {
  type: 'stdout' | 'stderr' | 'image' | 'html' | 'error';
  text?: string;
  mime?: string;
  data?: string;
  content?: string;
  ename?: string;
  evalue?: string;
  traceback?: string;
};

export type VariableInfo = {
  name: string;
  type_name: string;
  repr: string;
};

type ExecutionListener = {
  onOutput: (block: OutputBlock) => void;
  onVariables?: (variables: VariableInfo[]) => void;
  onDone: (success: boolean, executionTime: number) => void;
};

type GlobalListener = {
  onReady?: () => void;
  onLoading?: (stage: string) => void;
  onError?: (message: string) => void;
};

let worker: Worker | null = null;
let _ready = false;
const executionListeners = new Map<string, ExecutionListener>();
const readyCallbacks: Array<() => void> = [];
const globalListeners = new Set<GlobalListener>();

function ensureWorker(): Worker {
  if (worker) return worker;

  worker = new Worker(
    new URL('./pyodide-worker.ts', import.meta.url),
    { type: 'module' },
  );

  worker.onerror = () => {
    for (const [, listener] of executionListeners) {
      listener.onOutput({ type: 'error', ename: 'WorkerError', evalue: 'Pyodide worker failed to load' });
      listener.onDone(false, 0);
    }
    executionListeners.clear();
  };

  worker.onmessage = (e) => {
    const msg = e.data;
    if (!msg?.type) return;

    if (msg.type === 'ready') {
      _ready = true;
      readyCallbacks.forEach((cb) => cb());
      readyCallbacks.length = 0;
      globalListeners.forEach((gl) => gl.onReady?.());
      return;
    }
    if (msg.type === 'loading') {
      globalListeners.forEach((gl) => gl.onLoading?.(msg.stage || 'Loading...'));
      return;
    }
    if (msg.type === 'init_error') {
      // Pyodide failed to load — notify listeners so UI can fallback to server mode
      globalListeners.forEach((gl) => gl.onError?.(msg.message || 'Python runtime failed to load'));
      return;
    }

    const listener = msg.id ? executionListeners.get(msg.id) : null;
    if (!listener) return;

    switch (msg.type) {
      case 'output':
        if (msg.block) listener.onOutput(msg.block);
        break;
      case 'variables':
        if (msg.variables) listener.onVariables?.(msg.variables);
        break;
      case 'done':
        listener.onDone(msg.success, msg.executionTime);
        executionListeners.delete(msg.id);
        break;
      case 'error':
        listener.onOutput({ type: 'error', ename: 'Error', evalue: msg.message });
        listener.onDone(false, 0);
        executionListeners.delete(msg.id);
        break;
    }
  };

  worker.postMessage({ type: 'init' });
  return worker;
}

let execCounter = 0;

/** Whether the Pyodide runtime is initialized and ready. */
export function isReady(): boolean { return _ready; }

/**
 * Execute Python code. Used by both inline Run button and CodeInterpreter.
 * Returns the execution ID.
 */
export function execute(
  code: string,
  listener: ExecutionListener,
): string {
  const w = ensureWorker();
  const id = `exec-${Date.now()}-${++execCounter}`;

  executionListeners.set(id, listener);

  const run = () => { w.postMessage({ type: 'execute', id, code }); };

  if (_ready) {
    run();
  } else {
    readyCallbacks.push(run);
  }

  return id;
}

/** Simplified execute for inline code run buttons (no variables). */
export function executeInline(
  code: string,
  onOutput: (block: OutputBlock) => void,
  onDone: (success: boolean, executionTime: number) => void,
): string {
  return execute(code, { onOutput, onDone });
}

/** Reset the Python runtime (clear all variables). */
export function resetRuntime(): void {
  ensureWorker().postMessage({ type: 'reset' });
}

/** Subscribe to global events (ready, loading). Returns unsubscribe function. */
export function subscribe(listener: GlobalListener): () => void {
  globalListeners.add(listener);
  // If already ready, fire immediately
  if (_ready) listener.onReady?.();
  return () => { globalListeners.delete(listener); };
}
