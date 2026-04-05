import type { OutputBlock } from './pyodideExecutor';

export function executeJS(
  code: string,
  callbacks: {
    onOutput: (block: OutputBlock) => void;
    onDone: (success: boolean, time: number) => void;
  }
): void {
  const start = performance.now();
  const { onOutput, onDone } = callbacks;

  // Mock console that captures output
  const mockConsole = {
    log: (...args: unknown[]) => onOutput({ type: 'stdout', text: args.map(String).join(' ') }),
    warn: (...args: unknown[]) => onOutput({ type: 'stderr', text: `⚠ ${args.map(String).join(' ')}` }),
    error: (...args: unknown[]) => onOutput({ type: 'stderr', text: `✗ ${args.map(String).join(' ')}` }),
    info: (...args: unknown[]) => onOutput({ type: 'stdout', text: args.map(String).join(' ') }),
    table: (data: unknown) => onOutput({ type: 'stdout', text: JSON.stringify(data, null, 2) }),
  };

  try {
    // Use Function constructor for isolated scope
    const fn = new Function('console', code);
    const result = fn(mockConsole);

    // Handle async results (Promises)
    if (result && typeof result.then === 'function') {
      result
        .then((val: unknown) => {
          if (val !== undefined) onOutput({ type: 'stdout', text: String(val) });
          onDone(true, performance.now() - start);
        })
        .catch((err: Error) => {
          onOutput({ type: 'error', ename: 'AsyncError', evalue: err.message, traceback: err.stack });
          onDone(false, performance.now() - start);
        });
    } else {
      if (result !== undefined) onOutput({ type: 'stdout', text: String(result) });
      onDone(true, performance.now() - start);
    }
  } catch (err: any) {
    onOutput({
      type: 'error',
      ename: err.name || 'Error',
      evalue: err.message || String(err),
      traceback: err.stack,
    });
    onDone(false, performance.now() - start);
  }
}
