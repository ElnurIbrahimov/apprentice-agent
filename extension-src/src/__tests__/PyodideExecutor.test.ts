/**
 * Tests for PyodideExecutor — import parsing, routing, lifecycle.
 */

// Mock api module before import
jest.mock('../api', () => ({
  HTTP: 'http://localhost:8000',
  getAuthHeaders: () => ({ 'X-API-Key': 'test-key' }),
}));

import { PyodideExecutor } from '../utils/PyodideExecutor';

// Mock fetch for backend fallback tests
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('PyodideExecutor.parseImports', () => {
  test('parses simple import', () => {
    expect(PyodideExecutor.parseImports('import numpy')).toEqual(['numpy']);
  });

  test('parses from...import', () => {
    expect(PyodideExecutor.parseImports('from pandas import DataFrame')).toEqual(['pandas']);
  });

  test('parses dotted import (takes root package)', () => {
    expect(PyodideExecutor.parseImports('import os.path')).toEqual(['os']);
  });

  test('parses multiple imports', () => {
    const code = `import numpy
from scipy.stats import norm
import json
from matplotlib import pyplot`;
    const imports = PyodideExecutor.parseImports(code);
    expect(imports).toContain('numpy');
    expect(imports).toContain('scipy');
    expect(imports).toContain('json');
    expect(imports).toContain('matplotlib');
  });

  test('ignores non-import lines', () => {
    const code = `x = 1
print("import fake")
# import commented`;
    expect(PyodideExecutor.parseImports(code)).toEqual([]);
  });

  test('handles empty code', () => {
    expect(PyodideExecutor.parseImports('')).toEqual([]);
  });

  test('deduplicates imports', () => {
    const code = `import numpy
import numpy
from numpy import array`;
    expect(PyodideExecutor.parseImports(code)).toEqual(['numpy']);
  });
});

describe('PyodideExecutor.needsBackend', () => {
  test('returns true for torch', () => {
    expect(PyodideExecutor.needsBackend('import torch')).toBe(true);
  });

  test('returns true for tensorflow', () => {
    expect(PyodideExecutor.needsBackend('from tensorflow import keras')).toBe(true);
  });

  test('returns true for requests', () => {
    expect(PyodideExecutor.needsBackend('import requests')).toBe(true);
  });

  test('returns false for numpy', () => {
    expect(PyodideExecutor.needsBackend('import numpy')).toBe(false);
  });

  test('returns false for stdlib', () => {
    expect(PyodideExecutor.needsBackend('import json\nimport math\nimport re')).toBe(false);
  });

  test('returns false for pandas + matplotlib', () => {
    expect(PyodideExecutor.needsBackend('import pandas\nfrom matplotlib import pyplot')).toBe(false);
  });

  test('returns true if any import needs backend', () => {
    expect(PyodideExecutor.needsBackend('import numpy\nimport torch')).toBe(true);
  });
});

describe('PyodideExecutor lifecycle', () => {
  test('starts in idle state', () => {
    const exec = new PyodideExecutor();
    expect(exec.currentState).toBe('idle');
    expect(exec.isReady).toBe(false);
    exec.dispose();
  });

  test('dispose calls pending callbacks with done(false)', () => {
    const exec = new PyodideExecutor();
    const onDone = jest.fn();

    // Simulate pending callback by accessing internal map
    (exec as any).pendingCallbacks.set('test-id', {
      onOutput: jest.fn(),
      onVariables: jest.fn(),
      onDone,
    });

    exec.dispose();
    expect(onDone).toHaveBeenCalledWith(false, 0);
  });

  test('backend fallback works when forceBackend is true', async () => {
    const exec = new PyodideExecutor();
    const onOutput = jest.fn();
    const onVariables = jest.fn();
    const onDone = jest.fn();

    mockFetch.mockResolvedValueOnce({
      ok: true, status: 200,
      json: async () => ({
        success: true,
        outputs: [{ type: 'stdout', text: 'hello\n' }],
        variables: [{ name: 'x', type_name: 'int', repr: '42' }],
        execution_time: 0.5,
      }),
    });

    await exec.execute('print("hello")', { onOutput, onVariables, onDone }, { forceBackend: true });

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/code/execute',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(onOutput).toHaveBeenCalledWith({ type: 'stdout', text: 'hello\n' });
    expect(onVariables).toHaveBeenCalledWith([{ name: 'x', type_name: 'int', repr: '42' }]);
    expect(onDone).toHaveBeenCalledWith(true, expect.any(Number));

    exec.dispose();
  });

  test('backend fallback handles HTTP errors', async () => {
    const exec = new PyodideExecutor();
    const onOutput = jest.fn();
    const onDone = jest.fn();

    mockFetch.mockResolvedValueOnce({
      ok: false, status: 500,
      json: async () => ({ detail: 'Internal Server Error' }),
    });

    await exec.execute('broken', { onOutput, onVariables: jest.fn(), onDone }, { forceBackend: true });

    expect(onOutput).toHaveBeenCalledWith(expect.objectContaining({
      type: 'error',
      ename: 'BackendError',
    }));
    expect(onDone).toHaveBeenCalledWith(false, expect.any(Number));

    exec.dispose();
  });

  test('backend fallback handles network failure', async () => {
    const exec = new PyodideExecutor();
    const onOutput = jest.fn();
    const onDone = jest.fn();

    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    await exec.execute('code', { onOutput, onVariables: jest.fn(), onDone }, { forceBackend: true });

    expect(onOutput).toHaveBeenCalledWith(expect.objectContaining({
      type: 'error',
      evalue: 'Network error',
    }));
    expect(onDone).toHaveBeenCalledWith(false, expect.any(Number));

    exec.dispose();
  });

  test('dispose blocks further worker executions', () => {
    const exec = new PyodideExecutor();
    exec.dispose();
    expect(exec.currentState).toBe('idle');
    expect(exec.isReady).toBe(false);

    // Calling execute after dispose should use backend fallback (not crash)
    const onDone = jest.fn();
    mockFetch.mockResolvedValueOnce({
      ok: true, status: 200,
      json: async () => ({ success: true, outputs: [], variables: [] }),
    });

    // Won't crash — falls back to backend since not ready
    exec.execute('print(1)', { onOutput: jest.fn(), onVariables: jest.fn(), onDone }, { forceBackend: true });
  });

  test('needsBackend handles mixed imports correctly', () => {
    // Only stdlib + compatible = no backend needed
    expect(PyodideExecutor.needsBackend('import json\nimport numpy\nimport pandas')).toBe(false);
    // One backend-only package triggers backend
    expect(PyodideExecutor.needsBackend('import json\nimport torch')).toBe(true);
    // Empty code = no backend needed
    expect(PyodideExecutor.needsBackend('x = 1 + 2')).toBe(false);
  });
});
