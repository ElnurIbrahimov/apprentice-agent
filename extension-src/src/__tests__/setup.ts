/**
 * Jest setup — mock chrome extension APIs and Web API polyfills.
 */
import { TextEncoder, TextDecoder } from 'util';

// Polyfill Web APIs missing from jsdom
Object.assign(globalThis, { TextEncoder, TextDecoder });

// In-memory storage mock
const storage: Record<string, any> = {};

const chromeMock = {
  storage: {
    local: {
      get: jest.fn((keys: string[], cb: (d: Record<string, any>) => void) => {
        const result: Record<string, any> = {};
        for (const k of keys) {
          if (k in storage) result[k] = storage[k];
        }
        cb(result);
      }),
      set: jest.fn((data: Record<string, any>, cb?: () => void) => {
        Object.assign(storage, data);
        cb?.();
      }),
      remove: jest.fn((keys: string[], cb?: () => void) => {
        for (const k of keys) delete storage[k];
        cb?.();
      }),
    },
  },
  runtime: {
    getURL: jest.fn((path: string) => `chrome-extension://mock-id/${path}`),
    sendMessage: jest.fn(),
    onMessage: {
      addListener: jest.fn(),
      removeListener: jest.fn(),
    },
  },
};

Object.defineProperty(globalThis, 'chrome', { value: chromeMock, writable: true });
Object.defineProperty(globalThis, 'browser', { value: undefined, writable: true });

// Mock URL.createObjectURL / revokeObjectURL for export tests
global.URL.createObjectURL = jest.fn(() => 'blob:mock-url');
global.URL.revokeObjectURL = jest.fn();

export { storage, chromeMock };
