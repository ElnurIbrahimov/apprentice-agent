/**
 * useWebContainer — React hook for managing WebContainer lifecycle.
 *
 * Handles: boot, file sync from VirtualFS, npm install, dev server,
 * terminal output, and auto-detection of when WebContainer is needed.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import WebContainerManager, { type WCStatus, type WCEventHandlers } from '../utils/WebContainerManager';
import type { TerminalLine } from '../components/TerminalOutput';
import { scanProjectImports } from '../utils/importScanner';

export interface UseWebContainerOptions {
  /** Enable WebContainer (false = no boot, no resources) */
  enabled: boolean;
  /** Project files as { path: content } map */
  files?: Record<string, string>;
  /** Callback when preview URL is ready */
  onServerReady?: (url: string) => void;
}

export interface UseWebContainerResult {
  /** Current WebContainer status */
  status: WCStatus;
  /** Whether WebContainers are supported in this environment */
  supported: boolean;
  /** Preview URL from the dev server (null if not running) */
  serverUrl: string | null;
  /** Terminal output lines */
  terminalLines: TerminalLine[];
  /** Whether a package.json exists in the project files */
  hasPackageJson: boolean;
  /** Boot the WebContainer manually */
  boot: () => Promise<void>;
  /** Sync files to the WebContainer */
  syncFiles: (files: Record<string, string>) => Promise<void>;
  /** Install dependencies */
  installDeps: () => Promise<void>;
  /** Start the dev server */
  startServer: () => Promise<void>;
  /** Full setup: boot → mount → install → start server */
  fullSetup: (files: Record<string, string>) => Promise<void>;
  /** Write a single file (for incremental updates) */
  writeFile: (path: string, content: string) => Promise<void>;
  /** Tear down everything */
  teardown: () => Promise<void>;
  /** Clear terminal output */
  clearTerminal: () => void;
}

const MAX_TERMINAL_LINES = 500;

export function useWebContainer(options: UseWebContainerOptions): UseWebContainerResult {
  const { enabled, files, onServerReady } = options;
  const [status, setStatus] = useState<WCStatus>('idle');
  const [serverUrl, setServerUrl] = useState<string | null>(null);
  const [terminalLines, setTerminalLines] = useState<TerminalLine[]>([]);
  const supported = WebContainerManager.isSupported();
  const managerRef = useRef<WebContainerManager | null>(null);
  const onServerReadyRef = useRef(onServerReady);
  onServerReadyRef.current = onServerReady;

  const hasPackageJson = !!(files && 'package.json' in files);

  const addTerminalLine = useCallback((text: string, type: TerminalLine['type']) => {
    setTerminalLines(prev => {
      const next = [...prev, { text, type, timestamp: Date.now() }];
      return next.length > MAX_TERMINAL_LINES ? next.slice(-MAX_TERMINAL_LINES) : next;
    });
  }, []);

  const getManager = useCallback(() => {
    if (!managerRef.current) {
      managerRef.current = WebContainerManager.getInstance();
      const handlers: WCEventHandlers = {
        onOutput: (text, stream) => addTerminalLine(text, stream === 'stderr' ? 'stderr' : 'stdout'),
        onServerReady: (_port, url) => {
          setServerUrl(url);
          onServerReadyRef.current?.(url);
        },
        onStatusChange: (s) => setStatus(s),
        onError: (err) => addTerminalLine(`Error: ${err.message}`, 'stderr'),
      };
      managerRef.current.setHandlers(handlers);
    }
    return managerRef.current;
  }, [addTerminalLine]);

  const boot = useCallback(async () => {
    if (!supported) {
      addTerminalLine('WebContainers not supported (SharedArrayBuffer unavailable)', 'stderr');
      return;
    }
    addTerminalLine('[System] Booting WebContainer...', 'system');
    try {
      await getManager().boot();
      addTerminalLine('[System] WebContainer ready', 'system');
    } catch (err: any) {
      addTerminalLine(`[System] Boot failed: ${err.message}`, 'stderr');
    }
  }, [supported, getManager, addTerminalLine]);

  const syncFiles = useCallback(async (projectFiles: Record<string, string>) => {
    const mgr = getManager();
    if (!mgr.isBooted()) await boot();
    addTerminalLine(`[System] Mounting ${Object.keys(projectFiles).length} files...`, 'system');
    await mgr.mountFiles(projectFiles);
    addTerminalLine('[System] Files mounted', 'system');
  }, [getManager, boot, addTerminalLine]);

  const installDeps = useCallback(async () => {
    const mgr = getManager();
    if (!mgr.isBooted()) throw new Error('WebContainer not booted');
    addTerminalLine('[System] Running npm install...', 'system');
    const { exitCode } = await mgr.installDeps();
    if (exitCode === 0) {
      addTerminalLine('[System] Dependencies installed', 'system');
    } else {
      addTerminalLine(`[System] npm install exited with code ${exitCode}`, 'stderr');
    }
  }, [getManager, addTerminalLine]);

  const startServer = useCallback(async () => {
    const mgr = getManager();
    if (!mgr.isBooted()) throw new Error('WebContainer not booted');
    addTerminalLine('[System] Starting dev server...', 'system');
    try {
      const url = await mgr.startDevServer();
      addTerminalLine(`[System] Dev server ready at ${url}`, 'system');
    } catch (err: any) {
      addTerminalLine(`[System] Dev server failed: ${err.message}`, 'stderr');
    }
  }, [getManager, addTerminalLine]);

  const fullSetup = useCallback(async (projectFiles: Record<string, string>) => {
    try {
      await boot();
      await syncFiles(projectFiles);

      // Auto-detect and install missing packages
      const pkgJsonStr = projectFiles['package.json'];
      if (pkgJsonStr) {
        const pkgJson = JSON.parse(pkgJsonStr);
        const { missing } = scanProjectImports(projectFiles, pkgJson);
        if (missing.length > 0) {
          addTerminalLine(`[System] Auto-detected missing packages: ${missing.join(', ')}`, 'system');
          // Add missing packages to package.json before install
          const deps = pkgJson.dependencies || {};
          for (const pkg of missing) {
            deps[pkg] = 'latest';
          }
          pkgJson.dependencies = deps;
          await getManager().writeFile('package.json', JSON.stringify(pkgJson, null, 2));
        }
        await installDeps();
        await startServer();
      }
    } catch (err: any) {
      addTerminalLine(`[System] Setup failed: ${err.message}`, 'stderr');
    }
  }, [boot, syncFiles, installDeps, startServer, getManager, addTerminalLine]);

  const writeFile = useCallback(async (path: string, content: string) => {
    const mgr = getManager();
    if (mgr.isBooted()) {
      await mgr.writeFile(path, content);
    }
  }, [getManager]);

  const teardown = useCallback(async () => {
    const mgr = managerRef.current;
    if (mgr) {
      await mgr.teardown();
      setServerUrl(null);
      setStatus('idle');
    }
  }, []);

  const clearTerminal = useCallback(() => {
    setTerminalLines([]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Don't teardown on unmount — WebContainer is a singleton shared across panels.
      // It will auto-shutdown after idle timeout.
    };
  }, []);

  // Auto-teardown when disabled
  useEffect(() => {
    if (!enabled && managerRef.current?.isBooted()) {
      teardown();
    }
  }, [enabled, teardown]);

  return {
    status,
    supported,
    serverUrl,
    terminalLines,
    hasPackageJson,
    boot,
    syncFiles,
    installDeps,
    startServer,
    fullSetup,
    writeFile,
    teardown,
    clearTerminal,
  };
}
