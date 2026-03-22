import { useState, useCallback, useRef } from 'react';

export interface Version {
  id: string;
  timestamp: number;
  prompt: string;
  code: string;
  label?: string;
}

/**
 * Hook for managing version history of generated content.
 * Stores versions in memory (max 20 per session).
 */
export function useVersionHistory(maxVersions = 20) {
  const [versions, setVersions] = useState<Version[]>([]);
  const [currentIdx, setCurrentIdx] = useState(-1);
  // Refs to avoid stale closures in undo/redo callbacks
  const versionsRef = useRef(versions);
  versionsRef.current = versions;
  const idxRef = useRef(currentIdx);
  idxRef.current = currentIdx;

  const pushVersion = useCallback((prompt: string, code: string, label?: string) => {
    const version: Version = {
      id: `v-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: Date.now(),
      prompt,
      code,
      label,
    };
    setVersions(prev => {
      const next = [...prev, version].slice(-maxVersions);
      return next;
    });
    // Set index separately to avoid cross-setter batching issues
    setCurrentIdx(prev => {
      const len = versionsRef.current.length + 1;
      return Math.min(len - 1, maxVersions - 1);
    });
    return version;
  }, [maxVersions]);

  const goToVersion = useCallback((idx: number) => {
    const v = versionsRef.current[idx] || null;
    if (v) setCurrentIdx(idx);
    return v;
  }, []);

  const currentVersion = currentIdx >= 0 ? versions[currentIdx] : null;

  const canUndo = currentIdx > 0;
  const canRedo = currentIdx < versions.length - 1;

  const undo = useCallback(() => {
    const idx = idxRef.current;
    const vers = versionsRef.current;
    if (idx > 0) {
      setCurrentIdx(idx - 1);
      return vers[idx - 1] || null;
    }
    return null;
  }, []);

  const redo = useCallback(() => {
    const idx = idxRef.current;
    const vers = versionsRef.current;
    if (idx < vers.length - 1) {
      setCurrentIdx(idx + 1);
      return vers[idx + 1] || null;
    }
    return null;
  }, []);

  const clear = useCallback(() => {
    setVersions([]);
    setCurrentIdx(-1);
  }, []);

  return {
    versions,
    currentVersion,
    currentIdx,
    pushVersion,
    goToVersion,
    undo,
    redo,
    canUndo,
    canRedo,
    clear,
  };
}
