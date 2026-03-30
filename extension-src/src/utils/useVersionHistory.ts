import { useState, useCallback, useRef, useEffect } from 'react';

declare const browser: any;
const ext = typeof chrome !== 'undefined' ? chrome : typeof browser !== 'undefined' ? browser : null;

export interface Version {
  id: string;
  timestamp: number;
  prompt: string;
  code: string;
  label?: string;
}

/* ─── Chrome storage helpers ─── */
function storageLoad(key: string): Promise<{ versions: Version[]; currentIdx: number } | null> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) { resolve(null); return; }
    ext.storage.local.get([key], (d: any) => {
      try { resolve(d?.[key] ? JSON.parse(d[key]) : null); }
      catch { resolve(null); }
    });
  });
}

function storageSave(key: string, versions: Version[], currentIdx: number) {
  if (!ext?.storage?.local || versions.length === 0) return;
  // Only save essential fields, cap at 10 versions to conserve storage
  const trimmed = versions.slice(-10).map(v => ({
    id: v.id,
    timestamp: v.timestamp,
    prompt: v.prompt.slice(0, 200),
    code: v.code.slice(0, 50000), // cap code at 50KB per version
    label: v.label,
  }));
  ext.storage.local.set({ [key]: JSON.stringify({ versions: trimmed, currentIdx }) });
}

function storageClear(key: string) {
  ext?.storage?.local?.remove([key]);
}

/**
 * Hook for managing version history of generated content.
 * Persists to chrome.storage.local when a storageKey is provided.
 */
export function useVersionHistory(maxVersions = 20, storageKey?: string) {
  const [versions, setVersions] = useState<Version[]>([]);
  const [currentIdx, setCurrentIdx] = useState(-1);
  const versionsRef = useRef(versions);
  versionsRef.current = versions;
  const idxRef = useRef(currentIdx);
  idxRef.current = currentIdx;
  const restoredRef = useRef(false);

  // Restore from storage on mount
  useEffect(() => {
    if (!storageKey || restoredRef.current) return;
    restoredRef.current = true;
    storageLoad(storageKey).then(saved => {
      if (!saved || !saved.versions?.length) return;
      setVersions(saved.versions);
      setCurrentIdx(saved.currentIdx >= 0 ? Math.min(saved.currentIdx, saved.versions.length - 1) : saved.versions.length - 1);
    });
  }, [storageKey]);

  // Auto-save to storage on changes (debounced)
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!storageKey || !restoredRef.current) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      storageSave(storageKey, versions, currentIdx);
    }, 500);
    return () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); };
  }, [versions, currentIdx, storageKey]);

  const pushVersion = useCallback((prompt: string, code: string, label?: string) => {
    const version: Version = {
      id: `v-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: Date.now(),
      prompt,
      code,
      label,
    };
    setVersions(prev => [...prev, version].slice(-maxVersions));
    setCurrentIdx(prev => Math.min(prev + 1, maxVersions - 1));
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
    if (storageKey) storageClear(storageKey);
  }, [storageKey]);

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
