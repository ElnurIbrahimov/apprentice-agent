/**
 * projectContext — unified project abstraction wrapping VirtualFS + shared context.
 * Ties together WebCreator, CodePanel, Artifacts, and component library state.
 */

import ext from '../ext';
import type { DesignTokens } from './designTokens';

export interface ProjectMeta {
  id: string;
  name: string;
  description: string;
  framework: string;
  designTokens?: DesignTokens;
  createdAt: number;
  updatedAt: number;
}

export interface SharedDataOutput {
  key: string;
  fromPanel: string;
  type: 'json' | 'html' | 'csv' | 'text' | 'image';
  data: string;
  label: string;
  timestamp: number;
}

const PROJECT_META_KEY = 'aura_project_meta';
const SHARED_DATA_KEY = 'aura_shared_data';
const MAX_SHARED_ITEMS = 20;

// ── Project meta persistence ──

export async function loadProjectMeta(): Promise<ProjectMeta | null> {
  if (!ext?.storage?.local) return null;
  return new Promise(resolve => {
    ext.storage.local.get([PROJECT_META_KEY], (data: any) => {
      resolve(data[PROJECT_META_KEY] || null);
    });
  });
}

export async function saveProjectMeta(meta: ProjectMeta): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.set({ [PROJECT_META_KEY]: { ...meta, updatedAt: Date.now() } }, resolve);
  });
}

export async function clearProjectMeta(): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.remove([PROJECT_META_KEY], resolve);
  });
}

// ── Shared data outputs (cross-panel) ──

export async function publishData(output: Omit<SharedDataOutput, 'timestamp'>): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.get([SHARED_DATA_KEY], (raw: any) => {
      const list: SharedDataOutput[] = raw[SHARED_DATA_KEY] || [];
      // Replace existing with same key, or prepend
      const filtered = list.filter(item => item.key !== output.key);
      filtered.unshift({ ...output, timestamp: Date.now() });
      ext.storage.local.set({ [SHARED_DATA_KEY]: filtered.slice(0, MAX_SHARED_ITEMS) }, resolve);
    });
  });
}

export async function consumeData(key: string): Promise<SharedDataOutput | null> {
  if (!ext?.storage?.local) return null;
  return new Promise(resolve => {
    ext.storage.local.get([SHARED_DATA_KEY], (raw: any) => {
      const list: SharedDataOutput[] = raw[SHARED_DATA_KEY] || [];
      resolve(list.find(item => item.key === key) || null);
    });
  });
}

export async function listSharedData(): Promise<SharedDataOutput[]> {
  if (!ext?.storage?.local) return [];
  return new Promise(resolve => {
    ext.storage.local.get([SHARED_DATA_KEY], (raw: any) => {
      resolve(raw[SHARED_DATA_KEY] || []);
    });
  });
}

export async function clearSharedData(): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.remove([SHARED_DATA_KEY], resolve);
  });
}

// ── Project list for switcher ──

const PROJECT_LIST_KEY = 'aura_project_list';
const MAX_PROJECTS = 20;

export interface ProjectListEntry {
  id: string;
  name: string;
  framework: string;
  updatedAt: number;
}

export async function loadProjectList(): Promise<ProjectListEntry[]> {
  if (!ext?.storage?.local) return [];
  return new Promise(resolve => {
    ext.storage.local.get([PROJECT_LIST_KEY], (data: any) => {
      resolve(Array.isArray(data[PROJECT_LIST_KEY]) ? data[PROJECT_LIST_KEY] : []);
    });
  });
}

export async function saveProjectToList(entry: ProjectListEntry): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.get([PROJECT_LIST_KEY], (data: any) => {
      const list: ProjectListEntry[] = data[PROJECT_LIST_KEY] || [];
      const filtered = list.filter(e => e.id !== entry.id);
      filtered.unshift(entry);
      ext.storage.local.set({ [PROJECT_LIST_KEY]: filtered.slice(0, MAX_PROJECTS) }, resolve);
    });
  });
}

export async function removeProjectFromList(id: string): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.get([PROJECT_LIST_KEY], (data: any) => {
      const list: ProjectListEntry[] = (data[PROJECT_LIST_KEY] || []).filter((e: ProjectListEntry) => e.id !== id);
      ext.storage.local.set({ [PROJECT_LIST_KEY]: list }, resolve);
    });
  });
}
