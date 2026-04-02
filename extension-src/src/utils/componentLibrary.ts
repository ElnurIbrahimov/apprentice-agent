/**
 * componentLibrary — reusable UI components saved by the user.
 * AI references these when generating new pages (VOYAGER-style composition).
 */

import ext from '../ext';

export interface UIComponent {
  id: string;
  name: string;
  description: string;
  category: ComponentCategory;
  html: string;
  css: string;
  js?: string;
  thumbnail?: string;
  tags: string[];
  usageCount: number;
  createdAt: number;
  updatedAt: number;
  source: 'generated' | 'captured' | 'manual';
}

export type ComponentCategory =
  | 'navigation'
  | 'cards'
  | 'forms'
  | 'layout'
  | 'hero'
  | 'footer'
  | 'buttons'
  | 'modals'
  | 'tables'
  | 'other';

export const CATEGORIES: { value: ComponentCategory; label: string }[] = [
  { value: 'navigation', label: 'Navigation' },
  { value: 'hero', label: 'Hero / Header' },
  { value: 'cards', label: 'Cards' },
  { value: 'forms', label: 'Forms' },
  { value: 'layout', label: 'Layout' },
  { value: 'footer', label: 'Footer' },
  { value: 'buttons', label: 'Buttons' },
  { value: 'modals', label: 'Modals' },
  { value: 'tables', label: 'Tables' },
  { value: 'other', label: 'Other' },
];

const LIBRARY_KEY = 'aura_component_library';
const MAX_COMPONENTS = 100;
const MAX_HTML_SIZE = 50_000;

function generateId(): string {
  return `comp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

async function loadAll(): Promise<UIComponent[]> {
  if (!ext?.storage?.local) return [];
  return new Promise(resolve => {
    ext.storage.local.get([LIBRARY_KEY], (data: any) => {
      resolve(Array.isArray(data[LIBRARY_KEY]) ? data[LIBRARY_KEY] : []);
    });
  });
}

async function saveAll(components: UIComponent[]): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.set({ [LIBRARY_KEY]: components.slice(0, MAX_COMPONENTS) }, resolve);
  });
}

export const componentLibrary = {
  async list(category?: ComponentCategory): Promise<UIComponent[]> {
    const all = await loadAll();
    if (!category) return all;
    return all.filter(c => c.category === category);
  },

  async get(id: string): Promise<UIComponent | null> {
    const all = await loadAll();
    return all.find(c => c.id === id) || null;
  },

  async save(component: Omit<UIComponent, 'id' | 'createdAt' | 'updatedAt' | 'usageCount'>): Promise<string> {
    if (component.html.length > MAX_HTML_SIZE) {
      throw new Error(`Component HTML too large (max ${MAX_HTML_SIZE} chars)`);
    }
    const all = await loadAll();
    const id = generateId();
    const now = Date.now();
    all.unshift({
      ...component,
      id,
      usageCount: 0,
      createdAt: now,
      updatedAt: now,
    });
    await saveAll(all);
    return id;
  },

  async update(id: string, updates: Partial<UIComponent>): Promise<void> {
    const all = await loadAll();
    const idx = all.findIndex(c => c.id === id);
    if (idx === -1) return;
    all[idx] = { ...all[idx], ...updates, updatedAt: Date.now() };
    await saveAll(all);
  },

  async delete(id: string): Promise<void> {
    const all = await loadAll();
    await saveAll(all.filter(c => c.id !== id));
  },

  async incrementUsage(id: string): Promise<void> {
    const all = await loadAll();
    const comp = all.find(c => c.id === id);
    if (comp) {
      comp.usageCount += 1;
      comp.updatedAt = Date.now();
      await saveAll(all);
    }
  },

  async search(query: string): Promise<UIComponent[]> {
    const q = query.toLowerCase();
    const all = await loadAll();
    return all.filter(c =>
      c.name.toLowerCase().includes(q) ||
      c.description.toLowerCase().includes(q) ||
      c.tags.some(t => t.toLowerCase().includes(q))
    );
  },

  /** Generate a summary for including in LLM system prompts. */
  async getSummaryForLLM(): Promise<string> {
    const all = await loadAll();
    if (all.length === 0) return '';
    const lines = all
      .sort((a, b) => b.usageCount - a.usageCount)
      .slice(0, 20) // Top 20 by usage
      .map(c => `- ${c.name} (${c.category}): ${c.description}`)
      .join('\n');
    return `\n[REUSABLE COMPONENTS]\nYou have these saved components. Use them when appropriate:\n${lines}\n[/REUSABLE COMPONENTS]`;
  },
};
