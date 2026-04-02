import ext from '../ext';

export interface SavedWebPage {
  id: string;
  name: string;
  html: string;
  prompt?: string;
  thumbnail?: string;
  createdAt: number;
  updatedAt: number;
  tags?: string[];
}

const GALLERY_KEY = 'aura_webcreator_gallery';
const MAX_PAGES = 50;
const MAX_HTML_SIZE = 150_000;

function normalizePage(input: SavedWebPage): SavedWebPage {
  return {
    ...input,
    name: (input.name || 'Untitled page').trim() || 'Untitled page',
    html: (input.html || '').slice(0, MAX_HTML_SIZE),
    prompt: typeof input.prompt === 'string' ? input.prompt : '',
    thumbnail: typeof input.thumbnail === 'string' && input.thumbnail ? input.thumbnail : undefined,
    createdAt: Number.isFinite(input.createdAt) ? input.createdAt : Date.now(),
    updatedAt: Number.isFinite(input.updatedAt) ? input.updatedAt : Date.now(),
    tags: Array.isArray(input.tags) ? input.tags.filter(Boolean).slice(0, 10) : [],
  };
}

async function readGallery(): Promise<SavedWebPage[]> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve([]);
      return;
    }

    ext.storage.local.get([GALLERY_KEY], (data: any) => {
      const raw = Array.isArray(data?.[GALLERY_KEY]) ? data[GALLERY_KEY] : [];
      resolve(
        raw
          .filter((item: any) => item && typeof item === 'object')
          .map((item: SavedWebPage) => normalizePage(item))
          .sort((a, b) => b.updatedAt - a.updatedAt),
      );
    });
  });
}

async function writeGallery(items: SavedWebPage[]): Promise<void> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve();
      return;
    }

    ext.storage.local.set({
      [GALLERY_KEY]: items
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, MAX_PAGES)
        .map(normalizePage),
    }, () => resolve());
  });
}

export const webCreatorGalleryStore = {
  async list(): Promise<SavedWebPage[]> {
    return readGallery();
  },

  async save(page: Omit<SavedWebPage, 'id' | 'createdAt' | 'updatedAt'>): Promise<string> {
    const current = await readGallery();
    const now = Date.now();
    const next = normalizePage({
      ...page,
      id: crypto.randomUUID(),
      createdAt: now,
      updatedAt: now,
    });
    await writeGallery([next, ...current]);
    return next.id;
  },

  async get(id: string): Promise<SavedWebPage | null> {
    const items = await readGallery();
    return items.find((item) => item.id === id) || null;
  },

  async update(id: string, updates: Partial<SavedWebPage>): Promise<void> {
    const items = await readGallery();
    await writeGallery(items.map((item) => (
      item.id === id
        ? normalizePage({
          ...item,
          ...updates,
          id: item.id,
          createdAt: item.createdAt,
          updatedAt: Date.now(),
        })
        : item
    )));
  },

  async delete(id: string): Promise<void> {
    const items = await readGallery();
    await writeGallery(items.filter((item) => item.id !== id));
  },
};
