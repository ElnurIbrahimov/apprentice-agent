import ext from '../ext';

export type SavedArtifactType = 'html' | 'react' | 'svg' | 'mermaid' | 'chart' | 'markdown' | 'css';

export interface SavedArtifact {
  id: string;
  name: string;
  type: SavedArtifactType;
  code: string;
  prompt?: string;
  thumbnail?: string;
  createdAt: number;
  updatedAt: number;
  tags?: string[];
}

const GALLERY_KEY = 'aura_artifact_gallery';
const MAX_ARTIFACTS = 50;
const MAX_CODE_SIZE = 100_000;

function normalizeArtifact(input: SavedArtifact): SavedArtifact {
  return {
    ...input,
    code: input.code.slice(0, MAX_CODE_SIZE),
    updatedAt: Number.isFinite(input.updatedAt) ? input.updatedAt : Date.now(),
    createdAt: Number.isFinite(input.createdAt) ? input.createdAt : Date.now(),
    name: (input.name || 'Untitled artifact').trim() || 'Untitled artifact',
    thumbnail: typeof input.thumbnail === 'string' && input.thumbnail ? input.thumbnail : undefined,
    tags: Array.isArray(input.tags) ? input.tags.filter(Boolean).slice(0, 10) : [],
  };
}

async function readGallery(): Promise<SavedArtifact[]> {
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
          .map((item: SavedArtifact) => normalizeArtifact(item))
          .sort((a, b) => b.updatedAt - a.updatedAt),
      );
    });
  });
}

async function writeGallery(items: SavedArtifact[]): Promise<void> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve();
      return;
    }

    const trimmed = items
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_ARTIFACTS)
      .map(normalizeArtifact);

    ext.storage.local.set({ [GALLERY_KEY]: trimmed }, () => resolve());
  });
}

export const galleryStore = {
  async list(): Promise<SavedArtifact[]> {
    return readGallery();
  },

  async save(
    artifact: Omit<SavedArtifact, 'id' | 'createdAt' | 'updatedAt'>,
  ): Promise<string> {
    const current = await readGallery();
    const now = Date.now();
    const next: SavedArtifact = normalizeArtifact({
      ...artifact,
      id: crypto.randomUUID(),
      createdAt: now,
      updatedAt: now,
    });
    await writeGallery([next, ...current]);
    return next.id;
  },

  async get(id: string): Promise<SavedArtifact | null> {
    const items = await readGallery();
    return items.find((item) => item.id === id) || null;
  },

  async delete(id: string): Promise<void> {
    const items = await readGallery();
    await writeGallery(items.filter((item) => item.id !== id));
  },

  async update(id: string, updates: Partial<SavedArtifact>): Promise<void> {
    const items = await readGallery();
    const next = items.map((item) => {
      if (item.id !== id) return item;
      return normalizeArtifact({
        ...item,
        ...updates,
        id: item.id,
        createdAt: item.createdAt,
        updatedAt: Date.now(),
      });
    });
    await writeGallery(next);
  },
};
