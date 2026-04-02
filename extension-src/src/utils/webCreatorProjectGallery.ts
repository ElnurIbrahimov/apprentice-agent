import ext from '../ext';

export interface SavedWebProject {
  id: string;
  name: string;
  project: string;
  entryPoint: string;
  fileCount: number;
  framework: string;
  prompt?: string;
  thumbnail?: string;
  createdAt: number;
  updatedAt: number;
  tags?: string[];
}

const GALLERY_KEY = 'aura_webcreator_project_gallery';
const MAX_PROJECTS = 40;
const MAX_PROJECT_JSON_SIZE = 1_500_000;

function normalizeProject(input: SavedWebProject): SavedWebProject {
  return {
    ...input,
    name: (input.name || 'Untitled project').trim() || 'Untitled project',
    project: typeof input.project === 'string' ? input.project.slice(0, MAX_PROJECT_JSON_SIZE) : '',
    entryPoint: typeof input.entryPoint === 'string' && input.entryPoint ? input.entryPoint : 'index.html',
    fileCount: Number.isFinite(input.fileCount) ? input.fileCount : 0,
    framework: typeof input.framework === 'string' && input.framework ? input.framework : 'static',
    prompt: typeof input.prompt === 'string' ? input.prompt : '',
    thumbnail: typeof input.thumbnail === 'string' && input.thumbnail ? input.thumbnail : undefined,
    createdAt: Number.isFinite(input.createdAt) ? input.createdAt : Date.now(),
    updatedAt: Number.isFinite(input.updatedAt) ? input.updatedAt : Date.now(),
    tags: Array.isArray(input.tags) ? input.tags.filter(Boolean).slice(0, 10) : [],
  };
}

async function readGallery(): Promise<SavedWebProject[]> {
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
          .map((item: SavedWebProject) => normalizeProject(item))
          .sort((a, b) => b.updatedAt - a.updatedAt),
      );
    });
  });
}

async function writeGallery(items: SavedWebProject[]): Promise<void> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve();
      return;
    }

    ext.storage.local.set({
      [GALLERY_KEY]: items
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, MAX_PROJECTS)
        .map(normalizeProject),
    }, () => resolve());
  });
}

export const webCreatorProjectGalleryStore = {
  async list(): Promise<SavedWebProject[]> {
    return readGallery();
  },

  async save(project: Omit<SavedWebProject, 'id' | 'createdAt' | 'updatedAt'>): Promise<string> {
    const current = await readGallery();
    const now = Date.now();
    const next = normalizeProject({
      ...project,
      id: crypto.randomUUID(),
      createdAt: now,
      updatedAt: now,
    });
    await writeGallery([next, ...current]);
    return next.id;
  },

  async get(id: string): Promise<SavedWebProject | null> {
    const items = await readGallery();
    return items.find((item) => item.id === id) || null;
  },

  async update(id: string, updates: Partial<SavedWebProject>): Promise<void> {
    const items = await readGallery();
    await writeGallery(items.map((item) => (
      item.id === id
        ? normalizeProject({
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
