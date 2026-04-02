import ext from '../ext';

export interface CreationSettings {
  autoFixErrors: boolean;
  autoOpenConsoleOnError: boolean;
  showDiffBeforeApply: boolean;
  autoAcceptDiffLineThreshold: number;
  forceDiffReviewChangePercent: number;
}

const CREATION_SETTINGS_KEY = 'aura_creation_settings';

const DEFAULT_CREATION_SETTINGS: CreationSettings = {
  autoFixErrors: true,
  autoOpenConsoleOnError: true,
  showDiffBeforeApply: true,
  autoAcceptDiffLineThreshold: 4,
  forceDiffReviewChangePercent: 30,
};

export function getDefaultCreationSettings(): CreationSettings {
  return { ...DEFAULT_CREATION_SETTINGS };
}

export async function loadCreationSettings(): Promise<CreationSettings> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve(getDefaultCreationSettings());
      return;
    }

    ext.storage.local.get([CREATION_SETTINGS_KEY], (data: any) => {
      const saved = data?.[CREATION_SETTINGS_KEY];
      resolve({
        ...DEFAULT_CREATION_SETTINGS,
        ...(saved && typeof saved === 'object' ? saved : {}),
      });
    });
  });
}

export async function saveCreationSettings(
  updates: Partial<CreationSettings>,
): Promise<CreationSettings> {
  const next = {
    ...(await loadCreationSettings()),
    ...updates,
  };

  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve(next);
      return;
    }

    ext.storage.local.set({ [CREATION_SETTINGS_KEY]: next }, () => resolve(next));
  });
}
