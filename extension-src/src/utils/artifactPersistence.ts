import ext from '../ext';

export interface PersistedArtifactsState {
  code: string;
  prompt: string;
  type: string;
  timestamp: number;
  messages?: Array<{ role: string; content: string }>;
  activeFile?: string;
}

const ARTIFACTS_STATE_KEY = 'aura_artifacts_state';

export async function loadArtifactsPanelState(): Promise<PersistedArtifactsState | null> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve(null);
      return;
    }

    ext.storage.local.get([ARTIFACTS_STATE_KEY], (data: any) => {
      const saved = data?.[ARTIFACTS_STATE_KEY];
      if (!saved || typeof saved !== 'object') {
        resolve(null);
        return;
      }

      resolve({
        code: typeof saved.code === 'string' ? saved.code : '',
        prompt: typeof saved.prompt === 'string' ? saved.prompt : '',
        type: typeof saved.type === 'string' ? saved.type : 'html',
        timestamp: typeof saved.timestamp === 'number' ? saved.timestamp : Date.now(),
        messages: Array.isArray(saved.messages) ? saved.messages : undefined,
        activeFile: typeof saved.activeFile === 'string' ? saved.activeFile : undefined,
      });
    });
  });
}

export async function saveArtifactsPanelState(
  state: PersistedArtifactsState,
): Promise<void> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve();
      return;
    }

    const isEmpty = !state.code.trim() && !state.prompt.trim();
    if (isEmpty) {
      ext.storage.local.remove([ARTIFACTS_STATE_KEY], () => resolve());
      return;
    }

    ext.storage.local.set({ [ARTIFACTS_STATE_KEY]: state }, () => resolve());
  });
}

export async function clearArtifactsPanelState(): Promise<void> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      resolve();
      return;
    }

    ext.storage.local.remove([ARTIFACTS_STATE_KEY], () => resolve());
  });
}
