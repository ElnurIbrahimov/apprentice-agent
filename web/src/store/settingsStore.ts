import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type ColorPreset = 'aura' | 'ocean' | 'forest' | 'sunset' | 'rose' | 'mono';

export interface Settings {
  theme: 'dark' | 'light' | 'system';
  fontSize: 'small' | 'medium' | 'large';
  showThinking: boolean;
  autoScroll: boolean;
  soundEnabled: boolean;
  colorPreset: ColorPreset;
  onboardingDone: boolean;
  userName: string;
  backendUrl: string;
}

interface SettingsState {
  settings: Settings;
  updateSettings: (settings: Partial<Settings>) => void;
  resetSettings: () => void;
}

const defaultSettings: Settings = {
  theme: 'dark',
  fontSize: 'medium',
  showThinking: true,
  autoScroll: true,
  soundEnabled: false,
  colorPreset: 'aura',
  onboardingDone: false,
  userName: '',
  backendUrl: '',
};

// Map font size to CSS class
export const fontSizeClasses: Record<Settings['fontSize'], string> = {
  small: 'text-sm',
  medium: 'text-base',
  large: 'text-lg',
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      settings: defaultSettings,
      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),
      resetSettings: () => set({ settings: defaultSettings }),
    }),
    {
      name: 'aura-settings',
    }
  )
);

// Apply theme to document
export const applyTheme = (theme: Settings['theme']) => {
  const root = document.documentElement;
  root.classList.remove('dark', 'light', 'system');
  if (theme === 'system') {
    root.classList.add('system');
    // Also add dark/light for Tailwind dark: prefix compatibility
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.classList.add(isDark ? 'dark' : 'light');
  } else {
    root.classList.add(theme);
  }
};

// Apply font size to document
export const applyFontSize = (fontSize: Settings['fontSize']) => {
  const root = document.documentElement;
  root.classList.remove('text-sm', 'text-base', 'text-lg');
  root.classList.add(fontSizeClasses[fontSize]);
};

// Apply color preset to document
export const applyColorPreset = (preset: ColorPreset) => {
  const root = document.documentElement;
  root.classList.remove('preset-aura', 'preset-ocean', 'preset-forest', 'preset-sunset', 'preset-rose', 'preset-mono');
  if (preset !== 'aura') {
    root.classList.add(`preset-${preset}`);
  }
};
