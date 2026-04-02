/**
 * designTokens — shared design system tokens for consistent styling
 * across panels and AI-generated content.
 */

import ext from '../ext';

export interface DesignTokens {
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    error: string;
    success: string;
  };
  fonts: {
    heading: string;
    body: string;
    mono: string;
  };
  spacing: {
    unit: number;
    radius: string;
  };
  darkMode: boolean;
}

export interface TokenPreset {
  id: string;
  name: string;
  tokens: DesignTokens;
}

export const TOKEN_PRESETS: TokenPreset[] = [
  {
    id: 'aura-dark',
    name: 'Aura Dark',
    tokens: {
      colors: {
        primary: '#7c3aed', secondary: '#8b5cf6', accent: '#06b6d4',
        background: '#0a0a0f', surface: '#151520', text: '#ededed',
        textSecondary: '#94a3b8', border: '#1e293b', error: '#ef4444', success: '#10b981',
      },
      fonts: { heading: "'Inter', system-ui, sans-serif", body: "'Inter', system-ui, sans-serif", mono: "'JetBrains Mono', monospace" },
      spacing: { unit: 4, radius: '8px' },
      darkMode: true,
    },
  },
  {
    id: 'aura-light',
    name: 'Aura Light',
    tokens: {
      colors: {
        primary: '#7c3aed', secondary: '#8b5cf6', accent: '#0ea5e9',
        background: '#ffffff', surface: '#f8fafc', text: '#0f172a',
        textSecondary: '#64748b', border: '#e2e8f0', error: '#ef4444', success: '#10b981',
      },
      fonts: { heading: "'Inter', system-ui, sans-serif", body: "'Inter', system-ui, sans-serif", mono: "'JetBrains Mono', monospace" },
      spacing: { unit: 4, radius: '8px' },
      darkMode: false,
    },
  },
  {
    id: 'minimal',
    name: 'Minimal',
    tokens: {
      colors: {
        primary: '#171717', secondary: '#404040', accent: '#737373',
        background: '#ffffff', surface: '#fafafa', text: '#0a0a0a',
        textSecondary: '#737373', border: '#e5e5e5', error: '#dc2626', success: '#16a34a',
      },
      fonts: { heading: "'Inter', system-ui, sans-serif", body: "system-ui, sans-serif", mono: "monospace" },
      spacing: { unit: 4, radius: '4px' },
      darkMode: false,
    },
  },
  {
    id: 'ocean',
    name: 'Ocean',
    tokens: {
      colors: {
        primary: '#0369a1', secondary: '#0284c7', accent: '#06b6d4',
        background: '#0c1222', surface: '#162032', text: '#e0f2fe',
        textSecondary: '#7dd3fc', border: '#1e3a5f', error: '#f87171', success: '#34d399',
      },
      fonts: { heading: "'Inter', system-ui, sans-serif", body: "'Inter', system-ui, sans-serif", mono: "'Fira Code', monospace" },
      spacing: { unit: 4, radius: '12px' },
      darkMode: true,
    },
  },
  {
    id: 'forest',
    name: 'Forest',
    tokens: {
      colors: {
        primary: '#15803d', secondary: '#16a34a', accent: '#a3e635',
        background: '#0a1f0a', surface: '#14291a', text: '#dcfce7',
        textSecondary: '#86efac', border: '#1a3a20', error: '#fb923c', success: '#4ade80',
      },
      fonts: { heading: "'Inter', system-ui, sans-serif", body: "'Inter', system-ui, sans-serif", mono: "monospace" },
      spacing: { unit: 4, radius: '6px' },
      darkMode: true,
    },
  },
];

/** Convert tokens to CSS variables block */
export function tokensToCssVariables(tokens: DesignTokens): string {
  return `:root {
  --color-primary: ${tokens.colors.primary};
  --color-secondary: ${tokens.colors.secondary};
  --color-accent: ${tokens.colors.accent};
  --color-background: ${tokens.colors.background};
  --color-surface: ${tokens.colors.surface};
  --color-text: ${tokens.colors.text};
  --color-text-secondary: ${tokens.colors.textSecondary};
  --color-border: ${tokens.colors.border};
  --color-error: ${tokens.colors.error};
  --color-success: ${tokens.colors.success};
  --font-heading: ${tokens.fonts.heading};
  --font-body: ${tokens.fonts.body};
  --font-mono: ${tokens.fonts.mono};
  --spacing-unit: ${tokens.spacing.unit}px;
  --radius: ${tokens.spacing.radius};
}
body {
  background: var(--color-background);
  color: var(--color-text);
  font-family: var(--font-body);
}`;
}

/** Build a system prompt addition describing the active tokens */
export function tokensToSystemPrompt(tokens: DesignTokens): string {
  return `\n[DESIGN TOKENS]
Use these CSS variables for all styling. Do NOT use hardcoded colors — always use var(--color-*).
Primary: ${tokens.colors.primary} → var(--color-primary)
Secondary: ${tokens.colors.secondary} → var(--color-secondary)
Accent: ${tokens.colors.accent} → var(--color-accent)
Background: ${tokens.colors.background} → var(--color-background)
Surface: ${tokens.colors.surface} → var(--color-surface)
Text: ${tokens.colors.text} → var(--color-text)
Text secondary: ${tokens.colors.textSecondary} → var(--color-text-secondary)
Border: ${tokens.colors.border} → var(--color-border)
Fonts: heading=${tokens.fonts.heading}, body=${tokens.fonts.body}
Border radius: ${tokens.spacing.radius}
Mode: ${tokens.darkMode ? 'dark' : 'light'}
[/DESIGN TOKENS]`;
}

// ── Persistence ──
const TOKENS_KEY = 'aura_design_tokens';

export async function loadDesignTokens(): Promise<DesignTokens | null> {
  if (!ext?.storage?.local) return null;
  return new Promise(resolve => {
    ext.storage.local.get([TOKENS_KEY], (data: any) => {
      resolve(data[TOKENS_KEY] || null);
    });
  });
}

export async function saveDesignTokens(tokens: DesignTokens): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.set({ [TOKENS_KEY]: tokens }, resolve);
  });
}
