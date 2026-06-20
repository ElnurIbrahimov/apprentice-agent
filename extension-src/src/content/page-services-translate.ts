/**
 * Full Page Translation — extracted from page-services.ts
 * Provides in-page translation overlay with bilingual/translated mode toggle.
 */

const TRANSLATABLE_SELECTORS = 'p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, figcaption';
const AURA_TRANSLATE_ATTR = 'data-aura-translated';
const BATCH_SIZE = 10;
const MAX_CONCURRENT = 10;

interface TranslationState {
  mode: 'bilingual' | 'translated';
  targetLang: string;
  active: boolean;
  badge: HTMLDivElement | null;
  elements: { original: HTMLElement; translation: HTMLDivElement }[];
  activeCount: number;
}

export function initTranslation(ext: typeof chrome): {
  start(targetLang: string): Promise<void>;
  remove(): void;
  setMode(mode: 'bilingual' | 'translated'): void;
} {
  const state: TranslationState = {
    mode: 'bilingual',
    targetLang: 'English',
    active: false,
    badge: null,
    elements: [],
    activeCount: 0,
  };

  function getTranslatableElements(): HTMLElement[] {
    const all = document.querySelectorAll<HTMLElement>(TRANSLATABLE_SELECTORS);
    const results: HTMLElement[] = [];
    for (const el of all) {
      if (el.hasAttribute(AURA_TRANSLATE_ATTR)) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) continue;
      if (el.closest('#aura-host, #aura-dock-shadow, #aura-quick-action-host, .aura-translate-badge')) continue;
      if (el.tagName === 'SPAN' && (el.textContent || '').trim().length <= 20) continue;
      const text = (el.textContent || '').trim();
      if (text.length < 5) continue;
      results.push(el);
    }
    return results;
  }

  function createTranslationElement(originalEl: HTMLElement): HTMLDivElement {
    const translationDiv = document.createElement('div');
    translationDiv.className = 'aura-page-translation';
    translationDiv.setAttribute('data-aura-translation', 'true');
    Object.assign(translationDiv.style, {
      borderLeft: '2px solid rgba(124, 58, 237, 0.6)',
      background: 'rgba(124, 58, 237, 0.05)',
      padding: '6px 10px',
      marginTop: '4px',
      marginBottom: '4px',
      fontSize: '0.95em',
      color: 'inherit',
      opacity: '0',
      fontFamily: 'inherit',
      lineHeight: '1.5',
      borderRadius: '0 4px 4px 0',
      transition: 'opacity 0.3s ease',
      fontStyle: 'italic',
    });
    translationDiv.textContent = 'Translating...';
    translationDiv.style.color = 'rgba(124, 58, 237, 0.5)';

    originalEl.setAttribute(AURA_TRANSLATE_ATTR, 'true');
    originalEl.after(translationDiv);

    // Fade in the placeholder
    requestAnimationFrame(() => { translationDiv.style.opacity = '0.6'; });
    return translationDiv;
  }

  function fadeInTranslation(el: HTMLDivElement, text: string): void {
    el.style.opacity = '0';
    el.textContent = text;
    el.style.fontStyle = 'normal';
    el.style.color = 'inherit';
    requestAnimationFrame(() => { el.style.opacity = '0.85'; });
  }

  function translateBatchRequest(texts: string[], lang: string): Promise<string[]> {
    return new Promise((resolve) => {
      try {
        ext.runtime.sendMessage(
          { type: 'TRANSLATE_BATCH', texts, targetLang: lang },
          (response: { ok: boolean; translations?: string[]; error?: string }) => {
            if (ext.runtime.lastError) {
              resolve(texts.map(() => '[Translation failed]'));
              return;
            }
            if (response?.ok && response.translations) {
              resolve(response.translations);
            } else {
              resolve(texts.map(() => response?.error || '[Translation failed]'));
            }
          }
        );
      } catch {
        resolve(texts.map(() => '[Translation failed]'));
      }
    });
  }

  function updateBadgeText(): void {
    if (!state.badge) return;
    const modeBtn = state.badge.querySelector('[data-badge-mode]') as HTMLElement | null;
    if (modeBtn) {
      modeBtn.textContent = state.mode === 'bilingual' ? 'Bilingual' : 'Translated';
    }
  }

  function setMode(mode: 'bilingual' | 'translated'): void {
    state.mode = mode;
    for (const pair of state.elements) {
      if (mode === 'translated') {
        pair.original.style.display = 'none';
        pair.translation.style.marginTop = '0';
      } else {
        pair.original.style.display = '';
        pair.translation.style.marginTop = '4px';
      }
    }
    updateBadgeText();
  }

  function remove(): void {
    state.active = false;
    for (const pair of state.elements) {
      pair.translation.remove();
      pair.original.removeAttribute(AURA_TRANSLATE_ATTR);
      pair.original.style.display = '';
    }
    state.elements = [];
    if (state.badge) {
      state.badge.remove();
      state.badge = null;
    }
  }

  function showTranslateBadge(): void {
    if (state.badge) { state.badge.remove(); state.badge = null; }

    state.badge = document.createElement('div');
    state.badge.className = 'aura-translate-badge';
    Object.assign(state.badge.style, {
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      zIndex: '2147483646',
      background: 'rgba(10, 8, 24, 0.92)',
      backdropFilter: 'blur(20px) saturate(1.5)',
      WebkitBackdropFilter: 'blur(20px) saturate(1.5)',
      border: '1px solid rgba(124, 58, 237, 0.35)',
      borderRadius: '12px',
      padding: '8px 12px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05) inset',
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif",
      fontSize: '12px',
      color: 'rgba(226, 232, 240, 0.9)',
    });

    // Purple status dot
    const dot = document.createElement('span');
    Object.assign(dot.style, {
      width: '6px', height: '6px', borderRadius: '50%',
      background: '#7c3aed', flexShrink: '0',
    });
    state.badge.appendChild(dot);

    // Label
    const label = document.createElement('span');
    label.style.color = 'rgba(160, 148, 210, 0.8)';
    label.textContent = 'Translation active';
    state.badge.appendChild(label);

    // Separator
    const sep1 = document.createElement('span');
    Object.assign(sep1.style, { width: '1px', height: '14px', background: 'rgba(255,255,255,0.1)', flexShrink: '0' });
    state.badge.appendChild(sep1);

    // Language display
    const langSpan = document.createElement('span');
    langSpan.setAttribute('data-badge-lang', '');
    langSpan.textContent = state.targetLang;
    langSpan.style.color = 'rgba(124, 58, 237, 0.9)';
    langSpan.style.fontWeight = '600';
    state.badge.appendChild(langSpan);

    const badgeBtnBase: Record<string, string> = {
      background: 'rgba(124, 58, 237, 0.15)',
      border: '1px solid rgba(124, 58, 237, 0.3)',
      borderRadius: '6px',
      color: 'rgba(226, 232, 240, 0.9)',
      padding: '3px 8px',
      cursor: 'pointer',
      fontSize: '11px',
      fontFamily: 'inherit',
      transition: 'background 0.15s, border-color 0.15s',
    };

    // Mode toggle button
    const modeBtn = document.createElement('button');
    modeBtn.setAttribute('data-badge-mode', '');
    modeBtn.textContent = 'Bilingual';
    Object.assign(modeBtn.style, badgeBtnBase);
    modeBtn.addEventListener('mouseenter', () => { modeBtn.style.background = 'rgba(124, 58, 237, 0.3)'; });
    modeBtn.addEventListener('mouseleave', () => { modeBtn.style.background = 'rgba(124, 58, 237, 0.15)'; });
    modeBtn.addEventListener('click', () => {
      setMode(state.mode === 'bilingual' ? 'translated' : 'bilingual');
    });
    state.badge.appendChild(modeBtn);

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.textContent = '\u2715';
    Object.assign(removeBtn.style, { ...badgeBtnBase, padding: '3px 6px', color: 'rgba(226, 232, 240, 0.6)' });
    removeBtn.title = 'Remove translation';
    removeBtn.addEventListener('mouseenter', () => {
      removeBtn.style.background = 'rgba(239, 68, 68, 0.2)';
      removeBtn.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      removeBtn.style.color = 'rgba(239, 68, 68, 0.9)';
    });
    removeBtn.addEventListener('mouseleave', () => {
      removeBtn.style.background = 'rgba(124, 58, 237, 0.15)';
      removeBtn.style.borderColor = 'rgba(124, 58, 237, 0.3)';
      removeBtn.style.color = 'rgba(226, 232, 240, 0.6)';
    });
    removeBtn.addEventListener('click', () => { remove(); });
    state.badge.appendChild(removeBtn);

    document.body.appendChild(state.badge);
  }

  async function start(targetLang: string): Promise<void> {
    state.targetLang = targetLang;
    state.active = true;
    state.mode = 'bilingual';
    state.elements = [];
    state.activeCount = 0;

    showTranslateBadge();

    const elements = getTranslatableElements();
    if (elements.length === 0) return;

    // Create translation placeholders for all elements
    const pairs: { original: HTMLElement; translation: HTMLDivElement; text: string }[] = [];
    for (const el of elements) {
      const text = (el.textContent || '').trim();
      if (!text) continue;
      const translationDiv = createTranslationElement(el);
      state.elements.push({ original: el, translation: translationDiv });
      pairs.push({ original: el, translation: translationDiv, text });
    }

    // Split into batches
    const batches: typeof pairs[] = [];
    for (let i = 0; i < pairs.length; i += BATCH_SIZE) {
      batches.push(pairs.slice(i, i + BATCH_SIZE));
    }

    const processBatch = async (batch: typeof pairs): Promise<void> => {
      while (state.activeCount >= MAX_CONCURRENT) {
        await new Promise(r => setTimeout(r, 100));
      }
      if (!state.active) return;

      state.activeCount++;
      try {
        const texts = batch.map(p => p.text);
        const translations = await translateBatchRequest(texts, state.targetLang);
        if (!state.active) return;

        batch.forEach((pair, idx) => {
          if (!state.active) return;
          fadeInTranslation(pair.translation, translations[idx] || '[No translation]');
          if (state.mode === 'translated') {
            pair.original.style.display = 'none';
          }
        });
      } finally {
        state.activeCount--;
      }
    };

    const promises = batches.map(batch => processBatch(batch));
    await Promise.all(promises);
  }

  return { start, remove, setMode };
}