import type { ContentModule, ContextStore } from './types';
import { morph, fadeIn, dissolve } from './animator';
import { MODAL, ANIM, GLASS, FONT_STACK, Z_TOP } from './tokens';

// ── Public API ────────────────────────────────────────────────────────────────

export interface ModalAPI {
  openWithText(text: string, originRect: DOMRect): void;
  openWithImage(imageUrl: string, originRect: DOMRect): void;
  close(): Promise<void>;
  onAction(cb: (action: string, text: string, model: string) => void): void;
}

// ── Internal state ────────────────────────────────────────────────────────────

interface ModalInternals {
  overlay: HTMLElement | null;
  modal: HTMLElement | null;
  originRect: DOMRect | null;
  content: string;        // text or imageUrl
  isOpen: boolean;
  closing: boolean;
  opening: boolean;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function truncateText(text: string): string {
  if (text.length <= MODAL.previewMaxChars) return text;
  const remaining = text.length - MODAL.previewMaxChars;
  return text.slice(0, MODAL.previewMaxChars) + `... (${remaining} more chars)`;
}

function getPlaceholder(storeType: string): string {
  switch (storeType) {
    case 'article': return 'Ask about this article...';
    case 'code':    return 'Ask about this code...';
    default:        return 'Ask anything about this text...';
  }
}

function centeredRect(): DOMRect {
  const w = Math.min(MODAL.maxWidth, window.innerWidth - 32);
  const h = Math.min(MODAL.maxHeight, window.innerHeight - 32);
  const left = (window.innerWidth - w) / 2;
  const top = (window.innerHeight - h) / 2;
  return {
    left, top,
    right: left + w,
    bottom: top + h,
    width: w,
    height: h,
    x: left, y: top,
    toJSON: () => ({}),
  } as DOMRect;
}

function applyGlassStyle(el: HTMLElement, rect: DOMRect) {
  Object.assign(el.style, {
    position: 'fixed',
    left: '0',
    top: '0',
    width: `${rect.width}px`,
    height: `${rect.height}px`,
    transform: `translate(${rect.left}px, ${rect.top}px)`,
    background: GLASS.bg,
    backdropFilter: GLASS.backdrop,
    WebkitBackdropFilter: GLASS.backdrop,
    border: `1px solid rgba(255,255,255,${GLASS.borderOpacity})`,
    borderRadius: '16px',
    boxShadow: GLASS.shadowBase,
    fontFamily: FONT_STACK,
    color: '#e5e7eb',
    overflow: 'hidden',
    zIndex: String(Z_TOP),
    boxSizing: 'border-box',
  } as unknown as CSSStyleDeclaration);
}

// ── Build modal content DOM ───────────────────────────────────────────────────

function buildTextContent(text: string, placeholder: string): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'modal-content-wrap';
  Object.assign(wrap.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '16px',
    height: '100%',
    boxSizing: 'border-box',
    opacity: '0',
  });

  // Preview
  const preview = document.createElement('div');
  preview.className = 'modal-preview';
  Object.assign(preview.style, {
    fontSize: '13px',
    lineHeight: '1.5',
    color: 'rgba(229,231,235,0.75)',
    overflow: 'hidden',
    display: '-webkit-box',
    WebkitLineClamp: String(MODAL.previewMaxLines),
    WebkitBoxOrient: 'vertical',
    maxHeight: `${MODAL.previewMaxLines * 20}px`,
    flexShrink: '0',
  } as unknown as CSSStyleDeclaration);
  preview.textContent = truncateText(text);

  // Input row
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'modal-input';
  input.placeholder = placeholder;
  Object.assign(input.style, {
    background: 'rgba(255,255,255,0.07)',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: '8px',
    padding: '8px 12px',
    color: '#e5e7eb',
    fontSize: '14px',
    fontFamily: FONT_STACK,
    outline: 'none',
    flexShrink: '0',
  });

  // Action buttons
  const actions = document.createElement('div');
  actions.className = 'modal-actions';
  Object.assign(actions.style, {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
    flexShrink: '0',
  });

  const actionDefs = [
    { label: 'Explain',         value: 'explain' },
    { label: 'Summarize',       value: 'summarize' },
    { label: 'Chat with AURA',  value: 'chat' },
    { label: 'Save to Memory',  value: 'save' },
    { label: 'Translate',       value: 'translate' },
  ];

  for (const def of actionDefs) {
    const btn = document.createElement('button');
    btn.className = 'modal-action-btn';
    btn.textContent = def.label;
    btn.dataset.action = def.value;
    Object.assign(btn.style, {
      background: 'rgba(255,255,255,0.08)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: '6px',
      padding: '5px 10px',
      color: '#e5e7eb',
      fontSize: '12px',
      fontFamily: FONT_STACK,
      cursor: 'pointer',
    });
    actions.appendChild(btn);
  }

  // Model row
  const modelRow = document.createElement('div');
  modelRow.className = 'modal-model-row';
  Object.assign(modelRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: 'auto',
    flexShrink: '0',
  });

  const modelLabel = document.createElement('span');
  modelLabel.textContent = 'Model';
  Object.assign(modelLabel.style, { fontSize: '12px', color: 'rgba(229,231,235,0.5)' });

  const select = document.createElement('select');
  select.className = 'modal-model-select';
  Object.assign(select.style, {
    background: 'rgba(255,255,255,0.07)',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: '6px',
    padding: '4px 8px',
    color: '#e5e7eb',
    fontSize: '12px',
    fontFamily: FONT_STACK,
    cursor: 'pointer',
  });

  const modelOptions = [
    { label: 'Auto',      value: 'auto' },
    { label: 'Fast',      value: 'fast' },
    { label: 'Balanced',  value: 'balanced' },
    { label: 'Powerful',  value: 'powerful' },
  ];
  for (const opt of modelOptions) {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    select.appendChild(option);
  }

  modelRow.appendChild(modelLabel);
  modelRow.appendChild(select);

  wrap.appendChild(preview);
  wrap.appendChild(input);
  wrap.appendChild(actions);
  wrap.appendChild(modelRow);

  return wrap;
}

function buildImageContent(imageUrl: string): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'modal-content-wrap';
  Object.assign(wrap.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '16px',
    height: '100%',
    boxSizing: 'border-box',
    opacity: '0',
  });

  // Image preview
  const preview = document.createElement('div');
  preview.className = 'modal-preview';
  Object.assign(preview.style, {
    flexShrink: '0',
    overflow: 'hidden',
    borderRadius: '8px',
  });

  const img = document.createElement('img');
  img.src = imageUrl;
  Object.assign(img.style, {
    maxWidth: '100%',
    maxHeight: `${MODAL.imagePreviewMaxHeight}px`,
    objectFit: 'contain',
    display: 'block',
  });
  preview.appendChild(img);

  // Action buttons (image-specific)
  const actions = document.createElement('div');
  actions.className = 'modal-actions';
  Object.assign(actions.style, {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
    flexShrink: '0',
  });

  const actionDefs = [
    { label: 'Describe',        value: 'describe' },
    { label: 'Summarize',       value: 'summarize' },
    { label: 'Chat with AURA',  value: 'chat' },
    { label: 'Save to Memory',  value: 'save' },
    { label: 'Translate',       value: 'translate' },
  ];

  for (const def of actionDefs) {
    const btn = document.createElement('button');
    btn.className = 'modal-action-btn';
    btn.textContent = def.label;
    btn.dataset.action = def.value;
    Object.assign(btn.style, {
      background: 'rgba(255,255,255,0.08)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: '6px',
      padding: '5px 10px',
      color: '#e5e7eb',
      fontSize: '12px',
      fontFamily: FONT_STACK,
      cursor: 'pointer',
    });
    actions.appendChild(btn);
  }

  // Input row
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'modal-input';
  input.placeholder = 'Ask about this image...';
  Object.assign(input.style, {
    background: 'rgba(255,255,255,0.07)',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: '8px',
    padding: '8px 12px',
    color: '#e5e7eb',
    fontSize: '14px',
    fontFamily: FONT_STACK,
    outline: 'none',
    flexShrink: '0',
  });

  // Model row
  const modelRow = document.createElement('div');
  modelRow.className = 'modal-model-row';
  Object.assign(modelRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: 'auto',
    flexShrink: '0',
  });

  const modelLabel = document.createElement('span');
  modelLabel.textContent = 'Model';
  Object.assign(modelLabel.style, { fontSize: '12px', color: 'rgba(229,231,235,0.5)' });

  const select = document.createElement('select');
  select.className = 'modal-model-select';
  Object.assign(select.style, {
    background: 'rgba(255,255,255,0.07)',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: '6px',
    padding: '4px 8px',
    color: '#e5e7eb',
    fontSize: '12px',
    fontFamily: FONT_STACK,
    cursor: 'pointer',
  });

  for (const opt of [
    { label: 'Auto', value: 'auto' },
    { label: 'Fast', value: 'fast' },
    { label: 'Balanced', value: 'balanced' },
    { label: 'Powerful', value: 'powerful' },
  ]) {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    select.appendChild(option);
  }

  modelRow.appendChild(modelLabel);
  modelRow.appendChild(select);

  wrap.appendChild(preview);
  wrap.appendChild(actions);
  wrap.appendChild(input);
  wrap.appendChild(modelRow);

  return wrap;
}

// ── createModal ───────────────────────────────────────────────────────────────

export function createModal(): ContentModule & ModalAPI {
  let internals: ModalInternals = {
    overlay: null,
    modal: null,
    originRect: null,
    content: '',
    isOpen: false,
    closing: false,
    opening: false,
  };

  let store_: ContextStore | null = null;
  let actionCallback: ((action: string, text: string, model: string) => void) | null = null;

  // ── Keydown handler (close on Escape) ──
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && internals.isOpen) {
      close();
    }
  };

  // ── open() — shared setup ──
  async function open(contentEl: HTMLElement, originRect: DOMRect, content: string): Promise<void> {
    if (internals.opening) return; // prevent concurrent opens
    if (internals.isOpen) await close();

    internals.opening = true;
    internals.originRect = originRect;
    internals.content = content; // set AFTER close() so it's not reset
    internals.isOpen = true;

    // 1. Overlay
    const overlay = document.createElement('div');
    overlay.className = 'aura-modal-overlay';
    Object.assign(overlay.style, {
      position: 'fixed',
      inset: '0',
      background: 'rgba(0,0,0,0.3)',
      zIndex: String(Z_TOP - 1),
      opacity: '0',
    });
    document.body.appendChild(overlay);
    internals.overlay = overlay;

    // 2. Modal shell — start at originRect
    const modal = document.createElement('div');
    modal.className = 'aura-modal';
    applyGlassStyle(modal, originRect);
    document.body.appendChild(modal);
    internals.modal = modal;

    // 3. Fade in overlay
    fadeIn(overlay, {
      duration: ANIM.flowDuration,
      easing: 'ease-out',
    }).then(() => {
      overlay.style.opacity = '1';
    });

    // 4. Morph modal from originRect → centered
    const target = centeredRect();
    await morph(modal, originRect, target, {
      duration: ANIM.morphDuration,
      easing: ANIM.morphEasing,
    });

    // Apply final position permanently
    applyGlassStyle(modal, target);

    // 5. Inject content
    modal.appendChild(contentEl);

    // 6. Fade in content
    fadeIn(contentEl, {
      duration: ANIM.crossFadeDuration,
      easing: 'ease-out',
    }).then(() => {
      contentEl.style.opacity = '1';
    });

    // 7. Wire action buttons
    modal.querySelectorAll<HTMLElement>('.modal-action-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action ?? 'ask';
        const select = modal.querySelector<HTMLSelectElement>('.modal-model-select');
        const model = select?.value ?? 'auto';
        actionCallback?.(action, internals.content, model);
      });
    });

    // 8. Wire input Enter key
    const inputEl = modal.querySelector<HTMLInputElement>('.modal-input');
    if (inputEl) {
      inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const select = modal.querySelector<HTMLSelectElement>('.modal-model-select');
          const model = select?.value ?? 'auto';
          actionCallback?.('ask', inputEl.value, model);
        }
      });
    }

    // 9. Overlay click → close
    overlay.addEventListener('click', () => close());

    // 10. Escape key
    document.addEventListener('keydown', onKeyDown);

    internals.opening = false;
  }

  // ── close() ──
  async function close(): Promise<void> {
    if (!internals.isOpen || internals.closing) return;
    internals.closing = true;

    const { modal, overlay, originRect } = internals;

    // Remove keydown listener immediately
    document.removeEventListener('keydown', onKeyDown);

    // Reverse morph + fade overlay in parallel
    const animations: Promise<void>[] = [];
    if (modal && originRect) {
      const currentRect = centeredRect();
      animations.push(
        morph(modal, currentRect, originRect, {
          duration: ANIM.morphDuration,
          easing: ANIM.morphEasing,
        }).catch(() => {})
      );
    }
    if (overlay) {
      animations.push(
        dissolve(overlay, {
          duration: ANIM.morphDuration,
          easing: 'ease-in',
        }).catch(() => {})
      );
    }
    await Promise.all(animations);

    // Remove all DOM
    modal?.remove();
    overlay?.remove();

    internals = {
      overlay: null,
      modal: null,
      originRect: null,
      content: '',
      isOpen: false,
      closing: false,
      opening: false,
    };
  }

  return {
    // ── ContentModule ──
    init(_container: HTMLElement, store: ContextStore, _ext: typeof chrome): void {
      store_ = store;
    },

    destroy(): void {
      close();
    },

    // ── ModalAPI ──
    openWithText(text: string, originRect: DOMRect): void {
      const placeholder = getPlaceholder(store_?.get().type ?? 'general');
      const contentEl = buildTextContent(text, placeholder);
      open(contentEl, originRect, text);
    },

    openWithImage(imageUrl: string, originRect: DOMRect): void {
      const contentEl = buildImageContent(imageUrl);
      open(contentEl, originRect, imageUrl);
    },

    close,

    onAction(cb: (action: string, text: string, model: string) => void): void {
      actionCallback = cb;
    },
  };
}
