import { GLASS, FAB, GHOST_BAR, MODAL, ANIM, FONT_STACK, Z_TOP, Z_MID } from './tokens';

export function buildStylesheet(): string {
  return `
/* ── Host / Root ─────────────────────────────────────────────────────────── */
:host {
  --aura-accent: #7c3aed;
  --aura-glow: rgba(124, 58, 237, 0.3);
  all: initial;
  pointer-events: none;
  font-family: ${FONT_STACK};
}

/* ── FAB ──────────────────────────────────────────────────────────────────── */
.aura-fab {
  position: fixed;
  right: 0;
  bottom: 30px;
  z-index: ${Z_TOP};
  transform: translateX(100%);
  transition: transform ${ANIM.morphDuration}ms ${ANIM.morphEasing};
  pointer-events: none;
  will-change: transform, opacity;
}

.aura-fab.show {
  transform: translateX(0);
}

.aura-fab.left {
  left: 0;
  right: auto;
  transform: translateX(-100%);
}

.aura-fab.left.show {
  transform: translateX(0);
}

.fab-pill {
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: ${FAB.pillPadding};
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 20px;
  box-shadow: ${GLASS.shadowBase};
  cursor: pointer;
  pointer-events: auto;
  transition: padding ${FAB.expandDuration}ms ${ANIM.morphEasing},
              border-radius ${FAB.expandDuration}ms ${ANIM.morphEasing};
  will-change: transform, opacity;
  font-family: ${FONT_STACK};
  color: #fff;
  user-select: none;
}

.fab-pill.hover {
  padding: ${FAB.pillPadding};
}

.fab-pill.dragging {
  border-radius: 50%;
  cursor: move;
  padding: ${FAB.pillPadding};
}

.fab-glow {
  position: absolute;
  inset: -6px;
  border-radius: inherit;
  background: var(--aura-glow);
  filter: blur(10px);
  animation: aura-glow-pulse ${ANIM.glowPulse}ms ease-in-out infinite;
  pointer-events: none;
  z-index: -1;
  will-change: transform, opacity;
}

.fab-logo {
  width: ${FAB.logoSize}px;
  height: ${FAB.logoSize}px;
  pointer-events: none;
  flex-shrink: 0;
}

.fab-close {
  position: absolute;
  bottom: -18px;
  left: 50%;
  transform: translateX(-50%);
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s ease;
  pointer-events: auto;
  font-size: 9px;
  color: rgba(255, 255, 255, 0.7);
}

.fab-pill:hover .fab-close,
.aura-fab:hover .fab-close {
  opacity: 1;
}

.fab-popout {
  position: absolute;
  bottom: calc(100% + 8px);
  right: 0;
  min-width: 160px;
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 12px;
  box-shadow: ${GLASS.shadowBase};
  padding: 6px;
  opacity: 0;
  pointer-events: none;
  transform: translateY(4px);
  transition: opacity 0.2s ease, transform 0.2s ease;
  will-change: transform, opacity;
}

.fab-popout.visible {
  opacity: 1;
  pointer-events: auto;
  transform: translateY(0);
}

.aura-fab.left .fab-popout {
  right: auto;
  left: 0;
}

.fab-action-btn {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: transparent;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: rgba(255, 255, 255, 0.8);
  transition: background 0.15s ease, color 0.15s ease;
  pointer-events: auto;
  font-family: ${FONT_STACK};
}

.fab-action-btn:hover {
  background: var(--aura-accent);
  color: #fff;
}

/* ── Ghost Bar ────────────────────────────────────────────────────────────── */
.ghost-bar {
  position: fixed;
  z-index: ${Z_MID};
  display: flex;
  align-items: center;
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  box-shadow: ${GLASS.shadowBase};
  pointer-events: auto;
  will-change: transform, opacity;
  font-family: ${FONT_STACK};
  color: #fff;
}

.ghost-bar-text {
  height: ${GHOST_BAR.height}px;
  padding: 0 8px;
  border-radius: 0 0 8px 8px;
  display: flex;
  align-items: center;
  gap: 2px;
}

.ghost-bar-image {
  height: ${GHOST_BAR.imageBarHeight}px;
  padding: 0 10px;
  background: ${GLASS.bgHeavy};
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.gb-action {
  width: ${GHOST_BAR.iconSize}px;
  height: ${GHOST_BAR.iconSize}px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: rgba(255, 255, 255, 0.75);
  border-radius: 4px;
  transition: color 0.15s ease, background 0.15s ease;
  pointer-events: auto;
  background: transparent;
  border: none;
  padding: 0;
}

.gb-action:hover {
  color: var(--aura-accent);
  background: rgba(255, 255, 255, 0.08);
}

.gb-separator {
  width: 1px;
  height: 12px;
  background: rgba(255, 255, 255, 0.2);
  flex-shrink: 0;
  margin: 0 2px;
}

.gb-extended {
  display: none;
  align-items: center;
  gap: 2px;
  padding-top: 4px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 4px;
}

.gb-extended.visible {
  display: flex;
}

/* ── Modal ────────────────────────────────────────────────────────────────── */
.aura-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  z-index: ${Z_MID};
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: auto;
}

.aura-modal {
  max-width: ${MODAL.maxWidth}px;
  max-height: ${MODAL.maxHeight}px;
  width: 90vw;
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 16px;
  box-shadow: ${GLASS.shadowBase};
  overflow: hidden;
  display: flex;
  flex-direction: column;
  will-change: transform, opacity;
  font-family: ${FONT_STACK};
  color: #fff;
}

.modal-preview {
  padding: 12px 16px;
  max-height: calc(${MODAL.previewMaxLines} * 1.5em + 24px);
  overflow-y: auto;
  font-size: 13px;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.75);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-input {
  width: 100%;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  padding: 10px 12px;
  color: #fff;
  font-family: ${FONT_STACK};
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s ease;
  box-sizing: border-box;
}

.modal-input:focus {
  border-color: var(--aura-accent);
}

.modal-action-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 16px;
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: transparent;
  color: rgba(255, 255, 255, 0.85);
  font-family: ${FONT_STACK};
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.modal-action-btn:hover {
  background: var(--aura-accent);
  border-color: var(--aura-accent);
  color: #fff;
}

.modal-model-select {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  padding: 6px 10px;
  color: #fff;
  font-family: ${FONT_STACK};
  font-size: 13px;
  outline: none;
  cursor: pointer;
  transition: border-color 0.2s ease;
}

.modal-model-select:focus {
  border-color: var(--aura-accent);
}

/* ── Highlights ───────────────────────────────────────────────────────────── */
.hl-tooltip {
  position: fixed;
  z-index: ${Z_TOP};
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 8px;
  box-shadow: ${GLASS.shadowBase};
  padding: 6px 10px;
  font-family: ${FONT_STACK};
  font-size: 12px;
  color: rgba(255, 255, 255, 0.9);
  pointer-events: none;
  will-change: transform, opacity;
}

/* ── Toast ────────────────────────────────────────────────────────────────── */
.aura-toast {
  position: fixed;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%);
  z-index: ${Z_TOP};
  background: rgba(16, 185, 129, 0.9);
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 20px;
  box-shadow: ${GLASS.shadowBase};
  padding: 8px 18px;
  font-family: ${FONT_STACK};
  font-size: 13px;
  color: #fff;
  pointer-events: none;
  will-change: transform, opacity;
}

/* ── Keyframes ────────────────────────────────────────────────────────────── */
@keyframes aura-glow-pulse {
  0%   { opacity: ${FAB.glowIntensityMin}; }
  50%  { opacity: ${FAB.glowIntensityMax}; }
  100% { opacity: ${FAB.glowIntensityMin}; }
}
`.trim();
}
