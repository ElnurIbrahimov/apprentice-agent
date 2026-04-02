/**
 * panelHandoff — cross-panel data transfer protocol.
 * Panels use this to send work to other panels with context.
 */

import type { PanelId } from '../types';

export interface PanelHandoff {
  from: PanelId;
  to: PanelId;
  action: 'edit' | 'render' | 'execute' | 'capture' | 'present';
  data: {
    code?: string;
    files?: Record<string, string>;
    context?: string;
    selectedElement?: { html: string; cssPath: string };
    language?: string;
  };
  timestamp: number;
}

/** In-memory singleton for pending handoffs (consumed once by the target panel). */
let _pending: PanelHandoff | null = null;

export function sendHandoff(handoff: Omit<PanelHandoff, 'timestamp'>): void {
  _pending = { ...handoff, timestamp: Date.now() };
}

export function consumeHandoff(targetPanel: PanelId): PanelHandoff | null {
  if (_pending && _pending.to === targetPanel) {
    const h = _pending;
    _pending = null;
    return h;
  }
  return null;
}

export function peekHandoff(): PanelHandoff | null {
  return _pending;
}

export function clearHandoff(): void {
  _pending = null;
}
