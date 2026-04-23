/**
 * Rough token estimation for the in-composer context-budget meter.
 *
 * chars/4 is the standard back-of-envelope heuristic for English-biased
 * text with a BPE tokenizer. Good enough for a progress bar; anything
 * more precise would need to load a real tokenizer (100s of KB).
 *
 * Returns tokens as a plain number so callers can compare against a
 * model-specific context window and render the right colour band.
 */

import type { Message } from '../types';

/** Crude chars/4 estimator with a 4-token-per-message overhead for role framing. */
export function estimateTokens(messages: Pick<Message, 'content' | 'toolTrace'>[]): number {
  let total = 0;
  for (const m of messages) {
    total += Math.ceil((m.content?.length ?? 0) / 4) + 4;
    // Tool trace details can be long — count them too.
    if (m.toolTrace) {
      for (const t of m.toolTrace) {
        total += Math.ceil((t.detail?.length ?? 0) / 4);
      }
    }
  }
  return total;
}

/**
 * Context window in tokens. Default 128k (covers most modern models).
 * Could be wired to the selected-model registry later; kept simple for now.
 */
export const DEFAULT_CONTEXT_WINDOW = 128_000;

/** Colour band thresholds for the UI meter. */
export type UsageBand = 'safe' | 'warn' | 'danger';

export function usageBand(ratio: number): UsageBand {
  if (ratio >= 0.9) return 'danger';
  if (ratio >= 0.75) return 'warn';
  return 'safe';
}

/** Whether to even show the meter. Hidden below this ratio to reduce UI noise. */
export const METER_VISIBILITY_THRESHOLD = 0.5;
