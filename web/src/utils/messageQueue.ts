/**
 * Offline outbox for chat messages.
 *
 * When the WebSocket is disconnected, outgoing messages are stashed here so
 * the user doesn't lose them. On reconnect, `useWebSocket` drains the queue
 * in-order and replays each send.
 *
 * Storage is plain localStorage for simplicity (small text; tens of messages
 * at most). Falls back to an in-memory array if localStorage is unavailable
 * (private mode).
 */

import type { FileAttachment } from '../types';

const STORAGE_KEY = 'aura-msg-queue-v1';

export interface QueuedMessage {
  id: string;
  message: string;
  attachments?: FileAttachment[];
  modelOverride?: string | null;
  actionMode?: string | null;
  conversationId?: string | null;
  enqueuedAt: number;
}

let memoryFallback: QueuedMessage[] = [];
let useMemory = false;

function read(): QueuedMessage[] {
  if (useMemory) return memoryFallback.slice();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    useMemory = true;
    return memoryFallback.slice();
  }
}

function write(queue: QueuedMessage[]): void {
  if (useMemory) { memoryFallback = queue.slice(); return; }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(queue));
  } catch {
    useMemory = true;
    memoryFallback = queue.slice();
  }
}

export function enqueueMessage(entry: Omit<QueuedMessage, 'id' | 'enqueuedAt'>): QueuedMessage {
  const queued: QueuedMessage = {
    ...entry,
    id: `q_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    enqueuedAt: Date.now(),
  };
  const q = read();
  q.push(queued);
  write(q);
  return queued;
}

export function listQueued(): QueuedMessage[] {
  return read();
}

export function removeQueued(id: string): void {
  write(read().filter((q) => q.id !== id));
}

export function clearQueued(): void {
  write([]);
}

export function queuedCount(): number {
  return read().length;
}
