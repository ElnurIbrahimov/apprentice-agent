/**
 * IndexedDB wrapper for Aura conversation storage.
 * Replaces chrome.storage.local for conversations — supports higher limits
 * and indexed queries (by folder, pinned status, timestamp).
 */
import type { ConversationMeta, Message } from '../types';

const DB_NAME = 'AuraDB';
const DB_VERSION = 1;
const STORE_CONVERSATIONS = 'conversations';
const STORE_MESSAGES = 'messages';

let _db: IDBDatabase | null = null;

export function isIndexedDBAvailable(): boolean {
  try {
    return typeof indexedDB !== 'undefined' && !!indexedDB;
  } catch {
    return false;
  }
}

export async function initDB(): Promise<IDBDatabase> {
  if (_db) return _db;
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(new Error(`IndexedDB open failed: ${req.error?.message}`));
    req.onsuccess = () => { _db = req.result; resolve(_db); };
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_CONVERSATIONS)) {
        const cs = db.createObjectStore(STORE_CONVERSATIONS, { keyPath: 'id' });
        cs.createIndex('timestamp', 'timestamp', { unique: false });
        cs.createIndex('folder', 'folder', { unique: false });
        cs.createIndex('pinned', 'pinned', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORE_MESSAGES)) {
        const ms = db.createObjectStore(STORE_MESSAGES, { autoIncrement: true });
        ms.createIndex('conversationId', 'conversationId', { unique: false });
      }
    };
  });
}

export async function saveConversation(meta: ConversationMeta, messages: Message[]): Promise<void> {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_CONVERSATIONS, STORE_MESSAGES], 'readwrite');
    tx.objectStore(STORE_CONVERSATIONS).put(meta);

    // Delete old messages for this conversation, then write new ones
    const msgStore = tx.objectStore(STORE_MESSAGES);
    const idx = msgStore.index('conversationId');
    const range = IDBKeyRange.only(meta.id);
    idx.openCursor(range).onsuccess = (e) => {
      const cursor = (e.target as IDBRequest<IDBCursorWithValue | null>).result;
      if (cursor) { cursor.delete(); cursor.continue(); }
    };
    for (const msg of messages) {
      msgStore.add({ ...msg, conversationId: meta.id });
    }
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(new Error(`Save failed: ${tx.error?.message}`));
  });
}

export async function saveConversationMeta(meta: ConversationMeta): Promise<void> {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_CONVERSATIONS], 'readwrite');
    tx.objectStore(STORE_CONVERSATIONS).put(meta);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(new Error(`Save meta failed: ${tx.error?.message}`));
  });
}

export async function loadConversation(id: string): Promise<{ meta: ConversationMeta; messages: Message[] } | null> {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_CONVERSATIONS, STORE_MESSAGES], 'readonly');
    let meta: ConversationMeta | null = null;
    const messages: Message[] = [];

    tx.objectStore(STORE_CONVERSATIONS).get(id).onsuccess = (e) => {
      meta = (e.target as IDBRequest).result || null;
    };
    tx.objectStore(STORE_MESSAGES).index('conversationId').getAll(id).onsuccess = (e) => {
      const rows = (e.target as IDBRequest).result as Array<Message & { conversationId: string }>;
      for (const { conversationId: _, ...msg } of rows) messages.push(msg);
      messages.sort((a, b) => a.timestamp - b.timestamp);
    };
    tx.oncomplete = () => resolve(meta ? { meta, messages } : null);
    tx.onerror = () => reject(new Error(`Load failed: ${tx.error?.message}`));
  });
}

export async function deleteConversation(id: string): Promise<void> {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_CONVERSATIONS, STORE_MESSAGES], 'readwrite');
    tx.objectStore(STORE_CONVERSATIONS).delete(id);
    const idx = tx.objectStore(STORE_MESSAGES).index('conversationId');
    idx.openCursor(IDBKeyRange.only(id)).onsuccess = (e) => {
      const cursor = (e.target as IDBRequest<IDBCursorWithValue | null>).result;
      if (cursor) { cursor.delete(); cursor.continue(); }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(new Error(`Delete failed: ${tx.error?.message}`));
  });
}

export async function getAllConversations(): Promise<ConversationMeta[]> {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_CONVERSATIONS], 'readonly');
    const req = tx.objectStore(STORE_CONVERSATIONS).getAll();
    req.onsuccess = () => {
      const convs = req.result as ConversationMeta[];
      // Sort: pinned first, then by timestamp desc
      convs.sort((a, b) => {
        if (a.pinned && !b.pinned) return -1;
        if (!a.pinned && b.pinned) return 1;
        return b.timestamp - a.timestamp;
      });
      resolve(convs);
    };
    req.onerror = () => reject(new Error(`GetAll failed: ${req.error?.message}`));
  });
}
