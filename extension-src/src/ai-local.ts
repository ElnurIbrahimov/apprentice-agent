/**
 * On-Device AI API — High-level interface to the AI Web Worker.
 *
 * Usage:
 *   import { localAI } from './ai-local';
 *
 *   // Embed text
 *   const embeddings = await localAI.embed(['hello world', 'foo bar']);
 *
 *   // Semantic search
 *   const results = await localAI.semanticSearch('query', ['doc1', 'doc2'], 3);
 *
 *   // Classify
 *   const cls = await localAI.classify('I love this!', ['positive', 'negative']);
 *
 *   // Summarize
 *   const summary = await localAI.summarize('Long text...');
 *
 *   // Language detection
 *   const lang = await localAI.detectLanguage('Salam dünya');
 *
 *   // Similarity
 *   const score = await localAI.similarity('hello', 'hi there');
 */

type PendingRequest = {
  resolve: (data: any) => void;
  reject: (err: Error) => void;
};

class LocalAI {
  private worker: Worker | null = null;
  private pending = new Map<string, PendingRequest>();
  private readyPromise: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private _isReady = false;
  private _idCounter = 0;
  private _loadingCallbacks: ((stage: string, progress?: number) => void)[] = [];

  /** Whether the AI worker is loaded and ready */
  get isReady() { return this._isReady; }

  /** Register a callback for loading progress updates */
  onLoading(cb: (stage: string, progress?: number) => void) {
    this._loadingCallbacks.push(cb);
    return () => {
      this._loadingCallbacks = this._loadingCallbacks.filter(c => c !== cb);
    };
  }

  /** Initialize the worker and pre-load the embedding model */
  async init(): Promise<void> {
    if (this._isReady) return;
    if (this.readyPromise) return this.readyPromise;

    this.readyPromise = new Promise<void>((resolve) => {
      this.readyResolve = resolve;
    });

    try {
      // Create worker from the built file
      this.worker = new Worker(
        chrome.runtime.getURL('ai-worker.js'),
        { type: 'classic' }
      );

      this.worker.onmessage = (event) => this._handleMessage(event.data);
      this.worker.onerror = (err) => {
        console.error('[LocalAI] Worker error:', err);
      };

      // Ask worker to init (pre-load embedding model)
      this.worker.postMessage({ type: 'init' });
    } catch (err) {
      console.error('[LocalAI] Failed to create worker:', err);
      this.readyPromise = null;
      throw err;
    }

    return this.readyPromise;
  }

  /** Lazy init — creates worker on first use if not already initialized */
  private async ensureReady(): Promise<void> {
    if (!this._isReady) await this.init();
  }

  private _handleMessage(msg: any) {
    switch (msg.type) {
      case 'ready':
        this._isReady = true;
        this.readyResolve?.();
        break;

      case 'loading':
        this._loadingCallbacks.forEach(cb => cb(msg.stage, msg.progress));
        break;

      case 'result': {
        const p = this.pending.get(msg.id);
        if (p) {
          p.resolve(msg.data);
          this.pending.delete(msg.id);
        }
        break;
      }

      case 'error': {
        const p = this.pending.get(msg.id);
        if (p) {
          p.reject(new Error(msg.message));
          this.pending.delete(msg.id);
        } else {
          console.error('[LocalAI] Worker error:', msg.message);
        }
        break;
      }
    }
  }

  private _nextId(): string {
    return `ai_${++this._idCounter}_${Date.now()}`;
  }

  private _send(msg: any, timeout = 60000): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = msg.id || this._nextId();
      msg.id = id;
      this.pending.set(id, { resolve, reject });

      // Timeout safety
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`LocalAI request timed out after ${timeout}ms`));
        }
      }, timeout);

      // Clear timeout on completion
      const origResolve = resolve;
      const origReject = reject;
      this.pending.set(id, {
        resolve: (data) => { clearTimeout(timer); origResolve(data); },
        reject: (err) => { clearTimeout(timer); origReject(err); },
      });

      this.worker!.postMessage(msg);
    });
  }

  // ── Public API ────────────────────────────────────────────────────────

  /**
   * Generate embeddings for one or more texts.
   * Returns array of 384-dimensional vectors (all-MiniLM-L6-v2).
   */
  async embed(texts: string[]): Promise<number[][]> {
    await this.ensureReady();
    const { embeddings } = await this._send({ type: 'embed', texts });
    return embeddings;
  }

  /**
   * Zero-shot text classification.
   * Returns { labels, scores, topLabel, topScore }.
   */
  async classify(text: string, labels: string[]): Promise<{
    labels: string[];
    scores: number[];
    topLabel: string;
    topScore: number;
  }> {
    await this.ensureReady();
    return this._send({ type: 'classify', text, labels });
  }

  /**
   * Summarize text (up to ~4096 tokens input).
   * Returns summary string.
   */
  async summarize(text: string): Promise<string> {
    await this.ensureReady();
    const { summary } = await this._send({ type: 'summarize', text }, 120000);
    return summary;
  }

  /**
   * Detect the language of input text.
   * Returns { language: 'en'|'ru'|'az'|..., confidence: number }.
   */
  async detectLanguage(text: string): Promise<{ language: string; confidence: number }> {
    await this.ensureReady();
    return this._send({ type: 'detect_language', text });
  }

  /**
   * Compute cosine similarity between two texts.
   * Returns number in [-1, 1].
   */
  async similarity(a: string, b: string): Promise<number> {
    await this.ensureReady();
    const { similarity } = await this._send({ type: 'similarity', a, b });
    return similarity;
  }

  /**
   * Semantic search: find the most similar texts in a corpus to the query.
   * Returns ranked array of { text, index, score }.
   */
  async semanticSearch(query: string, corpus: string[], topK = 5): Promise<{
    text: string;
    index: number;
    score: number;
  }[]> {
    await this.ensureReady();
    const { results } = await this._send({ type: 'semantic_search', query, corpus, topK });
    return results;
  }

  /** Terminate the worker and free resources */
  destroy() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this._isReady = false;
    this.readyPromise = null;
    this.pending.clear();
  }
}

/** Singleton instance — shared across all panels */
export const localAI = new LocalAI();
export default localAI;
