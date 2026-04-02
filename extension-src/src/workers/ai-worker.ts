/**
 * On-Device AI Worker — Runs Transformers.js models in a Web Worker.
 *
 * Capabilities:
 *   - Text embeddings (all-MiniLM-L6-v2, 384-dim) for semantic search
 *   - Text classification / sentiment analysis
 *   - Summarization (small model for offline use)
 *   - Language detection
 *
 * Messages IN:
 *   { type: 'init' }                                          — Load models
 *   { type: 'embed', id: string, texts: string[] }            — Generate embeddings
 *   { type: 'classify', id: string, text: string, labels: string[] } — Zero-shot classify
 *   { type: 'summarize', id: string, text: string }           — Summarize text
 *   { type: 'detect_language', id: string, text: string }     — Detect language
 *   { type: 'similarity', id: string, a: string, b: string }  — Cosine similarity
 *   { type: 'semantic_search', id: string, query: string, corpus: string[], topK?: number }
 *
 * Messages OUT:
 *   { type: 'ready', models: string[] }
 *   { type: 'loading', stage: string, progress?: number }
 *   { type: 'result', id: string, data: any }
 *   { type: 'error', id: string, message: string }
 */

declare const self: Worker & typeof globalThis;

// Transformers.js loaded dynamically
let pipeline: any = null;
let env: any = null;

// Cached pipelines (lazy-loaded per task)
let embedder: any = null;
let classifier: any = null;
let summarizer: any = null;

const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const CLASSIFY_MODEL = 'Xenova/nli-deberta-v3-xsmall';
const SUMMARIZE_MODEL = 'Xenova/distilbart-cnn-6-6';

let isReady = false;
let loadedModels: string[] = [];

// ── Utilities ─────────────────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

// ── Model Loading ─────────────────────────────────────────────────────────

async function loadTransformers() {
  if (pipeline) return;
  self.postMessage({ type: 'loading', stage: 'Loading Transformers.js runtime...' });

  // Dynamic import from CDN (works in extension workers)
  const tf = await import(
    // @ts-ignore CDN import resolved at runtime
    /* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
  );

  pipeline = tf.pipeline;
  env = tf.env;

  // Use extension's local cache, disable remote model fetching warnings
  env.allowLocalModels = false;
  env.useBrowserCache = true;

  self.postMessage({ type: 'loading', stage: 'Runtime loaded' });
}

async function getEmbedder() {
  if (embedder) return embedder;
  await loadTransformers();
  self.postMessage({ type: 'loading', stage: 'Loading embedding model...', progress: 0 });
  embedder = await pipeline('feature-extraction', EMBED_MODEL, {
    progress_callback: (p: any) => {
      if (p.status === 'progress') {
        self.postMessage({ type: 'loading', stage: `Embedding model: ${Math.round(p.progress)}%`, progress: p.progress });
      }
    }
  });
  loadedModels.push('embeddings');
  self.postMessage({ type: 'loading', stage: 'Embedding model ready' });
  return embedder;
}

async function getClassifier() {
  if (classifier) return classifier;
  await loadTransformers();
  self.postMessage({ type: 'loading', stage: 'Loading classifier model...' });
  classifier = await pipeline('zero-shot-classification', CLASSIFY_MODEL, {
    progress_callback: (p: any) => {
      if (p.status === 'progress') {
        self.postMessage({ type: 'loading', stage: `Classifier: ${Math.round(p.progress)}%`, progress: p.progress });
      }
    }
  });
  loadedModels.push('classifier');
  return classifier;
}

async function getSummarizer() {
  if (summarizer) return summarizer;
  await loadTransformers();
  self.postMessage({ type: 'loading', stage: 'Loading summarization model...' });
  summarizer = await pipeline('summarization', SUMMARIZE_MODEL, {
    progress_callback: (p: any) => {
      if (p.status === 'progress') {
        self.postMessage({ type: 'loading', stage: `Summarizer: ${Math.round(p.progress)}%`, progress: p.progress });
      }
    }
  });
  loadedModels.push('summarizer');
  return summarizer;
}

// ── Message Handler ───────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent) => {
  const msg = event.data;

  try {
    switch (msg.type) {
      case 'init': {
        // Pre-load the embedding model (most common use case)
        await getEmbedder();
        isReady = true;
        self.postMessage({ type: 'ready', models: loadedModels });
        break;
      }

      case 'embed': {
        const model = await getEmbedder();
        const results = await model(msg.texts, { pooling: 'mean', normalize: true });
        // Convert to regular arrays
        const embeddings: number[][] = [];
        for (let i = 0; i < msg.texts.length; i++) {
          embeddings.push(Array.from(results[i].data));
        }
        self.postMessage({ type: 'result', id: msg.id, data: { embeddings } });
        break;
      }

      case 'classify': {
        const model = await getClassifier();
        const result = await model(msg.text, msg.labels);
        self.postMessage({
          type: 'result', id: msg.id,
          data: {
            labels: result.labels,
            scores: result.scores,
            topLabel: result.labels[0],
            topScore: result.scores[0],
          }
        });
        break;
      }

      case 'summarize': {
        const model = await getSummarizer();
        const text = msg.text.slice(0, 4096); // Model input limit
        const result = await model(text, {
          max_length: 150,
          min_length: 30,
          do_sample: false,
        });
        self.postMessage({
          type: 'result', id: msg.id,
          data: { summary: result[0].summary_text }
        });
        break;
      }

      case 'detect_language': {
        // Use embeddings + heuristic: embed the text and compare against
        // language-specific seed phrases
        const seeds: Record<string, string> = {
          en: 'This is an English text about technology and science',
          ru: 'Это русский текст о технологиях и науке',
          az: 'Bu Azərbaycan dilində texnologiya və elm haqqında mətndir',
          es: 'Este es un texto en español sobre tecnología y ciencia',
          fr: 'Ceci est un texte français sur la technologie et la science',
          de: 'Dies ist ein deutscher Text über Technologie und Wissenschaft',
          tr: 'Bu Türkçe bir teknoloji ve bilim metnidir',
          zh: '这是一篇关于技术和科学的中文文本',
          ja: 'これは技術と科学に関する日本語のテキストです',
          ar: 'هذا نص عربي عن التكنولوجيا والعلوم',
        };

        const model = await getEmbedder();
        const allTexts = [msg.text, ...Object.values(seeds)];
        const results = await model(allTexts, { pooling: 'mean', normalize: true });

        const queryEmb = Array.from(results[0].data) as number[];
        const langs = Object.keys(seeds);
        let bestLang = 'en';
        let bestScore = -1;

        for (let i = 0; i < langs.length; i++) {
          const seedEmb = Array.from(results[i + 1].data) as number[];
          const score = cosineSimilarity(queryEmb, seedEmb);
          if (score > bestScore) {
            bestScore = score;
            bestLang = langs[i];
          }
        }

        self.postMessage({
          type: 'result', id: msg.id,
          data: { language: bestLang, confidence: bestScore }
        });
        break;
      }

      case 'similarity': {
        const model = await getEmbedder();
        const results = await model([msg.a, msg.b], { pooling: 'mean', normalize: true });
        const embA = Array.from(results[0].data) as number[];
        const embB = Array.from(results[1].data) as number[];
        const score = cosineSimilarity(embA, embB);
        self.postMessage({ type: 'result', id: msg.id, data: { similarity: score } });
        break;
      }

      case 'semantic_search': {
        const model = await getEmbedder();
        const topK = msg.topK || 5;
        const allTexts = [msg.query, ...msg.corpus];
        const results = await model(allTexts, { pooling: 'mean', normalize: true });

        const queryEmb = Array.from(results[0].data) as number[];
        const scored = msg.corpus.map((text: string, i: number) => {
          const docEmb = Array.from(results[i + 1].data) as number[];
          return { text, index: i, score: cosineSimilarity(queryEmb, docEmb) };
        });

        scored.sort((a: any, b: any) => b.score - a.score);
        self.postMessage({
          type: 'result', id: msg.id,
          data: { results: scored.slice(0, topK) }
        });
        break;
      }

      default:
        self.postMessage({ type: 'error', id: msg.id, message: `Unknown message type: ${msg.type}` });
    }
  } catch (err: any) {
    self.postMessage({
      type: 'error',
      id: msg.id || 'unknown',
      message: err.message || String(err),
    });
  }
};

// Signal worker is alive
self.postMessage({ type: 'loading', stage: 'AI Worker initialized, awaiting commands' });
