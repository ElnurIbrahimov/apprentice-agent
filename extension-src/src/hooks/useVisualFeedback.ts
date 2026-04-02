/**
 * useVisualFeedback — captures iframe screenshots and sends them
 * to a vision model for quality analysis and auto-refinement.
 */

import { useCallback, useRef, useState } from 'react';
import { HTTP, getAuthHeaders } from '../api';
import { captureScreenshot, stripDataUri } from '../utils/screenshotCapture';
import { useStore } from '../store';

// ── Types ──

export interface VisualIssue {
  type: 'layout' | 'color' | 'typography' | 'missing' | 'broken';
  description: string;
  severity: 'high' | 'medium' | 'low';
  suggestion: string;
}

export interface VisualFeedback {
  matches_intent: boolean;
  score: number;
  issues: VisualIssue[];
  overall: string;
}

export type FeedbackStatus = 'idle' | 'capturing' | 'analyzing' | 'refining' | 'done' | 'error' | 'unavailable';

interface UseVisualFeedbackResult {
  status: FeedbackStatus;
  feedback: VisualFeedback | null;
  error: string;
  /** Capture + analyze the current iframe */
  analyze: (iframe: HTMLIFrameElement, userPrompt: string) => Promise<VisualFeedback | null>;
  /** Run full auto-refinement loop: analyze → fix → re-analyze (max 2 rounds) */
  autoRefine: (
    iframe: HTMLIFrameElement,
    userPrompt: string,
    currentCode: string,
    onCodeUpdate: (newCode: string) => void,
  ) => Promise<VisualFeedback | null>;
  /** Build a refinement prompt from feedback issues */
  buildFixPrompt: (feedback: VisualFeedback, currentCode: string) => string;
  /** Reset state */
  reset: () => void;
}

const MAX_REFINE_ROUNDS = 2;
const MIN_ACCEPTABLE_SCORE = 8;

const VISUAL_ANALYSIS_PROMPT = `Analyze this web page screenshot and evaluate the visual quality.

USER'S ORIGINAL REQUEST: "{prompt}"

Evaluate:
1. Does the result match what the user asked for?
2. Are there layout problems (overlapping, misaligned, overflow)?
3. Are colors/contrast/readability good?
4. Is the design professional and polished?
5. Any missing elements the user likely expected?

Return ONLY a JSON object (no markdown, no explanation):
{
  "matches_intent": true,
  "score": 7,
  "issues": [
    { "type": "layout", "description": "...", "severity": "high", "suggestion": "..." }
  ],
  "overall": "brief one-line assessment"
}

Valid issue types: layout, color, typography, missing, broken
Valid severities: high, medium, low
Score: 1-10 (10 = perfect)`;

export function useVisualFeedback(): UseVisualFeedbackResult {
  const [status, setStatus] = useState<FeedbackStatus>('idle');
  const [feedback, setFeedback] = useState<VisualFeedback | null>(null);
  const [error, setError] = useState('');
  const { getModel } = useStore();
  const abortRef = useRef<AbortController | null>(null);

  const getVisionModel = useCallback((): string => {
    // Use configured model for visual feedback, or fall back to known vision models
    return getModel('vision') || 'moondream';
  }, [getModel]);

  const analyze = useCallback(async (
    iframe: HTMLIFrameElement,
    userPrompt: string,
  ): Promise<VisualFeedback | null> => {
    setStatus('capturing');
    setError('');

    try {
      // 1. Wait for render to settle
      await new Promise(r => setTimeout(r, 500));

      // 2. Capture screenshot
      const screenshot = await captureScreenshot(iframe);
      const base64 = stripDataUri(screenshot);

      setStatus('analyzing');

      // 3. Send to vision model
      if (abortRef.current) abortRef.current.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      const prompt = VISUAL_ANALYSIS_PROMPT.replace('{prompt}', userPrompt);
      const model = getVisionModel();

      const resp = await fetch(`${HTTP}/api/generate/raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          message: prompt,
          model,
          images: [base64],
        }),
        signal: ctrl.signal,
      });

      if (!resp.ok) {
        if (resp.status === 400 || resp.status === 404) {
          setStatus('unavailable');
          setError(`Vision model "${model}" not available. Install it via Ollama or configure a vision-capable model.`);
          return null;
        }
        throw new Error(`Analysis failed: HTTP ${resp.status}`);
      }

      // 4. Parse SSE stream
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '', text = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'chunk' && parsed.content) text += parsed.content;
          } catch { /* skip parse errors */ }
        }
      }

      // 5. Parse JSON response
      const result = parseVisualFeedback(text);
      setFeedback(result);
      setStatus('done');
      return result;
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setStatus('idle');
        return null;
      }
      setError(err.message || 'Analysis failed');
      setStatus('error');
      return null;
    }
  }, [getVisionModel]);

  const buildFixPrompt = useCallback((fb: VisualFeedback, currentCode: string): string => {
    const issueList = fb.issues
      .filter(i => i.severity === 'high' || i.severity === 'medium')
      .map(i => `- ${i.type.toUpperCase()}: ${i.description}. Fix: ${i.suggestion}`)
      .join('\n');

    if (!issueList) return '';

    return `The current page has these visual issues detected by automated analysis:

${issueList}

Current code:
\`\`\`html
${currentCode}
\`\`\`

Fix ALL the listed issues. Return the complete corrected code.
Focus on: proper spacing, color contrast, professional appearance.`;
  }, []);

  const autoRefine = useCallback(async (
    iframe: HTMLIFrameElement,
    userPrompt: string,
    currentCode: string,
    onCodeUpdate: (newCode: string) => void,
  ): Promise<VisualFeedback | null> => {
    let code = currentCode;
    let lastFeedback: VisualFeedback | null = null;

    for (let round = 0; round < MAX_REFINE_ROUNDS; round++) {
      // Analyze
      const fb = await analyze(iframe, userPrompt);
      if (!fb) return lastFeedback;
      lastFeedback = fb;

      // Good enough — stop
      if (fb.score >= MIN_ACCEPTABLE_SCORE) {
        return fb;
      }

      // Build fix prompt
      const fixPrompt = buildFixPrompt(fb, code);
      if (!fixPrompt) return fb;

      setStatus('refining');

      // Generate fixed code
      try {
        if (abortRef.current) abortRef.current.abort();
        const ctrl = new AbortController();
        abortRef.current = ctrl;

        const model = getModel('webcreator') || undefined;
        const resp = await fetch(`${HTTP}/api/generate/raw`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
          body: JSON.stringify({ message: fixPrompt, model }),
          signal: ctrl.signal,
        });

        if (!resp.ok) throw new Error(`Fix generation failed: ${resp.status}`);

        const reader = resp.body!.getReader();
        const decoder = new TextDecoder();
        let buf = '', text = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              if (parsed.type === 'chunk' && parsed.content) text += parsed.content;
            } catch { /* skip */ }
          }
        }

        // Extract code from response
        const newCode = extractCode(text);
        if (newCode && newCode !== code) {
          code = newCode;
          onCodeUpdate(newCode);
          // Wait for re-render
          await new Promise(r => setTimeout(r, 800));
        }
      } catch (err: any) {
        if (err.name === 'AbortError') return lastFeedback;
        setError(err.message);
        setStatus('error');
        return lastFeedback;
      }
    }

    return lastFeedback;
  }, [analyze, buildFixPrompt, getModel]);

  const reset = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    setStatus('idle');
    setFeedback(null);
    setError('');
  }, []);

  return {
    status,
    feedback,
    error,
    analyze,
    autoRefine,
    buildFixPrompt,
    reset,
  };
}

// ── Internal helpers ──

function parseVisualFeedback(text: string): VisualFeedback {
  // Try to extract JSON from the response
  const cleaned = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();

  // Find the JSON object
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start === -1 || end === -1) {
    return { matches_intent: true, score: 5, issues: [], overall: cleaned.slice(0, 200) };
  }

  try {
    const parsed = JSON.parse(cleaned.slice(start, end + 1));
    return {
      matches_intent: !!parsed.matches_intent,
      score: typeof parsed.score === 'number' ? Math.max(1, Math.min(10, parsed.score)) : 5,
      issues: Array.isArray(parsed.issues) ? parsed.issues.filter(isValidIssue) : [],
      overall: typeof parsed.overall === 'string' ? parsed.overall : '',
    };
  } catch {
    return { matches_intent: true, score: 5, issues: [], overall: 'Could not parse visual feedback' };
  }
}

function isValidIssue(item: any): item is VisualIssue {
  return item
    && typeof item.description === 'string'
    && ['layout', 'color', 'typography', 'missing', 'broken'].includes(item.type)
    && ['high', 'medium', 'low'].includes(item.severity);
}

function extractCode(text: string): string {
  // Try to extract HTML from fenced code blocks
  const fenceMatch = text.match(/```(?:html)?\s*([\s\S]*?)```/);
  if (fenceMatch) return fenceMatch[1].trim();

  // If the response starts with <!DOCTYPE or <html, treat the whole thing as code
  const trimmed = text.trim();
  if (trimmed.startsWith('<!DOCTYPE') || trimmed.startsWith('<html')) return trimmed;

  return trimmed;
}
