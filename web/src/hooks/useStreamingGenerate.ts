import { useState, useRef, useCallback } from 'react';
import { apiFetch } from '../utils/apiFetch';

export interface GenerateOptions {
  model?: string;
  images?: string[];
}

export interface UseStreamingGenerateReturn {
  isGenerating: boolean;
  streamedText: string;
  setStreamedText: React.Dispatch<React.SetStateAction<string>>;
  generate: (message: string, systemPrompt: string, options?: GenerateOptions) => Promise<string>;
  stop: () => void;
}

export function useStreamingGenerate(): UseStreamingGenerateReturn {
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  // Abort any in-flight request on unmount
  // Panels that use this hook should call stop() themselves if needed,
  // but the ref is also available for cleanup via the returned stop fn.

  const generate = useCallback(
    async (
      message: string,
      systemPrompt: string,
      options: GenerateOptions = {},
    ): Promise<string> => {
      setStreamedText('');
      setIsGenerating(true);

      const controller = new AbortController();
      abortRef.current = controller;

      let fullResponse = '';

      try {
        const body: Record<string, unknown> = {
          message,
          system_prompt: systemPrompt,
        };
        if (options.model) body.model = options.model;
        if (options.images?.length) body.images = options.images;

        const res = await apiFetch('/api/generate/raw', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`API error: ${res.status}`);

        if (res.body) {
          const reader = res.body.getReader();
          const decoder = new TextDecoder();

          while (true) {
            if (controller.signal.aborted) break;
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });

            for (const line of chunk.split('\n')) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') continue;
                try {
                  const parsed = JSON.parse(data);
                  const text =
                    parsed.choices?.[0]?.delta?.content ||
                    parsed.content ||
                    parsed.chunk ||
                    '';
                  if (text) {
                    fullResponse += text;
                    setStreamedText(fullResponse);
                  }
                } catch {
                  fullResponse += data;
                  setStreamedText(fullResponse);
                }
              } else if (line.trim() && !line.startsWith(':')) {
                fullResponse += line;
                setStreamedText(fullResponse);
              }
            }
          }
        } else {
          fullResponse = await res.text();
          setStreamedText(fullResponse);
        }
      } finally {
        setIsGenerating(false);
        abortRef.current = null;
      }

      return fullResponse;
    },
    [],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  return { isGenerating, streamedText, setStreamedText, generate, stop };
}
