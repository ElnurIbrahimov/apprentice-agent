import { useState, useRef, useEffect } from 'react';

export interface UseModelSelectorReturn {
  selectedModel: string | null;
  setSelectedModel: (model: string | null) => void;
  availableModels: string[];
  showModelMenu: boolean;
  setShowModelMenu: React.Dispatch<React.SetStateAction<boolean>>;
  modelMenuRef: React.RefObject<HTMLDivElement>;
}

export function useModelSelector(): UseModelSelectorReturn {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  // Fetch models on mount
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all = [
          ...(data.chatgpt_models || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models || []),
          ...(data.local_models || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, []);

  // Close menu on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return { selectedModel, setSelectedModel, availableModels, showModelMenu, setShowModelMenu, modelMenuRef };
}
