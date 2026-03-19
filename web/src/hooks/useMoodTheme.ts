import { useEffect } from 'react';
import { useChatStore } from '../store/chatStore';
import { getMoodCSSVars } from '../utils/moodTheme';

export function useMoodTheme() {
  const valence = useChatStore(s => s.mood?.valence);
  const arousal = useChatStore(s => s.mood?.arousal);

  useEffect(() => {
    if (valence === undefined || valence === null || arousal === undefined || arousal === null) return;
    const vars = getMoodCSSVars(valence, arousal);
    const root = document.documentElement;
    for (const [key, value] of Object.entries(vars)) {
      root.style.setProperty(key, value);
    }
    return () => {
      for (const key of Object.keys(vars)) {
        root.style.removeProperty(key);
      }
    };
  }, [valence, arousal]);
}
