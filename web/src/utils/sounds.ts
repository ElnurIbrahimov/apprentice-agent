let ctx: AudioContext | null = null;

function getCtx() {
  if (!ctx) {
    const AudioCtx = typeof window !== 'undefined'
      ? window.AudioContext || (window as any).webkitAudioContext
      : null;
    if (!AudioCtx) return null;
    ctx = new AudioCtx();
  }
  return ctx;
}

export function playTone(freq: number, duration: number, vol: number = 0.15) {
  try {
    const c = getCtx();
    if (!c) return;
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.value = vol;
    gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + duration / 1000);
    osc.connect(gain).connect(c.destination);
    osc.start();
    osc.stop(c.currentTime + duration / 1000);
  } catch {}
}

export const sounds = {
  send: () => playTone(880, 80, 0.1),
  receive: () => playTone(660, 150, 0.08),
  error: () => playTone(220, 200, 0.12),
};
