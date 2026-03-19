// Map PAD dimensions to CSS custom properties for emotion-aware UI theming.
// valence: -1 (sad/angry) to 1 (happy)
// arousal: -1 (calm) to 1 (excited)

export function getMoodCSSVars(valence: number, arousal: number): Record<string, string> {
  // Hue rotation: neutral=270(purple), happy=45(warm), sad=220(cool blue), angry=0(red), calm=180(teal)
  const hue = valence >= 0
    ? lerp(270, 45, valence * (0.5 + arousal * 0.5))  // positive: warm shift
    : lerp(270, arousal > 0 ? 0 : 220, -valence * 0.6); // negative: red if aroused, blue if calm

  const saturation = 0.3 + Math.abs(valence) * 0.4; // More emotional = more saturated
  const lightness = 0.5 + valence * 0.1; // Happier = slightly brighter

  return {
    '--mood-accent': `hsl(${hue}, ${saturation * 100}%, ${lightness * 100}%)`,
    '--mood-glow': `hsla(${hue}, ${saturation * 100}%, ${lightness * 100}%, 0.4)`,
    '--mood-bg-tint': `hsla(${hue}, ${saturation * 100}%, 15%, 0.15)`,
    '--mood-mesh-1': `hsla(${hue}, ${saturation * 100}%, 10%, 0.5)`,
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * Math.max(0, Math.min(1, t));
}
