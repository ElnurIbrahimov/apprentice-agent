/**
 * Split in-progress markdown into a safe-to-parse prefix and an unsafe
 * trailing slice. Prevents the "**bo → **bold**" flicker by only feeding
 * complete blocks to the full markdown renderer.
 *
 * Block boundaries we trust:
 *   - A double newline (paragraph break) that isn't inside an open code fence
 *   - A closed triple-backtick fence (even count of ``` markers)
 *   - Completed table/list/heading lines
 *
 * The trailing slice (if any) should be rendered as plain streaming text —
 * styled to match but without any markdown re-parsing.
 */

export function splitAtSafePoint(content: string): { safe: string; trailing: string } {
  if (!content) return { safe: '', trailing: '' };

  // Count opening/closing fence markers. If odd, we're inside a fence → the
  // boundary is the line BEFORE the last opened fence.
  const fenceMatches = [...content.matchAll(/^```/gm)];
  if (fenceMatches.length % 2 === 1) {
    // Inside an open fence. Safe prefix ends at the start of that fence line.
    const lastFence = fenceMatches[fenceMatches.length - 1];
    const fenceStart = lastFence.index ?? content.length;
    return {
      safe: content.slice(0, Math.max(0, fenceStart)).replace(/\n+$/, ''),
      trailing: content.slice(fenceStart),
    };
  }

  // All fences are closed. Find the last paragraph break.
  const lastDouble = content.lastIndexOf('\n\n');
  if (lastDouble < 0) {
    // No paragraph break yet — nothing is safe to parse. Render entirely as
    // trailing plain text while streaming.
    return { safe: '', trailing: content };
  }

  const safeEnd = lastDouble + 2;
  return {
    safe: content.slice(0, safeEnd),
    trailing: content.slice(safeEnd),
  };
}
