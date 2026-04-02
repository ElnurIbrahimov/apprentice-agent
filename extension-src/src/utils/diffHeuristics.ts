import { presentableDiff } from '@codemirror/merge';

export interface DiffMetrics {
  changeRatio: number;
  changedCharCount: number;
  changedLineCount: number;
}

function countLinesInRange(text: string, from: number, to: number): number {
  if (to <= from) return 0;
  const slice = text.slice(from, to);
  return slice.split('\n').length;
}

export function summarizeDiff(original: string, modified: string): DiffMetrics {
  if (original === modified) {
    return {
      changeRatio: 0,
      changedCharCount: 0,
      changedLineCount: 0,
    };
  }

  const changes = presentableDiff(original, modified, { scanLimit: 500, timeout: 1000 });
  let changedCharCount = 0;
  let changedLineCount = 0;

  for (const change of changes) {
    const charsA = change.toA - change.fromA;
    const charsB = change.toB - change.fromB;
    changedCharCount += Math.max(charsA, charsB);

    const linesA = countLinesInRange(original, change.fromA, change.toA);
    const linesB = countLinesInRange(modified, change.fromB, change.toB);
    changedLineCount += Math.max(linesA, linesB, 1);
  }

  return {
    changeRatio: changedCharCount / Math.max(original.length, modified.length, 1),
    changedCharCount,
    changedLineCount,
  };
}
