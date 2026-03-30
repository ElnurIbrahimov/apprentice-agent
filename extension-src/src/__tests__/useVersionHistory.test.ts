/**
 * Tests for useVersionHistory hook.
 */
import { renderHook, act } from '@testing-library/react';
import { useVersionHistory } from '../utils/useVersionHistory';

describe('useVersionHistory', () => {
  test('starts with empty versions', () => {
    const { result } = renderHook(() => useVersionHistory());
    expect(result.current.versions).toEqual([]);
    expect(result.current.currentIdx).toBe(-1);
    expect(result.current.currentVersion).toBeNull();
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
  });

  test('pushVersion adds a version and sets currentIdx', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => {
      result.current.pushVersion('prompt1', 'code1');
    });

    expect(result.current.versions).toHaveLength(1);
    expect(result.current.versions[0].prompt).toBe('prompt1');
    expect(result.current.versions[0].code).toBe('code1');
    expect(result.current.currentIdx).toBe(0);
    expect(result.current.currentVersion?.code).toBe('code1');
  });

  test('pushVersion respects max cap', () => {
    const { result } = renderHook(() => useVersionHistory(3));

    act(() => {
      result.current.pushVersion('p1', 'c1');
      result.current.pushVersion('p2', 'c2');
      result.current.pushVersion('p3', 'c3');
      result.current.pushVersion('p4', 'c4');
    });

    expect(result.current.versions).toHaveLength(3);
    expect(result.current.versions[0].prompt).toBe('p2'); // p1 was evicted
    expect(result.current.versions[2].prompt).toBe('p4');
  });

  test('undo moves back one version', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => {
      result.current.pushVersion('p1', 'c1');
      result.current.pushVersion('p2', 'c2');
    });

    expect(result.current.currentIdx).toBe(1);
    expect(result.current.canUndo).toBe(true);

    let undone: any;
    act(() => { undone = result.current.undo(); });

    expect(result.current.currentIdx).toBe(0);
    expect(undone?.code).toBe('c1');
  });

  test('redo moves forward one version', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => {
      result.current.pushVersion('p1', 'c1');
      result.current.pushVersion('p2', 'c2');
    });

    act(() => { result.current.undo(); });
    expect(result.current.canRedo).toBe(true);

    let redone: any;
    act(() => { redone = result.current.redo(); });

    expect(result.current.currentIdx).toBe(1);
    expect(redone?.code).toBe('c2');
  });

  test('undo at start returns null', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => { result.current.pushVersion('p1', 'c1'); });

    let undone: any;
    act(() => { undone = result.current.undo(); });
    expect(undone).toBeNull();
    expect(result.current.currentIdx).toBe(0);
  });

  test('redo at end returns null', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => { result.current.pushVersion('p1', 'c1'); });

    let redone: any;
    act(() => { redone = result.current.redo(); });
    expect(redone).toBeNull();
  });

  test('goToVersion jumps to specific index', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => {
      result.current.pushVersion('p1', 'c1');
      result.current.pushVersion('p2', 'c2');
      result.current.pushVersion('p3', 'c3');
    });

    let v: any;
    act(() => { v = result.current.goToVersion(0); });
    expect(result.current.currentIdx).toBe(0);
    expect(v?.code).toBe('c1');
  });

  test('goToVersion with invalid index returns null', () => {
    const { result } = renderHook(() => useVersionHistory());

    let v: any;
    act(() => { v = result.current.goToVersion(99); });
    expect(v).toBeNull();
  });

  test('clear resets everything', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => {
      result.current.pushVersion('p1', 'c1');
      result.current.pushVersion('p2', 'c2');
    });

    act(() => { result.current.clear(); });

    expect(result.current.versions).toEqual([]);
    expect(result.current.currentIdx).toBe(-1);
    expect(result.current.currentVersion).toBeNull();
  });

  test('versions have unique IDs', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => {
      result.current.pushVersion('p1', 'c1');
      result.current.pushVersion('p2', 'c2');
    });

    const ids = result.current.versions.map(v => v.id);
    expect(new Set(ids).size).toBe(2);
  });

  test('versions have timestamps', () => {
    const { result } = renderHook(() => useVersionHistory());

    const before = Date.now();
    act(() => { result.current.pushVersion('p1', 'c1'); });
    const after = Date.now();

    expect(result.current.versions[0].timestamp).toBeGreaterThanOrEqual(before);
    expect(result.current.versions[0].timestamp).toBeLessThanOrEqual(after);
  });

  test('undo after push keeps canUndo/canRedo consistent', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => {
      result.current.pushVersion('p1', 'c1');
      result.current.pushVersion('p2', 'c2');
      result.current.pushVersion('p3', 'c3');
    });

    act(() => { result.current.undo(); }); // at p2
    act(() => { result.current.undo(); }); // at p1

    expect(result.current.currentIdx).toBe(0);
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(true);
  });

  test('pushVersion with label stores it', () => {
    const { result } = renderHook(() => useVersionHistory());

    act(() => {
      result.current.pushVersion('prompt', 'code', 'v1.0');
    });

    expect(result.current.versions[0].label).toBe('v1.0');
  });

  test('rapid sequential pushes maintain correct index', () => {
    const { result } = renderHook(() => useVersionHistory(5));

    act(() => {
      for (let i = 0; i < 10; i++) {
        result.current.pushVersion(`p${i}`, `c${i}`);
      }
    });

    // Capped at 5, index at last
    expect(result.current.versions.length).toBe(5);
    expect(result.current.versions[4].prompt).toBe('p9');
  });
});
