/**
 * Tests for liquid-morph animation primitives.
 */
import { flow, dissolve, crossFade, sequentialReveal } from '../../content/animator';

function makeEl(height = 100): HTMLElement {
  const el = document.createElement('div');
  // jsdom getBoundingClientRect returns zeros, so we stub offsetHeight
  Object.defineProperty(el, 'offsetHeight', { value: height, configurable: true });
  return el;
}

describe('animator', () => {
  let animateMock: jest.Mock;

  beforeEach(() => {
    animateMock = jest.fn(() => ({ finished: Promise.resolve(), cancel: jest.fn() }));
    Element.prototype.animate = animateMock;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('flow() direction=down calls animate with 0→height keyframes', async () => {
    const el = makeEl(80);
    await flow(el, { duration: 300, easing: 'ease', direction: 'down' });

    expect(animateMock).toHaveBeenCalledTimes(1);
    const [keyframes] = animateMock.mock.calls[0];
    expect(keyframes[0]).toMatchObject({ height: '0px', opacity: 0 });
    expect(keyframes[1]).toMatchObject({ height: '80px', opacity: 1 });
  });

  test('flow() direction=up reverses keyframes (height→0)', async () => {
    const el = makeEl(80);
    await flow(el, { duration: 300, easing: 'ease', direction: 'up' });

    expect(animateMock).toHaveBeenCalledTimes(1);
    const [keyframes] = animateMock.mock.calls[0];
    expect(keyframes[0]).toMatchObject({ height: '80px', opacity: 1 });
    expect(keyframes[1]).toMatchObject({ height: '0px', opacity: 0 });
  });

  test('dissolve() animates opacity from 1 to 0', async () => {
    const el = makeEl();
    await dissolve(el, { duration: 200, easing: 'ease-out' });

    expect(animateMock).toHaveBeenCalledTimes(1);
    const [keyframes] = animateMock.mock.calls[0];
    expect(keyframes[0]).toMatchObject({ opacity: 1 });
    expect(keyframes[1]).toMatchObject({ opacity: 0 });
  });

  test('crossFade() calls animate on both old and new elements simultaneously', async () => {
    const oldEl = makeEl();
    const newEl = makeEl();
    await crossFade(oldEl, newEl, { duration: 250, easing: 'ease' });

    // Both elements must be animated
    expect(animateMock).toHaveBeenCalledTimes(2);

    // First call on oldEl: opacity 1→0
    const [oldKeyframes] = animateMock.mock.calls[0];
    expect(oldKeyframes[0]).toMatchObject({ opacity: 1 });
    expect(oldKeyframes[1]).toMatchObject({ opacity: 0 });

    // Second call on newEl: opacity 0→1
    const [newKeyframes] = animateMock.mock.calls[1];
    expect(newKeyframes[0]).toMatchObject({ opacity: 0 });
    expect(newKeyframes[1]).toMatchObject({ opacity: 1 });
  });

  test('sequentialReveal() staggers children with correct delays (0, 40, 80)', async () => {
    const parent = document.createElement('div');
    const children = [
      document.createElement('div'),
      document.createElement('div'),
      document.createElement('div'),
    ];
    children.forEach(c => parent.appendChild(c));

    await sequentialReveal(parent, { duration: 200, easing: 'ease', stagger: 40 });

    expect(animateMock).toHaveBeenCalledTimes(3);

    const delays = animateMock.mock.calls.map((call: any[]) => call[1]?.delay ?? 0);
    expect(delays).toEqual([0, 40, 80]);
  });
});
