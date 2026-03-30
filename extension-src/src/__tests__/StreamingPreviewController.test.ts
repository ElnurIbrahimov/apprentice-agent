/**
 * Tests for StreamingPreviewController.
 */
import { StreamingPreviewController } from '../utils/StreamingPreviewController';

jest.useFakeTimers();

describe('StreamingPreviewController', () => {
  let renderFn: jest.Mock;
  let ctrl: StreamingPreviewController;

  beforeEach(() => {
    renderFn = jest.fn();
    ctrl = new StreamingPreviewController(renderFn, { debounceMs: 100, maxWaitMs: 500 });
  });

  afterEach(() => {
    ctrl.dispose();
    jest.clearAllTimers();
  });

  test('does not render immediately on append', () => {
    ctrl.append('<html>');
    expect(renderFn).not.toHaveBeenCalled();
  });

  test('renders after debounce at safe boundary', () => {
    ctrl.append('<html><body>Hello</body></html>');
    jest.advanceTimersByTime(100);
    expect(renderFn).toHaveBeenCalledWith('<html><body>Hello</body></html>');
  });

  test('does NOT render inside unclosed script tag', () => {
    ctrl.append('<html><script>var x = 1;');
    jest.advanceTimersByTime(100);
    // Unsafe — not rendered via debounce
    expect(renderFn).not.toHaveBeenCalled();
  });

  test('max-wait forces render even if unsafe', () => {
    ctrl.append('<html><script>var x = 1;');
    jest.advanceTimersByTime(100); // debounce fires, but unsafe
    expect(renderFn).not.toHaveBeenCalled();

    jest.advanceTimersByTime(500); // max-wait fires
    expect(renderFn).toHaveBeenCalledWith('<html><script>var x = 1;');
  });

  test('does NOT render inside unclosed style tag', () => {
    ctrl.append('<html><style>.foo { color: red;');
    jest.advanceTimersByTime(100);
    expect(renderFn).not.toHaveBeenCalled();
  });

  test('does NOT render inside unclosed HTML tag', () => {
    ctrl.append('<html><div class="foo');
    jest.advanceTimersByTime(100);
    expect(renderFn).not.toHaveBeenCalled();
  });

  test('renders after script tag is closed', () => {
    ctrl.append('<html><script>var x = 1;</script>');
    jest.advanceTimersByTime(100);
    expect(renderFn).toHaveBeenCalled();
  });

  test('flush forces immediate render', () => {
    ctrl.append('<html>partial');
    const result = ctrl.flush();
    expect(renderFn).toHaveBeenCalledWith('<html>partial');
    expect(result).toBe('<html>partial');
  });

  test('getBuffer returns accumulated content', () => {
    ctrl.append('chunk1');
    ctrl.append('chunk2');
    expect(ctrl.getBuffer()).toBe('chunk1chunk2');
  });

  test('reset clears buffer but preserves lastGoodHTML', () => {
    ctrl.append('<html>good</html>');
    jest.advanceTimersByTime(100);
    expect(renderFn).toHaveBeenCalled();

    ctrl.reset();
    expect(ctrl.getBuffer()).toBe('');
    expect(ctrl.getLastGoodHTML()).toBe('<html>good</html>');
  });

  test('dispose clears everything including lastGoodHTML', () => {
    ctrl.append('<html>good</html>');
    jest.advanceTimersByTime(100);

    ctrl.dispose();
    expect(ctrl.getBuffer()).toBe('');
    expect(ctrl.getLastGoodHTML()).toBe('');
  });

  test('lastGoodHTML not updated on render failure', () => {
    ctrl.append('<html>first</html>');
    jest.advanceTimersByTime(100);
    expect(ctrl.getLastGoodHTML()).toBe('<html>first</html>');

    // Make render throw
    renderFn.mockImplementationOnce(() => { throw new Error('render failed'); });
    ctrl.append('<html>second</html>');
    jest.advanceTimersByTime(100);

    // lastGoodHTML should still be 'first'
    expect(ctrl.getLastGoodHTML()).toBe('<html>first</html>');
  });

  test('onFirstChunk fires only once', () => {
    const onFirst = jest.fn();
    const c = new StreamingPreviewController(renderFn, { debounceMs: 100, onFirstChunk: onFirst });
    c.append('a');
    c.append('b');
    c.append('c');
    expect(onFirst).toHaveBeenCalledTimes(1);
    c.dispose();
  });

  test('accumulates multiple chunks before render', () => {
    ctrl.append('<ht');
    ctrl.append('ml>');
    ctrl.append('<body>Hi</body></html>');
    jest.advanceTimersByTime(100);
    expect(renderFn).toHaveBeenCalledWith('<html><body>Hi</body></html>');
  });

  test('handles deeply nested unclosed script', () => {
    ctrl.append('<html><div><div><script>alert("');
    jest.advanceTimersByTime(100);
    expect(renderFn).not.toHaveBeenCalled();
    // Max-wait forces render
    jest.advanceTimersByTime(500);
    expect(renderFn).toHaveBeenCalled();
  });

  test('handles empty string append', () => {
    ctrl.append('');
    jest.advanceTimersByTime(100);
    // Empty buffer — isSafeToRender returns true but nothing meaningful
    expect(renderFn).toHaveBeenCalledWith('');
  });

  test('multiple rapid appends use single debounce', () => {
    ctrl.append('a');
    ctrl.append('b');
    ctrl.append('c');
    jest.advanceTimersByTime(100);
    // Should render once with accumulated content
    expect(renderFn).toHaveBeenCalledTimes(1);
    expect(renderFn).toHaveBeenCalledWith('abc');
  });
});
