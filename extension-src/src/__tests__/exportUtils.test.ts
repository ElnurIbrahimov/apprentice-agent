/**
 * Tests for exportUtils.
 */
import { formatChatExport } from '../utils/exportUtils';

describe('formatChatExport', () => {
  test('formats messages with text field', () => {
    const result = formatChatExport([
      { role: 'user', text: 'Hello', timestamp: 1700000000000 },
      { role: 'ai', text: 'Hi there', timestamp: 1700000001000 },
    ]);

    expect(result).toEqual([
      { role: 'user', text: 'Hello', time: '2023-11-14T22:13:20.000Z' },
      { role: 'ai', text: 'Hi there', time: '2023-11-14T22:13:21.000Z' },
    ]);
  });

  test('uses content field as fallback', () => {
    const result = formatChatExport([
      { role: 'user', content: 'Hello' },
    ]);

    expect(result).toEqual([
      { role: 'user', text: 'Hello', time: undefined },
    ]);
  });

  test('handles empty messages', () => {
    const result = formatChatExport([
      { role: 'user' },
    ]);

    expect(result).toEqual([
      { role: 'user', text: '', time: undefined },
    ]);
  });

  test('handles empty array', () => {
    expect(formatChatExport([])).toEqual([]);
  });

  test('handles zero timestamp', () => {
    const result = formatChatExport([{ role: 'user', text: 'test', timestamp: 0 }]);
    // 0 is falsy — time should be undefined
    expect(result[0].time).toBeUndefined();
  });

  test('handles very long text', () => {
    const longText = 'x'.repeat(10000);
    const result = formatChatExport([{ role: 'user', text: longText }]);
    expect(result[0].text).toBe(longText);
    expect(result[0].text.length).toBe(10000);
  });

  test('prefers text over content', () => {
    const result = formatChatExport([{ role: 'user', text: 'from text', content: 'from content' }]);
    expect(result[0].text).toBe('from text');
  });
});
