/**
 * Tests for the API client helpers added for store consolidation.
 * Verifies each wrapper hits the right URL and shape.
 */

jest.mock('../../api', () => ({
  HTTP: 'http://mock',
  apiFetch: jest.fn(),
  getAuthHeaders: jest.fn(() => ({})),
}));

import { chat, proactive, models } from '../../api/client';
import { apiFetch } from '../../api';

const mockApiFetch = apiFetch as jest.MockedFunction<typeof apiFetch>;

beforeEach(() => {
  mockApiFetch.mockReset();
  mockApiFetch.mockResolvedValue({});
});

describe('chat.clear', () => {
  test('POSTs to /api/chat/clear', async () => {
    await chat.clear();
    expect(mockApiFetch).toHaveBeenCalledWith('http://mock/api/chat/clear', { method: 'POST' });
  });
});

describe('proactive.dismiss', () => {
  test('POSTs id in JSON body', async () => {
    await proactive.dismiss('pm-123');
    const [url, opts] = mockApiFetch.mock.calls[0];
    expect(url).toBe('http://mock/api/proactive/dismiss');
    expect((opts as any).method).toBe('POST');
    expect(JSON.parse((opts as any).body)).toEqual({ id: 'pm-123' });
    expect((opts as any).headers['Content-Type']).toBe('application/json');
  });
});

describe('models.listAvailable', () => {
  test('GETs /api/models/available with a default abort signal', async () => {
    await models.listAvailable();
    const [url, opts] = mockApiFetch.mock.calls[0];
    expect(url).toBe('http://mock/api/models/available');
    expect((opts as any).signal).toBeDefined();
  });

  test('honors a caller-provided signal', async () => {
    const controller = new AbortController();
    await models.listAvailable(controller.signal);
    expect((mockApiFetch.mock.calls[0][1] as any).signal).toBe(controller.signal);
  });
});
