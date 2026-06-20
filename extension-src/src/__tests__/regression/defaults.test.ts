import { DEFAULT_BACKEND_URL } from '../../defaults';

describe('DEFAULT_BACKEND_URL', () => {
  it('is localhost:8000', () => {
    expect(DEFAULT_BACKEND_URL).toBe('http://localhost:8000');
  });

  it('contains no DuckDNS', () => {
    expect(DEFAULT_BACKEND_URL.toLowerCase()).not.toContain('duckdns');
  });
});