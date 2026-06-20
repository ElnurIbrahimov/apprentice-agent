import { DEFAULT_BACKEND_URL } from '../../defaults';

describe('DuckDNS removal regression', () => {
  it('DEFAULT_BACKEND_URL is localhost:8000', () => {
    expect(DEFAULT_BACKEND_URL).toBe('http://localhost:8000');
  });

  it('DEFAULT_BACKEND_URL contains no duckdns or aura-elnur', () => {
    const lower = DEFAULT_BACKEND_URL.toLowerCase();
    expect(lower).not.toContain('duckdns');
    expect(lower).not.toContain('aura-elnur');
  });
});