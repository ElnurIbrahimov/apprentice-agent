/**
 * Tests for the MCP action factory. Mocks apiFetch; verifies endpoint
 * selection, error handling, and per-server update semantics.
 */

jest.mock('../../api', () => ({
  HTTP: 'http://mock',
  apiFetch: jest.fn(),
  API_KEY: 'test-key',
}));

import { createMcpActions } from '../../store/mcp-actions';
import { apiFetch } from '../../api';

const mockApiFetch = apiFetch as jest.MockedFunction<typeof apiFetch>;

function makeState() {
  const state: any = {
    mcpServers: [],
    mcpLoaded: false,
    mcpError: null,
  };
  const set = (partial: any) => {
    const patch = typeof partial === 'function' ? partial(state) : partial;
    Object.assign(state, patch);
  };
  const get = () => state;
  return { state, set, get };
}

beforeEach(() => {
  mockApiFetch.mockReset();
});

describe('mcp-actions', () => {
  test('loadMcpServers populates list and marks loaded on success', async () => {
    const { state, set, get } = makeState();
    mockApiFetch.mockResolvedValueOnce({ servers: [{ name: 's1', enabled: true }] });
    const actions = createMcpActions(set, get);
    await actions.loadMcpServers();
    expect(state.mcpServers).toHaveLength(1);
    expect(state.mcpLoaded).toBe(true);
    expect(state.mcpError).toBeNull();
  });

  test('loadMcpServers records error when apiFetch rejects', async () => {
    const { state, set, get } = makeState();
    mockApiFetch.mockRejectedValueOnce(new Error('down'));
    const actions = createMcpActions(set, get);
    await actions.loadMcpServers();
    expect(state.mcpError).toBe('down');
    expect(state.mcpLoaded).toBe(true);
  });

  test('removeMcpServer filters the named server out locally', async () => {
    const { state, set, get } = makeState();
    state.mcpServers = [{ name: 'a' }, { name: 'b' }];
    mockApiFetch.mockResolvedValueOnce({});
    const actions = createMcpActions(set, get);
    await actions.removeMcpServer('a');
    expect(state.mcpServers.map((s: any) => s.name)).toEqual(['b']);
  });

  test('setMcpServerEnabled picks enable vs disable endpoint', async () => {
    const { state, set, get } = makeState();
    state.mcpServers = [{ name: 'svc', enabled: false }];
    mockApiFetch.mockResolvedValue({});
    const actions = createMcpActions(set, get);

    await actions.setMcpServerEnabled('svc', true);
    expect(mockApiFetch.mock.calls[0][0]).toBe('http://mock/api/mcp/servers/svc/enable');
    expect(state.mcpServers[0].enabled).toBe(true);

    await actions.setMcpServerEnabled('svc', false);
    expect(mockApiFetch.mock.calls[1][0]).toBe('http://mock/api/mcp/servers/svc/disable');
    expect(state.mcpServers[0].enabled).toBe(false);
  });

  test('testMcpServer returns ok=false on thrown error and does not mutate state', async () => {
    const { state, set, get } = makeState();
    state.mcpServers = [{ name: 'x', connected: false, tool_count: 0 }];
    mockApiFetch.mockRejectedValueOnce(new Error('refused'));
    const actions = createMcpActions(set, get);
    const result = await actions.testMcpServer('x');
    expect(result).toEqual({ ok: false, tool_count: 0, error: 'refused' });
    expect(state.mcpServers[0].connected).toBe(false);
  });

  test('testMcpServer updates per-server connection info on success', async () => {
    const { state, set, get } = makeState();
    state.mcpServers = [{ name: 'x' }];
    mockApiFetch.mockResolvedValueOnce({ ok: true, tool_count: 3, tools: [{ name: 't1' }, { name: 't2' }, { name: 't3' }] });
    const actions = createMcpActions(set, get);
    const result = await actions.testMcpServer('x');
    expect(result.ok).toBe(true);
    expect(result.tool_count).toBe(3);
    expect(state.mcpServers[0].connected).toBe(true);
    expect(state.mcpServers[0].tools).toEqual(['t1', 't2', 't3']);
  });
});
