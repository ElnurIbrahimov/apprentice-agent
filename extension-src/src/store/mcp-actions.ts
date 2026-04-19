import { HTTP, apiFetch } from '../api';
import type { McpServerInfo, McpServerCreate } from '../types';

type SetFn = (partial: any) => void;
type GetFn = () => any;

export function createMcpActions(set: SetFn, get: GetFn) {
  return {
    loadMcpServers: async () => {
      try {
        const data = await apiFetch(`${HTTP}/api/mcp/servers`);
        set({ mcpServers: (data?.servers || []) as McpServerInfo[], mcpLoaded: true, mcpError: null });
      } catch (err: any) {
        set({ mcpError: err?.message || 'Failed to load MCP servers', mcpLoaded: true });
      }
    },

    addMcpServer: async (server: McpServerCreate) => {
      await apiFetch(`${HTTP}/api/mcp/servers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(server),
      });
      await get().loadMcpServers();
    },

    removeMcpServer: async (name: string) => {
      await apiFetch(`${HTTP}/api/mcp/servers/${encodeURIComponent(name)}`, { method: 'DELETE' });
      set((s: any) => ({ mcpServers: s.mcpServers.filter((srv: McpServerInfo) => srv.name !== name) }));
    },

    setMcpServerEnabled: async (name: string, enabled: boolean) => {
      const action = enabled ? 'enable' : 'disable';
      await apiFetch(`${HTTP}/api/mcp/servers/${encodeURIComponent(name)}/${action}`, { method: 'POST' });
      set((s: any) => ({
        mcpServers: s.mcpServers.map((srv: McpServerInfo) =>
          srv.name === name ? { ...srv, enabled } : srv,
        ),
      }));
    },

    testMcpServer: async (name: string) => {
      try {
        const data: any = await apiFetch(`${HTTP}/api/mcp/servers/${encodeURIComponent(name)}/test`, { method: 'POST' });
        set((s: any) => ({
          mcpServers: s.mcpServers.map((srv: McpServerInfo) =>
            srv.name === name
              ? { ...srv, connected: !!data?.ok, tool_count: Number(data?.tool_count || 0), tools: (data?.tools || []).map((t: any) => t.name || String(t)), error: data?.error }
              : srv,
          ),
        }));
        return { ok: !!data?.ok, tool_count: Number(data?.tool_count || 0), error: data?.error };
      } catch (err: any) {
        return { ok: false, tool_count: 0, error: err?.message || 'Test failed' };
      }
    },
  };
}
