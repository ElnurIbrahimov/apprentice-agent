package com.aura.mcp

import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages MCP server connections, tool discovery, and tool invocation.
 *
 * This is the initial lightweight implementation using direct HTTP.
 * When the official Kotlin MCP SDK (io.modelcontextprotocol:kotlin-sdk)
 * is validated on Android, this will be swapped to use the SDK's
 * transport and protocol types while keeping the same public API.
 *
 * Security rules:
 * - HTTPS by default; HTTP only for trusted local endpoints
 * - Remote risk annotations from MCP servers are hints, not authority
 * - All tool calls pass through local ToolPolicy classification
 * - Response size is bounded
 * - Connection timeout and retry are enforced
 */
@Singleton
class McpClientManager @Inject constructor(
    private val httpClient: okhttp3.OkHttpClient,
) {
    private val connections = mutableMapOf<kotlin.String, McpConnection>()

    /**
     * Connect to an MCP server. Returns the connection health.
     */
    suspend fun connect(config: McpServerConfig): McpServerHealth {
        // Validate URL
        if (!config.trustedLocal && !config.url.startsWith("https://")) {
            return McpServerHealth(
                serverId = config.id,
                state = McpConnectionState.ERROR,
                lastError = "Non-local servers must use HTTPS",
            )
        }

        val connection = McpConnection(config, httpClient)
        val health = connection.initialize()
        if (health.state == McpConnectionState.CONNECTED) {
            connections[config.id] = connection
        }
        return health
    }

    /**
     * Disconnect from a server and clean up resources.
     */
    suspend fun disconnect(serverId: kotlin.String) {
        connections.remove(serverId)?.disconnect()
    }

    /**
     * List all discovered tools from a connected server.
     */
    suspend fun listTools(serverId: kotlin.String): List<McpToolInfo> {
        val conn = connections[serverId]
            ?: return emptyList()
        return conn.listTools()
    }

    /**
     * Call a tool on a connected MCP server.
     */
    suspend fun callTool(
        serverId: kotlin.String,
        toolName: kotlin.String,
        arguments: Map<kotlin.String, Any?>,
        timeoutMs: kotlin.Long = 30_000L,
    ): McpToolResult {
        val conn = connections[serverId]
            ?: return McpToolResult.Failure("Server $serverId is not connected", "not_connected")
        val config = conn.config

        // Check deny list
        if (toolName in config.deniedTools) {
            return McpToolResult.Failure("Tool $toolName is denied for server ${config.name}", "denied")
        }

        // Check prefix allow list
        if (config.allowedToolPrefixes.isNotEmpty()) {
            val allowed = config.allowedToolPrefixes.any { prefix -> toolName.startsWith(prefix) }
            if (!allowed) {
                return McpToolResult.Failure("Tool $toolName does not match any allowed prefix", "prefix_denied")
            }
        }

        return conn.callTool(toolName, arguments, timeoutMs)
    }

    /**
     * Get health for all connected servers.
     */
    fun healthSnapshot(): List<McpServerHealth> =
        connections.values.map { it.health }

    /**
     * Get a list of all connected server IDs.
     */
    fun connectedServerIds(): List<kotlin.String> =
        connections.filter { it.value.isConnected() }.keys.toList()

    /**
     * List resources from a connected server.
     */
    suspend fun listResources(serverId: kotlin.String): List<McpResourceInfo> {
        val conn = connections[serverId] ?: return emptyList()
        return conn.listResources()
    }
}