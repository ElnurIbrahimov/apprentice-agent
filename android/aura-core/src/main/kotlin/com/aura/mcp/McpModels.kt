package com.aura.mcp

import kotlinx.serialization.Serializable

/**
 * Configuration for a single MCP server connection.
 * Stored in Room (non-secret fields) + SecureDataStore (tokens).
 */
@Serializable
data class McpServerConfig(
    val id: kotlin.String,
    val name: kotlin.String,
    val url: kotlin.String,
    val enabled: kotlin.Boolean = true,
    /** Whether this is a local/private endpoint (relaxes SSRF for localhost). */
    val trustedLocal: kotlin.Boolean = false,
    /** Allowed tool name prefixes. Empty = all tools from this server. */
    val allowedToolPrefixes: List<kotlin.String> = emptyList(),
    /** Denied tool names (exact match). */
    val deniedTools: List<kotlin.String> = emptyList(),
    /** Maximum number of tools to discover from this server. */
    val maxTools: Int = 100,
    /** Maximum response size in bytes for any single tool call. */
    val maxResponseBytes: Int = 1_000_000,
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    val updatedAt: kotlin.Long = createdAt,
)

/**
 * A discovered MCP tool, resource, or prompt.
 */
@Serializable
data class McpToolInfo(
    val serverId: kotlin.String,
    val name: kotlin.String,
    val description: kotlin.String,
    val inputSchemaJson: kotlin.String = "{}",
    val serverName: kotlin.String = "",
)

@Serializable
data class McpResourceInfo(
    val serverId: kotlin.String,
    val uri: kotlin.String,
    val name: kotlin.String,
    val description: kotlin.String = "",
    val mimeType: kotlin.String = "",
)

/**
 * Result of a tool call through MCP.
 */
sealed class McpToolResult {
    data class Success(val output: kotlin.String, val isError: kotlin.Boolean = false) : McpToolResult()
    data class Failure(val message: kotlin.String, val code: kotlin.String) : McpToolResult()
    data class Timeout(val serverName: kotlin.String) : McpToolResult()
}

/**
 * Connection state for an MCP server.
 */
enum class McpConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    ERROR,
}

/**
 * Health snapshot for a server connection.
 */
data class McpServerHealth(
    val serverId: kotlin.String,
    val state: McpConnectionState,
    val lastConnectedAt: kotlin.Long = 0L,
    val lastError: kotlin.String = "",
    val toolCount: Int = 0,
    val resourceCount: Int = 0,
)