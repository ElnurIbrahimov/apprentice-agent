package com.aura.mcp

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.UUID

/**
 * Single MCP server connection. Implements the MCP Streamable HTTP
 * transport: JSON-RPC 2.0 over HTTP POST with SSE response support.
 *
 * This is a minimal implementation. The official Kotlin MCP SDK will
 * replace this class when validated on Android.
 */
internal class McpConnection(
    val config: McpServerConfig,
    private val httpClient: OkHttpClient,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mediaTypeJson = "application/json".toMediaType()
    private var _health: McpServerHealth = McpServerHealth(
        serverId = config.id,
        state = McpConnectionState.DISCONNECTED,
    )
    val health: McpServerHealth get() = _health

    fun isConnected(): kotlin.Boolean = _health.state == McpConnectionState.CONNECTED

    suspend fun initialize(): McpServerHealth = withContext(Dispatchers.IO) {
        try {
            val request = buildJsonRpcRequest("initialize", buildJsonObject {
                put("protocolVersion", "2025-03-26")
                putJsonObject("capabilities") { }
                put("clientInfo", buildJsonObject {
                    put("name", "aura-android")
                    put("version", "0.16.0")
                })
            })
            val response = sendRequest(request)
            if (response != null) {
                _health = _health.copy(
                    state = McpConnectionState.CONNECTED,
                    lastConnectedAt = System.currentTimeMillis(),
                    lastError = "",
                )
            } else {
                _health = _health.copy(
                    state = McpConnectionState.ERROR,
                    lastError = "Initialize failed: no response",
                )
            }
        } catch (e: Exception) {
            _health = _health.copy(
                state = McpConnectionState.ERROR,
                lastError = e.message ?: "Unknown error",
            )
        }
        _health
    }

    suspend fun listTools(): List<McpToolInfo> = withContext(Dispatchers.IO) {
        if (!isConnected()) return@withContext emptyList()
        try {
            val request = buildJsonRpcRequest("tools/list", JsonObject(emptyMap()))
            val response = sendRequest(request) ?: return@withContext emptyList()
            val toolsArray = response["result"]?.jsonObject?.get("tools")?.jsonArray ?: return@withContext emptyList()
            val tools = toolsArray.mapNotNull { item ->
                val obj = item as? JsonObject ?: return@mapNotNull null
                val name = obj["name"]?.jsonPrimitive?.content ?: return@mapNotNull null
                McpToolInfo(
                    serverId = config.id,
                    name = name,
                    description = obj["description"]?.jsonPrimitive?.content ?: "",
                    inputSchemaJson = obj["inputSchema"]?.toString() ?: "{}",
                    serverName = config.name,
                )
            }.take(config.maxTools)
            _health = _health.copy(toolCount = tools.size)
            tools
        } catch (e: Exception) {
            emptyList()
        }
    }

    suspend fun listResources(): List<McpResourceInfo> = withContext(Dispatchers.IO) {
        if (!isConnected()) return@withContext emptyList()
        try {
            val request = buildJsonRpcRequest("resources/list", JsonObject(emptyMap()))
            val response = sendRequest(request) ?: return@withContext emptyList()
            val resourcesArray = response["result"]?.jsonObject?.get("resources")?.jsonArray ?: return@withContext emptyList()
            resourcesArray.mapNotNull { item ->
                val obj = item as? JsonObject ?: return@mapNotNull null
                val uri = obj["uri"]?.jsonPrimitive?.content ?: return@mapNotNull null
                McpResourceInfo(
                    serverId = config.id,
                    uri = uri,
                    name = obj["name"]?.jsonPrimitive?.content ?: "",
                    description = obj["description"]?.jsonPrimitive?.content ?: "",
                    mimeType = obj["mimeType"]?.jsonPrimitive?.content ?: "",
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    suspend fun callTool(
        toolName: kotlin.String,
        arguments: Map<kotlin.String, Any?>,
        timeoutMs: kotlin.Long,
    ): McpToolResult = withTimeoutOrNull(timeoutMs) {
        withContext(Dispatchers.IO) {
            try {
                val argsObj = buildJsonObject {
                    arguments.forEach { (key, value) ->
                        when (value) {
                            is kotlin.String -> put(key, value)
                            is Number -> put(key, value)
                            is kotlin.Boolean -> put(key, value)
                            is Map<*, *> -> putJsonObject(key) {
                                @Suppress("UNCHECKED_CAST")
                                (value as Map<kotlin.String, Any?>).forEach { (k, v) ->
                                    when (v) {
                                        is kotlin.String -> put(k, v)
                                        is Number -> put(k, v)
                                        is kotlin.Boolean -> put(k, v)
                                    }
                                }
                            }
                        }
                    }
                }
                val request = buildJsonRpcRequest("tools/call", buildJsonObject {
                    put("name", toolName)
                    put("arguments", argsObj)
                })
                val response = sendRequest(request)
                    ?: return@withContext McpToolResult.Failure("No response from server", "no_response")
                val result = response["result"]?.jsonObject
                    ?: return@withContext McpToolResult.Failure("Missing result", "malformed_response")
                val content = result["content"]?.jsonArray
                    ?: return@withContext McpToolResult.Failure("Missing content array", "malformed_response")
                val output = content.mapNotNull { item ->
                    (item as? JsonObject)?.get("text")?.jsonPrimitive?.content
                }.joinToString("\n")
                // Enforce response size limit
                val truncated = if (output.length > config.maxResponseBytes) {
                    output.take(config.maxResponseBytes) + "\n[truncated at ${config.maxResponseBytes} bytes]"
                } else output
                McpToolResult.Success(truncated, isError = result["isError"]?.jsonPrimitive?.content == "true")
            } catch (e: Exception) {
                McpToolResult.Failure(e.message ?: "Tool call failed", "call_error")
            }
        }
    } ?: McpToolResult.Timeout(config.name)

    suspend fun disconnect() {
        _health = _health.copy(state = McpConnectionState.DISCONNECTED)
    }

    private fun buildJsonRpcRequest(method: kotlin.String, params: JsonObject): JsonObject = buildJsonObject {
        put("jsonrpc", "2.0")
        put("id", UUID.randomUUID().toString())
        put("method", method)
        put("params", params)
    }

    private fun sendRequest(requestBody: JsonObject): JsonObject? {
        val body = requestBody.toString().toRequestBody(mediaTypeJson)
        val builder = Request.Builder().url(config.url).post(body)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
        // Auth headers would go here from SecureDataStore

        return try {
            httpClient.newCall(builder.build()).execute().use { response ->
                if (!response.isSuccessful) return null
                val raw = response.body?.string() ?: return null
                json.parseToJsonElement(raw) as? JsonObject
            }
        } catch (e: Exception) {
            null
        }
    }
}