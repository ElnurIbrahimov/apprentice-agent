package com.aura.mcp

import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import okhttp3.OkHttpClient
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class McpClientManagerTest {

    private val httpClient = mockk<OkHttpClient>(relaxed = true)
    private val manager = McpClientManager(httpClient)

    @Test
    fun connect_returns_error_for_non_https_non_local() = runTest {
        val config = McpServerConfig(
            id = "s1",
            name = "remote",
            url = "http://example.com/mcp",
            trustedLocal = false,
        )
        val health = manager.connect(config)
        assertEquals(McpConnectionState.ERROR, health.state)
        assertTrue(health.lastError.contains("HTTPS"))
    }

    @Test
    fun connect_allows_http_for_trusted_local() = runTest {
        val config = McpServerConfig(
            id = "s1",
            name = "local",
            url = "http://localhost:3000/mcp",
            trustedLocal = true,
        )
        // The connection will fail because mock HTTP client returns null,
        // but it should not fail with "HTTPS required"
        val health = manager.connect(config)
        // Should be ERROR but not due to HTTPS
        assertTrue(health.state == McpConnectionState.ERROR || health.state == McpConnectionState.CONNECTED)
        assertTrue(!health.lastError.contains("HTTPS"))
    }

    @Test
    fun listTools_returns_empty_when_not_connected() = runTest {
        val tools = manager.listTools("unknown")
        assertTrue(tools.isEmpty())
    }

    @Test
    fun callTool_returns_failure_when_not_connected() = runTest {
        val result = manager.callTool("unknown", "test_tool", emptyMap())
        assertTrue(result is McpToolResult.Failure)
        assertEquals("not_connected", (result as McpToolResult.Failure).code)
    }

    @Test
    fun connectedServerIds_returns_empty_when_none_connected() = runTest {
        val ids = manager.connectedServerIds()
        assertTrue(ids.isEmpty())
    }

    @Test
    fun healthSnapshot_returns_empty_when_none_connected() = runTest {
        val snapshot = manager.healthSnapshot()
        assertTrue(snapshot.isEmpty())
    }
}