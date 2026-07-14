package com.aura.tools

import com.aura.agent.ToolRisk
import com.aura.data.UserPreferences
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import io.mockk.mockk
import okhttp3.OkHttpClient
import org.junit.Test
import kotlin.test.assertEquals

class RemoteCostToolRiskTest {
    private val httpClient = mockk<OkHttpClient>()
    private val providerKeys = mockk<ProviderKeys>()

    @Test
    fun `metered tools require remote cost approval`() {
        val tools = listOf(
            ImageGenTool(httpClient, providerKeys).tool,
            TranscriptionTool(httpClient, providerKeys).tool,
            BraveSearchTool(httpClient, providerKeys).tool,
            TavilySearchTool(httpClient, providerKeys).tool,
            FirecrawlFetchTool(httpClient, providerKeys).tool,
            DeepResearchTool(
                httpClient,
                providerKeys,
                mockk<ProviderRegistry>(),
                mockk<UserPreferences>(),
            ).tool,
        )

        assertEquals(
            setOf("image_gen", "transcribe", "brave_search", "tavily_search", "fetch_url", "deep_research"),
            tools.map { it.name }.toSet(),
        )
        tools.forEach { tool ->
            assertEquals(ToolRisk.REMOTE_COST, tool.risk, "${tool.name} must require paid-API approval")
        }
    }
}
