package com.aura.tools

import android.content.Context
import android.net.Uri
import androidx.browser.customtabs.CustomTabColorSchemeParams
import androidx.browser.customtabs.CustomTabsIntent
import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.core.url.SsrfGuard
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Open a URL in an in-app Chrome Custom Tab. Falls back to an external
 * browser if no Custom Tabs provider is available.
 * Risk: WRITE_LOCAL (launches an activity).
 */
@Singleton
class OpenBrowserTabTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "open_browser_tab",
        description = "Open a URL in an in-app browser tab (Chrome Custom Tab).",
        parameters = ToolParameters(
            properties = mapOf(
                "url" to ToolProperty(type = "string", description = "URL to open"),
                "color" to ToolProperty(type = "string", description = "Optional toolbar color hex, e.g. '#1a1a1a'"),
            ),
            required = listOf("url"),
        ),
    )

    val tool = Tool(
        name = "open_browser_tab",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, _ ->
            val url = call.arguments["url"] as? String
                ?: return@Tool ToolResult.Error("missing 'url'", "bad_args")
            val ssrfError = SsrfGuard.validate(url)
            if (ssrfError != null) {
                return@Tool ToolResult.Error(ssrfError, "ssrf_guard")
            }
            val colorHex = call.arguments["color"] as? String
            try {
                val uri = Uri.parse(url)
                val builder = CustomTabsIntent.Builder()
                colorHex?.let { parseColor(it) }?.let { color ->
                    builder.setDefaultColorSchemeParams(
                        CustomTabColorSchemeParams.Builder()
                            .setToolbarColor(color)
                            .build()
                    )
                }
                val intent = builder.build()
                intent.intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                intent.launchUrl(context, uri)
                ToolResult.Ok("Opened in browser tab: $url")
            } catch (e: Exception) {
                ToolResult.Error("Could not open browser tab: ${e.message}", "exception")
            }
        },
        category = "web",
    )

    private fun parseColor(hex: String): Int? = try {
        android.graphics.Color.parseColor(hex)
    } catch (_: IllegalArgumentException) {
        null
    }
}
