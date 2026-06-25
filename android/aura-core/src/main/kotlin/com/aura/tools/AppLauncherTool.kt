package com.aura.tools

import android.content.Context
import android.content.Intent
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Launch another app by package name or activity class. Mirrors aura/tools/app_launcher.py (none — new).
 * Risk: WRITE_LOCAL (just opens another app).
 */
@Singleton
class AppLauncherTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "launch_app",
        description = "Launch another app by package name (e.g. 'com.google.android.youtube') or a deep-link URL (e.g. 'https://twitter.com').",
        parameters = ToolParameters(
            properties = mapOf(
                "target" to ToolProperty(type = "string", description = "Package name (e.g. com.example.app) or URL (https://...)"),
            ),
            required = listOf("target"),
        ),
    )

    val tool = Tool(
        name = "launch_app",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val target = call.arguments["target"] as? String ?: return@Tool ToolResult.Error("missing 'target'", "bad_args")
            try {
                val intent = if (target.startsWith("http://") || target.startsWith("https://")) {
                    Intent(Intent.ACTION_VIEW, android.net.Uri.parse(target)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                } else {
                    context.packageManager.getLaunchIntentForPackage(target)?.apply {
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    } ?: return@Tool ToolResult.Error("App not found: $target", "not_found")
                }
                context.startActivity(intent)
                ToolResult.Ok("Launched: $target")
            } catch (e: Exception) {
                ToolResult.Error("launch failed: ${e.message}", "exception")
            }
        },
    )
}
