package com.aura.tools

import android.content.Context
import android.content.Intent
import android.provider.MediaStore
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
 * Open the system camera app for the user to take a photo. v1: intent only
 * (no in-app camera UI). v1.5: CameraX in-app capture with vision analysis.
 * Risk: WRITE_LOCAL (just opens camera).
 */
@Singleton
class CameraCaptureTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "camera_capture",
        description = "Open the system camera app so the user can take a photo. Returns a placeholder; the photo lands in the system gallery.",
        parameters = ToolParameters(properties = emptyMap(), required = emptyList()),
    )

    val tool = Tool(
        name = "camera_capture",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            try {
                val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                if (intent.resolveActivity(context.packageManager) == null) {
                    return@Tool ToolResult.Error("No camera app installed", "no_camera")
                }
                context.startActivity(intent)
                ToolResult.Ok("Camera opened. Photo will land in the gallery.")
            } catch (e: Exception) {
                ToolResult.Error("camera failed: ${e.message}", "exception")
            }
        },
    )
}
