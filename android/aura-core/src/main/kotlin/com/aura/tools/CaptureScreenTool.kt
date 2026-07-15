package com.aura.tools

import android.graphics.Bitmap
import android.util.Base64
import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.security.ScreenCaptureHolder
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import java.io.ByteArrayOutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Capture the device screen and return it as base64 JPEG.
 * Requires the user to grant screen-capture permission first.
 * Risk: PRIVACY.
 */
@Singleton
class CaptureScreenTool @Inject constructor(
    private val screenCaptureHolder: ScreenCaptureHolder,
) {
    fun definition() = ToolDefinition(
        name = "capture_screen",
        description = "Capture the device screen and return it as a base64 JPEG. Requires screen-capture permission; if not granted, the tool returns instructions for the user to allow it.",
        parameters = ToolParameters(
            properties = mapOf(
                "width" to ToolProperty(type = "integer", description = "Capture width (default 1080)"),
                "height" to ToolProperty(type = "integer", description = "Capture height (default 1920)"),
                "quality" to ToolProperty(type = "integer", description = "JPEG quality 1-100 (default 80)"),
            ),
            required = emptyList(),
        ),
    )

    val tool = Tool(
        name = "capture_screen",
        description = definition().description,
        risk = ToolRisk.PRIVACY,
        parameters = definition().parameters,
        execute = { call, _ ->
            if (screenCaptureHolder.pendingResult.value != true) {
                screenCaptureHolder.requestPermission()
                return@Tool ToolResult.Ok(
                    "Screen-capture permission requested. Please confirm the system dialog, then ask me to capture again."
                )
            }
            val width = (call.arguments["width"] as? Int) ?: 1080
            val height = (call.arguments["height"] as? Int) ?: 1920
            val quality = ((call.arguments["quality"] as? Int) ?: 80).coerceIn(1, 100)

            val bitmap = screenCaptureHolder.capture(width, height)
                ?: return@Tool ToolResult.Error("Screen capture failed. Re-grant permission and try again.", "capture_failed")

            val base64 = bitmapToBase64(bitmap, quality)
            ToolResult.Ok("data:image/jpeg;base64,$base64")
        },
        category = "device",
    )

    private fun bitmapToBase64(bitmap: Bitmap, quality: Int): String {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)
        return Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)
    }
}
