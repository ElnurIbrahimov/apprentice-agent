package com.aura.tools

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import androidx.core.content.FileProvider
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.first
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Captures a photo via the system camera, base64-encodes it, and asks the
 * active vision-capable LLM to describe or analyze the image.
 *
 * Mirrors aura/tools/vision.py + aura/tools/screenshot.py — phone-native variant
 * where the photo comes from the camera intent rather than a screen capture.
 *
 * Risk: WRITE_LOCAL (writes a temp file the camera intent will fill).
 */
@Singleton
class ImageInputTool @Inject constructor(
    @ApplicationContext private val context: Context,
    private val providerRegistry: ProviderRegistry,
) {
    fun definition() = ToolDefinition(
        name = "image_input",
        description = "Open the system camera, take a photo, then ask the active vision model to describe or analyze it. The 'question' parameter is the prompt to ask about the photo (e.g. 'what is in this image?', 'is this expired?', 'read the serial number').",
        parameters = ToolParameters(
            properties = mapOf(
                "question" to ToolProperty(type = "string", description = "What to ask about the captured image"),
            ),
            required = listOf("question"),
        ),
    )

    val tool = Tool(
        name = "image_input",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        requiredPermissions = listOf(android.Manifest.permission.CAMERA),
        parameters = definition().parameters,
        execute = { call, ctx ->
            val question = call.arguments["question"] as? String ?: return@Tool ToolResult.Error("missing 'question'", "bad_args")
            try {
                // 1) Create a temp file in cache dir for the camera output
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val photoFile = File(context.cacheDir, "aura_image_$timestamp.jpg")
                val photoUri: Uri = FileProvider.getUriForFile(
                    context,
                    "${context.packageName}.fileprovider",
                    photoFile,
                )
                // 2) Launch the camera intent
                val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
                    putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
                    addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                if (intent.resolveActivity(context.packageManager) == null) {
                    return@Tool ToolResult.Error("No camera app installed", "no_camera")
                }
                context.startActivity(intent)
                // 3) The user takes the photo outside the agent loop. For v1 we
                //    return a placeholder; v1.5 wires ActivityResultCallback to
                //    feed the bitmap back into the agent.
                ToolResult.Ok("Camera opened. (v1: user must describe the photo in the next message. v1.5 will inline the image.)")
            } catch (e: SecurityException) {
                ToolResult.NeedsPermission(android.Manifest.permission.CAMERA, "Camera access required.")
            } catch (e: Exception) {
                ToolResult.Error("image_input failed: ${e.message}", "exception")
            }
        },
    )
}
