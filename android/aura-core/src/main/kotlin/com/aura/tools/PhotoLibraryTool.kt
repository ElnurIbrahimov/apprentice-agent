package com.aura.tools

import android.content.ContentUris
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

/**
 * List recent photos from the device's photo library. Mirrors aura/tools/photo_library.py (none — new).
 * Risk: PRIVACY (READ_MEDIA_IMAGES on Android 13+).
 */
@Singleton
class PhotoLibraryTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "photo_library",
        description = "List recent photos. Returns up to N photos (default 10) with their date taken, dimensions, and content URI. Useful for 'send my last photo' flows.",
        parameters = ToolParameters(
            properties = mapOf(
                "limit" to ToolProperty(type = "integer", description = "Max photos to return (default 10, max 50)"),
            ),
            required = emptyList(),
        ),
    )

    val tool = Tool(
        name = "photo_library",
        description = definition().description,
        risk = ToolRisk.PRIVACY,
        requiredPermissions = emptyList(), // READ_MEDIA_IMAGES is granular; permission is at OS level
        parameters = definition().parameters,
        execute = { call, ctx ->
            val limit = (call.arguments["limit"] as? Int ?: 10).coerceIn(1, 50)
            try {
                val photos = listRecent(limit)
                if (photos.isEmpty()) {
                    ToolResult.Ok("No photos found in the library.")
                } else {
                    val text = photos.mapIndexed { i, p ->
                        "${i + 1}. ${p.date} — ${p.width}x${p.height}\n   ${p.uri}"
                    }.joinToString("\n")
                    ToolResult.Ok(text)
                }
            } catch (e: SecurityException) {
                ToolResult.NeedsPermission("android.permission.READ_MEDIA_IMAGES", "Photo library access required.")
            } catch (e: Exception) {
                ToolResult.Error("photo_library failed: ${e.message}", "exception")
            }
        },
    )

    private data class Photo(val uri: Uri, val date: String, val width: Int, val height: Int)

    private fun listRecent(limit: Int): List<Photo> {
        val out = mutableListOf<Photo>()
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATE_TAKEN,
            MediaStore.Images.Media.WIDTH,
            MediaStore.Images.Media.HEIGHT,
        )
        val sort = "${MediaStore.Images.Media.DATE_TAKEN} DESC LIMIT $limit"
        val dateFmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US)
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection, null, null, sort
        )?.use { c ->
            while (c.moveToNext()) {
                val id = c.getLong(0)
                val date = c.getLong(1).let { if (it > 0) dateFmt.format(Date(it)) else "unknown" }
                val w = c.getInt(2)
                val h = c.getInt(3)
                val uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)
                out += Photo(uri, date, w, h)
            }
        }
        return out
    }
}
