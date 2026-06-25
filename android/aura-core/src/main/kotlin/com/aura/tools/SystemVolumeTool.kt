package com.aura.tools

import android.content.Context
import android.media.AudioManager
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
 * Get or set system volume for a stream (music, ring, alarm, notification).
 * Mirrors aura/tools/system_volume.py (none — new).
 * Risk: WRITE_LOCAL.
 */
@Singleton
class SystemVolumeTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "system_volume",
        description = "Get or set the system volume for a stream. 'stream' is one of: music, ring, alarm, notification. 'level' is 0-100. Omit 'level' to get current volume.",
        parameters = ToolParameters(
            properties = mapOf(
                "stream" to ToolProperty(type = "string", description = "music|ring|alarm|notification (default music)"),
                "level" to ToolProperty(type = "integer", description = "Volume 0-100 (omit to query)"),
            ),
            required = listOf("stream"),
        ),
    )

    val tool = Tool(
        name = "system_volume",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val streamName = (call.arguments["stream"] as? String ?: "music").lowercase()
            val streamType = when (streamName) {
                "music", "media" -> AudioManager.STREAM_MUSIC
                "ring" -> AudioManager.STREAM_RING
                "alarm" -> AudioManager.STREAM_ALARM
                "notification" -> AudioManager.STREAM_NOTIFICATION
                else -> return@Tool ToolResult.Error("unknown stream: $streamName", "bad_args")
            }
            val mgr = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
                ?: return@Tool ToolResult.Error("no AudioManager", "system_error")
            val level = call.arguments["level"] as? Int
            return@Tool if (level == null) {
                val current = mgr.getStreamVolume(streamType)
                val max = mgr.getStreamMaxVolume(streamType)
                ToolResult.Ok("$streamName volume: $current / $max ($current% of max)")
            } else {
                val max = mgr.getStreamMaxVolume(streamType)
                val target = ((level.toDouble() / 100.0) * max).toInt().coerceIn(0, max)
                mgr.setStreamVolume(streamType, target, 0)
                ToolResult.Ok("$streamName volume set to $level% (index $target / $max)")
            }
        },
    )
}
