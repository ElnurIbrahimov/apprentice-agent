package com.aura.tools

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
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
 * Post a system notification. v1 fires immediately; v1.5 adds scheduled
 * notifications via WorkManager. Mirrors aura/tools/notifications.py.
 * Risk: WRITE_LOCAL (POST_NOTIFICATIONS required on Android 13+).
 */
@Singleton
class NotificationsTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    companion object {
        const val CHANNEL_ID = "aura_general"
        const val CHANNEL_NAME = "Aura Assistant"
    }

    init {
        val mgr = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(CHANNEL_ID, CHANNEL_NAME, NotificationManager.IMPORTANCE_DEFAULT)
            mgr.createNotificationChannel(ch)
        }
    }

    fun definition() = ToolDefinition(
        name = "post_notification",
        description = "Show a system notification with a title and body.",
        parameters = ToolParameters(
            properties = mapOf(
                "title" to ToolProperty(type = "string", description = "Notification title"),
                "body" to ToolProperty(type = "string", description = "Notification body text"),
            ),
            required = listOf("title", "body"),
        ),
    )

    val tool = Tool(
        name = "post_notification",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        requiredPermissions = listOf("android.permission.POST_NOTIFICATIONS"),
        parameters = definition().parameters,
        execute = { call, ctx ->
            val title = call.arguments["title"] as? String ?: return@Tool ToolResult.Error("missing 'title'", "bad_args")
            val body = call.arguments["body"] as? String ?: return@Tool ToolResult.Error("missing 'body'", "bad_args")
            try {
                post(title, body)
                ToolResult.Ok("Notification posted: $title")
            } catch (e: SecurityException) {
                ToolResult.NeedsPermission("android.permission.POST_NOTIFICATIONS", "Notifications permission required.")
            } catch (e: Exception) {
                ToolResult.Error("failed: ${e.message}", "exception")
            }
        },
    )

    private fun post(title: String, body: String) {
        // Launch the app's main activity by package + launcher intent. Works
        // without depending on the app module's MainActivity class directly.
        val launchIntent = context.packageManager.getLaunchIntentForPackage(context.packageName)
            ?: Intent(Intent.ACTION_MAIN).apply { addCategory(Intent.CATEGORY_LAUNCHER) }
        launchIntent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        val pi = PendingIntent.getActivity(
            context, 0, launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val n = NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(body)
            .setSmallIcon(com.aura.core.R.drawable.ic_aura_notification)
            .setContentIntent(pi)
            .setAutoCancel(true)
            .build()
        val mgr = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        mgr.notify(System.currentTimeMillis().toInt(), n)
    }
}
