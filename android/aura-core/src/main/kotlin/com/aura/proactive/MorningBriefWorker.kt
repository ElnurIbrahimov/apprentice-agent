package com.aura.proactive

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.aura.agent.Brain
import com.aura.agent.Conversation
import com.aura.core.R
import com.aura.memory.MemoryStore
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderRegistry
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.flow.first

/**
 * WorkManager job that runs at ~7am local time, asks Aura for a morning brief
 * based on the user's recent memories + current time, and posts the result
 * as a notification. Mirrors aura/hands/morning_briefing.py + aura/proactive/.
 */
@HiltWorker
class MorningBriefWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val brain: Brain,
    private val providerRegistry: ProviderRegistry,
    private val memoryStore: MemoryStore,
    private val eventBus: ProactiveEventBus,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        // 1) Recall the user's last 10 memories
        val memories = memoryStore.recent(10).joinToString("\n") { "- [${it.category}] ${it.content}" }
        val now = java.text.SimpleDateFormat("EEEE, MMM d, HH:mm", java.util.Locale.US)
            .format(java.util.Date())
        val systemPrompt = """
            You are Aura. Write a 3-5 line morning brief for the user. Be warm, direct,
            and specific. If you remember anything about the user from past
            conversations, weave it in naturally. Today's date is $now.
        """.trimIndent()
        val userMessage = "Give me my morning brief.\n\n# What I remember about you:\n$memories"
        // 2) Run a single-shot agentic call
        val conversation = Conversation(systemPrompt = systemPrompt)
        conversation.addUser(userMessage)
        val options = ChatOptions(temperature = 0.7, maxTokens = 500)
        val responseText = StringBuilder()
        val activeModel = "ollama:deepseek-v3.2:cloud"
        val provider = providerRegistry.all().firstOrNull { it.isConfigured() }
        if (provider == null) {
            // No provider configured; skip silently
            return Result.success()
        }
        try {
            // Use Brain.stream to get a single response (no tool calls expected)
            val messages = conversation.toMessages()
            provider.chat(activeModel.substringAfter(":"), messages, options, emptyList<com.aura.providers.ToolDefinition>()).collect { chunk ->
                chunk.text?.let { responseText.append(it) }
            }
        } catch (e: Exception) {
            return Result.retry()
        }
        val text = responseText.toString().trim()
        if (text.isBlank()) return Result.success()

        // 3) Post the notification
        postNotification(applicationContext, "☀️ Good morning", text)
        eventBus.emit(ProactiveEventBus.Event.MorningBriefReady("☀️ Good morning", text))
        return Result.success()
    }

    private fun postNotification(ctx: Context, title: String, body: String) {
        val mgr = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(
                "aura_morning_brief",
                "Aura Morning Brief",
                NotificationManager.IMPORTANCE_DEFAULT,
            )
            mgr.createNotificationChannel(ch)
        }
        val launchIntent = ctx.packageManager.getLaunchIntentForPackage(ctx.packageName)
            ?: Intent(Intent.ACTION_MAIN).apply { addCategory(Intent.CATEGORY_LAUNCHER) }
        launchIntent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        val pi = PendingIntent.getActivity(
            ctx, 0, launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val n = NotificationCompat.Builder(ctx, "aura_morning_brief")
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setSmallIcon(R.drawable.ic_aura_notification)
            .setContentIntent(pi)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .build()
        mgr.notify(MORNING_BRIEF_ID, n)
    }

    companion object {
        const val MORNING_BRIEF_ID = 1001
        const val UNIQUE_NAME = "morning-brief-daily"
    }
}
