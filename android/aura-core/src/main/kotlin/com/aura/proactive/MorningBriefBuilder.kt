package com.aura.proactive

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import com.aura.agent.Conversation
import com.aura.core.R
import com.aura.kg.KnowledgeGraphRepository
import com.aura.memory.MemoryStore
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderRegistry
import com.aura.tasks.TaskDao
import com.aura.tools.CalendarReadTool
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import javax.inject.Inject
import javax.inject.Singleton

/**
 * The actual morning-brief build + post. Extracted from
 * [MorningBriefWorker] so the ProactiveRunner can call the
 * same logic on demand (the "fire morning brief now" debug
 * button on the Proactive history screen). Also called by
 * the worker — `doWork` is now a one-liner.
 *
 * Public contract:
 *   - runNow() returns a [Result] so callers can report
 *     success / retry / failure to the UI
 *   - Errors from any step (memory / KG / tasks / calendar /
 *     LLM greeting) are caught with `runCatching` so a
 *     single broken source doesn't kill the whole brief
 *   - Emits on the [ProactiveEventBus] AND posts a system
 *     notification, so the brief shows up regardless of
 *     whether the user is in the app
 */
@Singleton
class MorningBriefBuilder @Inject constructor(
    @ApplicationContext private val context: Context,
    private val providerRegistry: ProviderRegistry,
    private val memoryStore: MemoryStore,
    private val taskDao: TaskDao,
    private val kgRepository: KnowledgeGraphRepository,
    private val calendarReadTool: CalendarReadTool,
    private val eventBus: ProactiveEventBus,
) {
    suspend fun runNow(): androidx.work.ListenableWorker.Result {
        val now = System.currentTimeMillis()
        val since24h = now - 24L * 60L * 60L * 1000L
        val cal = java.util.Calendar.getInstance()
        cal.set(java.util.Calendar.HOUR_OF_DAY, 0)
        cal.set(java.util.Calendar.MINUTE, 0)
        cal.set(java.util.Calendar.SECOND, 0)
        cal.set(java.util.Calendar.MILLISECOND, 0)
        val startOfDay = cal.timeInMillis
        val endOfDay = startOfDay + 24L * 60L * 60L * 1000L

        val briefContext = runCatching {
            coroutineScope {
                val decayedDeferred = async {
                    runCatching { memoryStore.decayedBelow(DECAY_THRESHOLD, 10) }
                        .getOrDefault(emptyList())
                }
                val newMemoriesDeferred = async {
                    runCatching { memoryStore.recentSince(since24h, 10) }
                        .getOrDefault(emptyList())
                }
                val newKgDeferred = async {
                    runCatching { kgRepository.recentSince(since24h, 10) }
                        .getOrDefault(emptyList())
                }
                val tasksDeferred = async {
                    runCatching { taskDao.dueInRange(startOfDay, endOfDay) }
                        .getOrDefault(emptyList())
                }
                val calendarDeferred = async {
                    runCatching { calendarReadTool.readTodaysEvents() }
                        .getOrDefault(emptyList())
                }
                BriefContext(
                    decayedMemories = decayedDeferred.await(),
                    newMemories = newMemoriesDeferred.await(),
                    newKgNodes = newKgDeferred.await(),
                    tasksDueToday = tasksDeferred.await(),
                    calendarToday = calendarDeferred.await(),
                )
            }
        }.getOrDefault(BriefContext())

        val summary = buildSummary(briefContext)
        val greeting = if (briefContext.isEmpty) "" else runCatching {
            llmGreeting(now)
        }.getOrDefault("").trim()

        val notificationTitle = "☀️ Good morning"
        val notificationBody = listOf(greeting, summary)
            .filter { it.isNotBlank() }
            .joinToString(separator = "\n\n")

        if (notificationBody.isBlank()) {
            return androidx.work.ListenableWorker.Result.success()
        }

        eventBus.emit(ProactiveEventBus.Event.MorningBriefReady(notificationTitle, notificationBody))
        eventBus.emit(ProactiveEventBus.Event.MorningBriefStructured(briefContext))

        postNotification(
            context,
            title = notificationTitle,
            body = notificationBody,
            summary = summary,
        )
        return androidx.work.ListenableWorker.Result.success()
    }

    private fun buildSummary(context: BriefContext): String {
        val lines = mutableListOf<String>()
        if (context.decayedMemories.isNotEmpty()) {
            val n = context.decayedMemories.size
            val preview = context.decayedMemories.take(3)
                .joinToString(separator = " · ") { it.content.take(40) }
            lines += if (n == 1) "💭 1 memory fading: $preview" else "💭 $n memories fading: $preview"
        }
        if (context.newMemories.isNotEmpty()) {
            val n = context.newMemories.size
            lines += if (n == 1) "🧠 1 new thing you told me" else "🧠 $n new things you told me"
        }
        if (context.newKgNodes.isNotEmpty()) {
            val n = context.newKgNodes.size
            lines += if (n == 1) "🕸️ 1 fact learned" else "🕸️ $n facts learned"
        }
        if (context.tasksDueToday.isNotEmpty()) {
            val n = context.tasksDueToday.size
            val titles = context.tasksDueToday.take(3).joinToString(" · ") { it.title }
            lines += if (n == 1) "📋 1 task due today: $titles" else "📋 $n tasks due today: $titles"
        }
        if (context.calendarToday.isNotEmpty()) {
            lines += "📅 Today: ${context.calendarToday.take(3).joinToString(" · ")}"
        }
        return lines.joinToString(separator = "\n")
    }

    private suspend fun llmGreeting(now: Long): String {
        val moaProvider = providerRegistry.get("moa")
        val (provider, model) = if (moaProvider?.isConfigured() == true) {
            providerRegistry.parse("moa:default")
        } else {
            val solo = providerRegistry.all().firstOrNull { it.isConfigured() }
                ?: return ""
            val modelId = solo.listModels().firstOrNull()
                ?.let { "${solo.prefix}:$it" }
                ?: defaultModelIdForProvider(solo.prefix)
            if (modelId == null) return ""
            try {
                providerRegistry.parse(modelId)
            } catch (e: IllegalArgumentException) {
                return ""
            }
        }
        val dateStr = java.text.SimpleDateFormat("EEEE, MMM d", java.util.Locale.US)
            .format(java.util.Date(now))
        val systemPrompt = """
            You are Aura. Write a 1-2 sentence warm good-morning line for the user. Be specific
            to today ($dateStr). Do NOT enumerate facts, tasks, or events — those are shown
            separately. Just a short opener.
        """.trimIndent()
        val userMessage = "Good morning."
        val options = ChatOptions(temperature = 0.7, maxTokens = 120)
        val conversation = Conversation(systemPrompt = systemPrompt).addUser(userMessage)
        val text = StringBuilder()
        try {
            provider.chat(model, conversation.toMessages(), options, emptyList<com.aura.providers.ToolDefinition>()).collect { chunk ->
                chunk.text?.let { text.append(it) }
            }
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e
        } catch (e: Exception) {
            return ""
        }
        return text.toString().trim()
    }

    private suspend fun defaultModelIdForProvider(prefix: String): String? {
        val provider = providerRegistry.all().firstOrNull { it.prefix == prefix }
            ?: providerRegistry.configured().firstOrNull()
            ?: return null
        val model = runCatching { provider.listModels().firstOrNull() }.getOrNull()
        return if (model != null) "${provider.prefix}:$model" else null
    }

    private fun postNotification(ctx: Context, title: String, body: String, summary: String) {
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

        val packageName = ctx.packageName
        val mainActivityClass = runCatching {
            Class.forName("$packageName.MainActivity")
        }.getOrNull() ?: android.app.Activity::class.java

        val chatIntent = Intent(ctx, mainActivityClass).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            putExtra("openChat", true)
            putExtra("morningBriefSummary", summary)
        }
        val chatPending = PendingIntent.getActivity(
            ctx, MorningBriefWorker.REQUEST_CODE_CHAT, chatIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val snoozeIntent = Intent(ctx, MorningBriefReceiver::class.java).apply {
            action = MorningBriefWorker.ACTION_SNOOZE
            putExtra("title", title)
            putExtra("body", body)
            putExtra("summary", summary)
        }
        val snoozePending = PendingIntent.getBroadcast(
            ctx, MorningBriefWorker.REQUEST_CODE_SNOOZE, snoozeIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val contentIntent = PendingIntent.getActivity(
            ctx, 0, launchIntent.apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val n = NotificationCompat.Builder(ctx, "aura_morning_brief")
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setSmallIcon(R.drawable.ic_aura_notification)
            .setContentIntent(contentIntent)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .addAction(R.drawable.ic_aura_notification, "Tell me more", chatPending)
            .addAction(R.drawable.ic_aura_notification, "Snooze 1h", snoozePending)
            .build()
        mgr.notify(MorningBriefWorker.MORNING_BRIEF_ID, n)
    }

    companion object {
        const val DECAY_THRESHOLD = 0.2f
    }
}
