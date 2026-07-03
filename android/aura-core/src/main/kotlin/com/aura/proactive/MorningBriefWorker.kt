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
import com.aura.agent.Conversation
import com.aura.core.R
import com.aura.kg.KnowledgeGraphRepository
import com.aura.memory.MemoryStore
import com.aura.providers.ChatOptions
import com.aura.providers.ProviderRegistry
import com.aura.tasks.TaskDao
import com.aura.tools.CalendarReadTool
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope

/**
 * WorkManager job that runs at ~7am local time and posts a morning
 * brief based on the user's recent state. Mirrors
 * `aura/hands/morning_briefing.py` + `aura/proactive/`.
 *
 * The brief is built in two passes:
 *
 *   1. **Structured context** — query the memories, knowledge graph,
 *      tasks, and calendar in parallel and assemble a [BriefContext].
 *      The brief shows a deterministic one-line summary per
 *      non-empty section (decayed memories, new facts, tasks due
 *      today, calendar) BEFORE the LLM is asked to write anything.
 *      The LLM only writes the greeting + opener.
 *
 *   2. **LLM greeting** — if the structured context has any
 *      non-empty section, send the context as a structured prompt
 *      and let the LLM write a 1-2 line warm opener that references
 *      the user's name + today's date. The opener + the
 *      deterministic summary are concatenated for the notification
 *      body.
 *
 * Both passes emit events on the proactive bus:
 *   - [ProactiveEventBus.Event.MorningBriefReady] — the LLM
 *     greeting + a one-line summary, used by the notification +
 *     the legacy "show the body string" UI.
 *   - [ProactiveEventBus.Event.MorningBriefStructured] — the
 *     full [BriefContext], used by the Home screen to render a
 *     rich card with all five sections.
 */
@HiltWorker
class MorningBriefWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
    private val providerRegistry: ProviderRegistry,
    private val memoryStore: MemoryStore,
    private val taskDao: TaskDao,
    private val kgRepository: KnowledgeGraphRepository,
    private val calendarReadTool: CalendarReadTool,
    private val eventBus: ProactiveEventBus,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        // 1) Build the structured context in parallel. Each query
        // is best-effort: if any one fails, the section is empty
        // and the brief still ships.
        val now = System.currentTimeMillis()
        val since24h = now - 24L * 60L * 60L * 1000L
        // "Today" = local-time midnight to next midnight. Calendar
        // (a java.util.Calendar) gives us the local TZ-aware window.
        val cal = java.util.Calendar.getInstance()
        cal.set(java.util.Calendar.HOUR_OF_DAY, 0)
        cal.set(java.util.Calendar.MINUTE, 0)
        cal.set(java.util.Calendar.SECOND, 0)
        cal.set(java.util.Calendar.MILLISECOND, 0)
        val startOfDay = cal.timeInMillis
        val endOfDay = startOfDay + 24L * 60L * 60L * 1000L

        val context = runCatching {
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

        // 2) Build the deterministic summary from the structured
        // context. The notification body is this summary plus an
        // optional LLM greeting on top.
        val summary = buildSummary(context, now)

        // 3) If there's anything to say, ask the LLM for a 1-2 line
        // warm opener referencing the user's name. Otherwise emit
        // the deterministic summary alone.
        val greeting = if (context.isEmpty) "" else runCatching {
            llmGreeting(now)
        }.getOrDefault("").trim()

        val notificationTitle = "☀️ Good morning"
        val notificationBody = listOf(greeting, summary)
            .filter { it.isNotBlank() }
            .joinToString(separator = "\n\n")

        if (notificationBody.isBlank()) {
            // Nothing to say today (no memories, no events, no
            // anything). Don't spam a notification.
            return Result.success()
        }

        // 4) Emit BOTH events: the legacy string body for the
        // notification + HomeScreen's existing "show latest event"
        // path, AND the structured BriefContext for the new
        // rich-card surface.
        eventBus.emit(ProactiveEventBus.Event.MorningBriefReady(notificationTitle, notificationBody))
        eventBus.emit(ProactiveEventBus.Event.MorningBriefStructured(context))

        // 5) Post the notification.
        postNotification(applicationContext, notificationTitle, notificationBody)
        return Result.success()
    }

    /**
     * Build the deterministic one-line-per-section summary. Used
     * as the notification body (and as the bottom of the Home
     * card) when the LLM is unavailable or returns nothing useful.
     */
    private fun buildSummary(context: BriefContext, now: Long): String {
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

    /**
     * Ask the configured LLM (MoA first, then first solo provider)
     * for a 1-2 line warm opener. The prompt is now minimal — the
     * structured context is the body of the brief, not the LLM's
     * imagination. Returns "" on any error so the deterministic
     * summary still ships.
     */
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
            // Network blip, 5xx, etc. Fall back to the deterministic
            // summary alone.
            return ""
        }
        return text.toString().trim()
    }

    /**
     * Build a model id the provider can serve. SettingsViewModel
     * persists a default in DataStore, but the worker doesn't
     * depend on that state — it asks the provider for its model
     * list and picks the first.
     */
    private fun defaultModelIdForProvider(prefix: String): String =
        when (prefix) {
            "ollama" -> "ollama:deepseek-v4-pro:cloud"
            "anthropic" -> "anthropic:claude-sonnet-4-5"
            "openai" -> "openai:gpt-4.1"
            "deepseek" -> "deepseek:deepseek-chat"
            "gemini" -> "gemini:gemini-1.5-flash"
            else -> "ollama:deepseek-v4-pro:cloud"
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

        /**
         * Memories at or below this decay score count as "fading"
         * for the morning brief. Matches the threshold used by
         * the [com.aura.memory.MemoryStore.runDecayPass] warning
         * system.
         */
        private const val DECAY_THRESHOLD = 0.2f
    }
}
