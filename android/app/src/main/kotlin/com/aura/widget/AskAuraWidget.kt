package com.aura.widget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.os.Build
import android.widget.RemoteViews
import com.aura.MainActivity
import com.aura.R
import com.aura.data.UserPreferences
import com.aura.memory.MemoryStore
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Home-screen widget. Shows a quick-ask button + the most recent
 * memory. Tapping the widget body opens Aura; tapping the Ask
 * button does the same (the current ChatScreen handles a fresh
 * conversation flow).
 *
 * Update cadence: 30 minutes (set in widget_ask_aura_info.xml's
 * updatePeriodMillis). On every update we read the most-recent
 * memory and update the body text. We also update on
 * APPWIDGET_ENABLED (when the user adds the widget) and on
 * APPWIDGET_UPDATE broadcasts.
 *
 * Hilt injection: the AppWidgetProvider is created by the system,
 * not Hilt, so we use [EntryPointAccessors] to pull [MemoryStore]
 * and [UserPreferences] out of the SingletonComponent.
 */
class AskAuraWidget : AppWidgetProvider() {

    @EntryPoint
    @InstallIn(SingletonComponent::class)
    interface WidgetEntryPoint {
        fun memoryStore(): MemoryStore
        fun userPreferences(): UserPreferences
    }

    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray,
    ) {
        refreshWidgets(context, appWidgetManager, appWidgetIds)
    }

    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context, intent)
        // Listen for the broadcast the [ProactiveBootstrap] sends on
        // cold start so the widget body stays current even if the
        // system hasn't ticked over 30 minutes yet.
        if (intent.action == com.aura.proactive.ProactiveBootstrap.ACTION_REFRESH_WIDGET) {
            val mgr = AppWidgetManager.getInstance(context)
            val component = ComponentName(context, AskAuraWidget::class.java)
            val ids = mgr.getAppWidgetIds(component)
            if (ids.isNotEmpty()) {
                refreshWidgets(context, mgr, ids)
            }
        }
    }

    private fun refreshWidgets(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray,
    ) {
        // Trigger a refresh of every widget instance. We delegate to
        // a coroutine that reads the most recent memory and updates
        // the RemoteViews. The widget itself isn't long-lived so we
        // launch on a process scope.
        val entry = EntryPointAccessors.fromApplication(
            context.applicationContext,
            WidgetEntryPoint::class.java,
        )
        val memoryStore = entry.memoryStore()

        CoroutineScope(Dispatchers.IO).launch {
            val recent = try {
                memoryStore.recent(1).firstOrNull()
            } catch (e: Exception) {
                null
            }
            withContext(Dispatchers.Main) {
                for (id in appWidgetIds) {
                    updateOne(context, appWidgetManager, id, recent?.content)
                }
            }
        }
    }

    private fun updateOne(
        context: Context,
        mgr: AppWidgetManager,
        widgetId: Int,
        memoryContent: String?,
    ) {
        val views = RemoteViews(context.packageName, R.layout.widget_ask_aura)

        val bodyText = if (memoryContent.isNullOrBlank()) {
            "No memories yet — start a chat to teach Aura about you."
        } else {
            memoryContent.take(180)
        }
        views.setTextViewText(R.id.widget_memory_text, bodyText)

        // Tapping the widget body opens Aura.
        val openIntent = Intent(context, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        val openPending = PendingIntent.getActivity(
            context, 0, openIntent,
            pendingFlags(),
        )
        views.setOnClickPendingIntent(R.id.widget_root, openPending)

        // The Ask button opens QuickAskActivity — a lightweight
        // transparent activity with a text field for asking a
        // question without opening the full app.
        val askIntent = Intent(context, QuickAskActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        val askPending = PendingIntent.getActivity(
            context, 1, askIntent,
            pendingFlags(),
        )
        views.setOnClickPendingIntent(R.id.widget_ask_button, askPending)

        mgr.updateAppWidget(widgetId, views)
    }

    private fun pendingFlags(): Int {
        // FLAG_IMMUTABLE is required on Android 12+; FLAG_UPDATE_CURRENT
        // is recommended so the extras on the intent get refreshed on
        // every update.
        val base = PendingIntent.FLAG_UPDATE_CURRENT
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            base or PendingIntent.FLAG_IMMUTABLE
        } else {
            base
        }
    }

    companion object {
        /**
         * Broadcast a refresh to all widget instances. Call from
         * [com.aura.proactive.ProactiveBootstrap] or whenever the
         * memory store is updated so the widget body stays current
         * (independent of the 30-minute system refresh).
         */
        fun requestRefresh(context: Context) {
            val mgr = AppWidgetManager.getInstance(context)
            val component = ComponentName(context, AskAuraWidget::class.java)
            val ids = mgr.getAppWidgetIds(component)
            if (ids.isEmpty()) return
            val intent = Intent(context, AskAuraWidget::class.java).apply {
                action = AppWidgetManager.ACTION_APPWIDGET_UPDATE
                putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, ids)
            }
            context.sendBroadcast(intent)
        }
    }
}
