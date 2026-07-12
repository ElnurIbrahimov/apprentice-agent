package com.aura.proactive

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import java.util.concurrent.TimeUnit

/**
 * Receiver for morning-brief notification actions.
 *
 * Handles the "Snooze 1h" action by re-enqueueing a one-time
 * work request that will post the same brief again after 60 minutes.
 * The work request reuses the same notification content captured
 * from the original brief; no DB query is needed.
 */
class MorningBriefReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        if (intent.action != MorningBriefWorker.ACTION_SNOOZE) return

        val body = intent.getStringExtra("body") ?: return
        val summary = intent.getStringExtra("summary") ?: return

        val inputData = androidx.work.workDataOf(
            "snooze_body" to body,
            "snooze_summary" to summary,
        )
        val constraints = Constraints.Builder()
            .setRequiresBatteryNotLow(false)
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .build()
        val work = OneTimeWorkRequestBuilder<MorningBriefSnoozeWorker>()
            .setInitialDelay(60, TimeUnit.MINUTES)
            .setInputData(inputData)
            .setConstraints(constraints)
            .build()
        WorkManager.getInstance(ctx)
            .enqueueUniqueWork(
                "morning-brief-snooze",
                ExistingWorkPolicy.REPLACE,
                work,
            )

        // Dismiss the original notification.
        val mgr = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as android.app.NotificationManager
        mgr.cancel(MorningBriefWorker.MORNING_BRIEF_ID)
    }
}
