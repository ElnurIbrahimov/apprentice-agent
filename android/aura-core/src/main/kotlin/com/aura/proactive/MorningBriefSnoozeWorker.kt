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
import com.aura.core.R
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject

/**
 * One-time snooze worker that re-posts the captured morning brief
 * content after 60 minutes.
 */
@HiltWorker
class MorningBriefSnoozeWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted params: WorkerParameters,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val title = inputData.getString("snooze_title") ?: return Result.failure()
        val body = inputData.getString("snooze_body") ?: return Result.failure()
        val summary = inputData.getString("snooze_summary") ?: return Result.failure()

        val ctx = applicationContext
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

        val contentIntent = PendingIntent.getActivity(
            ctx, 0, launchIntent.apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val n = NotificationCompat.Builder(ctx, "aura_morning_brief")
            .setContentTitle("☀️ Morning brief (snoozed)")
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setSmallIcon(R.drawable.ic_aura_notification)
            .setContentIntent(contentIntent)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .addAction(R.drawable.ic_aura_notification, "Tell me more", chatPending)
            .build()
        mgr.notify(MorningBriefWorker.MORNING_BRIEF_ID, n)
        return Result.success()
    }
}
