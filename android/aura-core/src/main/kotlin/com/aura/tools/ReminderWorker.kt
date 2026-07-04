package com.aura.tools

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.work.WorkerParameters
import androidx.work.CoroutineWorker
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

/**
 * WorkManager worker that posts the reminder notification when the
 * scheduled time fires. CoroutineWorker so we could fetch memory
 * context in v1.5.
 */
class ReminderWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val title = inputData.getString("title") ?: "\u23F0 Reminder"
        val body = inputData.getString("body") ?: ""
        // Use the stable notificationId that SetReminderTool reserved for us.
        // If it is missing, fall back to a deterministic range derived from
        // the work request id hash so the worst case is still a collision
        // within the reminder namespace, not with morning-brief / proactive.
        val notificationId = inputData.getInt("notificationId", id.hashCode().let {
            val h = if (it < 0) -it else it
            (h % REMINDER_ID_RANGE) + REMINDER_ID_START
        })
        postNotification(applicationContext, title, body, notificationId)
        return Result.success()
    }

    private fun postNotification(ctx: Context, title: String, body: String, id: Int) {
        val mgr = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as android.app.NotificationManager
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            val ch = android.app.NotificationChannel(
                NotificationsTool.CHANNEL_ID,
                NotificationsTool.CHANNEL_NAME,
                android.app.NotificationManager.IMPORTANCE_HIGH,
            )
            mgr.createNotificationChannel(ch)
        }
        val launchIntent = ctx.packageManager.getLaunchIntentForPackage(ctx.packageName)
            ?: Intent(Intent.ACTION_MAIN).apply { addCategory(Intent.CATEGORY_LAUNCHER) }
        launchIntent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        val pi = android.app.PendingIntent.getActivity(
            ctx, 0, launchIntent,
            android.app.PendingIntent.FLAG_UPDATE_CURRENT or android.app.PendingIntent.FLAG_IMMUTABLE
        )
        val n = NotificationCompat.Builder(ctx, NotificationsTool.CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(body)
            .setSmallIcon(com.aura.core.R.drawable.ic_aura_notification)
            .setContentIntent(pi)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()
        mgr.notify(id, n)
    }

    companion object {
        /** Fallback notification ID namespace when inputData misses the reserved id. */
        private const val REMINDER_ID_START = 10_000
        private const val REMINDER_ID_RANGE = 90_000
    }
}
