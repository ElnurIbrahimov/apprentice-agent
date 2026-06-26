package com.aura.tools

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters

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
        // Use the work request ID as notification ID. Keyed by unique work name
        // so each reminder gets its own notification and they don't overwrite
        // each other when fired concurrently.
        val notificationId = inputData.getInt("notificationId", id.hashCode().let {
            if (it < 0) -it else it
        })
        val idInt = if (notificationId == 0) 1 else notificationId
        postNotification(applicationContext, title, body, idInt)
        return Result.success()
    }

    private fun postNotification(ctx: Context, title: String, body: String, id: Int) {
        val mgr = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(
                NotificationsTool.CHANNEL_ID,
                NotificationsTool.CHANNEL_NAME,
                NotificationManager.IMPORTANCE_HIGH,
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
}
