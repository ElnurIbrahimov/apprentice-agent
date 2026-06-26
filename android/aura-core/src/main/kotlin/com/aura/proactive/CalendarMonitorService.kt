package com.aura.proactive

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.aura.core.R
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

/**
 * Foreground service that hosts [CalendarMonitor] polling so it survives
 * process death. Calls [startForeground] within 5 seconds of creation and
 * returns [START_STICKY] so the OS restarts the service if the process is
 * killed.
 *
 * Safe to start multiple times — [CalendarMonitor.start] is idempotent.
 */
@AndroidEntryPoint
class CalendarMonitorService : Service() {

    @Inject
    lateinit var calendarMonitor: CalendarMonitor

    override fun onCreate() {
        super.onCreate()
        startForeground(NOTIFICATION_ID, buildNotification())
        calendarMonitor.start()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        calendarMonitor.stop()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun buildNotification(): Notification {
        val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(
                CHANNEL_ID,
                "Calendar Monitor",
                NotificationManager.IMPORTANCE_LOW,
            )
            mgr.createNotificationChannel(ch)
        }
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
            ?: Intent(Intent.ACTION_MAIN).apply { addCategory(Intent.CATEGORY_LAUNCHER) }
        launchIntent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        val pi = PendingIntent.getActivity(
            this, 0, launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Aura")
            .setContentText("Aura is monitoring your calendar")
            .setSmallIcon(R.drawable.ic_aura_notification)
            .setContentIntent(pi)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    companion object {
        const val CHANNEL_ID = "calendar_monitor"
        const val NOTIFICATION_ID = 1002

        /** Convenience method to start this foreground service. */
        fun start(context: Context) {
            val intent = Intent(context, CalendarMonitorService::class.java)
            ContextCompat.startForegroundService(context, intent)
        }
    }
}
