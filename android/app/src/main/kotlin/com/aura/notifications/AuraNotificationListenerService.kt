package com.aura.notifications

import android.app.Notification
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

/**
 * Opt-in bridge for device notifications. Android only binds this service after
 * the user grants Notification access in system settings.
 */
@AndroidEntryPoint
class AuraNotificationListenerService : NotificationListenerService() {

    @Inject lateinit var store: NotificationCaptureStore

    override fun onListenerConnected() {
        super.onListenerConnected()
        store.setConnected(true)
        val active = runCatching { activeNotifications.orEmpty().map(::capture) }
            .getOrDefault(emptyList())
        store.replaceAll(active)
    }

    override fun onListenerDisconnected() {
        store.setConnected(false)
        super.onListenerDisconnected()
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        sbn ?: return
        store.upsert(capture(sbn))
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification?) {
        sbn ?: return
        store.remove(sbn.key)
    }

    override fun onDestroy() {
        store.setConnected(false)
        super.onDestroy()
    }

    private fun capture(sbn: StatusBarNotification): CapturedNotification {
        val extras = sbn.notification.extras
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString().orEmpty()
        val text = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()
            ?: extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()
            ?: extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES)
                ?.joinToString(" ") { it.toString() }
            ?: ""
        return CapturedNotification(
            key = sbn.key,
            packageName = sbn.packageName,
            title = title,
            text = text,
            postedAt = sbn.postTime,
        )
    }
}
