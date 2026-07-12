package com.aura.notifications

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import javax.inject.Inject
import javax.inject.Singleton

/** A privacy-sensitive notification snapshot captured by the opt-in listener. */
data class CapturedNotification(
    val key: String,
    val packageName: String,
    val title: String,
    val text: String,
    val postedAt: Long,
)

/**
 * Process-local bridge between Android's NotificationListenerService and the
 * agent tool. The system owns listener persistence; Aura keeps only currently
 * active notifications in memory and never writes their contents to Room.
 */
@Singleton
class NotificationCaptureStore @Inject constructor() {
    private val _connected = MutableStateFlow(false)
    val connected: StateFlow<Boolean> = _connected.asStateFlow()

    private val _notifications = MutableStateFlow<List<CapturedNotification>>(emptyList())

    fun setConnected(value: Boolean) {
        _connected.value = value
        if (!value) clear()
    }

    fun replaceAll(rows: List<CapturedNotification>) {
        _notifications.value = rows
            .distinctBy { it.key }
            .sortedByDescending { it.postedAt }
            .take(MAX_NOTIFICATIONS)
    }

    fun upsert(row: CapturedNotification) {
        _notifications.update { existing ->
            (existing.filterNot { it.key == row.key } + row)
                .sortedByDescending { it.postedAt }
                .take(MAX_NOTIFICATIONS)
        }
    }

    fun remove(key: String) {
        _notifications.update { rows -> rows.filterNot { it.key == key } }
    }

    fun clear() {
        _notifications.value = emptyList()
    }

    fun snapshot(limit: Int): List<CapturedNotification> =
        _notifications.value.take(limit.coerceIn(1, MAX_TOOL_RESULTS)).toList()

    private companion object {
        const val MAX_NOTIFICATIONS = 200
        const val MAX_TOOL_RESULTS = 50
    }
}
