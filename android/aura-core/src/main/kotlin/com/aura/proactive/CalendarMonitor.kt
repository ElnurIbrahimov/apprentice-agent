package com.aura.proactive

import android.content.Context
import android.provider.CalendarContract
import androidx.core.content.ContextCompat
import android.Manifest
import android.content.pm.PackageManager
import com.aura.proactive.ProactiveEventBus.Event.CalendarEventSoon
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.Calendar
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Polls the device calendar every 5 minutes. For each event starting
 * within the next 15 minutes that we haven't already surfaced, emits
 * a CalendarEventSoon event on the proactive bus. The notification
 * handler is a separate concern.
 *
 * Mirrors aura/consciousness/proactive_awareness.py + aura/proactive/monitors/.
 */
@Singleton
class CalendarMonitor @Inject constructor(
    @ApplicationContext private val context: Context,
    private val eventBus: ProactiveEventBus,
) {
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var pollJob: Job? = null
    private val _running = MutableStateFlow(false)
    val running: StateFlow<Boolean> = _running.asStateFlow()

    /**
     * IDs of events we've already emitted for, so we don't fire the same
     * notification twice. Bounded FIFO: when we exceed the cap we drop the
     * oldest entry. This trades a tiny chance of a duplicate emission (when
     * an event ID re-appears after a long gap) for bounded memory.
     */
    private val surfaced: LinkedHashSet<Long> = LinkedHashSet()

    fun start() {
        if (pollJob?.isActive == true) return
        _running.value = true
        pollJob = scope.launch {
            while (true) {
                poll()
                delay(POLL_INTERVAL_MS)
            }
        }
    }

    fun stop() {
        pollJob?.cancel()
        pollJob = null
        _running.value = false
    }

    private suspend fun poll() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CALENDAR)
            != PackageManager.PERMISSION_GRANTED) return
        val now = System.currentTimeMillis()
        val fifteenMinLater = now + 15L * 60L * 1000L
        val projection = arrayOf(
            CalendarContract.Events._ID,
            CalendarContract.Events.TITLE,
            CalendarContract.Events.DTSTART,
        )
        val selection = "${CalendarContract.Events.DTSTART} >= ? AND ${CalendarContract.Events.DTSTART} <= ?"
        val args = arrayOf(now.toString(), fifteenMinLater.toString())
        try {
            context.contentResolver.query(
                CalendarContract.Events.CONTENT_URI, projection, selection, args, null
            )?.use { c ->
                while (c.moveToNext()) {
                    val id = c.getLong(0)
                    val title = c.getString(1) ?: "(no title)"
                    val start = c.getLong(2)
                    if (id in surfaced) continue
                    if (surfaced.size >= MAX_SURFACED_IDS) {
                        // LinkedHashSet iteration order is insertion order; remove
                        // the oldest entry by evicting the first iterator element.
                        surfaced.iterator().next().let { surfaced.remove(it) }
                    }
                    surfaced.add(id)
                    val minutesUntil = ((start - now) / 60_000L).toInt().coerceAtLeast(0)
                    eventBus.emit(CalendarEventSoon(title, minutesUntil))
                }
            }
        } catch (e: SecurityException) {
            // permission revoked
        } catch (e: Exception) {
            // ignore
        }
    }

    companion object {
        const val POLL_INTERVAL_MS = 5L * 60L * 1000L
        // Cap surfaced IDs at ~10k. At 5-min polling each event ID re-appears
        // every poll only if it stays in the next-15-min window, so the
        // practical hit rate is well under the cap.
        const val MAX_SURFACED_IDS = 10_000
    }
}
