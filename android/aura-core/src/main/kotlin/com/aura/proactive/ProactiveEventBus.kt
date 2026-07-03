package com.aura.proactive

import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Proactive message bus. Monitors and the morning brief push events here;
 * the UI / notification layer subscribes to surface them.
 *
 * Mirrors aura/proactive/event_bus.py.
 */
@Singleton
class ProactiveEventBus @Inject constructor() {

    sealed class Event {
        abstract val timestamp: Long
        data class MorningBriefReady(val title: String, val body: String, override val timestamp: Long = System.currentTimeMillis()) : Event()
        /**
         * Companion event to [MorningBriefReady] carrying the full
         * structured [BriefContext] the brief was built from. The
         * Home screen uses this to render a rich card with the
         * same sections (decayed memories, new facts, tasks,
         * calendar) instead of a freeform paragraph. The two
         * events are emitted together; the structured one is
         * optional (older clients can ignore it).
         */
        data class MorningBriefStructured(
            val context: com.aura.proactive.BriefContext,
            override val timestamp: Long = System.currentTimeMillis(),
        ) : Event()
        data class CalendarEventSoon(val title: String, val minutesUntil: Int, override val timestamp: Long = System.currentTimeMillis()) : Event()
        data class LocationArrived(val placeName: String, val recalledMemories: List<String>, override val timestamp: Long = System.currentTimeMillis()) : Event()
        data class MemoryDecayWarning(val memoryId: String, val preview: String, override val timestamp: Long = System.currentTimeMillis()) : Event()
    }

    private val _events = MutableSharedFlow<Event>(
        replay = 0,
        extraBufferCapacity = 32,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val events: SharedFlow<Event> = _events.asSharedFlow()

    suspend fun emit(event: Event) {
        _events.emit(event)
    }

    fun tryEmit(event: Event) {
        _events.tryEmit(event)
    }
}
