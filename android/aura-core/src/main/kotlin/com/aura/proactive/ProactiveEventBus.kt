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
        data class MorningBriefReady(val title: String, val body: String) : Event()
        data class CalendarEventSoon(val title: String, val minutesUntil: Int) : Event()
        data class LocationArrived(val placeName: String, val recalledMemories: List<String>) : Event()
        data class MemoryDecayWarning(val memoryId: String, val preview: String) : Event()
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
