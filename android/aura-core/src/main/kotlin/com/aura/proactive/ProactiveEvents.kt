package com.aura.proactive

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ProactiveEvents @Inject constructor(
    private val bus: ProactiveEventBus,
    private val dao: ProactiveEventDao,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val _latest = MutableStateFlow<ProactiveEventBus.Event?>(null)
    val latest: StateFlow<ProactiveEventBus.Event?> = _latest.asStateFlow()

    private val _history = MutableStateFlow<List<ProactiveEventBus.Event>>(emptyList())
    val history: StateFlow<List<ProactiveEventBus.Event>> = _history.asStateFlow()

    init {
        // Load persisted history first, then collect new events
        scope.launch {
            val persisted = dao.recent(100)
            _history.value = persisted.mapNotNull { it.toEvent() }
        }
        scope.launch {
            bus.events.collect { event ->
                _latest.value = event
                _history.value = (_history.value + event).takeLast(100)
                // Persist to Room
                runCatching {
                    dao.insert(event.toEntity())
                }
            }
        }
    }

    fun dismiss() {
        _latest.value = null
    }

    private fun ProactiveEventEntity.toEvent(): ProactiveEventBus.Event? {
        return when (eventType) {
            "MorningBriefReady" -> ProactiveEventBus.Event.MorningBriefReady(title, body, timestamp)
            "CalendarEventSoon" -> {
                val minutes = body.toIntOrNull() ?: return null
                ProactiveEventBus.Event.CalendarEventSoon(title, minutes, timestamp)
            }
            "LocationArrived" -> ProactiveEventBus.Event.LocationArrived(title, emptyList(), timestamp)
            "MemoryDecayWarning" -> ProactiveEventBus.Event.MemoryDecayWarning("", title, timestamp)
            else -> null
        }
    }

    private fun ProactiveEventBus.Event.toEntity(): ProactiveEventEntity = when (this) {
        is ProactiveEventBus.Event.MorningBriefReady -> ProactiveEventEntity(
            eventType = "MorningBriefReady", title = title, body = body, timestamp = timestamp,
        )
        is ProactiveEventBus.Event.CalendarEventSoon -> ProactiveEventEntity(
            eventType = "CalendarEventSoon", title = title, body = minutesUntil.toString(), timestamp = timestamp,
        )
        is ProactiveEventBus.Event.LocationArrived -> ProactiveEventEntity(
            eventType = "LocationArrived", title = placeName, body = "", timestamp = timestamp,
        )
        is ProactiveEventBus.Event.MemoryDecayWarning -> ProactiveEventEntity(
            eventType = "MemoryDecayWarning", title = memoryId, body = preview, timestamp = timestamp,
        )
    }
}
