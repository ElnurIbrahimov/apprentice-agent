package com.aura.proactive

import com.aura.data.UserPreferences
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Proactive event store. Owns the latest + history of proactive
 * events and exposes the unread-count for the Home badge.
 *
 * @param scope Internal coroutine scope for `init` and the
 *   unread-count flow. Defaults to a SupervisorJob on
 *   Dispatchers.Default for production. Tests inject
 *   `TestScope`/`StandardTestDispatcher` so the internal launches
 *   drain on `advanceUntilIdle()`.
 */
class ProactiveEvents(
    private val bus: ProactiveEventBus,
    private val dao: ProactiveEventDao,
    private val userPreferences: UserPreferences,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default),
) {
    @Inject constructor(
        bus: ProactiveEventBus,
        dao: ProactiveEventDao,
        userPreferences: UserPreferences,
    ) : this(bus, dao, userPreferences, CoroutineScope(SupervisorJob() + Dispatchers.Default))

    private val _latest = MutableStateFlow<ProactiveEventBus.Event?>(null)
    val latest: StateFlow<ProactiveEventBus.Event?> = _latest.asStateFlow()

    private val _history = MutableStateFlow<List<ProactiveEventBus.Event>>(emptyList())
    val history: StateFlow<List<ProactiveEventBus.Event>> = _history.asStateFlow()

    /**
     * Refresh trigger — bumped after every event write and after
     * [markSeen] so the unread-count flow re-queries Room.
     */
    private val _refreshTick = MutableStateFlow(0L)

    /**
     * Number of proactive events emitted since the user last opened
     * the proactive history screen (or dismissed the Home card).
     * Drives the "📬 N today" badge on Home.
     *
     * Combines three sources:
     *   - the refresh tick (re-fires when a new event lands or after
     *     [markSeen] persists a new last-seen-at)
     *   - the persisted `lastSeenProactiveAt` from DataStore
     *   - the SQL `countSince(lastSeen)` aggregate
     *
     * On a fresh install `lastSeenProactiveAt` defaults to 0L, so
     * the count covers everything since the Unix epoch — which is
     * the right behavior for the first session (the user has never
     * seen any of these).
     */
    val unreadCount: StateFlow<Int> = combine(
        _refreshTick,
        userPreferences.lastSeenProactiveAt,
    ) { _, lastSeenAt -> lastSeenAt }
        .let { combined ->
            // Re-query the SQL count whenever either source changes.
            // Run as a child of [scope] (configurable for tests) so
            // the SQL query executes on the test dispatcher too.
            val countFlow = MutableStateFlow(0)
            scope.launch {
                combined.collect { lastSeenAt ->
                    countFlow.value = runCatching { dao.countSince(lastSeenAt) }.getOrDefault(0)
                }
            }
            countFlow.asStateFlow()
        }

    init {
        // Load persisted history first, then collect new events
        scope.launch {
            val persisted = dao.recent(100)
            _history.value = persisted.mapNotNull { it.toEvent() }
            // Re-run the unread count after loading history (lastSeenAt
            // is already flowing, but the initial value of
            // countFlow is 0 — this is the first "real" computation).
            _refreshTick.value = System.currentTimeMillis()
        }
        scope.launch {
            bus.events.collect { event ->
                _latest.value = event
                _history.value = (_history.value + event).takeLast(100)
                // Persist to Room
                runCatching {
                    dao.insert(event.toEntity())
                }
                // Bump the tick so the unread-count flow re-queries.
                _refreshTick.value = event.timestamp
            }
        }
    }

    fun dismiss() {
        _latest.value = null
    }

    /**
     * Mark all currently-unread proactive events as seen. Called
     * when the user opens the proactive history screen or taps the
     * Home card to view all events. Persists `now` as the new
     * last-seen-at so the next unread count returns 0.
     */
    fun markSeen() {
        scope.launch {
            userPreferences.setLastSeenProactiveAt(System.currentTimeMillis())
            _refreshTick.value = System.currentTimeMillis()
        }
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