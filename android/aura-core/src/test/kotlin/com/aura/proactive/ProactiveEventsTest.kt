package com.aura.proactive

import com.aura.data.UserPreferences
import io.mockk.Runs
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for the proactive unread-count surface (item #1 of the
 * user-facing polish list).
 *
 * The contract:
 *   - unreadCount reflects "how many proactive events have fired
 *     since the user last opened the history screen"
 *   - markSeen() updates lastSeenProactiveAt so the next unreadCount
 *     is 0
 *   - a new event landed via the bus increments the count
 *
 * These tests use a fake DAO (ProactiveEventDao is a small
 * interface — faking is the cleanest path; no need for an
 * in-memory Room DB which is androidTest-only). The fake mirrors
 * the SQL `countSince(timestamp > since)` aggregate.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ProactiveEventsTest {

    private val testDispatcher = StandardTestDispatcher()

    private lateinit var bus: ProactiveEventBus
    private lateinit var dao: FakeProactiveEventDao
    private lateinit var userPreferences: UserPreferences
    private val lastSeenFlow = MutableStateFlow(0L)

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        bus = ProactiveEventBus()
        dao = FakeProactiveEventDao()
        userPreferences = mockk(relaxed = true)
        every { userPreferences.lastSeenProactiveAt } returns lastSeenFlow
        // side-effect stub: when the SUT calls setLastSeenProactiveAt,
        // mirror the change into lastSeenFlow so the StateFlow
        // re-emits and the unreadCount reflows.
        coEvery { userPreferences.setLastSeenProactiveAt(any()) } answers {
            lastSeenFlow.value = firstArg()
            Unit
        }
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private fun newProactiveEvents(): ProactiveEvents =
        ProactiveEvents(
            bus = bus,
            dao = dao,
            userPreferences = userPreferences,
            // Test-controlled scope: drain on advanceUntilIdle().
            scope = CoroutineScope(testDispatcher + kotlinx.coroutines.SupervisorJob()),
        )

    @Test
    fun `unreadCount starts at 0 on a fresh install with no events`() = runTest(testDispatcher) {
        val events = newProactiveEvents()
        advanceUntilIdle()
        assertEquals(0, events.unreadCount.value)
    }

    @Test
    fun `unreadCount increments when a new event is emitted via the bus`() = runTest(testDispatcher) {
        val events = newProactiveEvents()
        advanceUntilIdle()
        assertEquals(0, events.unreadCount.value)

        bus.tryEmit(
            ProactiveEventBus.Event.MorningBriefReady("☀️", "today's brief", timestamp = 1_000L),
        )
        advanceUntilIdle()

        assertEquals(1, events.unreadCount.value)
    }

    @Test
    fun `markSeen zeros the unreadCount and persists lastSeenProactiveAt`() = runTest(testDispatcher) {
        val events = newProactiveEvents()
        advanceUntilIdle()

        // Land 2 events.
        bus.tryEmit(
            ProactiveEventBus.Event.MorningBriefReady("☀️", "brief", timestamp = 1_000L),
        )
        bus.tryEmit(
            ProactiveEventBus.Event.CalendarEventSoon("Standup", 15, timestamp = 2_000L),
        )
        advanceUntilIdle()
        assertEquals(2, events.unreadCount.value)

        // User opens the history screen.
        events.markSeen()
        advanceUntilIdle()

        assertEquals(0, events.unreadCount.value)
        coVerify { userPreferences.setLastSeenProactiveAt(any()) }
    }

    @Test
    fun `unreadCount only counts events newer than lastSeenProactiveAt`() = runTest(testDispatcher) {
        val events = newProactiveEvents()
        advanceUntilIdle()

        // Land 3 events at t=1k, 2k, 3k.
        bus.tryEmit(
            ProactiveEventBus.Event.MorningBriefReady("☀️", "a", timestamp = 1_000L),
        )
        bus.tryEmit(
            ProactiveEventBus.Event.CalendarEventSoon("B", 5, timestamp = 2_000L),
        )
        bus.tryEmit(
            ProactiveEventBus.Event.LocationArrived("Office", emptyList(), timestamp = 3_000L),
        )
        advanceUntilIdle()
        assertEquals(3, events.unreadCount.value)

        // User marks seen at t=2k — only the t=3k event is unread.
        lastSeenFlow.value = 2_000L
        advanceUntilIdle()
        assertEquals(1, events.unreadCount.value)

        // A new event at t=4k lands — unread goes back to 2.
        bus.tryEmit(
            ProactiveEventBus.Event.MemoryDecayWarning("mem-1", "fading", timestamp = 4_000L),
        )
        advanceUntilIdle()
        assertEquals(2, events.unreadCount.value)
    }

    @Test
    fun `markSeen is idempotent and never increases the count`() = runTest(testDispatcher) {
        val events = newProactiveEvents()
        advanceUntilIdle()

        events.markSeen()
        events.markSeen()
        advanceUntilIdle()

        assertEquals(0, events.unreadCount.value)
        assertTrue(events.unreadCount.value == 0)
    }
}

/**
 * In-process fake of [ProactiveEventDao] for unit tests. Mirrors the
 * SQL semantics of the real queries (`recent(limit)` ordering by
 * timestamp DESC, `countSince(since)` over timestamp > since).
 */
private class FakeProactiveEventDao : ProactiveEventDao {
    private val rows = mutableListOf<ProactiveEventEntity>()

    override suspend fun insert(event: ProactiveEventEntity): Long {
        val id = (rows.maxOfOrNull { it.id } ?: 0L) + 1L
        rows.add(event.copy(id = id))
        return id
    }

    override suspend fun recent(limit: Int): List<ProactiveEventEntity> =
        rows.sortedByDescending { it.timestamp }.take(limit)

    override suspend fun countSince(since: Long): Int =
        rows.count { it.timestamp > since }
}

