package com.aura.tasks

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals

class ReminderStoreTest {
    private val dao = mockk<ReminderDao>(relaxed = true)
    private val scheduler = mockk<ReminderScheduler>(relaxed = true)
    private val store = ReminderStore(dao, scheduler)

    @Test
    fun `observeUpcoming forwards the dao query`() = runTest {
        val expected = listOf(
            ReminderEntity(id = "a", workId = "wa", message = "call mom", triggerAt = 999999L),
            ReminderEntity(id = "b", workId = "wb", message = "buy milk", triggerAt = 888888L),
        )
        coEvery { dao.observeUpcoming(any()) } returns flowOf(expected)

        val actual = store.observeUpcoming().first()

        coVerify { dao.observeUpcoming(match { it > 0L }) }
        assertEquals(expected, actual)
    }

    @Test
    fun `cancel keeps lifecycle ownership in scheduler`() = runTest {
        store.cancel("reminder-1")

        coVerify(exactly = 1) { scheduler.cancel("reminder-1") }
    }

    @Test
    fun `update preserves identity and reschedules`() = runTest {
        val existing = ReminderEntity(
            id = "reminder-1",
            workId = "old-work",
            message = "old",
            triggerAt = 100L,
            createdAt = 10L,
        )
        coEvery { dao.get("reminder-1") } returns existing
        coEvery { scheduler.schedule(any()) } answers { firstArg() }

        val updated = store.update("reminder-1", "new", 200L, "daily")

        assertEquals("reminder-1", updated?.id)
        assertEquals("new", updated?.message)
        assertEquals("daily", updated?.recurrence)
        assertEquals(10L, updated?.createdAt)
        coVerify { scheduler.schedule(match { it.id == "reminder-1" && it.triggerAt == 200L }) }
    }
}
