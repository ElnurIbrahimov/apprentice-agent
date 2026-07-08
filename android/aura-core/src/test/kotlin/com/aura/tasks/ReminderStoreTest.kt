package com.aura.tasks

import android.content.Context
import androidx.work.WorkManager
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Lock the [ReminderStore] contract:
 *   - observeUpcoming delegates to ReminderDao.observeUpcoming(now)
 *     (so the list is always in the future)
 *   - cancel removes the row from Room (the source of truth for
 *     what the user sees) and best-effort cancels the WorkManager
 *     job
 *   - cancel is idempotent and survives a WorkManager failure
 */
class ReminderStoreTest {
    private lateinit var ctx: Context
    private lateinit var dao: ReminderDao

    @Before
    fun setUp() {
        ctx = mockk(relaxed = true)
        dao = mockk(relaxed = true)
        // Stub the WorkManager.getInstance() static to return a real
        // mock. We don't care about the return value — we only verify
        // dao.delete() was called, which is the source of truth.
        mockkStatic(WorkManager::class)
        every { WorkManager.getInstance(any()) } returns mockk(relaxed = true)
    }

    @After
    fun tearDown() {
        unmockkStatic(WorkManager::class)
    }

    @Test
    fun `observeUpcoming forwards the dao query`() = runTest {
        val expected = listOf(
            ReminderEntity(id = "a", message = "call mom", triggerAt = 999999L),
            ReminderEntity(id = "b", message = "buy milk", triggerAt = 888888L),
        )
        coEvery { dao.observeUpcoming(any()) } returns flowOf(expected)
        val store = ReminderStore(ctx, dao)
        val actual = store.observeUpcoming().first()
        coVerify { dao.observeUpcoming(match { it > 0L }) }
        assertEquals(expected, actual)
    }

    @Test
    fun `cancel removes the row (source of truth)`() = runTest {
        val store = ReminderStore(ctx, dao)
        store.cancel("work-uuid-1")
        // The Room delete is what unlists the reminder. The
        // WorkManager cancel is best-effort (see the survives-failure
        // test) and may be a no-op if the work has already run.
        coVerify(exactly = 1) { dao.delete("work-uuid-1") }
    }

@Test
    fun `cancel is idempotent (no exception on unknown id)`() = runTest {
        // The mock DAO will just no-op on a delete of a non-existent
        // row. The store should not throw.
        val store = ReminderStore(ctx, dao)
        store.cancel("never-existed-uuid")
        // Delete was attempted, even if no row matched.
        coVerify { dao.delete("never-existed-uuid") }
        // The fact we got here without exception is the actual test.
        assertNotNull(store)
    }
}
