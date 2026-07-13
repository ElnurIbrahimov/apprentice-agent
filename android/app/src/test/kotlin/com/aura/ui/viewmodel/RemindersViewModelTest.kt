package com.aura.ui.viewmodel

import com.aura.tasks.ReminderEntity
import com.aura.tasks.ReminderStore
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals

@OptIn(ExperimentalCoroutinesApi::class)
class RemindersViewModelTest {
    private val store = mockk<ReminderStore>(relaxed = true)
    private val upcoming = MutableStateFlow(emptyList<ReminderEntity>())
    private val history = MutableStateFlow(emptyList<ReminderEntity>())

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        every { store.observeUpcoming() } returns upcoming
        every { store.observeHistory() } returns history
    }

    @After
    fun tearDown() = Dispatchers.resetMain()

    @Test
    fun `combines upcoming and history into one lifecycle state`() = runTest {
        val vm = RemindersViewModel(store)
        val scheduled = reminder("scheduled", "r1")
        val fired = reminder("fired", "r2")

        upcoming.value = listOf(scheduled)
        history.value = listOf(fired)

        assertEquals(listOf(scheduled), vm.state.value.upcoming)
        assertEquals(listOf(fired), vm.state.value.history)
        assertEquals(false, vm.state.value.loading)
    }

    @Test
    fun `edit and cancel delegate to lifecycle store`() = runTest {
        val vm = RemindersViewModel(store)

        vm.update("r1", "new", 2_000L, "daily")
        vm.cancel("r1")

        coVerify { store.update("r1", "new", 2_000L, "daily") }
        coVerify { store.cancel("r1") }
    }

    private fun reminder(status: String, id: String) = ReminderEntity(
        id = id,
        workId = "work-$id",
        message = id,
        triggerAt = 10_000L,
        status = status,
    )
}
