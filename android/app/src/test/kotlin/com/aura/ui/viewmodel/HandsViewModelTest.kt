package com.aura.ui.viewmodel

import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.agent.ToolResult
import com.aura.hands.Hand
import com.aura.hands.HandRepository
import com.aura.hands.HandRun
import com.aura.hands.HandRunStatus
import com.aura.hands.HandScheduler
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

@OptIn(ExperimentalCoroutinesApi::class)
class HandsViewModelTest {
    private val dispatcher = UnconfinedTestDispatcher()
    private val repository = mockk<HandRepository>()
    private val executor = mockk<ToolExecutor>()
    private val registry = mockk<ToolRegistry>(relaxed = true)
    private val scheduler = mockk<HandScheduler>(relaxed = true)

    @Before
    fun setup() {
        Dispatchers.setMain(dispatcher)
        coEvery { repository.getAll() } returns emptyList()
        every { repository.observeRecentRuns(any()) } returns flowOf(emptyList())
        every { repository.variablesToJson(any()) } returns "{\"city\":\"Baku\"}"
        every { repository.conditionsToJson(any()) } returns "[]"
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `initial load includes durable run history`() = runTest {
        val hand = Hand("h1", "Morning")
        val run = HandRun(
            id = "r1",
            handId = "h1",
            handName = "Morning",
            trigger = "manual",
            status = HandRunStatus.SUCCESS.value,
        )
        coEvery { repository.getAll() } returns listOf(hand)
        every { repository.observeRecentRuns(100) } returns flowOf(listOf(run))

        val vm = viewModel()

        assertEquals(listOf(hand), vm.state.value.hands)
        assertEquals(listOf(run), vm.state.value.runs)
    }

    @Test
    fun `save persists full automation and schedules it`() = runTest {
        coEvery { repository.insert(any()) } returns Unit
        val vm = viewModel()
        val draft = HandDraft(
            name = "Morning",
            triggerPhrase = "start day",
            stepsJson = "[]",
            variables = mapOf("city" to "Baku"),
            conditions = emptyList(),
            scheduleType = "daily",
            scheduleHour = 8,
            scheduleMinute = 30,
            scheduleDayOfWeek = 1,
        )

        vm.save(null, draft)

        coVerify {
            repository.insert(match {
                it.name == "Morning" && it.variables.contains("Baku") &&
                    it.scheduleType == "daily" && it.scheduleHour == 8
            })
        }
        coVerify { scheduler.schedule(match { it.name == "Morning" }, any()) }
    }

    @Test
    fun `disabling a hand cancels scheduled work`() = runTest {
        val hand = Hand("h1", "Morning", enabled = true, scheduleType = "daily")
        coEvery { repository.update(any()) } returns Unit
        val vm = viewModel()

        vm.toggle(hand)

        coVerify { repository.update(match { !it.enabled }) }
        coVerify { scheduler.cancel("h1") }
    }

    @Test
    fun `delete cancels and removes only the stable hand id`() = runTest {
        val hand = Hand("h1", "Duplicate name")
        coEvery { repository.deleteById(hand.id) } returns Unit
        val vm = viewModel()

        vm.delete(hand)

        coVerify { scheduler.cancel("h1") }
        coVerify { repository.deleteById("h1") }
        coVerify(exactly = 0) { repository.deleteByName(any()) }
    }

    @Test
    fun `manual run passes variable overrides and clears running state`() = runTest {
        val hand = Hand("h1", "Weather")
        coEvery { repository.run(hand, executor, any<ToolContext>(), mapOf("city" to "Tokyo"), "manual") } returns
            ToolResult.Ok("done")
        val vm = viewModel()

        vm.runHand(hand, mapOf("city" to "Tokyo"))

        assertNull(vm.state.value.running)
        assertEquals("done", vm.state.value.lastResult)
    }

    @Test
    fun `clear history delegates and removes visible runs`() = runTest {
        coEvery { repository.deleteRunHistory() } returns Unit
        val vm = viewModel()

        vm.clearHistory()

        coVerify { repository.deleteRunHistory() }
    }

    private fun viewModel() = HandsViewModel(repository, executor, registry, scheduler)
}
