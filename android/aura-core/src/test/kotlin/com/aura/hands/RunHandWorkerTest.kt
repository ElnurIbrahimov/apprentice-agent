package com.aura.hands

import android.content.Context
import android.util.Log
import androidx.work.ListenableWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.aura.agent.ToolExecutor
import com.aura.tools.NotificationsTool
import dagger.Lazy
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import org.junit.Test
import kotlin.test.assertEquals

class RunHandWorkerTest {
    @Test
    fun `terminal worker reloads latest hand before scheduling next occurrence`() = runBlocking {
        mockkStatic(Log::class)
        every { Log.d(any(), any()) } returns 0
        every { Log.e(any(), any()) } returns 0
        every { Log.e(any(), any(), any()) } returns 0
        val context = mockk<Context>(relaxed = true)
        val params = mockk<WorkerParameters>(relaxed = true)
        val repository = mockk<HandRepository>()
        val executor = mockk<ToolExecutor>()
        val notifications = mockk<NotificationsTool>(relaxed = true)
        val scheduler = mockk<HandScheduler>(relaxed = true)
        val original = Hand(id = "h1", name = "Morning", enabled = true, scheduleType = "daily")
        val disabledWhileRunning = original.copy(enabled = false)
        every { params.inputData } returns workDataOf(RunHandWorker.KEY_HAND_ID to original.id)
        coEvery { repository.getById(original.id) } returnsMany listOf(original, disabledWhileRunning)
        coEvery {
            repository.run(original, executor, any(), emptyMap(), HandRunTrigger.SCHEDULE.value)
        } returns com.aura.agent.ToolResult.Ok("done")

        val worker = RunHandWorker(
            context,
            params,
            repository,
            Lazy { executor },
            notifications,
            scheduler,
        )

        assertEquals(ListenableWorker.Result.success(), worker.doWork())
        verify { scheduler.scheduleNextAfterRun(disabledWhileRunning, any()) }
        verify(exactly = 0) { scheduler.scheduleNextAfterRun(original, any()) }
    }

    @Test
    fun `terminal hand failure still completes worker so appended schedule survives`() = runBlocking {
        mockkStatic(Log::class)
        every { Log.d(any(), any()) } returns 0
        every { Log.e(any(), any()) } returns 0
        every { Log.e(any(), any(), any()) } returns 0
        val context = mockk<Context>(relaxed = true)
        val params = mockk<WorkerParameters>(relaxed = true)
        val repository = mockk<HandRepository>()
        val executor = mockk<ToolExecutor>()
        val lazyExecutor = Lazy { executor }
        val notifications = mockk<NotificationsTool>(relaxed = true)
        val scheduler = mockk<HandScheduler>(relaxed = true)
        val hand = Hand(id = "h1", name = "Morning", scheduleType = "daily")
        every { params.inputData } returns workDataOf(RunHandWorker.KEY_HAND_ID to hand.id)
        coEvery { repository.getById(hand.id) } returns hand
        coEvery {
            repository.run(hand, executor, any(), emptyMap(), HandRunTrigger.SCHEDULE.value)
        } returns com.aura.agent.ToolResult.Error("step failed", "step_error")

        val worker = RunHandWorker(context, params, repository, lazyExecutor, notifications, scheduler)

        val result = worker.doWork()

        assertEquals(ListenableWorker.Result.success(), result)
        verify { scheduler.scheduleNextAfterRun(hand, any()) }
        verify { notifications.post(match { it.contains("Failed") }, "step failed") }
    }

    @Test
    fun `unexpected runtime failure is reported and next occurrence remains scheduled`() = runBlocking {
        mockkStatic(Log::class)
        every { Log.d(any(), any()) } returns 0
        every { Log.e(any(), any()) } returns 0
        every { Log.e(any(), any(), any()) } returns 0
        val context = mockk<Context>(relaxed = true)
        val params = mockk<WorkerParameters>(relaxed = true)
        val repository = mockk<HandRepository>()
        val executor = mockk<ToolExecutor>()
        val lazyExecutor = Lazy { executor }
        val notifications = mockk<NotificationsTool>(relaxed = true)
        val scheduler = mockk<HandScheduler>(relaxed = true)
        val hand = Hand(id = "h1", name = "Morning", scheduleType = "daily")
        every { params.inputData } returns workDataOf(
            RunHandWorker.KEY_HAND_ID to hand.id,
            RunHandWorker.KEY_TRIGGER to HandRunTrigger.SCHEDULE.value,
        )
        coEvery { repository.getById(hand.id) } returns hand
        coEvery {
            repository.run(hand, executor, any(), emptyMap(), HandRunTrigger.SCHEDULE.value)
        } throws IllegalStateException("boom")

        val worker = RunHandWorker(context, params, repository, lazyExecutor, notifications, scheduler)

        val result = worker.doWork()

        assertEquals(ListenableWorker.Result.success(), result)
        verify { scheduler.scheduleNextAfterRun(hand, any()) }
        verify { notifications.post(match { it.contains("Failed") }, match { it.contains("boom") }) }
    }
}
