package com.aura.hands

import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import io.mockk.mockk
import io.mockk.verify
import java.time.ZoneId
import java.time.ZonedDateTime
import org.junit.Test

class HandSchedulerPolicyTest {
    private val now = ZonedDateTime.of(2026, 7, 13, 10, 0, 0, 0, ZoneId.of("UTC"))
    private val hand = Hand(
        id = "h1",
        name = "Morning",
        scheduleType = HandScheduleType.DAILY.value,
        scheduleHour = 11,
    )

    @Test
    fun `editor scheduling replaces pending occurrence`() {
        val workManager = mockk<WorkManager>(relaxed = true)
        val scheduler = HandScheduler(workManager)

        scheduler.schedule(hand, now)

        verify {
            workManager.enqueueUniqueWork(
                HandScheduler.uniqueWorkName(hand.id),
                ExistingWorkPolicy.REPLACE,
                any<OneTimeWorkRequest>(),
            )
        }
    }

    @Test
    fun `terminal worker appends next occurrence without cancelling itself`() {
        val workManager = mockk<WorkManager>(relaxed = true)
        val scheduler = HandScheduler(workManager)

        scheduler.scheduleNextAfterRun(hand, now)

        verify {
            workManager.enqueueUniqueWork(
                HandScheduler.uniqueWorkName(hand.id),
                ExistingWorkPolicy.APPEND_OR_REPLACE,
                any<OneTimeWorkRequest>(),
            )
        }
    }
}
