package com.aura.hands

import android.content.Context
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import dagger.hilt.android.qualifiers.ApplicationContext
import java.time.DayOfWeek
import java.time.Duration
import java.time.ZonedDateTime
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/** Schedules exactly one next occurrence per hand using unique WorkManager work. */
@Singleton
class HandScheduler internal constructor(
    private val workManager: WorkManager,
) {
    @Inject
    constructor(@ApplicationContext context: Context) : this(WorkManager.getInstance(context))

    /** Editor/toggle/restore path: replace any pending occurrence with fresh configuration. */
    fun schedule(hand: Hand, now: ZonedDateTime = ZonedDateTime.now()): Long? =
        enqueue(hand, now, ExistingWorkPolicy.REPLACE)

    /** Worker terminal path: append without replacing—and cancelling—the running worker itself. */
    fun scheduleNextAfterRun(hand: Hand, now: ZonedDateTime = ZonedDateTime.now()): Long? =
        enqueue(hand, now, ExistingWorkPolicy.APPEND_OR_REPLACE)

    private fun enqueue(
        hand: Hand,
        now: ZonedDateTime,
        policy: ExistingWorkPolicy,
    ): Long? {
        val uniqueName = uniqueWorkName(hand.id)
        val next = nextRunAt(hand, now)
        if (next == null) {
            workManager.cancelUniqueWork(uniqueName)
            return null
        }
        val delayMs = Duration.between(now, next).toMillis().coerceAtLeast(0L)
        val request = OneTimeWorkRequestBuilder<RunHandWorker>()
            .setInputData(
                Data.Builder()
                    .putString(RunHandWorker.KEY_HAND_ID, hand.id)
                    .putString(RunHandWorker.KEY_TRIGGER, HandRunTrigger.SCHEDULE.value)
                    .build(),
            )
            .setInitialDelay(delayMs, TimeUnit.MILLISECONDS)
            .addTag(HAND_SCHEDULE_TAG)
            .build()
        workManager.enqueueUniqueWork(uniqueName, policy, request)
        return next.toInstant().toEpochMilli()
    }

    fun cancel(handId: String) {
        workManager.cancelUniqueWork(uniqueWorkName(handId))
    }

    companion object {
        private const val HAND_SCHEDULE_TAG = "aura-hand-schedule"

        internal fun uniqueWorkName(handId: String): String = "aura-hand-schedule-$handId"

        fun nextRunAt(hand: Hand, now: ZonedDateTime = ZonedDateTime.now()): ZonedDateTime? {
            if (!hand.enabled) return null
            val type = HandScheduleType.from(hand.scheduleType)
            if (type == HandScheduleType.NONE) return null
            val hour = hand.scheduleHour.coerceIn(0, 23)
            val minute = hand.scheduleMinute.coerceIn(0, 59)
            val base = now.withHour(hour).withMinute(minute).withSecond(0).withNano(0)

            return when (type) {
                HandScheduleType.NONE -> null
                HandScheduleType.DAILY -> if (base.isAfter(now)) base else base.plusDays(1)
                HandScheduleType.WEEKDAYS -> {
                    var candidate = if (base.isAfter(now)) base else base.plusDays(1)
                    while (candidate.dayOfWeek == DayOfWeek.SATURDAY ||
                        candidate.dayOfWeek == DayOfWeek.SUNDAY
                    ) {
                        candidate = candidate.plusDays(1)
                    }
                    candidate
                }
                HandScheduleType.WEEKLY -> {
                    val target = runCatching { DayOfWeek.of(hand.scheduleDayOfWeek) }
                        .getOrDefault(DayOfWeek.MONDAY)
                    var days = (target.value - now.dayOfWeek.value + 7) % 7
                    if (days == 0 && !base.isAfter(now)) days = 7
                    base.plusDays(days.toLong())
                }
            }
        }
    }
}
