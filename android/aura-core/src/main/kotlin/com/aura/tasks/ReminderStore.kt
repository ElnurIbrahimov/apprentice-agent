package com.aura.tasks

import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

/** User-facing reminder lifecycle facade shared by Tasks and Reminders screens. */
@Singleton
class ReminderStore @Inject constructor(
    private val reminderDao: ReminderDao,
    private val reminderScheduler: ReminderScheduler,
) {
    fun observeUpcoming(): Flow<List<ReminderEntity>> =
        reminderDao.observeUpcoming(System.currentTimeMillis())

    fun observeHistory(): Flow<List<ReminderEntity>> = reminderDao.observeHistory()

    suspend fun create(
        message: String,
        triggerAt: Long,
        recurrence: String = "none",
    ): ReminderEntity = reminderScheduler.create(message, triggerAt, recurrence)

    suspend fun update(
        id: String,
        message: String,
        triggerAt: Long,
        recurrence: String,
    ): ReminderEntity? {
        val existing = reminderDao.get(id) ?: return null
        return reminderScheduler.schedule(
            existing.copy(
                message = message.trim(),
                triggerAt = triggerAt,
                recurrence = ReminderRecurrence.normalize(recurrence),
                status = "scheduled",
            ),
        )
    }

    /** Keep the row as history while cancelling the underlying work. */
    suspend fun cancel(id: String) = reminderScheduler.cancel(id)

    /** Permanently remove a lifecycle row and any outstanding work. */
    suspend fun delete(id: String) = reminderScheduler.delete(id)

    suspend fun clearHistory() = reminderDao.deleteHistoryBefore(Long.MAX_VALUE)
}
