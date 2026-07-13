package com.aura.tasks

import android.content.Context
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.aura.tools.ReminderWorker
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.UUID
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/** One scheduling path shared by UI, tools, recurrence, restore, and cancellation. */
@Singleton
class ReminderScheduler @Inject constructor(
    @ApplicationContext private val context: Context,
    private val reminderDao: ReminderDao,
) {
    suspend fun schedule(reminder: ReminderEntity): ReminderEntity {
        val recurrence = ReminderRecurrence.normalize(reminder.recurrence)
        val delayMs = (reminder.triggerAt - System.currentTimeMillis()).coerceAtLeast(0L)
        val work = OneTimeWorkRequestBuilder<ReminderWorker>()
            .setInitialDelay(delayMs, TimeUnit.MILLISECONDS)
            .setInputData(
                Data.Builder()
                    .putString(ReminderWorker.KEY_REMINDER_ID, reminder.id)
                    .putString(ReminderWorker.KEY_TITLE, "⏰ Reminder")
                    .putString(ReminderWorker.KEY_BODY, reminder.message)
                    .putInt(ReminderWorker.KEY_NOTIFICATION_ID, notificationId(reminder.id))
                    .build(),
            )
            .addTag(REMINDER_TAG)
            .build()
        val scheduled = reminder.copy(
            workId = work.id.toString(),
            recurrence = recurrence,
            status = "scheduled",
        )
        reminderDao.insert(scheduled)
        WorkManager.getInstance(context).enqueueUniqueWork(
            uniqueName(reminder.id),
            ExistingWorkPolicy.REPLACE,
            work,
        )
        return scheduled
    }

    suspend fun create(
        message: String,
        triggerAt: Long,
        recurrence: String = "none",
        taskId: String = "",
        id: String = UUID.randomUUID().toString(),
    ): ReminderEntity = schedule(
        ReminderEntity(
            id = id,
            workId = "",
            message = message.trim(),
            triggerAt = triggerAt,
            taskId = taskId,
            recurrence = recurrence,
        ),
    )

    suspend fun cancel(id: String) {
        WorkManager.getInstance(context).cancelUniqueWork(uniqueName(id))
        val existing = reminderDao.get(id) ?: return
        reminderDao.insert(existing.copy(status = "cancelled"))
    }

    suspend fun delete(id: String) {
        WorkManager.getInstance(context).cancelUniqueWork(uniqueName(id))
        reminderDao.delete(id)
    }

    private fun uniqueName(id: String): String = "reminder-$id"

    private fun notificationId(id: String): Int {
        val positive = id.hashCode().toLong().let { if (it < 0) -it else it }
        return (positive % NOTIFICATION_ID_RANGE).toInt() + NOTIFICATION_ID_START
    }

    companion object {
        const val REMINDER_TAG = "reminder"
        private const val NOTIFICATION_ID_START = 10_000
        private const val NOTIFICATION_ID_RANGE = 90_000
    }
}
