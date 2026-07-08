package com.aura.tasks

import android.content.Context
import androidx.work.WorkManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Single source of truth for the user-facing "my reminders" view.
 *
 * Wraps the [ReminderDao] (Room) and WorkManager (cancellation) so
 * callers don't have to know about either. The upcoming list is
 * a Flow so the screen redraws on insert / delete.
 *
 * Why one place for both? `ReminderEntity.id` is the WorkManager
 * request id (a UUID), so cancelling a reminder means:
 *
 *   1. WorkManager.cancelUniqueWork(...) — actually stops the
 *      scheduled fire (just deleting the Room row would leak the
 *      WorkManager job and the notification would still appear).
 *   2. ReminderDao.delete(id) — removes the row so the UI updates.
 *
 * The DAO has `deleteExpired` for a sweep job, but the live UI uses
 * [observeUpcoming] which is already filtered to `triggerAt > now`.
 */
@Singleton
class ReminderStore @Inject constructor(
    @ApplicationContext private val context: Context,
    private val reminderDao: ReminderDao,
) {
    /** Live list of reminders scheduled in the future, soonest first. */
    fun observeUpcoming(): Flow<List<ReminderEntity>> =
        reminderDao.observeUpcoming(System.currentTimeMillis())

    /**
     * Cancel a reminder. Idempotent: cancelling a non-existent
     * work request is a no-op for WorkManager, and the DAO delete
     * is also a no-op if the row is already gone.
     *
     * WorkManager has two cancellation paths:
     *   - cancelUniqueWork(uniqueName) — requires the original
     *     `workName` we used at enqueue time
     *     (`"reminder-${currentTimeMillis()}"`)
     *   - cancelWorkById(uuid) — takes the WorkManager request id
     *     directly
     *
     * SetReminderTool persists the **request id** (the UUID) as
     * the Room row id, not the work name. So we cancel by UUID —
     * that's the supported path for "I have the id, stop the
     * work" and survives the case where the work has already
     * been enqueued with a different name (e.g. during a config
     * change).
     *
     * Both calls are best-effort: if WorkManager doesn't know
     * the id (e.g. it already ran), the runCatching swallows the
     * throw. The Room delete is the source of truth for the UI
     * — that's what unlists the reminder.
     */
    suspend fun cancel(id: String) {
        runCatching {
            WorkManager.getInstance(context).cancelWorkById(java.util.UUID.fromString(id))
        }
        reminderDao.delete(id)
    }
}
