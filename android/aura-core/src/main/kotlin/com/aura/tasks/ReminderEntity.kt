package com.aura.tasks

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * A scheduled reminder, persisted so the UI can list and cancel upcoming
 * reminders before they fire. The [id] matches the WorkManager request
 * id so cancellation can hit both Room and WorkManager by the same key.
 */
@Entity(
    tableName = "reminders",
    indices = [Index(value = ["triggerAt"])],
)
data class ReminderEntity(
    @PrimaryKey val id: String,
    val message: String,
    val triggerAt: Long,
    val createdAt: Long = System.currentTimeMillis(),
    /**
     * Optional link to a task id when the reminder was created via
     * manage_tasks. Empty for standalone set_reminder reminders.
     */
    val taskId: String = "",
)
