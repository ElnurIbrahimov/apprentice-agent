package com.aura.tasks

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/** Durable reminder lifecycle record, independent of rotating WorkManager IDs. */
@Entity(
    tableName = "reminders",
    indices = [
        Index(value = ["triggerAt"]),
        Index(value = ["status", "triggerAt"]),
    ],
)
data class ReminderEntity(
    @PrimaryKey val id: String,
    /** Current WorkManager request id. Rotates after every recurring fire. */
    val workId: String,
    val message: String,
    val triggerAt: Long,
    val createdAt: Long = System.currentTimeMillis(),
    val taskId: String = "",
    /** none | daily | weekly | monthly */
    val recurrence: String = "none",
    /** scheduled | fired | cancelled */
    val status: String = "scheduled",
    val firedAt: Long? = null,
)
