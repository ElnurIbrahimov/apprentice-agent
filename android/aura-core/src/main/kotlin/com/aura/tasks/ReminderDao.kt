package com.aura.tasks

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ReminderDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(reminder: ReminderEntity)

    /**
     * Reminders whose scheduled time is in the future. Ordered soonest
     * first so the next reminder is always at the top of the list.
     */
    @Query("SELECT * FROM reminders WHERE triggerAt > :now ORDER BY triggerAt ASC")
    fun observeUpcoming(now: Long): Flow<List<ReminderEntity>>

    @Query("SELECT * FROM reminders WHERE triggerAt > :now ORDER BY triggerAt ASC")
    suspend fun upcoming(now: Long): List<ReminderEntity>

    @Query("SELECT * FROM reminders WHERE id = :id")
    suspend fun get(id: String): ReminderEntity?

    @Query("DELETE FROM reminders WHERE id = :id")
    suspend fun delete(id: String)

    @Query("DELETE FROM reminders WHERE triggerAt <= :now")
    suspend fun deleteExpired(now: Long)

    @Query("DELETE FROM reminders WHERE taskId = :taskId")
    suspend fun deleteByTaskId(taskId: String)
}
