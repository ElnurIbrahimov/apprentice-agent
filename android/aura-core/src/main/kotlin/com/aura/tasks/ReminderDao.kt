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

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(reminders: List<ReminderEntity>)

    @Query("SELECT * FROM reminders WHERE status = 'scheduled' AND triggerAt > :now ORDER BY triggerAt ASC")
    fun observeUpcoming(now: Long): Flow<List<ReminderEntity>>

    @Query("SELECT * FROM reminders WHERE status = 'scheduled' AND triggerAt > :now ORDER BY triggerAt ASC")
    suspend fun upcoming(now: Long): List<ReminderEntity>

    @Query("SELECT * FROM reminders WHERE status != 'scheduled' ORDER BY COALESCE(firedAt, triggerAt) DESC LIMIT :limit")
    fun observeHistory(limit: Int = 100): Flow<List<ReminderEntity>>

    @Query("SELECT * FROM reminders ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<ReminderEntity>

    @Query("SELECT * FROM reminders WHERE id = :id")
    suspend fun get(id: String): ReminderEntity?

    @Query("DELETE FROM reminders WHERE id = :id")
    suspend fun delete(id: String)

    @Query("DELETE FROM reminders")
    suspend fun deleteAll()

    @Query("DELETE FROM reminders WHERE status != 'scheduled' AND COALESCE(firedAt, triggerAt) <= :before")
    suspend fun deleteHistoryBefore(before: Long)

    @Query("DELETE FROM reminders WHERE taskId = :taskId")
    suspend fun deleteByTaskId(taskId: String)
}
