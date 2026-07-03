package com.aura.tasks

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface TaskDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(task: TaskEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(tasks: List<TaskEntity>)

    @Update
    suspend fun update(task: TaskEntity)

    @Query("SELECT * FROM tasks WHERE id = :id")
    suspend fun get(id: String): TaskEntity?

    @Query("SELECT * FROM tasks ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<TaskEntity>>

    @Query("SELECT * FROM tasks WHERE status = 'pending' ORDER BY createdAt DESC")
    suspend fun allPending(): List<TaskEntity>

    /**
     * Tasks that are pending and have [dueAt] in the inclusive range
     * [startMs, endMs]. Used by the morning brief to surface
     * "tasks due today" without scanning the full table. Ordered
     * by dueAt ASC so the soonest-due task is first.
     */
    @Query("""
        SELECT * FROM tasks
        WHERE status = 'pending' AND dueAt IS NOT NULL
          AND dueAt >= :startMs AND dueAt < :endMs
        ORDER BY dueAt ASC
    """)
    suspend fun dueInRange(startMs: Long, endMs: Long): List<TaskEntity>

    @Query("SELECT * FROM tasks ORDER BY createdAt DESC")
    suspend fun all(): List<TaskEntity>

    @Query("UPDATE tasks SET status = 'done', completedAt = :ts WHERE id = :id")
    suspend fun markComplete(id: String, ts: Long)

    @Query("DELETE FROM tasks WHERE id = :id")
    suspend fun delete(id: String)

    @Query("DELETE FROM tasks")
    suspend fun deleteAll()
}
