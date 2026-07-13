package com.aura.memory

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface MemoryEditDao {
    @Insert
    suspend fun insert(edit: MemoryEditEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(edits: List<MemoryEditEntity>)

    @Query("SELECT * FROM memory_edits ORDER BY editedAt ASC")
    suspend fun allForBackup(): List<MemoryEditEntity>

    @Query("DELETE FROM memory_edits")
    suspend fun deleteAll()

    @Query("SELECT * FROM memory_edits WHERE memoryId = :memoryId ORDER BY editedAt DESC")
    suspend fun getForMemory(memoryId: String): List<MemoryEditEntity>
}