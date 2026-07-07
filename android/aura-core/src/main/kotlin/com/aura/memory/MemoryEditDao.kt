package com.aura.memory

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

@Dao
interface MemoryEditDao {
    @Insert
    suspend fun insert(edit: MemoryEditEntity): Long

    @Query("SELECT * FROM memory_edits WHERE memoryId = :memoryId ORDER BY editedAt DESC")
    suspend fun getForMemory(memoryId: String): List<MemoryEditEntity>
}