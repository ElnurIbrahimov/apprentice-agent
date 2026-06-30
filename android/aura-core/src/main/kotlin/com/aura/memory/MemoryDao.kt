package com.aura.memory

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface MemoryDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(memory: MemoryEntity)

    @Update
    suspend fun update(memory: MemoryEntity)

    @Query("SELECT * FROM memories WHERE id = :id")
    suspend fun getById(id: String): MemoryEntity?

    @Query("SELECT * FROM memories ORDER BY createdAt DESC LIMIT :limit")
    suspend fun recent(limit: Int = 50): List<MemoryEntity>

    @Query("SELECT * FROM memories WHERE category = :category ORDER BY createdAt DESC LIMIT :limit")
    suspend fun byCategory(category: String, limit: Int = 50): List<MemoryEntity>

    @Query("SELECT * FROM memories WHERE content LIKE :query ESCAPE '\\' ORDER BY decayScore DESC LIMIT :limit")
    suspend fun searchByText(query: String, limit: Int = 50): List<MemoryEntity>

    @Query("SELECT * FROM memories ORDER BY decayScore DESC LIMIT :limit")
    suspend fun top(limit: Int = 50): List<MemoryEntity>

    /**
     * All memories, ordered by createdAt ascending so the export is
     * deterministic. Used by the backup module. No limit — the
     * export wants everything, even if the personal-use table is
     * bounded to a few thousand rows.
     */
    @Query("SELECT * FROM memories ORDER BY createdAt ASC")
    suspend fun allForExport(): List<MemoryEntity>

    /**
     * Bulk insert for the import path. Skips individual inserts to
     * avoid N round-trips to Room.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(rows: List<MemoryEntity>)

    @Query("UPDATE memories SET accessedAt = :now, accessCount = accessCount + 1, decayScore = MIN(1.0, decayScore + 0.1) WHERE id = :id")
    suspend fun touch(id: String, now: Long = System.currentTimeMillis())

    @Query("DELETE FROM memories WHERE id = :id")
    suspend fun delete(id: String)

    @Query("DELETE FROM memories")
    suspend fun deleteAll()

    @Query("UPDATE memories SET decayScore = decayScore * :factor WHERE createdAt < :cutoff")
    suspend fun applyDecay(cutoff: Long, factor: Float)

    @Query("SELECT COUNT(*) FROM memories")
    fun count(): Flow<Int>

    @Query("SELECT COUNT(*) FROM memories")
    suspend fun countOnce(): Int
}
