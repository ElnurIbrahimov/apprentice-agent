package com.aura.agent

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface ConversationDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(conv: ConversationEntity)

    @Update
    suspend fun update(conv: ConversationEntity)

    @Query("SELECT * FROM conversations WHERE id = :id")
    suspend fun getById(id: String): ConversationEntity?

    @Query("SELECT * FROM conversations ORDER BY updatedAt DESC LIMIT :limit")
    suspend fun recent(limit: Int = 50): List<ConversationEntity>

    @Query("SELECT * FROM conversations ORDER BY updatedAt DESC LIMIT 1")
    suspend fun mostRecent(): ConversationEntity?

    @Query("SELECT COUNT(*) FROM conversations")
    fun count(): Flow<Int>

    @Query("DELETE FROM conversations WHERE id = :id")
    suspend fun delete(id: String)
}
