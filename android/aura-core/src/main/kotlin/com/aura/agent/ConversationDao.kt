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

    @Query("SELECT * FROM conversations ORDER BY createdAt ASC")
    suspend fun allForExport(): List<ConversationEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(rows: List<ConversationEntity>)

    /**
     * Full-text-ish search across conversation title + serialized turn
     * content. Uses SQL LIKE because the turns are stored as a single
     * JSON column — adding an FTS virtual table would require a
     * migration and a per-turn indexer, which is overkill for personal
     * use. LIKE is O(n) on a personal-size table (hundreds of rows),
     * which is fine.
     *
     * [escapedQuery] must have `%`/`_`/`\` escaped (see
     * [com.aura.memory.MemoryStore.escapeLikeWildcards]) so a user
     * query containing those characters matches literally rather than
     * acting as a pattern.
     */
    @Query(
        """
        SELECT * FROM conversations
        WHERE title LIKE '%' || :escapedQuery || '%' ESCAPE '\'
           OR turnsJson LIKE '%' || :escapedQuery || '%' ESCAPE '\'
        ORDER BY updatedAt DESC
        LIMIT :limit
        """
    )
    suspend fun search(escapedQuery: kotlin.String, limit: Int = 50): List<ConversationEntity>

    @Query("SELECT * FROM conversations ORDER BY updatedAt DESC LIMIT 1")
    suspend fun mostRecent(): ConversationEntity?

    @Query("SELECT COUNT(*) FROM conversations")
    fun count(): Flow<Int>

    @Query("DELETE FROM conversations WHERE id = :id")
    suspend fun delete(id: String)

    @Query("DELETE FROM conversations")
    suspend fun deleteAll()

    /**
     * All conversations that have a non-null embedding. Used by
     * [ConversationStore.semanticSearch] to scan for cosine similarity
     * matches.
     */
    @Query("SELECT * FROM conversations WHERE embedding IS NOT NULL ORDER BY updatedAt DESC")
    suspend fun allWithEmbeddings(): List<ConversationEntity>

    /** Bounded legacy-row source for semantic-search backfill. */
    @Query("SELECT * FROM conversations WHERE embedding IS NULL ORDER BY updatedAt DESC LIMIT :limit")
    suspend fun missingEmbeddings(limit: Int): List<ConversationEntity>

    /**
     * Update just the embedding column for a conversation. Used by
     * save-time refresh and bounded legacy backfill.
     */
    @Query("UPDATE conversations SET embedding = :embedding WHERE id = :id")
    suspend fun updateEmbedding(id: String, embedding: ByteArray)
}
