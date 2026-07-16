package com.aura.documents

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface DocumentChunkDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(chunk: DocumentChunkEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(chunks: List<DocumentChunkEntity>)

    @Update
    suspend fun update(chunk: DocumentChunkEntity)

    @Query("SELECT * FROM document_chunks WHERE documentId = :documentId ORDER BY ordinal ASC")
    fun observeForDocument(documentId: kotlin.String): Flow<List<DocumentChunkEntity>>

    @Query("SELECT * FROM document_chunks WHERE documentId = :documentId ORDER BY ordinal ASC")
    suspend fun forDocument(documentId: kotlin.String): List<DocumentChunkEntity>

    @Query("SELECT * FROM document_chunks WHERE documentId = :documentId AND ordinal = :ordinal LIMIT 1")
    suspend fun getByOrdinal(documentId: kotlin.String, ordinal: Int): DocumentChunkEntity?

    @Query("SELECT * FROM document_chunks WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): DocumentChunkEntity?

    @Query("SELECT * FROM document_chunks WHERE embedding IS NOT NULL")
    suspend fun allWithEmbeddings(): List<DocumentChunkEntity>

    @Query("SELECT * FROM document_chunks WHERE embedding IS NULL")
    suspend fun allWithoutEmbeddings(): List<DocumentChunkEntity>

    @Query("SELECT COUNT(*) FROM document_chunks WHERE documentId = :documentId")
    suspend fun countForDocument(documentId: kotlin.String): Int

    @Query("DELETE FROM document_chunks WHERE documentId = :documentId")
    suspend fun deleteForDocument(documentId: kotlin.String)

    @Query("DELETE FROM document_chunks WHERE documentId IN (SELECT id FROM documents)")
    suspend fun deleteAll()

    @Query("UPDATE document_chunks SET embedding = :embedding, embeddingModel = :model, embeddingVersion = :version, embeddedAt = :timestamp WHERE id = :id")
    suspend fun updateEmbedding(id: kotlin.String, embedding: ByteArray, model: kotlin.String, version: Int, timestamp: kotlin.Long)

    @Query("SELECT * FROM document_chunks ORDER BY documentId, ordinal")
    suspend fun allForBackup(): List<DocumentChunkEntity>
}