package com.aura.memory

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import kotlinx.serialization.Serializable

@Serializable
@Entity(
    tableName = "memories",
    indices = [Index("createdAt"), Index("source"), Index("category"), Index("sourceConversationId")]
)
data class MemoryEntity(
    @PrimaryKey val id: String,
    val content: String,
    val source: String,         // "user", "assistant", "tool", "system"
    val category: String,       // "fact", "preference", "episode", "person", "project", "idea", "task"
    val importance: Float = 0.5f, // 0.0 - 1.0
    @ColumnInfo(name = "embedding") val embedding: ByteArray? = null, // 384 floats × 4 bytes
    @ColumnInfo(name = "createdAt") val createdAt: Long = System.currentTimeMillis(),
    @ColumnInfo(name = "accessedAt") val accessedAt: Long = System.currentTimeMillis(),
    @ColumnInfo(name = "accessCount") val accessCount: Int = 0,
    @ColumnInfo(name = "decayScore") val decayScore: Float = 1.0f, // 0.0 = forgotten, 1.0 = fresh
    val tags: String = "",       // comma-separated
    val metadata: String = "",   // JSON
    val sourceConversationId: kotlin.String = "",
    val sourceTurnTimestamp: kotlin.Long = 0L,
    /** Which embedding model produced [embedding]. Null for legacy rows. */
    val embeddingModel: kotlin.String? = null,
    /** Version of the embedding model for cache invalidation. */
    val embeddingVersion: Int = 0,
) {
    // Room requires equals/hashCode; ByteArray needs special handling
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MemoryEntity) return false
        return id == other.id
    }
    override fun hashCode(): Int = id.hashCode()
}
