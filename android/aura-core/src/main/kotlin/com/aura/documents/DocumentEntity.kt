package com.aura.documents

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "documents",
    indices = [
        Index(value = ["name"]),
        Index(value = ["importedAt"]),
    ],
)
data class DocumentEntity(
    /** SHA-256 of the imported bytes, making repeat imports idempotent. */
    @PrimaryKey val id: String,
    val name: String,
    val mimeType: String,
    /** Persisted Storage Access Framework URI. */
    val sourceUri: String,
    val importedAt: Long = System.currentTimeMillis(),
    val characterCount: Int,
    val chunkCount: Int,
)

data class DocumentImportResult(
    val document: DocumentEntity,
    val chunkCount: Int,
)