package com.aura.documents

import com.aura.memory.MemoryStore
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class DocumentRepository @Inject constructor(
    private val documentDao: DocumentDao,
    private val memoryStore: MemoryStore,
) {
    fun observeAll(): Flow<List<DocumentEntity>> = documentDao.observeAll()

    suspend fun import(
        id: String,
        name: String,
        mimeType: String,
        sourceUri: String,
        text: String,
    ): DocumentImportResult {
        require(text.isNotBlank()) { "The document contains no extractable text." }
        require(text.length <= MAX_TEXT_CHARS) {
            "The document is too large after extraction (maximum ${MAX_TEXT_CHARS / 1_000_000}M characters)."
        }
        val safeName = name.trim().take(240).ifBlank { "Untitled document" }
        val chunks = DocumentChunker.chunk(text)
        require(chunks.isNotEmpty()) { "The document contains no extractable text." }
        val source = sourceFor(id)
        val stemTag = safeName.substringBeforeLast('.')
            .lowercase()
            .replace(Regex("[^a-z0-9]+"), "-")
            .trim('-')
            .take(48)
            .ifBlank { "imported" }

        try {
            chunks.forEachIndexed { index, chunk ->
                memoryStore.storeIfAbsent(
                    content = buildString {
                        append("Document: ").append(safeName)
                        append(" · Part ").append(index + 1).append('/').append(chunks.size)
                        append("\n\n").append(chunk)
                    },
                    source = source,
                    category = "document",
                    importance = 0.65f,
                    tags = listOf("document", stemTag),
                )
            }
            val document = DocumentEntity(
                id = id,
                name = safeName,
                mimeType = mimeType.ifBlank { "application/octet-stream" },
                sourceUri = sourceUri,
                characterCount = text.length,
                chunkCount = chunks.size,
            )
            documentDao.insert(document)
            return DocumentImportResult(document, chunks.size)
        } catch (failure: Throwable) {
            memoryStore.deleteBySource(source)
            throw failure
        }
    }

    suspend fun delete(id: String) {
        memoryStore.deleteBySource(sourceFor(id))
        documentDao.deleteById(id)
    }

    suspend fun get(id: String): DocumentEntity? = documentDao.getById(id)

    companion object {
        const val MAX_TEXT_CHARS = 2_000_000
        fun sourceFor(id: String): String = "document:$id"
    }
}