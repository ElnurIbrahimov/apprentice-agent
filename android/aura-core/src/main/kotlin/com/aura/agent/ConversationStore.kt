package com.aura.agent

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

private val convJson = Json { ignoreUnknownKeys = true; encodeDefaults = true }

@Singleton
class ConversationStore @Inject constructor(
    private val dao: ConversationDao,
) {
    suspend fun save(conversation: Conversation) {
        val entity = ConversationEntity(
            id = conversation.id,
            title = conversation.title,
            createdAt = conversation.createdAt,
            updatedAt = System.currentTimeMillis(),
            systemPrompt = conversation.systemPrompt,
            model = conversation.model,
            metadataJson = convJson.encodeToString(conversation.metadata),
            turnsJson = convJson.encodeToString(conversation.turns),
        )
        // Use insert (REPLACE strategy) for upsert
        dao.insert(entity)
    }

    suspend fun load(id: String): Conversation? {
        val entity = dao.getById(id) ?: return null
        return entityToConversation(entity)
    }

    suspend fun mostRecent(): Conversation? {
        val entity = dao.mostRecent() ?: return null
        return entityToConversation(entity)
    }

    suspend fun recent(limit: Int = 50): List<Conversation> =
        dao.recent(limit).map { entityToConversation(it) }

    /**
     * Search across conversation title + serialized turn content. Returns
     * the most-recently-updated matches first. Like [com.aura.memory.MemoryStore.query],
     * the input is escaped so SQL LIKE wildcards in the user query are
     * treated as literal characters.
     */
    suspend fun search(query: String, limit: Int = 50): List<Conversation> {
        val trimmed = query.trim()
        if (trimmed.isEmpty()) return emptyList()
        val escaped = trimmed
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        return dao.search(escaped, limit).map { entityToConversation(it) }
    }

    suspend fun delete(id: String) = dao.delete(id)

    suspend fun deleteAll() = dao.deleteAll()

    private fun entityToConversation(e: ConversationEntity): Conversation {
        // runCatching here is a guard against a corrupted row. The
        // entity columns are typed, but metadataJson and turnsJson are
        // free-form text that may have been written by an older app
        // version with a different schema, or corrupted by a partial
        // disk write. We log and fall through to the empty default
        // rather than throwing — the user should still be able to see
        // their conversation title even if a turn is unrecoverable.
        val metadata: Map<String, String> = runCatching {
            convJson.decodeFromString<Map<String, String>>(e.metadataJson)
        }.getOrElse {
            android.util.Log.w("ConversationStore", "corrupt metadataJson for conv ${e.id}: ${it.message}")
            emptyMap()
        }

        val turns: List<Turn> = runCatching {
            convJson.decodeFromString<List<Turn>>(e.turnsJson)
        }.getOrElse {
            android.util.Log.w("ConversationStore", "corrupt turnsJson for conv ${e.id}: ${it.message}")
            emptyList()
        }

        return Conversation(
            id = e.id,
            title = e.title,
            createdAt = e.createdAt,
            updatedAt = e.updatedAt,
            systemPrompt = e.systemPrompt,
            model = e.model,
            metadata = metadata,
            turns = turns,
        )
    }
}
