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

    suspend fun delete(id: String) = dao.delete(id)

    private fun entityToConversation(e: ConversationEntity): Conversation {
        val metadata: Map<String, String> = runCatching {
            convJson.decodeFromString<Map<String, String>>(e.metadataJson)
        }.getOrDefault(emptyMap())

        val turns: List<Turn> = runCatching {
            convJson.decodeFromString<List<Turn>>(e.turnsJson)
        }.getOrDefault(emptyList())

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
