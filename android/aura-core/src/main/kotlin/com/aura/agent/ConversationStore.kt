package com.aura.agent

import com.aura.memory.escapeLikeWildcards
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

private val convJson = Json { ignoreUnknownKeys = true; encodeDefaults = true }

@Singleton
class ConversationStore @Inject constructor(
    private val dao: ConversationDao,
    private val embedder: com.aura.memory.Embedder,
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
            contextSummary = conversation.contextSummary,
            summaryThroughTurn = conversation.summaryThroughTurn.coerceIn(0, conversation.turns.size),
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
     * the input is escaped (via the shared `escapeLikeWildcards` helper)
     * so SQL LIKE wildcards in the user query are treated as literal
     * characters.
     */
    suspend fun search(query: String, limit: Int = 50): List<Conversation> {
        val trimmed = query.trim()
        if (trimmed.isEmpty()) return emptyList()
        return dao.search(escapeLikeWildcards(trimmed), limit).map { entityToConversation(it) }
    }

    suspend fun delete(id: String) = dao.delete(id)

    suspend fun deleteAll() = dao.deleteAll()

    /**
     * Rename a conversation. The new title must be non-blank and
     * not exceed 120 chars (the entity column caps it). Returns
     * true on success, false if the conversation doesn't exist
     * or the title is invalid.
     */
    suspend fun setTitle(id: String, newTitle: String): Boolean {
        val trimmed = newTitle.trim().take(120)
        if (trimmed.isBlank()) return false
        val entity = dao.getById(id) ?: return false
        val updated = entity.copy(title = trimmed, updatedAt = System.currentTimeMillis())
        dao.insert(updated)  // INSERT OR REPLACE — see ConversationDao
        return true
    }

    /**
     * Pin or unpin a conversation. Pinned conversations sort to
     * the top of the History list so the user can find them
     * without scrolling. Pin state lives in the metadata JSON
     * column under the key "pinned" — adding a typed column
     * would have required a Room migration.
     */
    suspend fun setPinned(id: String, pinned: Boolean): Boolean {
        val entity = dao.getById(id) ?: return false
        val metadata = runCatching {
            convJson.decodeFromString<Map<String, String>>(entity.metadataJson)
        }.getOrElse { emptyMap() }
        val updated = metadata.toMutableMap().apply {
            if (pinned) put("pinned", "true") else remove("pinned")
        }
        dao.insert(entity.copy(
            metadataJson = convJson.encodeToString(updated),
            updatedAt = System.currentTimeMillis(),
        ))
        return true
    }

    /**
     * Read the pinned flag from a conversation's metadata. Returns
     * false for conversations without the flag (default = unpinned).
     */
    fun isPinned(conv: Conversation): Boolean =
        conv.metadata["pinned"] == "true"

    /**
     * Recent conversations with pinned ones sorted to the top.
     * The list is otherwise ordered by updatedAt desc, same as
     * [recent]. Pinned items also have a stable insertion order
     * (most recently pinned first) because their updatedAt is
     * bumped when pinned.
     */
    suspend fun recentPinnedFirst(limit: Int = 50): List<Conversation> =
        recent(limit).sortedByDescending { isPinned(it) }

    /**
     * Semantic search across conversations. Embeds the query and
     * compares against conversation embeddings (lazy-populated from
     * the last user message). Falls back to SQL LIKE search if no
     * conversations have embeddings yet.
     *
     * The embedding scan is O(n) over conversations with embeddings.
     * For personal use (hundreds of conversations) this is fast.
     */
    suspend fun semanticSearch(query: String, limit: Int = 20): List<Conversation> {
        val trimmed = query.trim()
        if (trimmed.isEmpty()) return emptyList()

        val existing = runCatching { dao.allWithEmbeddings() }.getOrDefault(emptyList())
        if (existing.isEmpty()) {
            // No embeddings yet — fall back to LIKE search.
            return search(trimmed, limit)
        }

        val queryEmbedding = runCatching { embedder.embed(trimmed) }.getOrNull()
            ?: return search(trimmed, limit)

        // Rank conversations by cosine similarity to the query.
        data class Scored(val conv: ConversationEntity, val score: Float)
        val scored = existing.mapNotNull { entity ->
            val emb = entity.embedding ?: return@mapNotNull null
            val score = cosineSimilarity(queryEmbedding, com.aura.memory.Embedder.fromBytes(emb))
            if (score > 0.05f) Scored(entity, score) else null
        }.sortedByDescending { it.score }.take(limit)

        return scored.map { entityToConversation(it.conv) }
    }

    /**
     * Fork a conversation from a specific turn index. Creates a new
     * conversation with a new ID, copying turns 0..fromTurnIndex
     * (inclusive). The original is untouched. The fork's title is
     * "{original title} (fork)" and it inherits the system prompt +
     * model. Used by the "Fork from here" action in ChatScreen.
     */
    suspend fun fork(id: String, fromTurnIndex: Int): String? {
        val original = dao.getById(id) ?: return null
        val metadata = runCatching {
            convJson.decodeFromString<Map<String, String>>(original.metadataJson)
        }.getOrElse { emptyMap() }
        val allTurns = runCatching {
            convJson.decodeFromString<List<Turn>>(original.turnsJson)
        }.getOrElse { emptyList() }
        if (fromTurnIndex !in allTurns.indices) return null
        val forkedTurns = allTurns.take(fromTurnIndex + 1)
        val forkId = java.util.UUID.randomUUID().toString()
        val forkTitle = "${original.title} (fork)"
        val forkTurnCount = fromTurnIndex + 1
        val canReuseSummary = original.contextSummary.isNotBlank() &&
            original.summaryThroughTurn in 1..forkTurnCount
        dao.insert(ConversationEntity(
            id = forkId,
            title = forkTitle,
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis(),
            systemPrompt = original.systemPrompt,
            model = original.model,
            metadataJson = convJson.encodeToString(metadata),
            turnsJson = convJson.encodeToString(forkedTurns),
            contextSummary = if (canReuseSummary) original.contextSummary else "",
            summaryThroughTurn = if (canReuseSummary) original.summaryThroughTurn else 0,
        ))
        return forkId
    }

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
            contextSummary = e.contextSummary,
            summaryThroughTurn = e.summaryThroughTurn.coerceIn(0, turns.size),
            metadata = metadata,
            turns = turns,
        )
    }
}


/** Fast cosine similarity between two same-dimension float arrays. */
private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return 0f
    var dot = 0f
    var aNorm = 0f
    var bNorm = 0f
    for (i in a.indices) {
        dot += a[i] * b[i]
        aNorm += a[i] * a[i]
        bNorm += b[i] * b[i]
    }
    val aN = kotlin.math.sqrt(aNorm)
    val bN = kotlin.math.sqrt(bNorm)
    if (aN == 0f || bN == 0f) return 0f
    return dot / (aN * bN)
}
