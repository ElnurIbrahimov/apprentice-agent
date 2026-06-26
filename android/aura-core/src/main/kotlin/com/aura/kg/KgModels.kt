package com.aura.kg

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

enum class NodeType {
    CONCEPT, ENTITY, PERSON, PROJECT, TOOL, EVENT, SKILL, LOCATION, FILE, EMOTION, UNKNOWN;

    companion object {
        fun from(value: String): NodeType = entries.firstOrNull {
            it.name.equals(value, ignoreCase = true)
        } ?: UNKNOWN
    }
}

enum class EdgeType {
    RELATES_TO, IS_A, PART_OF, CAUSES, SOLVES, CREATED_BY, USES, TRIGGERS,
    LEARNED_FROM, PRECEDED_BY, FOLLOWED_BY, CONFLICTS_WITH, STRENGTHENS,
    WEAKENS, KNOWS, WORKS_ON, LOCATED_AT, UNKNOWN;

    companion object {
        fun from(value: String): EdgeType = entries.firstOrNull {
            it.name.equals(value.replace(" ", "_"), ignoreCase = true)
        } ?: UNKNOWN
    }
}

data class KgNode(
    val id: String,
    val label: String,
    val type: NodeType,
    val properties: JsonObject = JsonObject(emptyMap()),
    val confidence: Float = 0.8f,
    val sourceTurnId: String = "",
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
    val accessCount: Int = 0,
    val lastAccessed: Long = System.currentTimeMillis(),
) {
    fun toEntity() = NodeEntity(
        id = id,
        label = label,
        type = type.name.lowercase(),
        properties = Json.encodeToString(JsonObject.serializer(), properties),
        confidence = confidence,
        sourceTurnId = sourceTurnId,
        createdAt = createdAt,
        updatedAt = updatedAt,
        accessCount = accessCount,
        lastAccessed = lastAccessed,
    )

    companion object {
        fun fromEntity(e: NodeEntity): KgNode {
            val props = try {
                Json.parseToJsonElement(e.properties).let {
                    it as? JsonObject ?: JsonObject(emptyMap())
                }
            } catch (_: Exception) {
                JsonObject(emptyMap())
            }
            return KgNode(
                id = e.id,
                label = e.label,
                type = NodeType.from(e.type),
                properties = props,
                confidence = e.confidence,
                sourceTurnId = e.sourceTurnId,
                createdAt = e.createdAt,
                updatedAt = e.updatedAt,
                accessCount = e.accessCount,
                lastAccessed = e.lastAccessed,
            )
        }
    }
}

data class KgEdge(
    val id: String,
    val type: EdgeType,
    val sourceId: String,
    val targetId: String,
    val weight: Float = 0.5f,
    val properties: JsonObject = JsonObject(emptyMap()),
    val confidence: Float = 0.8f,
    val sourceTurnId: String = "",
    val createdAt: Long = System.currentTimeMillis(),
    val lastReinforced: Long = System.currentTimeMillis(),
) {
    fun toEntity() = EdgeEntity(
        id = id,
        type = type.name.lowercase(),
        sourceId = sourceId,
        targetId = targetId,
        weight = weight,
        properties = Json.encodeToString(JsonObject.serializer(), properties),
        confidence = confidence,
        sourceTurnId = sourceTurnId,
        createdAt = createdAt,
        lastReinforced = lastReinforced,
    )

    companion object {
        fun fromEntity(e: EdgeEntity): KgEdge {
            val props = try {
                Json.parseToJsonElement(e.properties).let {
                    it as? JsonObject ?: JsonObject(emptyMap())
                }
            } catch (_: Exception) {
                JsonObject(emptyMap())
            }
            return KgEdge(
                id = e.id,
                type = EdgeType.from(e.type),
                sourceId = e.sourceId,
                targetId = e.targetId,
                weight = e.weight,
                properties = props,
                confidence = e.confidence,
                sourceTurnId = e.sourceTurnId,
                createdAt = e.createdAt,
                lastReinforced = e.lastReinforced,
            )
        }
    }
}
