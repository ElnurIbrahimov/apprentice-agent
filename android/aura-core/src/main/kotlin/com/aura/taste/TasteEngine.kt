package com.aura.taste

import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * The Taste Twin — learns from user preference signals and
 * produces weighted style attributes that influence prompt
 * construction, candidate ranking, and model routing.
 *
 * Implementation is purely statistical (weighted averages over
 * structured signals) — no ML model training required.
 *
 * All signals are local, inspectable, and reversible.
 */
@Singleton
class TasteEngine @Inject constructor(
    private val signalDao: PreferenceSignalDao,
    private val profileDao: StyleProfileDao,
    private val routingDao: RoutingOutcomeDao,
) {
    private val json = Json { ignoreUnknownKeys = true; prettyPrint = true }
    private val mutex = Mutex()

    /**
     * Record a preference signal. Positive weight for accept/like,
     * negative for reject/dislike.
     */
    suspend fun recordSignal(
        projectId: kotlin.String = "",
        signalType: kotlin.String,
        category: kotlin.String,
        artifactId: kotlin.String? = null,
        attributes: Map<kotlin.String, kotlin.String> = emptyMap(),
        weight: Float = 1.0f,
    ) = mutex.withLock {
        signalDao.upsert(
            PreferenceSignalEntity(
                id = UUID.randomUUID().toString(),
                projectId = projectId,
                signalType = signalType,
                category = category,
                artifactId = artifactId,
                attributesJson = json.encodeToString(attributes),
                weight = weight,
            ),
        )
    }

    /**
     * Record a reaction (emoji or thumbs up/down).
     */
    suspend fun recordReaction(
        projectId: kotlin.String = "",
        artifactId: kotlin.String,
        reaction: kotlin.String,
        positive: kotlin.Boolean,
    ) {
        recordSignal(
            projectId = projectId,
            signalType = "reaction",
            category = "general",
            artifactId = artifactId,
            attributes = mapOf("reaction" to reaction),
            weight = if (positive) 1.0f else -1.0f,
        )
    }

    /**
     * Record a user edit (user modified the generated output).
     * The edit implies the original was insufficient.
     */
    suspend fun recordEdit(
        projectId: kotlin.String = "",
        artifactId: kotlin.String,
        editType: kotlin.String = "text",
    ) {
        recordSignal(
            projectId = projectId,
            signalType = "edit",
            category = editType,
            artifactId = artifactId,
            weight = -0.5f,
        )
    }

    /**
     * Record a model routing outcome for learning.
     */
    suspend fun recordRoutingOutcome(
        modelRole: kotlin.String,
        modelId: kotlin.String,
        success: kotlin.Boolean,
        latencyMs: kotlin.Long = 0L,
        costClass: kotlin.String = "unknown",
        outcomeType: kotlin.String = "user_accepted",
    ) {
        routingDao.upsert(
            RoutingOutcomeEntity(
                id = UUID.randomUUID().toString(),
                modelRole = modelRole,
                modelId = modelId,
                success = success,
                latencyMs = latencyMs,
                costClass = costClass,
                outcomeType = outcomeType,
            ),
        )
    }

    /**
     * Recompute the style profile for a project (or global if
     * projectId is empty). Aggregates all signals into weighted
     * attributes.
     */
    suspend fun recomputeProfile(projectId: kotlin.String = "") = mutex.withLock {
        val signals = if (projectId.isBlank()) {
            signalDao.global(500)
        } else {
            signalDao.forProject(projectId, 500)
        }

        if (signals.isEmpty()) return@withLock

        // Aggregate: group by category, compute weighted averages
        val byCategory = signals.groupBy { it.category }
        val attributes = mutableMapOf<kotlin.String, MutableMap<kotlin.String, Float>>()

        for ((category, categorySignals) in byCategory) {
            val attrs = mutableMapOf<kotlin.String, Float>()
            for (signal in categorySignals) {
                val parsed = runCatching {
                    json.decodeFromString<Map<kotlin.String, kotlin.String>>(signal.attributesJson)
                }.getOrDefault(emptyMap())

                for ((key, value) in parsed) {
                    val current = attrs.getOrDefault(value, 0f)
                    attrs[value] = current + signal.weight
                }
            }
            if (attrs.isNotEmpty()) {
                // Normalize: divide by total signal count for this category
                val totalWeight = categorySignals.sumOf { it.weight.toDouble() }.toFloat().coerceAtLeast(1f)
                attrs.forEach { (k, v) -> attrs[k] = v / totalWeight }
                attributes[category] = attrs
            }
        }

        val profileId = profileDao.forProject(projectId)?.id ?: UUID.randomUUID().toString()
        profileDao.upsert(
            StyleProfileEntity(
                id = profileId,
                projectId = projectId,
                attributesJson = json.encodeToString(attributes),
                signalCount = signals.size,
                updatedAt = System.currentTimeMillis(),
            ),
        )
    }

    /**
     * Get the current style profile for a project (or global).
     * Falls back to global profile if project profile doesn't exist.
     */
    suspend fun getProfile(projectId: kotlin.String = ""): StyleProfileEntity? {
        if (projectId.isNotBlank()) {
            profileDao.forProject(projectId)?.let { return it }
        }
        return profileDao.global()
    }

    /**
     * Get the best model for a given role, based on routing outcomes.
     * Returns null if no data.
     */
    suspend fun bestModelForRole(role: kotlin.String): kotlin.String? {
        val stats = routingDao.statsForRole(role)
        if (stats.isEmpty()) return null
        return stats
            .filter { it.count >= 2 }
            .maxByOrNull { it.successes.toFloat() / it.count.toFloat() }
            ?.modelId
    }

    /**
     * Clear all signals for a project (or global).
     */
    suspend fun clearSignals(projectId: kotlin.String = "") {
        if (projectId.isBlank()) {
            signalDao.deleteGlobal()
        } else {
            signalDao.deleteForProject(projectId)
        }
    }

    /**
     * Delete a single signal by ID.
     */
    suspend fun deleteSignal(id: kotlin.String) {
        signalDao.delete(id)
    }
}

/**
 * Serializable style profile attributes for prompt construction.
 */
@Serializable
data class StyleAttributes(
    val tone: kotlin.String = "",
    val pacing: kotlin.String = "",
    val vocabulary: kotlin.String = "",
    val palette: kotlin.String = "",
    val composition: kotlin.String = "",
    val voiceStyle: kotlin.String = "",
    val musicStyle: kotlin.String = "",
)