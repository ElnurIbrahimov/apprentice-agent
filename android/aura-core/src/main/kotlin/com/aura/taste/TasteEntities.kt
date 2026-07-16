package com.aura.taste

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * A preference signal captured from user behavior. Signals are
 * evidence for the Taste Twin — they influence prompt construction,
 * candidate ranking, and style recommendations.
 *
 * Signals are local-only, inspectable, and reversible. The user can
 * delete individual signals or clear all signals at any time.
 */
@Entity(
    tableName = "preference_signals",
    indices = [
        Index(value = ["projectId"]),
        Index(value = ["signalType"]),
        Index(value = ["createdAt"]),
    ],
)
data class PreferenceSignalEntity(
    @PrimaryKey val id: kotlin.String,
    /** Project ID. Null/empty = global (applies to all projects). */
    val projectId: kotlin.String = "",
    /** "accept", "reject", "edit", "reaction", "rewrite", "select" */
    val signalType: kotlin.String,
    /** What the signal is about: "text_style", "visual_palette", "voice", "music", "pacing". */
    val category: kotlin.String,
    /** The content or artifact ID the signal references. */
    val artifactId: kotlin.String? = null,
    /** JSON: the specific attributes of the preference. */
    val attributesJson: kotlin.String = "{}",
    /** Weight: positive for accept/like, negative for reject/dislike. */
    val weight: Float = 1.0f,
    val createdAt: kotlin.Long = System.currentTimeMillis(),
)

/**
 * A style profile derived from preference signals. Profiles are
 * per-project or global. They contain weighted attributes that
 * the TasteEngine uses to influence generation.
 */
@Entity(
    tableName = "style_profiles",
    indices = [
        Index(value = ["projectId"]),
    ],
)
data class StyleProfileEntity(
    @PrimaryKey val id: kotlin.String,
    val projectId: kotlin.String = "",
    /** JSON: weighted style attributes (tone, pacing, vocabulary, palette, etc.). */
    val attributesJson: kotlin.String = "{}",
    /** Number of signals aggregated into this profile. */
    val signalCount: Int = 0,
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    val updatedAt: kotlin.Long = createdAt,
)

/**
 * A reference identity for cross-media consistency. Defines the
 * canonical appearance/voice/style of a character, location, or object.
 */
@Entity(
    tableName = "reference_identities",
    indices = [
        Index(value = ["projectId"]),
        Index(value = ["identityType"]),
        Index(value = ["name"]),
    ],
    foreignKeys = [
        androidx.room.ForeignKey(
            entity = com.aura.creative.CreativeProjectEntity::class,
            parentColumns = ["id"],
            childColumns = ["projectId"],
            onDelete = androidx.room.ForeignKey.CASCADE,
        ),
    ],
)
data class ReferenceIdentityEntity(
    @PrimaryKey val id: kotlin.String,
    val projectId: kotlin.String,
    /** "character", "location", "object", "palette", "voice", "music_motif", "cinematography" */
    val identityType: kotlin.String,
    val name: kotlin.String,
    /** JSON: attributes (eye_color, hair, build, clothing, architecture, etc.) */
    val attributesJson: kotlin.String = "{}",
    /** Reference artifact IDs (images, audio, etc.) that define this identity. */
    val referenceArtifactIdsJson: kotlin.String = "[]",
    /** Whether this identity is locked (changes require explicit approval). */
    val locked: kotlin.Boolean = false,
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    val updatedAt: kotlin.Long = createdAt,
)

/**
 * Recorded outcome of a model routing decision. Used to learn
 * which models perform best per task role.
 */
@Entity(
    tableName = "routing_outcomes",
    indices = [
        Index(value = ["modelRole"]),
        Index(value = ["modelId"]),
        Index(value = ["success"]),
    ],
)
data class RoutingOutcomeEntity(
    @PrimaryKey val id: kotlin.String,
    val modelRole: kotlin.String,
    val modelId: kotlin.String,
    val success: kotlin.Boolean,
    val latencyMs: kotlin.Long = 0L,
    val costClass: kotlin.String = "unknown",
    /** "user_accepted", "user_rejected", "user_edited", "postcondition_passed", "postcondition_failed" */
    val outcomeType: kotlin.String = "user_accepted",
    val createdAt: kotlin.Long = System.currentTimeMillis(),
)