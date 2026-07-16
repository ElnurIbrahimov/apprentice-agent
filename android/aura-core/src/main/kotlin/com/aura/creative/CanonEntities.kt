package com.aura.creative

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * A typed, versioned fact in the creative world canon. Facts are
 * scoped to a project and branch. Each fact has a subject type/id,
 * predicate, value, validity window, confidence, and source revision.
 *
 * Canon facts are the normalized form of the WorldBible — instead of
 * free-form JSON, every world element is a typed fact that can be
 * queried, diffed, and checked for contradictions.
 */
@Entity(
    tableName = "canon_facts",
    indices = [
        Index(value = ["projectId"]),
        Index(value = ["projectId", "branchId"]),
        Index(value = ["subjectType", "subjectId"]),
        Index(value = ["predicate"]),
        Index(value = ["status"]),
    ],
    foreignKeys = [
        androidx.room.ForeignKey(
            entity = CreativeProjectEntity::class,
            parentColumns = ["id"],
            childColumns = ["projectId"],
            onDelete = androidx.room.ForeignKey.CASCADE,
        ),
    ],
)
data class CanonFactEntity(
    @PrimaryKey val id: kotlin.String,
    val projectId: kotlin.String,
    val branchId: kotlin.String,
    /** "character", "location", "faction", "object", "rule", "timeline", "relationship" */
    val subjectType: kotlin.String,
    /** Entity name or ID this fact is about. */
    val subjectId: kotlin.String,
    /** What this fact says: "location", "age", "allegiance", "owns", etc. */
    val predicate: kotlin.String,
    /** JSON value: string, number, array, or object. */
    val valueJson: kotlin.String,
    /** Epoch ms. 0 = no start bound. */
    val validFrom: kotlin.Long = 0L,
    /** Epoch ms. 0 = no end bound (still valid). */
    val validTo: kotlin.Long = 0L,
    val confidence: Float = 1.0f,
    /** Revision that established this fact. */
    val sourceRevisionId: kotlin.String? = null,
    /** "active", "superseded", "retired" */
    val status: kotlin.String = "active",
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    val updatedAt: kotlin.Long = createdAt,
)

/**
 * A simulation of a potential world state change. Simulations produce
 * narrative outcomes, typed state deltas, and causal explanations.
 * They are never auto-canonized — the user approves selected deltas.
 */
@Entity(
    tableName = "creative_simulations",
    indices = [
        Index(value = ["projectId"]),
        Index(value = ["projectId", "branchId"]),
        Index(value = ["canonizedAt"]),
    ],
    foreignKeys = [
        androidx.room.ForeignKey(
            entity = CreativeProjectEntity::class,
            parentColumns = ["id"],
            childColumns = ["projectId"],
            onDelete = androidx.room.ForeignKey.CASCADE,
        ),
    ],
)
data class CreativeSimulationEntity(
    @PrimaryKey val id: kotlin.String,
    val projectId: kotlin.String,
    val branchId: kotlin.String,
    val premise: kotlin.String,
    /** JSON array of assumptions. */
    val assumptionsJson: kotlin.String = "[]",
    val narrative: kotlin.String = "",
    /** JSON: typed state deltas (subjectType, subjectId, predicate, oldValue, newValue) */
    val stateDeltaJson: kotlin.String = "[]",
    /** JSON: causal graph edges (cause -> effect with explanation) */
    val causalGraphJson: kotlin.String = "[]",
    val confidence: Float = 1.0f,
    /** JSON array of contradiction descriptions, if any. */
    val contradictionsJson: kotlin.String = "[]",
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    /** 0 = not canonized. Non-zero = when user approved selected deltas. */
    val canonizedAt: kotlin.Long = 0L,
    /** JSON array of canonized fact IDs (subset of state delta). */
    val canonizedFactIdsJson: kotlin.String = "[]",
)

/**
 * A typed continuity issue detected by the continuity compiler.
 * Issues have severity, evidence (linking to canon facts), and
 * suggested patches.
 */
@Entity(
    tableName = "continuity_issues",
    indices = [
        Index(value = ["projectId"]),
        Index(value = ["projectId", "branchId"]),
        Index(value = ["artifactId"]),
        Index(value = ["severity"]),
        Index(value = ["status"]),
    ],
)
data class ContinuityIssueEntity(
    @PrimaryKey val id: kotlin.String,
    val projectId: kotlin.String,
    val branchId: kotlin.String,
    val artifactId: kotlin.String? = null,
    /** "identity", "location", "timeline", "knowledge", "relationship", "rule", "visual", "voice" */
    val category: kotlin.String,
    /** "info", "warning", "error" */
    val severity: kotlin.String,
    val message: kotlin.String,
    /** JSON array of canon fact IDs that provide evidence. */
    val evidenceFactIdsJson: kotlin.String = "[]",
    /** JSON suggested patch (canon fact change or text edit). */
    val suggestedPatchJson: kotlin.String = "{}",
    /** "open", "accepted", "dismissed", "intentional_exception" */
    val status: kotlin.String = "open",
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    val resolvedAt: kotlin.Long? = null,
    val resolvedBy: kotlin.String = "",
)

/**
 * A dependency between two artifacts. When a canon fact or reference
 * changes, dependent artifacts are marked for review.
 */
@Entity(
    tableName = "artifact_dependencies",
    indices = [
        Index(value = ["sourceArtifactId"]),
        Index(value = ["targetArtifactId"]),
        Index(value = ["relation"]),
    ],
    foreignKeys = [
        androidx.room.ForeignKey(
            entity = CreativeArtifactEntity::class,
            parentColumns = ["id"],
            childColumns = ["sourceArtifactId"],
            onDelete = androidx.room.ForeignKey.CASCADE,
        ),
        androidx.room.ForeignKey(
            entity = CreativeArtifactEntity::class,
            parentColumns = ["id"],
            childColumns = ["targetArtifactId"],
            onDelete = androidx.room.ForeignKey.CASCADE,
        ),
    ],
)
data class ArtifactDependencyEntity(
    @PrimaryKey val id: kotlin.String,
    val sourceArtifactId: kotlin.String,
    val targetArtifactId: kotlin.String,
    /** "references", "derived_from", "depends_on", "invalidates" */
    val relation: kotlin.String,
    /** "cascade" (delete target when source deleted), "mark_review" (mark target for review) */
    val invalidationPolicy: kotlin.String = "mark_review",
    val createdAt: kotlin.Long = System.currentTimeMillis(),
)