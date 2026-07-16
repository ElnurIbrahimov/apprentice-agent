package com.aura.creative

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * A durable unit of creative output. One project has many artifacts.
 * Artifacts are typed (text, image, audio, video, 3D, document, reference, export)
 * and each has a current revision pointing to the latest [CreativeRevisionEntity].
 *
 * Bytes are never stored in Room — only the [storageUri] pointing to an
 * app-private file. The [contentHash] verifies file integrity.
 */
@Entity(
    tableName = "creative_artifacts",
    indices = [
        Index(value = ["projectId"]),
        Index(value = ["projectId", "kind"]),
        Index(value = ["status"]),
        Index(value = ["updatedAt"]),
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
data class CreativeArtifactEntity(
    @PrimaryKey val id: kotlin.String,
    val projectId: kotlin.String,
    val branchId: kotlin.String,
    /** "text", "image", "audio", "video", "3d", "document", "reference", "export" */
    val kind: kotlin.String,
    val title: kotlin.String,
    /** Current revision ID (points to CreativeRevisionEntity). */
    val currentRevisionId: kotlin.String? = null,
    /** Short text preview for list cards (first ~200 chars). */
    val previewText: kotlin.String = "",
    /** MIME type for media artifacts. */
    val mimeType: kotlin.String = "",
    /** App-private file URI for media artifacts. Null for text-only. */
    val storageUri: kotlin.String? = null,
    /** SHA-256 of the stored file content. */
    val contentHash: kotlin.String = "",
    /** "pending", "ready", "failed", "archived" */
    val status: kotlin.String = "pending",
    /** JSON metadata: provider, model, prompt, settings, cost, duration. */
    val metadataJson: kotlin.String = "{}",
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    val updatedAt: kotlin.Long = createdAt,
)

/**
 * An immutable revision of a [CreativeArtifactEntity]. Revisions are
 * append-only — creating a new revision doesn't delete the old one.
 * Parent revision chain enables diff, restore, and lineage.
 */
@Entity(
    tableName = "creative_revisions",
    indices = [
        Index(value = ["artifactId"]),
        Index(value = ["branchId"]),
        Index(value = ["parentRevisionId"]),
        Index(value = ["createdAt"]),
    ],
    foreignKeys = [
        androidx.room.ForeignKey(
            entity = CreativeArtifactEntity::class,
            parentColumns = ["id"],
            childColumns = ["artifactId"],
            onDelete = androidx.room.ForeignKey.CASCADE,
        ),
    ],
)
data class CreativeRevisionEntity(
    @PrimaryKey val id: kotlin.String,
    val artifactId: kotlin.String,
    val branchId: kotlin.String,
    /** Parent revision ID. Null for the first revision. */
    val parentRevisionId: kotlin.String? = null,
    /** Text content for text artifacts. Null/empty for media. */
    val contentText: kotlin.String = "",
    /** File URI for media artifacts. Null for text-only. */
    val storageUri: kotlin.String? = null,
    val contentHash: kotlin.String = "",
    /** "manual", "generation", "simulation_promoted", "import", "edit" */
    val authorKind: kotlin.String = "manual",
    val providerPrefix: kotlin.String = "",
    val modelId: kotlin.String = "",
    val prompt: kotlin.String = "",
    val settingsJson: kotlin.String = "{}",
    val createdAt: kotlin.Long = System.currentTimeMillis(),
)

/**
 * A divergent line of revisions within a project. The main branch is
 * created automatically when a project is initialized.
 */
@Entity(
    tableName = "creative_branches",
    indices = [
        Index(value = ["projectId"]),
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
data class CreativeBranchEntity(
    @PrimaryKey val id: kotlin.String,
    val projectId: kotlin.String,
    val name: kotlin.String,
    /** Revision this branch was created from. */
    val baseRevisionId: kotlin.String? = null,
    /** Current head revision of this branch. */
    val headRevisionId: kotlin.String? = null,
    /** "active", "merged", "archived" */
    val status: kotlin.String = "active",
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    val updatedAt: kotlin.Long = createdAt,
)

/**
 * A durable generation job for media (image, video, 3D, audio).
 * Survives process death so the user can resume after a crash.
 */
@Entity(
    tableName = "creative_generation_jobs",
    indices = [
        Index(value = ["projectId"]),
        Index(value = ["branchId"]),
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
data class CreativeGenerationJobEntity(
    @PrimaryKey val id: kotlin.String,
    val projectId: kotlin.String,
    val branchId: kotlin.String,
    /** [com.aura.capabilities.CapabilityKind] name. */
    val capabilityKind: kotlin.String,
    /** JSON request: prompt, negativePrompt, size, aspect, etc. */
    val requestJson: kotlin.String,
    /** "queued", "running", "waiting_provider", "succeeded", "failed", "cancelled" */
    val status: kotlin.String = "queued",
    /** 0-100 progress estimate. */
    val progress: Int = 0,
    val providerPrefix: kotlin.String = "",
    /** Provider-side operation ID for polling. */
    val providerOperationId: kotlin.String? = null,
    /** Resulting artifact IDs when succeeded. JSON array. */
    val resultArtifactIdsJson: kotlin.String = "[]",
    val errorCode: kotlin.String = "",
    val errorMessage: kotlin.String = "",
    val attempts: Int = 0,
    val createdAt: kotlin.Long = System.currentTimeMillis(),
    val updatedAt: kotlin.Long = createdAt,
)