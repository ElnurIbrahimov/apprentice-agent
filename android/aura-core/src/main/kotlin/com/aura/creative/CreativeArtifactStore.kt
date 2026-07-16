package com.aura.creative

import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Domain store for [CreativeArtifactEntity]. Wraps the DAO with
 * project-scoped mutex, status transitions, and soft-delete (archive).
 */
@Singleton
class CreativeArtifactStore @Inject constructor(
    private val artifactDao: CreativeArtifactDao,
    private val revisionDao: CreativeRevisionDao,
    private val branchDao: CreativeBranchDao,
) {
    private val projectMutex = Mutex()

    suspend fun create(
        projectId: String,
        branchId: String,
        kind: String,
        title: String,
        initialContent: String = "",
        authorKind: String = "manual",
        providerPrefix: String = "",
        modelId: String = "",
        prompt: String = "",
    ): CreativeArtifactEntity = projectMutex.withLock {
        val artifactId = UUID.randomUUID().toString()
        val revisionId = UUID.randomUUID().toString()
        val now = System.currentTimeMillis()
        val contentHash = sha256(initialContent)

        val revision = CreativeRevisionEntity(
            id = revisionId,
            artifactId = artifactId,
            branchId = branchId,
            parentRevisionId = null,
            contentText = initialContent,
            contentHash = contentHash,
            authorKind = authorKind,
            providerPrefix = providerPrefix,
            modelId = modelId,
            prompt = prompt,
            createdAt = now,
        )
        revisionDao.upsert(revision)

        val artifact = CreativeArtifactEntity(
            id = artifactId,
            projectId = projectId,
            branchId = branchId,
            kind = kind,
            title = title,
            currentRevisionId = revisionId,
            previewText = initialContent.take(200),
            status = "ready",
            createdAt = now,
            updatedAt = now,
        )
        artifactDao.upsert(artifact)
        artifact
    }

    suspend fun revise(
        artifactId: String,
        content: String,
        parentRevisionId: String? = null,
        authorKind: String = "manual",
        providerPrefix: String = "",
        modelId: String = "",
        prompt: String = "",
    ): CreativeRevisionEntity = projectMutex.withLock {
        val artifact = artifactDao.getById(artifactId)
            ?: error("Artifact $artifactId not found")
        val effectiveParent = parentRevisionId ?: artifact.currentRevisionId
        val revisionId = UUID.randomUUID().toString()
        val now = System.currentTimeMillis()
        val revision = CreativeRevisionEntity(
            id = revisionId,
            artifactId = artifactId,
            branchId = artifact.branchId,
            parentRevisionId = effectiveParent,
            contentText = content,
            contentHash = sha256(content),
            authorKind = authorKind,
            providerPrefix = providerPrefix,
            modelId = modelId,
            prompt = prompt,
            createdAt = now,
        )
        revisionDao.upsert(revision)
        artifactDao.upsert(artifact.copy(
            currentRevisionId = revisionId,
            previewText = content.take(200),
            updatedAt = now,
        ))
        revision
    }

    suspend fun get(id: String): CreativeArtifactEntity? = artifactDao.getById(id)

    suspend fun forProject(projectId: String): List<CreativeArtifactEntity> =
        artifactDao.allForProject(projectId)

    suspend fun forProjectByKind(projectId: String, kind: String): List<CreativeArtifactEntity> =
        artifactDao.forProjectByKind(projectId, kind)

    suspend fun archive(id: String) {
        val artifact = artifactDao.getById(id) ?: return
        artifactDao.upsert(artifact.copy(status = "archived", updatedAt = System.currentTimeMillis()))
    }

    suspend fun restore(id: String) {
        val artifact = artifactDao.getById(id) ?: return
        artifactDao.upsert(artifact.copy(status = "ready", updatedAt = System.currentTimeMillis()))
    }

    suspend fun lineage(revisionId: String): List<String> =
        revisionDao.ancestryChain(revisionId)

    suspend fun revisionsForArtifact(artifactId: String): List<CreativeRevisionEntity> =
        revisionDao.forArtifact(artifactId)

    fun observeForProject(projectId: String) = artifactDao.observeForProject(projectId)

    private fun sha256(text: String): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        return digest.digest(text.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
    }
}