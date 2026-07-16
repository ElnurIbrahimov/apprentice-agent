package com.aura.creative

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CreativeArtifactStoreTest {

    private val artifactDao = mockk<CreativeArtifactDao>(relaxed = true)
    private val revisionDao = mockk<CreativeRevisionDao>(relaxed = true)
    private val branchDao = mockk<CreativeBranchDao>(relaxed = true)
    private val store = CreativeArtifactStore(artifactDao, revisionDao, branchDao)

    @Test
    fun create_persists_artifact_and_initial_revision() = runTest {
        val artifactSlot = slot<CreativeArtifactEntity>()
        val revisionSlot = slot<CreativeRevisionEntity>()
        coEvery { artifactDao.upsert(capture(artifactSlot)) } returns Unit
        coEvery { revisionDao.upsert(capture(revisionSlot)) } returns Unit

        val artifact = store.create(
            projectId = "p1",
            branchId = "b1",
            kind = "text",
            title = "Chapter 1",
            initialContent = "It was a dark and stormy night.",
        )

        assertEquals("p1", artifact.projectId)
        assertEquals("text", artifact.kind)
        assertEquals("Chapter 1", artifact.title)
        assertEquals("ready", artifact.status)
        assertTrue(artifact.previewText.isNotBlank())
        assertNotNull(artifact.currentRevisionId)

        val capturedRevision = revisionSlot.captured
        assertEquals(artifact.id, capturedRevision.artifactId)
        assertEquals("manual", capturedRevision.authorKind)
        assertNotNull(capturedRevision.contentHash)
    }

    @Test
    fun revise_creates_new_revision_and_updates_current() = runTest {
        val existing = CreativeArtifactEntity(
            id = "a1",
            projectId = "p1",
            branchId = "b1",
            kind = "text",
            title = "Chapter 1",
            currentRevisionId = "rev1",
            status = "ready",
        )
        coEvery { artifactDao.getById("a1") } returns existing
        val revisionSlot = slot<CreativeRevisionEntity>()
        val artifactSlot = slot<CreativeArtifactEntity>()
        coEvery { revisionDao.upsert(capture(revisionSlot)) } returns Unit
        coEvery { artifactDao.upsert(capture(artifactSlot)) } returns Unit

        val revision = store.revise(
            artifactId = "a1",
            content = "Updated content",
            authorKind = "generation",
        )

        assertEquals("a1", revision.artifactId)
        assertEquals("rev1", revision.parentRevisionId)
        assertEquals("generation", revision.authorKind)
        assertEquals("Updated content", revision.contentText)

        val updatedArtifact = artifactSlot.captured
        assertEquals(revision.id, updatedArtifact.currentRevisionId)
        assertEquals("Updated content", updatedArtifact.previewText)
    }

    @Test
    fun archive_sets_status_without_deleting() = runTest {
        val artifact = CreativeArtifactEntity(
            id = "a1",
            projectId = "p1",
            branchId = "b1",
            kind = "image",
            title = "Cover",
            status = "ready",
        )
        coEvery { artifactDao.getById("a1") } returns artifact
        val slot = slot<CreativeArtifactEntity>()
        coEvery { artifactDao.upsert(capture(slot)) } returns Unit

        store.archive("a1")

        assertEquals("archived", slot.captured.status)
    }

    @Test
    fun restore_sets_status_back_to_ready() = runTest {
        val artifact = CreativeArtifactEntity(
            id = "a1",
            projectId = "p1",
            branchId = "b1",
            kind = "image",
            title = "Cover",
            status = "archived",
        )
        coEvery { artifactDao.getById("a1") } returns artifact
        val slot = slot<CreativeArtifactEntity>()
        coEvery { artifactDao.upsert(capture(slot)) } returns Unit

        store.restore("a1")

        assertEquals("ready", slot.captured.status)
    }

    @Test
    fun lineage_calls_ancestry_chain() = runTest {
        coEvery { revisionDao.ancestryChain("rev3") } returns listOf("rev3", "rev2", "rev1")
        val lineage = store.lineage("rev3")
        assertEquals(listOf("rev3", "rev2", "rev1"), lineage)
    }
}