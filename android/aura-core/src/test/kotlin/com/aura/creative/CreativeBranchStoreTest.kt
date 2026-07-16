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

class CreativeBranchStoreTest {

    private val branchDao = mockk<CreativeBranchDao>(relaxed = true)
    private val store = CreativeBranchStore(branchDao)

    @Test
    fun createMainBranch_creates_branch_when_none_exist() = runTest {
        coEvery { branchDao.forProject("p1") } returns emptyList()
        val slot = slot<CreativeBranchEntity>()
        coEvery { branchDao.upsert(capture(slot)) } returns Unit

        val branch = store.createMainBranch("p1")

        assertEquals("main", branch.name)
        assertEquals("p1", branch.projectId)
        assertEquals("active", branch.status)
        assertNotNull(branch.id)
    }

    @Test
    fun createMainBranch_returns_existing_when_already_present() = runTest {
        val existing = CreativeBranchEntity(
            id = "b1",
            projectId = "p1",
            name = "main",
            status = "active",
        )
        coEvery { branchDao.forProject("p1") } returns listOf(existing)

        val branch = store.createMainBranch("p1")

        assertEquals("b1", branch.id)
        assertEquals("main", branch.name)
    }

    @Test
    fun branchFrom_creates_new_branch() = runTest {
        coEvery { branchDao.forProject("p1") } returns emptyList()
        val slot = slot<CreativeBranchEntity>()
        coEvery { branchDao.upsert(capture(slot)) } returns Unit

        val branch = store.branchFrom("p1", "rev5", "alternate-ending")

        assertEquals("alternate-ending", branch.name)
        assertEquals("rev5", branch.baseRevisionId)
        assertEquals("p1", branch.projectId)
    }

    @Test
    fun archive_sets_status_to_archived() = runTest {
        val existing = CreativeBranchEntity(
            id = "b1",
            projectId = "p1",
            name = "alt",
            status = "active",
        )
        coEvery { branchDao.getById("b1") } returns existing
        val slot = slot<CreativeBranchEntity>()
        coEvery { branchDao.upsert(capture(slot)) } returns Unit

        store.archive("b1")

        assertEquals("archived", slot.captured.status)
    }

    @Test
    fun updateHead_updates_head_revision_id() = runTest {
        val existing = CreativeBranchEntity(
            id = "b1",
            projectId = "p1",
            name = "main",
            headRevisionId = "rev1",
            status = "active",
        )
        coEvery { branchDao.getById("b1") } returns existing
        val slot = slot<CreativeBranchEntity>()
        coEvery { branchDao.upsert(capture(slot)) } returns Unit

        store.updateHead("b1", "rev2")

        assertEquals("rev2", slot.captured.headRevisionId)
    }
}