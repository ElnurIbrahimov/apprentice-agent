package com.aura.creative

import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Domain store for [CreativeBranchEntity]. Creates, lists, and manages
 * branch lifecycle within a project. The main branch is created
 * automatically when a project is initialized.
 */
@Singleton
class CreativeBranchStore @Inject constructor(
    private val branchDao: CreativeBranchDao,
) {
    private val mutex = Mutex()

    suspend fun createMainBranch(projectId: String): CreativeBranchEntity = mutex.withLock {
        val existing = branchDao.forProject(projectId)
        if (existing.any { it.name == "main" }) {
            return@withLock existing.first { it.name == "main" }
        }
        val branch = CreativeBranchEntity(
            id = UUID.randomUUID().toString(),
            projectId = projectId,
            name = "main",
            status = "active",
        )
        branchDao.upsert(branch)
        branch
    }

    suspend fun branchFrom(
        projectId: String,
        baseRevisionId: String?,
        name: String,
    ): CreativeBranchEntity = mutex.withLock {
        val branch = CreativeBranchEntity(
            id = UUID.randomUUID().toString(),
            projectId = projectId,
            name = name,
            baseRevisionId = baseRevisionId,
            status = "active",
        )
        branchDao.upsert(branch)
        branch
    }

    suspend fun forProject(projectId: String): List<CreativeBranchEntity> =
        branchDao.forProject(projectId)

    suspend fun getById(id: String): CreativeBranchEntity? = branchDao.getById(id)

    suspend fun archive(id: String) {
        val branch = branchDao.getById(id) ?: return
        branchDao.upsert(branch.copy(status = "archived", updatedAt = System.currentTimeMillis()))
    }

    suspend fun updateHead(branchId: String, headRevisionId: String) {
        val branch = branchDao.getById(branchId) ?: return
        branchDao.upsert(branch.copy(headRevisionId = headRevisionId, updatedAt = System.currentTimeMillis()))
    }
}