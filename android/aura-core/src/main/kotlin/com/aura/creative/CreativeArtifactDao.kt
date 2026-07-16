package com.aura.creative

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface CreativeArtifactDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(artifact: CreativeArtifactEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(artifacts: List<CreativeArtifactEntity>)

    @Query("SELECT * FROM creative_artifacts WHERE projectId = :projectId ORDER BY updatedAt DESC")
    fun observeForProject(projectId: kotlin.String): Flow<List<CreativeArtifactEntity>>

    @Query("SELECT * FROM creative_artifacts WHERE projectId = :projectId AND kind = :kind ORDER BY updatedAt DESC")
    suspend fun forProjectByKind(projectId: kotlin.String, kind: kotlin.String): List<CreativeArtifactEntity>

    @Query("SELECT * FROM creative_artifacts WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): CreativeArtifactEntity?

    @Query("SELECT * FROM creative_artifacts WHERE projectId = :projectId ORDER BY updatedAt DESC")
    suspend fun allForProject(projectId: kotlin.String): List<CreativeArtifactEntity>

    @Query("SELECT * FROM creative_artifacts ORDER BY updatedAt DESC")
    suspend fun allForBackup(): List<CreativeArtifactEntity>

    @Query("SELECT COUNT(*) FROM creative_artifacts WHERE projectId = :projectId")
    suspend fun countForProject(projectId: kotlin.String): Int

    @Query("DELETE FROM creative_artifacts WHERE id = :id")
    suspend fun delete(id: kotlin.String)

    @Query("DELETE FROM creative_artifacts WHERE projectId = :projectId")
    suspend fun deleteAllForProject(projectId: kotlin.String)
}

@Dao
interface CreativeRevisionDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(revision: CreativeRevisionEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(revisions: List<CreativeRevisionEntity>)

    @Query("SELECT * FROM creative_revisions WHERE artifactId = :artifactId ORDER BY createdAt DESC")
    suspend fun forArtifact(artifactId: kotlin.String): List<CreativeRevisionEntity>

    @Query("SELECT * FROM creative_revisions WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): CreativeRevisionEntity?

    @Query("""WITH RECURSIVE ancestry(id, parentRevisionId, depth) AS (
                SELECT id, parentRevisionId, 0 FROM creative_revisions WHERE id = :revisionId
                UNION ALL
                SELECT cr.id, cr.parentRevisionId, a.depth + 1
                FROM creative_revisions cr INNER JOIN ancestry a ON cr.id = a.parentRevisionId
                WHERE a.depth < 100
              ) SELECT id FROM ancestry ORDER BY depth""")
    suspend fun ancestryChain(revisionId: kotlin.String): List<kotlin.String>

    @Query("SELECT * FROM creative_revisions ORDER BY createdAt DESC")
    suspend fun allForBackup(): List<CreativeRevisionEntity>

    @Query("DELETE FROM creative_revisions WHERE artifactId = :artifactId")
    suspend fun deleteForArtifact(artifactId: kotlin.String)
}

@Dao
interface CreativeBranchDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(branch: CreativeBranchEntity)

    @Query("SELECT * FROM creative_branches WHERE projectId = :projectId ORDER BY createdAt ASC")
    suspend fun forProject(projectId: kotlin.String): List<CreativeBranchEntity>

    @Query("SELECT * FROM creative_branches WHERE projectId = :projectId AND status = 'active' ORDER BY createdAt ASC")
    fun observeActive(projectId: kotlin.String): Flow<List<CreativeBranchEntity>>

    @Query("SELECT * FROM creative_branches WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): CreativeBranchEntity?

    @Query("SELECT * FROM creative_branches ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<CreativeBranchEntity>

    @Query("DELETE FROM creative_branches WHERE id = :id")
    suspend fun delete(id: kotlin.String)
}

@Dao
interface CreativeGenerationJobDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(job: CreativeGenerationJobEntity)

    @Query("SELECT * FROM creative_generation_jobs WHERE status IN ('queued', 'running', 'waiting_provider') ORDER BY createdAt ASC")
    suspend fun pendingJobs(): List<CreativeGenerationJobEntity>

    @Query("SELECT * FROM creative_generation_jobs WHERE projectId = :projectId ORDER BY createdAt DESC")
    fun observeForProject(projectId: kotlin.String): Flow<List<CreativeGenerationJobEntity>>

    @Query("SELECT * FROM creative_generation_jobs WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): CreativeGenerationJobEntity?

    @Query("UPDATE creative_generation_jobs SET status = :status, progress = :progress, updatedAt = :timestamp WHERE id = :id")
    suspend fun updateStatus(id: kotlin.String, status: kotlin.String, progress: Int, timestamp: kotlin.Long)

    @Query("UPDATE creative_generation_jobs SET status = :status, resultArtifactIdsJson = :resultJson, updatedAt = :timestamp WHERE id = :id")
    suspend fun complete(id: kotlin.String, status: kotlin.String, resultJson: kotlin.String, timestamp: kotlin.Long)

    @Query("DELETE FROM creative_generation_jobs WHERE status IN ('succeeded', 'failed', 'cancelled') AND createdAt < :cutoff")
    suspend fun cleanupOld(cutoff: kotlin.Long)
}