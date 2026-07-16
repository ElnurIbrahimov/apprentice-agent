package com.aura.creative

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface CanonFactDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(fact: CanonFactEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertAll(facts: List<CanonFactEntity>)

    @Query("SELECT * FROM canon_facts WHERE projectId = :projectId AND branchId = :branchId AND status = 'active' ORDER BY subjectType, subjectId")
    suspend fun activeForBranch(projectId: kotlin.String, branchId: kotlin.String): List<CanonFactEntity>

    @Query("SELECT * FROM canon_facts WHERE projectId = :projectId AND branchId = :branchId AND subjectType = :type AND subjectId = :subjectId AND status = 'active'")
    suspend fun forSubject(projectId: kotlin.String, branchId: kotlin.String, type: kotlin.String, subjectId: kotlin.String): List<CanonFactEntity>

    @Query("SELECT * FROM canon_facts WHERE projectId = :projectId AND branchId = :branchId AND predicate = :predicate AND status = 'active'")
    suspend fun byPredicate(projectId: kotlin.String, branchId: kotlin.String, predicate: kotlin.String): List<CanonFactEntity>

    @Query("SELECT * FROM canon_facts WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): CanonFactEntity?

    @Query("UPDATE canon_facts SET status = :status, updatedAt = :timestamp WHERE id = :id")
    suspend fun updateStatus(id: kotlin.String, status: kotlin.String, timestamp: kotlin.Long)

    @Query("SELECT * FROM canon_facts ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<CanonFactEntity>

    @Query("DELETE FROM canon_facts WHERE projectId = :projectId")
    suspend fun deleteForProject(projectId: kotlin.String)
}

@Dao
interface CreativeSimulationDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(sim: CreativeSimulationEntity)

    @Query("SELECT * FROM creative_simulations WHERE projectId = :projectId ORDER BY createdAt DESC")
    suspend fun forProject(projectId: kotlin.String): List<CreativeSimulationEntity>

    @Query("SELECT * FROM creative_simulations WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): CreativeSimulationEntity?

    @Query("UPDATE creative_simulations SET canonizedAt = :timestamp, canonizedFactIdsJson = :factIds WHERE id = :id")
    suspend fun canonize(id: kotlin.String, factIds: kotlin.String, timestamp: kotlin.Long)

    @Query("SELECT * FROM creative_simulations ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<CreativeSimulationEntity>
}

@Dao
interface ContinuityIssueDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(issue: ContinuityIssueEntity)

    @Query("SELECT * FROM continuity_issues WHERE projectId = :projectId AND branchId = :branchId AND status = 'open' ORDER BY severity DESC, createdAt DESC")
    fun observeOpen(projectId: kotlin.String, branchId: kotlin.String): Flow<List<ContinuityIssueEntity>>

    @Query("SELECT * FROM continuity_issues WHERE projectId = :projectId AND branchId = :branchId ORDER BY severity DESC, createdAt DESC")
    suspend fun forBranch(projectId: kotlin.String, branchId: kotlin.String): List<ContinuityIssueEntity>

    @Query("SELECT * FROM continuity_issues WHERE artifactId = :artifactId AND status = 'open'")
    suspend fun forArtifact(artifactId: kotlin.String): List<ContinuityIssueEntity>

    @Query("UPDATE continuity_issues SET status = :status, resolvedAt = :timestamp, resolvedBy = :resolver WHERE id = :id")
    suspend fun resolve(id: kotlin.String, status: kotlin.String, timestamp: kotlin.Long, resolver: kotlin.String)

    @Query("SELECT * FROM continuity_issues ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<ContinuityIssueEntity>
}

@Dao
interface ArtifactDependencyDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(dep: ArtifactDependencyEntity)

    @Query("SELECT * FROM artifact_dependencies WHERE sourceArtifactId = :artifactId")
    suspend fun dependentsOf(artifactId: kotlin.String): List<ArtifactDependencyEntity>

    @Query("SELECT * FROM artifact_dependencies WHERE targetArtifactId = :artifactId")
    suspend fun dependenciesOf(artifactId: kotlin.String): List<ArtifactDependencyEntity>

    @Query("DELETE FROM artifact_dependencies WHERE sourceArtifactId = :sourceId AND targetArtifactId = :targetId")
    suspend fun delete(sourceId: kotlin.String, targetId: kotlin.String)

    @Query("SELECT * FROM artifact_dependencies ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<ArtifactDependencyEntity>
}