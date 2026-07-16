package com.aura.taste

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface PreferenceSignalDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(signal: PreferenceSignalEntity)

    @Query("SELECT * FROM preference_signals WHERE projectId = :projectId ORDER BY createdAt DESC LIMIT :limit")
    suspend fun forProject(projectId: kotlin.String, limit: Int = 200): List<PreferenceSignalEntity>

    @Query("SELECT * FROM preference_signals WHERE projectId = '' OR projectId IS NULL ORDER BY createdAt DESC LIMIT :limit")
    suspend fun global(limit: Int = 200): List<PreferenceSignalEntity>

    @Query("SELECT * FROM preference_signals WHERE category = :category ORDER BY createdAt DESC LIMIT :limit")
    suspend fun byCategory(category: kotlin.String, limit: Int = 100): List<PreferenceSignalEntity>

    @Query("SELECT * FROM preference_signals ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<PreferenceSignalEntity>

    @Query("DELETE FROM preference_signals WHERE id = :id")
    suspend fun delete(id: kotlin.String)

    @Query("DELETE FROM preference_signals WHERE projectId = :projectId")
    suspend fun deleteForProject(projectId: kotlin.String)

    @Query("DELETE FROM preference_signals WHERE projectId = '' OR projectId IS NULL")
    suspend fun deleteGlobal()

    @Query("SELECT COUNT(*) FROM preference_signals")
    fun count(): Flow<Int>
}

@Dao
interface StyleProfileDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(profile: StyleProfileEntity)

    @Query("SELECT * FROM style_profiles WHERE projectId = :projectId LIMIT 1")
    suspend fun forProject(projectId: kotlin.String): StyleProfileEntity?

    @Query("SELECT * FROM style_profiles WHERE projectId = '' OR projectId IS NULL LIMIT 1")
    suspend fun global(): StyleProfileEntity?

    @Query("SELECT * FROM style_profiles ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<StyleProfileEntity>
}

@Dao
interface ReferenceIdentityDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(identity: ReferenceIdentityEntity)

    @Update
    suspend fun update(identity: ReferenceIdentityEntity)

    @Query("SELECT * FROM reference_identities WHERE projectId = :projectId ORDER BY identityType, name")
    suspend fun forProject(projectId: kotlin.String): List<ReferenceIdentityEntity>

    @Query("SELECT * FROM reference_identities WHERE projectId = :projectId AND identityType = :type ORDER BY name")
    suspend fun byType(projectId: kotlin.String, type: kotlin.String): List<ReferenceIdentityEntity>

    @Query("SELECT * FROM reference_identities WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): ReferenceIdentityEntity?

    @Query("SELECT * FROM reference_identities ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<ReferenceIdentityEntity>

    @Query("DELETE FROM reference_identities WHERE id = :id")
    suspend fun delete(id: kotlin.String)
}

@Dao
interface RoutingOutcomeDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(outcome: RoutingOutcomeEntity)

    @Query("SELECT * FROM routing_outcomes WHERE modelRole = :role ORDER BY createdAt DESC LIMIT :limit")
    suspend fun forRole(role: kotlin.String, limit: Int = 100): List<RoutingOutcomeEntity>

    @Query("SELECT modelId, COUNT(*) as count, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes FROM routing_outcomes WHERE modelRole = :role GROUP BY modelId")
    suspend fun statsForRole(role: kotlin.String): List<RoutingStats>

    @Query("SELECT * FROM routing_outcomes ORDER BY createdAt ASC")
    suspend fun allForBackup(): List<RoutingOutcomeEntity>
}

data class RoutingStats(
    val modelId: kotlin.String,
    val count: Int,
    val successes: Int,
)