package com.aura.creative

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface CreativeProjectDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(project: CreativeProjectEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(projects: List<CreativeProjectEntity>)

    @Query("SELECT * FROM creative_projects ORDER BY updatedAt DESC")
    fun observeAll(): Flow<List<CreativeProjectEntity>>

    @Query("SELECT * FROM creative_projects ORDER BY updatedAt DESC")
    suspend fun allForBackup(): List<CreativeProjectEntity>

    @Query("SELECT * FROM creative_projects WHERE id = :id LIMIT 1")
    suspend fun getById(id: String): CreativeProjectEntity?

    @Query("DELETE FROM creative_projects WHERE id = :id")
    suspend fun delete(id: String)

    @Query("DELETE FROM creative_projects")
    suspend fun deleteAll()
}