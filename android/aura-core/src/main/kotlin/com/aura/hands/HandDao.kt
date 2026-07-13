package com.aura.hands

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface HandDao {

    @Query("SELECT * FROM hands ORDER BY createdAt DESC")
    suspend fun getAll(): List<Hand>

    @Query("SELECT * FROM hands ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<Hand>>

    @Query("SELECT * FROM hands WHERE enabled = 1 ORDER BY createdAt DESC")
    suspend fun getEnabled(): List<Hand>

    @Query("SELECT * FROM hands WHERE name = :name LIMIT 1")
    suspend fun getByName(name: String): Hand?

    @Query("SELECT * FROM hands WHERE id = :id LIMIT 1")
    suspend fun getById(id: String): Hand?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(hand: Hand)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(hands: List<Hand>)

    @Update
    suspend fun update(hand: Hand)

    @Query("DELETE FROM hands WHERE name = :name")
    suspend fun deleteByName(name: String)

    @Query("DELETE FROM hands WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("DELETE FROM hands")
    suspend fun deleteAll()

    @Query("SELECT * FROM hand_runs ORDER BY startedAt DESC LIMIT :limit")
    fun observeRecentRuns(limit: Int = 100): Flow<List<HandRun>>

    @Query("SELECT * FROM hand_runs WHERE handId = :handId ORDER BY startedAt DESC LIMIT :limit")
    fun observeRunsForHand(handId: String, limit: Int = 50): Flow<List<HandRun>>

    @Query("SELECT * FROM hand_runs ORDER BY startedAt DESC")
    suspend fun allRunsForBackup(): List<HandRun>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertRun(run: HandRun)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAllRuns(runs: List<HandRun>)

    @Update
    suspend fun updateRun(run: HandRun)

    @Query("DELETE FROM hand_runs")
    suspend fun deleteRunHistory()
}
