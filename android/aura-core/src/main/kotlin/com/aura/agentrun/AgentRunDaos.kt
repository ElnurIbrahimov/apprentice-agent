package com.aura.agentrun

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface AgentRunDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(run: AgentRunEntity)

    @Query("SELECT * FROM agent_runs ORDER BY startedAt DESC LIMIT :limit")
    suspend fun recent(limit: Int = 50): List<AgentRunEntity>

    @Query("SELECT * FROM agent_runs WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): AgentRunEntity?

    @Query("SELECT * FROM agent_runs WHERE status IN ('PENDING', 'PLANNING', 'RUNNING', 'PAUSED') ORDER BY startedAt DESC")
    suspend fun activeRuns(): List<AgentRunEntity>

    @Query("SELECT * FROM agent_runs ORDER BY startedAt DESC")
    fun observeAll(): Flow<List<AgentRunEntity>>

    @Query("UPDATE agent_runs SET status = :status, updatedAt = :timestamp WHERE id = :id")
    suspend fun updateStatus(id: kotlin.String, status: kotlin.String, timestamp: kotlin.Long)

    @Query("UPDATE agent_runs SET finishedAt = :timestamp, errorMessage = :error, status = :status WHERE id = :id")
    suspend fun finish(id: kotlin.String, status: kotlin.String, error: kotlin.String, timestamp: kotlin.Long)

    @Query("DELETE FROM agent_runs WHERE id = :id")
    suspend fun delete(id: kotlin.String)
}

@Dao
interface GoalDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(goal: GoalEntity)

    @Query("SELECT * FROM agent_goals WHERE agentRunId = :runId LIMIT 1")
    suspend fun forRun(runId: kotlin.String): GoalEntity?

    @Query("SELECT * FROM agent_goals WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): GoalEntity?

    @Query("UPDATE agent_goals SET isAchieved = :achieved, achievedAt = :timestamp WHERE id = :id")
    suspend fun markAchieved(id: kotlin.String, achieved: kotlin.Boolean, timestamp: kotlin.Long)
}

@Dao
interface StepDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(step: StepEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertAll(steps: List<StepEntity>)

    @Query("SELECT * FROM agent_steps WHERE agentRunId = :runId ORDER BY position ASC")
    suspend fun forRun(runId: kotlin.String): List<StepEntity>

    @Query("SELECT * FROM agent_steps WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): StepEntity?

    @Query("UPDATE agent_steps SET status = :status, result = :result, finishedAt = :timestamp WHERE id = :id")
    suspend fun complete(id: kotlin.String, status: kotlin.String, result: kotlin.String, timestamp: kotlin.Long)

    @Query("UPDATE agent_steps SET status = :status, errorMessage = :error, finishedAt = :timestamp WHERE id = :id")
    suspend fun fail(id: kotlin.String, status: kotlin.String, error: kotlin.String, timestamp: kotlin.Long)

    @Query("UPDATE agent_steps SET status = :status, startedAt = :timestamp WHERE id = :id")
    suspend fun markStarted(id: kotlin.String, status: kotlin.String, timestamp: kotlin.Long)

    @Query("UPDATE agent_steps SET postconditionResult = :result WHERE id = :id")
    suspend fun setPostcondition(id: kotlin.String, result: kotlin.String)
}

@Dao
interface AgentEventDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(event: AgentEventEntity)

    @Query("SELECT * FROM agent_events WHERE agentRunId = :runId ORDER BY timestamp ASC")
    suspend fun forRun(runId: kotlin.String): List<AgentEventEntity>

    @Query("SELECT * FROM agent_events WHERE agentRunId = :runId ORDER BY timestamp DESC LIMIT :limit")
    suspend fun recentForRun(runId: kotlin.String, limit: Int = 100): List<AgentEventEntity>
}

@Dao
interface ApprovalRequestDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(approval: ApprovalRequestEntity)

    @Query("SELECT * FROM agent_approvals WHERE agentRunId = :runId AND status = 'PENDING' ORDER BY id ASC")
    suspend fun pendingForRun(runId: kotlin.String): List<ApprovalRequestEntity>

    @Query("SELECT * FROM agent_approvals WHERE id = :id LIMIT 1")
    suspend fun getById(id: kotlin.String): ApprovalRequestEntity?

    @Query("UPDATE agent_approvals SET status = :status, decisionAt = :timestamp, denyReason = :reason WHERE id = :id")
    suspend fun decide(id: kotlin.String, status: kotlin.String, reason: kotlin.String, timestamp: kotlin.Long)
}

@Dao
interface RunCheckpointDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(checkpoint: RunCheckpointEntity)

    @Query("SELECT * FROM agent_checkpoints WHERE agentRunId = :runId ORDER BY createdAt DESC LIMIT 1")
    suspend fun latestForRun(runId: kotlin.String): RunCheckpointEntity?

    @Query("DELETE FROM agent_checkpoints WHERE agentRunId = :runId AND id != :keepId")
    suspend fun cleanupOld(runId: kotlin.String, keepId: kotlin.String)
}