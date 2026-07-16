package com.aura.agentrun

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * A durable agent execution. An AgentRun is a top-level execution triggered
 * by a user query, schedule, proactive engine, or subagent spawn. Each run
 * has one [GoalEntity] (the outcome definition) and multiple [StepEntity]
 * records (planned + completed actions).
 *
 * AgentRuns survive process death — they are persisted in Room and can be
 * resumed after a crash or reboot.
 */
@Entity(
    tableName = "agent_runs",
    indices = [
        Index(value = ["status"]),
        Index(value = ["triggerType"]),
        Index(value = ["startedAt"]),
    ],
)
data class AgentRunEntity(
    @PrimaryKey val id: kotlin.String,
    val goalId: kotlin.String,
    /** PENDING, PLANNING, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED */
    val status: kotlin.String = "PENDING",
    /** USER_QUERY, SCHEDULED, PROACTIVE, SUBAGENT, RESUME */
    val triggerType: kotlin.String,
    val triggerPayload: kotlin.String = "",
    val modelId: kotlin.String = "",
    val specialistName: kotlin.String? = null,
    val conversationId: kotlin.String = "",
    val parentRunId: kotlin.String? = null,
    val startedAt: kotlin.Long = System.currentTimeMillis(),
    val updatedAt: kotlin.Long = startedAt,
    val finishedAt: kotlin.Long? = null,
    val errorMessage: kotlin.String = "",
    /** JSON: retry count, tags, budget metadata */
    val metadata: kotlin.String = "{}",
)

/**
 * The desired outcome for an [AgentRunEntity]. Contains the natural language
 * goal and the definition of done (postcondition checks).
 */
@Entity(
    tableName = "agent_goals",
    indices = [Index(value = ["agentRunId"])],
)
data class GoalEntity(
    @PrimaryKey val id: kotlin.String,
    val agentRunId: kotlin.String,
    val description: kotlin.String,
    /** JSON array of postcondition checks */
    val doneCriteria: kotlin.String = "[]",
    val successEvaluation: kotlin.String = "",
    val isAchieved: kotlin.Boolean = false,
    val achievedAt: kotlin.Long? = null,
)

/**
 * A single step in an [AgentRunEntity]. Steps form a DAG via [dependsOn]
 * (JSON array of step IDs). A step is ready when all dependencies have
 * status=SUCCESS.
 */
@Entity(
    tableName = "agent_steps",
    indices = [
        Index(value = ["agentRunId"]),
        Index(value = ["parentStepId"]),
        Index(value = ["status"]),
    ],
)
data class StepEntity(
    @PrimaryKey val id: kotlin.String,
    val agentRunId: kotlin.String,
    val parentStepId: kotlin.String? = null,
    val toolName: kotlin.String = "",
    val toolArgs: kotlin.String = "{}",
    /** PENDING, RUNNING, SUCCESS, FAILED, SKIPPED, BLOCKED */
    val status: kotlin.String = "PENDING",
    /** JSON array of step IDs this step depends on (DAG) */
    val dependsOn: kotlin.String = "[]",
    val result: kotlin.String = "",
    val errorMessage: kotlin.String = "",
    val startedAt: kotlin.Long? = null,
    val finishedAt: kotlin.Long? = null,
    /** Postcondition verification result JSON */
    val postconditionResult: kotlin.String = "",
    /** Ordering hint for linear display */
    val position: Int = 0,
)

/**
 * Append-only event ledger for agent runs. Every action (tool call, approval,
 * checkpoint, etc.) is recorded as an event for replay and audit.
 */
@Entity(
    tableName = "agent_events",
    indices = [
        Index(value = ["agentRunId"]),
        Index(value = ["timestamp"]),
    ],
)
data class AgentEventEntity(
    @PrimaryKey val id: kotlin.String,
    val agentRunId: kotlin.String,
    val stepId: kotlin.String? = null,
    val parentEventId: kotlin.String? = null,
    val timestamp: kotlin.Long = System.currentTimeMillis(),
    val type: kotlin.String,
    val toolName: kotlin.String? = null,
    val redactedPayload: kotlin.String = "",
    val durationMs: kotlin.Long = 0L,
    val success: kotlin.Boolean = true,
    val errorCode: kotlin.String? = null,
)

/**
 * A durable approval request. Scoped to one step/tool/resource/action.
 * Has an expiry and a decision.
 */
@Entity(
    tableName = "agent_approvals",
    indices = [
        Index(value = ["agentRunId"]),
        Index(value = ["status"]),
    ],
)
data class ApprovalRequestEntity(
    @PrimaryKey val id: kotlin.String,
    val agentRunId: kotlin.String,
    val stepId: kotlin.String,
    val toolName: kotlin.String,
    val rationale: kotlin.String,
    /** PENDING, APPROVED, DENIED, EXPIRED */
    val status: kotlin.String = "PENDING",
    val decisionAt: kotlin.Long? = null,
    val denyReason: kotlin.String = "",
    val expiresAt: kotlin.Long = 0L,
)

/**
 * A serialized snapshot of a run's execution state. Used for resuming
 * after process death. Contains the active frontier (ready steps),
 * resolved variables, and pending callbacks.
 */
@Entity(
    tableName = "agent_checkpoints",
    indices = [Index(value = ["agentRunId"])],
)
data class RunCheckpointEntity(
    @PrimaryKey val id: kotlin.String,
    val agentRunId: kotlin.String,
    /** JSON: active step IDs, variables, pending callbacks, definition version */
    val stateJson: kotlin.String,
    val createdAt: kotlin.Long = System.currentTimeMillis(),
)