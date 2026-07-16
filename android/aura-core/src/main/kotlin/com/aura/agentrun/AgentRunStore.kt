package com.aura.agentrun

import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Domain store wrapping [AgentRunDao], [GoalDao], [StepDao],
 * [AgentEventDao], [ApprovalRequestDao], and [RunCheckpointDao].
 *
 * Provides CRUD, DAG step planning, checkpoint/resume, and goal
 * verification. All mutations are mutex-protected per run.
 */
@Singleton
class AgentRunStore @Inject constructor(
    private val runDao: AgentRunDao,
    private val goalDao: GoalDao,
    private val stepDao: StepDao,
    private val eventDao: AgentEventDao,
    private val approvalDao: ApprovalRequestDao,
    private val checkpointDao: RunCheckpointDao,
    private val dagResolver: DagResolver,
) {
    private val json = Json { ignoreUnknownKeys = true }
    private val mutex = Mutex()

    suspend fun createRun(
        trigger: kotlin.String,
        goalDescription: kotlin.String,
        conversationId: kotlin.String = "",
        modelId: kotlin.String = "",
    ): AgentRunEntity = mutex.withLock {
        val goalId = UUID.randomUUID().toString()
        val runId = UUID.randomUUID().toString()
        val now = System.currentTimeMillis()
        val goal = GoalEntity(
            id = goalId,
            agentRunId = runId,
            description = goalDescription,
        )
        goalDao.upsert(goal)
        val run = AgentRunEntity(
            id = runId,
            goalId = goalId,
            status = "RUNNING",
            triggerType = trigger,
            conversationId = conversationId,
            modelId = modelId,
            startedAt = now,
            updatedAt = now,
        )
        runDao.upsert(run)
        emitEvent(runId, "RUN_STARTED")
        run
    }

    suspend fun loadRun(id: kotlin.String): AgentRunEntity? = runDao.getById(id)

    suspend fun listRecent(limit: Int = 50): List<AgentRunEntity> = runDao.recent(limit)

    suspend fun updateStatus(id: kotlin.String, status: kotlin.String) {
        runDao.updateStatus(id, status, System.currentTimeMillis())
    }

    suspend fun finish(id: kotlin.String, status: kotlin.String, error: kotlin.String = "") {
        runDao.finish(id, status, error, System.currentTimeMillis())
        emitEvent(id, if (status == "COMPLETED") "RUN_COMPLETED" else "RUN_FAILED")
    }

    suspend fun planSteps(runId: kotlin.String, steps: List<StepSpec>) = mutex.withLock {
        stepDao.upsertAll(steps.mapIndexed { index, spec ->
            StepEntity(
                id = UUID.randomUUID().toString(),
                agentRunId = runId,
                toolName = spec.toolName,
                toolArgs = spec.toolArgs,
                dependsOn = spec.dependsOn,
                position = index,
            )
        })
    }

    suspend fun readySteps(runId: kotlin.String): List<StepEntity> {
        val steps = stepDao.forRun(runId)
        return dagResolver.readySteps(steps)
    }

    suspend fun completeStep(stepId: kotlin.String, result: kotlin.String) {
        val step = stepDao.getById(stepId) ?: return
        stepDao.complete(stepId, "SUCCESS", result, System.currentTimeMillis())
        emitEvent(step.agentRunId, "STEP_COMPLETED", stepId = stepId)
    }

    suspend fun failStep(stepId: kotlin.String, error: kotlin.String) {
        val step = stepDao.getById(stepId) ?: return
        stepDao.fail(stepId, "FAILED", error, System.currentTimeMillis())
        emitEvent(step.agentRunId, "STEP_FAILED", stepId = stepId, success = false)
    }

    suspend fun checkpoint(runId: kotlin.String): RunCheckpointEntity = mutex.withLock {
        val steps = stepDao.forRun(runId)
        val state = CheckpointState(
            activeStepIds = steps.filter { it.status == "PENDING" || it.status == "RUNNING" }.map { it.id },
        )
        val checkpoint = RunCheckpointEntity(
            id = UUID.randomUUID().toString(),
            agentRunId = runId,
            stateJson = json.encodeToString(state),
        )
        checkpointDao.upsert(checkpoint)
        checkpointDao.cleanupOld(runId, checkpoint.id)
        emitEvent(runId, "CHECKPOINT")
        checkpoint
    }

    suspend fun resumeFromCheckpoint(runId: kotlin.String): List<StepEntity> {
        val checkpoint = checkpointDao.latestForRun(runId) ?: return emptyList()
        val state = runCatching { json.decodeFromString<CheckpointState>(checkpoint.stateJson) }.getOrNull()
            ?: return emptyList()
        val steps = stepDao.forRun(runId)
        return steps.filter { it.id in state.activeStepIds && it.status == "PENDING" }
    }

    suspend fun eventsForRun(runId: kotlin.String): List<AgentEventEntity> =
        eventDao.forRun(runId)

    suspend fun pendingApprovals(runId: kotlin.String): List<ApprovalRequestEntity> =
        approvalDao.pendingForRun(runId)

    suspend fun requestApproval(
        runId: kotlin.String,
        stepId: kotlin.String,
        toolName: kotlin.String,
        rationale: kotlin.String,
        expiresAt: kotlin.Long = 0L,
    ): ApprovalRequestEntity = mutex.withLock {
        val approval = ApprovalRequestEntity(
            id = UUID.randomUUID().toString(),
            agentRunId = runId,
            stepId = stepId,
            toolName = toolName,
            rationale = rationale,
            expiresAt = expiresAt,
        )
        approvalDao.upsert(approval)
        emitEvent(runId, "APPROVAL_REQUESTED", stepId = stepId, toolName = toolName)
        approval
    }

    suspend fun approve(id: kotlin.String) {
        approvalDao.decide(id, "APPROVED", "", System.currentTimeMillis())
        val approval = approvalDao.getById(id) ?: return
        emitEvent(approval.agentRunId, "APPROVAL_DECIDED", stepId = approval.stepId)
    }

    suspend fun deny(id: kotlin.String, reason: kotlin.String = "") {
        approvalDao.decide(id, "DENIED", reason, System.currentTimeMillis())
        val approval = approvalDao.getById(id) ?: return
        emitEvent(approval.agentRunId, "APPROVAL_DECIDED", stepId = approval.stepId, success = false)
    }

    private suspend fun emitEvent(
        runId: kotlin.String,
        type: kotlin.String,
        stepId: kotlin.String? = null,
        toolName: kotlin.String? = null,
        success: kotlin.Boolean = true,
    ) {
        eventDao.insert(
            AgentEventEntity(
                id = UUID.randomUUID().toString(),
                agentRunId = runId,
                stepId = stepId,
                timestamp = System.currentTimeMillis(),
                type = type,
                toolName = toolName,
                success = success,
            )
        )
    }
}

data class StepSpec(
    val toolName: kotlin.String,
    val toolArgs: kotlin.String = "{}",
    val dependsOn: kotlin.String = "[]",
)

@Serializable
data class CheckpointState(
    val activeStepIds: List<kotlin.String> = emptyList(),
)