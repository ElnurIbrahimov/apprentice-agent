package com.aura.agent.runtime

import kotlinx.serialization.Serializable

/**
 * Correlation IDs for attributing agent actions to runs and steps.
 * Passed through [com.aura.agent.ToolContext] and the event ledger.
 */
data class RunContext(
    val runId: kotlin.String,
    val stepId: kotlin.String? = null,
    val conversationId: kotlin.String? = null,
    val projectId: kotlin.String? = null,
    val parentEventId: kotlin.String? = null,
)

/**
 * Append-only event in the agent trace ledger. Never mutated after
 * creation. Corrections create new events.
 *
 * Payloads are redacted — never include secrets, full API responses,
 * or private user data. Include only enough for replay and debugging.
 */
@Serializable
data class AgentTraceEvent(
    val id: kotlin.String,
    val runId: kotlin.String,
    val stepId: kotlin.String? = null,
    val parentEventId: kotlin.String? = null,
    val timestamp: kotlin.Long = System.currentTimeMillis(),
    val type: TraceEventType,
    val toolName: kotlin.String? = null,
    val redactedPayload: kotlin.String = "",
    val durationMs: kotlin.Long = 0L,
    val success: kotlin.Boolean = true,
    val errorCode: kotlin.String? = null,
)

@Serializable
enum class TraceEventType {
    RUN_STARTED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_CANCELLED,
    STEP_STARTED,
    STEP_COMPLETED,
    STEP_FAILED,
    TOOL_CALL,
    TOOL_RESULT,
    APPROVAL_REQUESTED,
    APPROVAL_DECIDED,
    CHECKPOINT,
    SUBAGENT_SPAWNED,
    SUBAGENT_COMPLETED,
    ARTIFACT_CREATED,
    ARTIFACT_REVISED,
    PROVIDER_FAILOVER,
    CONTEXT_TRUNCATED,
    MEMORY_RECALLED,
    KNOWLEDGE_EXTRACTED,
}