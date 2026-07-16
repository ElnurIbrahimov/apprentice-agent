package com.aura.agents

import kotlinx.serialization.Serializable

/**
 * Specification for an isolated subagent worker. Subagents have their
 * own context, model role, tool allowlist, and budget. They do not
 * share mutable conversation history — they communicate through
 * artifacts and structured results.
 */
@Serializable
data class SubagentSpec(
    val role: kotlin.String,
    val objective: kotlin.String,
    /** Context bundle: artifact IDs, canon fact IDs, relevant text. */
    val contextArtifactIds: List<kotlin.String> = emptyList(),
    val contextText: kotlin.String = "",
    /** Model role from [com.aura.providers.ModelRole]. */
    val modelRole: kotlin.String = "CREATIVE_DRAFT",
    /** Tools this subagent is allowed to use. Empty = inherit parent's allowlist. */
    val toolAllowlist: List<kotlin.String> = emptyList(),
    /** Maximum execution time in milliseconds. 0 = inherit parent timeout. */
    val budgetMs: kotlin.Long = 0L,
    /** Maximum tool calls. 0 = no limit (parent budget still applies). */
    val maxToolCalls: Int = 10,
    /** Expected output schema name for structured results. */
    val outputSchema: kotlin.String = "",
)

/**
 * A task assigned to a subagent. Contains the spec and run metadata.
 */
@Serializable
data class SubagentTask(
    val id: kotlin.String,
    val spec: SubagentSpec,
    val parentRunId: kotlin.String,
    val createdAt: kotlin.Long = System.currentTimeMillis(),
)

/**
 * The result returned by a subagent. Contains structured output,
 * artifacts produced, and a concise rationale (not raw chain-of-thought).
 */
@Serializable
data class SubagentResult(
    val taskId: kotlin.String,
    val success: kotlin.Boolean,
    val output: kotlin.String = "",
    /** Structured JSON output if the spec requested a schema. */
    val structuredOutputJson: kotlin.String = "",
    /** Artifact IDs created by this subagent. */
    val createdArtifactIds: List<kotlin.String> = emptyList(),
    /** Concise rationale for the user (not internal reasoning). */
    val rationale: kotlin.String = "",
    val error: kotlin.String = "",
    val durationMs: kotlin.Long = 0L,
    val toolCalls: Int = 0,
)

/**
 * A bundle of immutable context passed to a subagent. Contains
 * artifact revisions, canon facts, and relevant text — never
 * mutable conversation history.
 */
@Serializable
data class ContextBundle(
    val artifactContents: Map<kotlin.String, kotlin.String> = emptyMap(),
    val canonFacts: List<kotlin.String> = emptyList(),
    val systemPrompt: kotlin.String = "",
    val userMessage: kotlin.String = "",
)