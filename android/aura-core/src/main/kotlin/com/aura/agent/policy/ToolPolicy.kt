package com.aura.agent.policy

import com.aura.agent.ToolRisk
import kotlinx.serialization.Serializable

/**
 * User-controlled policy for a single tool. Stored in DataStore (non-secret)
 * and applied by [PolicyEngine] before [com.aura.agent.ToolExecutor] dispatches.
 *
 * Policy can only tighten the built-in [ToolRisk] — it can never loosen it.
 * For example, a READ_ONLY tool can be disabled, but a READ_ONLY tool cannot
 * be elevated to allow WRITE_REMOTE behavior.
 */
@Serializable
data class ToolPolicy(
    val toolName: kotlin.String,
    val enabled: kotlin.Boolean = true,
    /** Minimum confirmation level for this tool. */
    val confirmation: ConfirmationLevel = ConfirmationLevel.NONE,
    /** Maximum cost (in arbitrary units) per call. 0 = no limit. */
    val costCeiling: kotlin.Double = 0.0,
    /** Allowed app/domain/path scopes. Empty = all allowed. */
    val allowedScopes: List<kotlin.String> = emptyList(),
    /** When true, the tool requires fresh user approval every run. */
    val requireApprovalPerRun: kotlin.Boolean = false,
    /** Approval expiry in milliseconds. 0 = never expires. */
    val approvalExpiryMs: kotlin.Long = 0L,
)

@Serializable
enum class ConfirmationLevel {
    NONE,
    IMPLICIT,
    EXPLICIT,
    BIOMETRIC,
}

/**
 * Resolution result for a tool policy check.
 */
sealed class PolicyResult {
    data class Allowed(val policy: ToolPolicy) : PolicyResult()
    data class Disabled(val toolName: kotlin.String) : PolicyResult()
    data class NeedsConfirmation(val level: ConfirmationLevel, val policy: ToolPolicy) : PolicyResult()
    data class NeedsApproval(val policy: ToolPolicy) : PolicyResult()
    data class CostExceeded(val toolName: kotlin.String, val ceiling: kotlin.Double) : PolicyResult()
    data class ScopeDenied(val toolName: kotlin.String, val scope: kotlin.String) : PolicyResult()
}