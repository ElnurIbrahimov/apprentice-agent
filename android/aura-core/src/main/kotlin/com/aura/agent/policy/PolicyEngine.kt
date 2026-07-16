package com.aura.agent.policy

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolRisk
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Evaluates [ToolPolicy] against a tool and execution context.
 * Policy can only tighten the built-in [ToolRisk] — never loosen it.
 *
 * Resolution order (innermost wins):
 * 1. Built-in ToolRisk (READ_ONLY → DESTRUCTIVE)
 * 2. Incognito gate (memoryEnabled=false blocks >= WRITE_LOCAL)
 * 3. User ToolPolicy (enabled/disabled, confirmation, cost, scope, approval)
 * 4. Per-run approval scope (approvedRemoteCostTools)
 */
@Singleton
class PolicyEngine @Inject constructor(
    private val policyStore: ToolPolicyStore,
) {
    /**
     * Evaluate whether [tool] can execute in [ctx].
     * Returns [PolicyResult.Allowed] if all checks pass, or a specific
     * denial/approval/confirmation result otherwise.
     */
    suspend fun evaluate(tool: Tool, ctx: ToolContext): PolicyResult {
        // 1. Incognito gate — hard lower boundary, cannot be overridden by policy
        if (!ctx.memoryEnabled && tool.risk.ordinal >= ToolRisk.WRITE_LOCAL.ordinal) {
            return PolicyResult.Disabled(tool.name)
        }

        // 2. User policy
        val policy = policyStore.getPolicy(tool.name) ?: ToolPolicy(toolName = tool.name)

        if (!policy.enabled) {
            return PolicyResult.Disabled(tool.name)
        }

        // 3. Confirmation level check
        if (policy.confirmation != ConfirmationLevel.NONE) {
            return PolicyResult.NeedsConfirmation(policy.confirmation, policy)
        }

        // 4. Per-run approval for REMOTE_COST tools
        if (tool.risk == ToolRisk.REMOTE_COST && tool.name !in ctx.approvedRemoteCostTools) {
            return PolicyResult.NeedsApproval(policy)
        }

        // 5. Per-run approval override from policy
        if (policy.requireApprovalPerRun && tool.name !in ctx.approvedRemoteCostTools) {
            return PolicyResult.NeedsApproval(policy)
        }

        return PolicyResult.Allowed(policy)
    }
}