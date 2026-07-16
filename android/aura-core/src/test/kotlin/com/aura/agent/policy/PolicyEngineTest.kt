package com.aura.agent.policy

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import io.mockk.coEvery
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PolicyEngineTest {

    private val policyStore = mockk<ToolPolicyStore>(relaxed = true)
    private val engine = PolicyEngine(policyStore)

    private fun makeTool(risk: ToolRisk, name: kotlin.String = "test_tool") = Tool(
        name = name,
        description = "Test tool",
        risk = risk,
        execute = { _, _ -> ToolResult.Ok("ok") },
    )

    private fun ctx(
        memoryEnabled: kotlin.Boolean = true,
        approvedTools: Set<kotlin.String> = emptySet(),
    ) = ToolContext(
        conversationId = "test",
        memoryEnabled = memoryEnabled,
        approvedRemoteCostTools = approvedTools,
    )

    @Test
    fun allows_read_only_tool_with_default_policy() = runTest {
        val tool = makeTool(ToolRisk.READ_ONLY)
        coEvery { policyStore.getPolicy("test_tool") } returns null
        val result = engine.evaluate(tool, ctx())
        assertTrue("Expected Allowed, got $result", result is PolicyResult.Allowed)
    }

    @Test
    fun blocks_write_local_in_incognito_mode() = runTest {
        val tool = makeTool(ToolRisk.WRITE_LOCAL)
        coEvery { policyStore.getPolicy("test_tool") } returns null
        val result = engine.evaluate(tool, ctx(memoryEnabled = false))
        assertTrue("Expected Disabled, got $result", result is PolicyResult.Disabled)
    }

    @Test
    fun allows_read_only_in_incognito_mode() = runTest {
        val tool = makeTool(ToolRisk.READ_ONLY)
        coEvery { policyStore.getPolicy("test_tool") } returns null
        val result = engine.evaluate(tool, ctx(memoryEnabled = false))
        assertTrue("Expected Allowed, got $result", result is PolicyResult.Allowed)
    }

    @Test
    fun blocks_disabled_tool() = runTest {
        val tool = makeTool(ToolRisk.READ_ONLY)
        coEvery { policyStore.getPolicy("test_tool") } returns ToolPolicy(
            toolName = "test_tool",
            enabled = false,
        )
        val result = engine.evaluate(tool, ctx())
        assertTrue("Expected Disabled, got $result", result is PolicyResult.Disabled)
    }

    @Test
    fun requires_confirmation_when_policy_set() = runTest {
        val tool = makeTool(ToolRisk.READ_ONLY)
        coEvery { policyStore.getPolicy("test_tool") } returns ToolPolicy(
            toolName = "test_tool",
            confirmation = ConfirmationLevel.EXPLICIT,
        )
        val result = engine.evaluate(tool, ctx())
        assertTrue("Expected NeedsConfirmation, got $result", result is PolicyResult.NeedsConfirmation)
        val confirmation = result as PolicyResult.NeedsConfirmation
        assertEquals(ConfirmationLevel.EXPLICIT, confirmation.level)
    }

    @Test
    fun remote_cost_needs_approval_without_explicit_approval() = runTest {
        val tool = makeTool(ToolRisk.REMOTE_COST, "web_search")
        coEvery { policyStore.getPolicy("web_search") } returns null
        val result = engine.evaluate(tool, ctx())
        assertTrue("Expected NeedsApproval, got $result", result is PolicyResult.NeedsApproval)
    }

    @Test
    fun remote_cost_allowed_with_explicit_approval() = runTest {
        val tool = makeTool(ToolRisk.REMOTE_COST, "web_search")
        coEvery { policyStore.getPolicy("web_search") } returns null
        val result = engine.evaluate(tool, ctx(approvedTools = setOf("web_search")))
        assertTrue("Expected Allowed, got $result", result is PolicyResult.Allowed)
    }

    @Test
    fun per_run_approval_required_when_policy_set() = runTest {
        val tool = makeTool(ToolRisk.WRITE_LOCAL, "set_reminder")
        coEvery { policyStore.getPolicy("set_reminder") } returns ToolPolicy(
            toolName = "set_reminder",
            requireApprovalPerRun = true,
        )
        val result = engine.evaluate(tool, ctx())
        assertTrue("Expected NeedsApproval, got $result", result is PolicyResult.NeedsApproval)
    }
}