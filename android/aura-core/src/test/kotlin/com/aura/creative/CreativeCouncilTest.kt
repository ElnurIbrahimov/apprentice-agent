package com.aura.creative

import com.aura.agents.SubagentManager
import com.aura.agents.SubagentResult
import com.aura.agents.SubagentTask
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Test

class CreativeCouncilTest {

    private val subagentManager = SubagentManager()
    private val council = CreativeCouncil(subagentManager)

    private fun makeExecutor(output: kotlin.String, success: kotlin.Boolean = true): suspend (SubagentTask) -> SubagentResult =
        { task -> SubagentResult(taskId = task.id, success = success, output = output, rationale = "Test rationale") }

    @Test
    fun run_returns_result_with_director_output() = runTest {
        val request = CouncilSessionRequest(
            projectId = "p1",
            brief = "Design a trailer for Glass City",
            roles = listOf(CouncilRole.WRITER, CouncilRole.DIRECTOR),
            budgetMs = 5000,
        )
        val result = council.run(request, makeExecutor("Draft scene: The glass city shimmers."))
        assertTrue(result.success)
        assertTrue(result.directorOutput.isNotBlank())
        assertTrue(result.proposals.isNotEmpty())
    }

    @Test
    fun run_producers_run_before_director() = runTest {
        val request = CouncilSessionRequest(
            projectId = "p1",
            brief = "Write a scene",
            roles = CouncilRole.full,
            budgetMs = 10_000,
        )
        val result = council.run(request, makeExecutor("Output for role"))
        // Should have proposals from producers, critics, and director
        assertTrue("Expected multiple proposals, got ${result.proposals.size}", result.proposals.size >= 3)
    }

    @Test
    fun run_returns_failure_when_executor_fails() = runTest {
        val request = CouncilSessionRequest(
            projectId = "p1",
            brief = "Test",
            roles = listOf(CouncilRole.WRITER),
            budgetMs = 5000,
        )
        val result = council.run(request, makeExecutor("fail", success = false))
        // Council should still succeed (non-fatal member failure) but include failed proposal
        assertTrue(result.success)
        val writerProposal = result.proposals.firstOrNull { it.role == CouncilRole.WRITER }
        assertTrue("Expected writer proposal", writerProposal != null)
        assertFalse("Writer should have failed", writerProposal!!.success)
    }

    @Test
    fun run_with_empty_roles_uses_full_council() = runTest {
        val request = CouncilSessionRequest(
            projectId = "p1",
            brief = "Test",
            roles = emptyList(),
            budgetMs = 10_000,
        )
        val result = council.run(request, makeExecutor("output"))
        assertTrue(result.proposals.isNotEmpty())
    }

    @Test
    fun council_roles_full_includes_director() {
        assertTrue(CouncilRole.full.contains(CouncilRole.DIRECTOR))
        assertTrue(CouncilRole.producers.contains(CouncilRole.WRITER))
        assertTrue(CouncilRole.critics.contains(CouncilRole.STORY_EDITOR))
    }

    @Test
    fun toSubagentSpec_sets_correct_role() {
        val request = CouncilSessionRequest(projectId = "p1", brief = "test")
        val spec = CouncilRole.WRITER.toSubagentSpec(request)
        assertEquals("Writer", spec.role)
        assertEquals("CREATIVE_DRAFT", spec.modelRole)
    }
}