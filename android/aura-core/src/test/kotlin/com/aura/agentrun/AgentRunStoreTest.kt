package com.aura.agentrun

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AgentRunStoreTest {

    private val runDao = mockk<AgentRunDao>(relaxed = true)
    private val goalDao = mockk<GoalDao>(relaxed = true)
    private val stepDao = mockk<StepDao>(relaxed = true)
    private val eventDao = mockk<AgentEventDao>(relaxed = true)
    private val approvalDao = mockk<ApprovalRequestDao>(relaxed = true)
    private val checkpointDao = mockk<RunCheckpointDao>(relaxed = true)
    private val dagResolver = mockk<DagResolver>(relaxed = true)
    private val store = AgentRunStore(runDao, goalDao, stepDao, eventDao, approvalDao, checkpointDao, dagResolver)

    @Test
    fun createRun_persists_run_and_goal() = runTest {
        val runSlot = slot<AgentRunEntity>()
        val goalSlot = slot<GoalEntity>()
        coEvery { runDao.upsert(capture(runSlot)) } returns Unit
        coEvery { goalDao.upsert(capture(goalSlot)) } returns Unit

        val run = store.createRun(
            trigger = "USER_QUERY",
            goalDescription = "Find the best coffee shop nearby",
        )

        assertEquals("USER_QUERY", run.triggerType)
        assertEquals("RUNNING", run.status)
        assertNotNull(run.goalId)
        assertEquals("Find the best coffee shop nearby", goalSlot.captured.description)
        assertEquals(run.id, goalSlot.captured.agentRunId)
    }

    @Test
    fun planSteps_inserts_steps_with_positions() = runTest {
        val stepsSlot = slot<List<StepEntity>>()
        coEvery { stepDao.upsertAll(capture(stepsSlot)) } returns Unit

        store.planSteps("run1", listOf(
            StepSpec(toolName = "web_search", toolArgs = """{"query":"coffee"}"""),
            StepSpec(toolName = "location", toolArgs = "{}", dependsOn = """["step1"]"""),
        ))

        assertEquals(2, stepsSlot.captured.size)
        assertEquals(0, stepsSlot.captured[0].position)
        assertEquals(1, stepsSlot.captured[1].position)
    }

    @Test
    fun completeStep_updates_status_and_emits_event() = runTest {
        val step = StepEntity(id = "s1", agentRunId = "run1", toolName = "web_search", status = "RUNNING")
        coEvery { stepDao.getById("s1") } returns step

        store.completeStep("s1", "results here")

        coVerify { stepDao.complete("s1", "SUCCESS", "results here", any()) }
        coVerify { eventDao.insert(any()) }
    }

    @Test
    fun checkpoint_creates_and_cleans_old() = runTest {
        coEvery { stepDao.forRun("run1") } returns listOf(
            StepEntity(id = "s1", agentRunId = "run1", status = "PENDING"),
            StepEntity(id = "s2", agentRunId = "run1", status = "SUCCESS"),
        )

        val cp = store.checkpoint("run1")

        assertNotNull(cp.id)
        assertEquals("run1", cp.agentRunId)
        assertTrue(cp.stateJson.contains("s1"))
        assertTrue(!cp.stateJson.contains("s2"))
        coVerify { checkpointDao.cleanupOld("run1", cp.id) }
    }

    @Test
    fun resumeFromCheckpoint_returns_pending_active_steps() = runTest {
        val cp = RunCheckpointEntity(
            id = "cp1",
            agentRunId = "run1",
            stateJson = """{"activeStepIds":["s1","s3"]}""",
        )
        coEvery { checkpointDao.latestForRun("run1") } returns cp
        coEvery { stepDao.forRun("run1") } returns listOf(
            StepEntity(id = "s1", agentRunId = "run1", status = "PENDING"),
            StepEntity(id = "s2", agentRunId = "run1", status = "SUCCESS"),
            StepEntity(id = "s3", agentRunId = "run1", status = "PENDING"),
        )

        val resumable = store.resumeFromCheckpoint("run1")
        assertEquals(2, resumable.size)
        assertTrue(resumable.any { it.id == "s1" })
        assertTrue(resumable.any { it.id == "s3" })
    }

    @Test
    fun requestApproval_creates_and_emits_event() = runTest {
        val approvalSlot = slot<ApprovalRequestEntity>()
        coEvery { approvalDao.upsert(capture(approvalSlot)) } returns Unit

        val approval = store.requestApproval(
            runId = "run1",
            stepId = "s1",
            toolName = "web_search",
            rationale = "Paid API call",
        )

        assertEquals("run1", approval.agentRunId)
        assertEquals("s1", approval.stepId)
        assertEquals("PENDING", approval.status)
        coVerify { eventDao.insert(any()) }
    }

    @Test
    fun approve_updates_status_and_emits_event() = runTest {
        val approval = ApprovalRequestEntity(
            id = "ap1",
            agentRunId = "run1",
            stepId = "s1",
            toolName = "web_search",
            rationale = "test",
            status = "PENDING",
        )
        coEvery { approvalDao.getById("ap1") } returns approval

        store.approve("ap1")

        coVerify { approvalDao.decide("ap1", "APPROVED", "", any()) }
        coVerify { eventDao.insert(any()) }
    }
}