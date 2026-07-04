package com.aura.tools

import android.content.Context
import androidx.work.ExistingWorkPolicy
import androidx.work.WorkManager
import com.aura.tasks.TaskDao
import com.aura.tasks.TaskEntity
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.slot
import io.mockk.unmockkStatic
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Contract tests for [TaskManagerTool].
 */
class TaskManagerToolTest {

    private val context = mockk<Context>(relaxed = true)
    private val taskDao = mockk<TaskDao>(relaxed = true)
    private val workManager = mockk<WorkManager>(relaxed = true)

    @Before
    fun setUp() {
        mockkStatic(WorkManager::class)
        every { WorkManager.getInstance(any()) } returns workManager
    }

    @After
    fun tearDown() {
        unmockkStatic(WorkManager::class)
    }

    @Test
    fun `create stores full task metadata`() = runBlocking {
        val tool = TaskManagerTool(context, taskDao)
        val call = com.aura.agent.ToolCall(
            id = "c1",
            name = "manage_tasks",
            arguments = mapOf(
                "action" to "create",
                "title" to "Ship features",
                "description" to "Close G3, G7, G8",
                "priority" to 2,
                "tags" to "aura, polish",
                "when" to "23:59",
            ),
        )

        val entitySlot = slot<TaskEntity>()
        coEvery { taskDao.insert(capture(entitySlot)) } returns Unit
        coEvery { taskDao.allPending() } returns emptyList()

        val result = tool.tool.execute(call, mockk(relaxed = true))

        assertTrue(result is com.aura.agent.ToolResult.Ok, "expected Ok, got $result")
        assertEquals("Ship features", entitySlot.captured.title)
        assertEquals("Close G3, G7, G8", entitySlot.captured.description)
        assertEquals(2, entitySlot.captured.priority)
        assertEquals("aura, polish", entitySlot.captured.tags)
        assertTrue(entitySlot.captured.dueAt != null, "dueAt should be set")
        assertEquals("pending", entitySlot.captured.status)
    }

    @Test
    fun `list returns pending tasks summary`() = runBlocking {
        val tool = TaskManagerTool(context, taskDao)
        coEvery { taskDao.allPending() } returns listOf(
            TaskEntity(id = "t1", title = "One", createdAt = 1L, dueAt = 3_600_000L),
        )

        val result = tool.tool.execute(
            com.aura.agent.ToolCall(
                id = "l1",
                name = "manage_tasks",
                arguments = mapOf("action" to "list"),
            ),
            mockk(relaxed = true),
        )

        assertTrue(result is com.aura.agent.ToolResult.Ok, "expected Ok, got $result")
        assertTrue(result.output.contains("1. One"))
    }

    @Test
    fun `complete marks task complete`() = runBlocking {
        val tool = TaskManagerTool(context, taskDao)
        coEvery { taskDao.markComplete("t1", any()) } returns Unit

        val result = tool.tool.execute(
            com.aura.agent.ToolCall(
                id = "d1",
                name = "manage_tasks",
                arguments = mapOf("action" to "complete", "id" to "t1"),
            ),
            mockk(relaxed = true),
        )

        assertTrue(result is com.aura.agent.ToolResult.Ok, "expected Ok, got $result")
        coVerify { taskDao.markComplete("t1", any()) }
    }

    @Test
    fun `delete removes task and cancels reminder`() = runBlocking {
        val tool = TaskManagerTool(context, taskDao)
        coEvery { taskDao.delete("t1") } returns Unit

        val result = tool.tool.execute(
            com.aura.agent.ToolCall(
                id = "x1",
                name = "manage_tasks",
                arguments = mapOf("action" to "delete", "id" to "t1"),
            ),
            mockk(relaxed = true),
        )

        assertTrue(result is com.aura.agent.ToolResult.Ok, "expected Ok, got $result")
        coVerify { taskDao.delete("t1") }
        coVerify { workManager.cancelUniqueWork("task-t1") }
    }

    @Test
    fun `create without title returns error`() = runBlocking {
        val tool = TaskManagerTool(context, taskDao)
        val result = tool.tool.execute(
            com.aura.agent.ToolCall(
                id = "c2",
                name = "manage_tasks",
                arguments = mapOf("action" to "create"),
            ),
            mockk(relaxed = true),
        )

        assertTrue(result is com.aura.agent.ToolResult.Error, "expected Error, got $result")
        assertEquals("bad_args", result.code)
    }
}
