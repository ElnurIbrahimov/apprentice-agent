package com.aura.tools

import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.creative.CreativeProject
import com.aura.creative.CreativeProjectStore
import com.aura.creative.WorldBible
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertTrue

class CreativeToolsTest {
    private val store = mockk<CreativeProjectStore>()

    @Test
    fun `read project returns structured canon`() = runTest {
        coEvery { store.get("p1") } returns CreativeProject(
            id = "p1", name = "Glass City", description = "", genre = "fantasy", tone = "",
            world = WorldBible(overview = "The city stores memories in glass."),
            templateId = "novel", turnCount = 0, createdAt = 1L, updatedAt = 1L,
        )
        val result = CreativeReadProjectTool(store).tool.execute(
            ToolCall("1", "creative_read_project", mapOf("projectId" to "p1")),
            ToolContext("c1"),
        )
        assertTrue(result is ToolResult.Ok)
        assertTrue((result as ToolResult.Ok).output.contains("Glass City"))
        assertTrue(result.output.contains("stores memories"))
    }

    @Test
    fun `add world item persists a character`() = runTest {
        val project = CreativeProject(
            id = "p1", name = "Glass City", description = "", genre = "fantasy", tone = "",
            world = WorldBible(), templateId = "novel", turnCount = 0, createdAt = 1L, updatedAt = 1L,
        )
        coEvery { store.get("p1") } returns project
        coEvery { store.updateWorld(any(), any()) } returns project

        val result = CreativeAddWorldItemTool(store).tool.execute(
            ToolCall(
                "2", "creative_add_world_item",
                mapOf("projectId" to "p1", "type" to "character", "name" to "Mara", "description" to "A mapmaker", "details" to "protagonist"),
            ),
            ToolContext("c1"),
        )

        assertTrue(result is ToolResult.Ok)
        coVerify {
            store.updateWorld("p1", match { world ->
                world.characters.single().name == "Mara" && world.characters.single().role == "protagonist"
            })
        }
    }
}