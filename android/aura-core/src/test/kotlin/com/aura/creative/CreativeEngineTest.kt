package com.aura.creative

import com.aura.data.UserPreferences
import com.aura.providers.ChatOptions
import com.aura.providers.FinishReason
import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderRegistry
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CreativeEngineTest {
    private val registry = mockk<ProviderRegistry>()
    private val preferences = mockk<UserPreferences>()
    private val store = mockk<CreativeProjectStore>(relaxed = true)
    private val engine = CreativeEngine(registry, preferences, store)

    private val project = CreativeProject(
        id = "p1", name = "Glass City", description = "", genre = "speculative", tone = "haunting",
        world = WorldBible(
            overview = "The city stores memories in glass.",
            characters = listOf(WorldCharacter(id = "c1", name = "Mara", role = "cartographer")),
            rules = listOf(WorldRule(id = "r1", name = "Maps change memory", description = "A redrawn street changes recollection")),
        ),
        templateId = "novel", turnCount = 0, createdAt = 1L, updatedAt = 1L,
    )

    @Test
    fun `generation uses configured default model and includes project canon`() = runTest {
        every { preferences.defaultModel } returns flowOf("openai:live-model")
        coEvery { store.get("p1") } returns project
        coEvery { registry.chat("openai:live-model", any(), any<ChatOptions>(), emptyList()) } returns flowOf(
            ProviderChunk(text = "Mara enters"),
            ProviderChunk(text = " the archive."),
            ProviderChunk(finishReason = FinishReason.stop),
        )

        val chunks = engine.generate("p1", CreativeMode.DRAFT, "Write the opening").toList()

        assertEquals(listOf("Mara enters", " the archive."), chunks)
        coVerify {
            registry.chat(
                "openai:live-model",
                match { messages ->
                    messages.first().content.contains("Glass City") &&
                        messages.first().content.contains("Maps change memory") &&
                        messages.last().content.contains("Write the opening")
                },
                any<ChatOptions>(),
                emptyList(),
            )
        }
        coVerify { store.incrementTurn("p1") }
    }

    @Test
    fun `simulation stores outcome as non-canon exploration`() = runTest {
        every { preferences.defaultModel } returns flowOf("openai:live-model")
        coEvery { store.get("p1") } returns project
        coEvery { registry.chat(any(), any(), any<ChatOptions>(), emptyList()) } returns flowOf(
            ProviderChunk(text = "The tower fractures."),
            ProviderChunk(finishReason = FinishReason.stop),
        )

        engine.generate("p1", CreativeMode.SIMULATE, "What if Mara burns the map?").toList()

        coVerify {
            store.recordSimulation(
                "p1",
                match { it.premise.contains("Mara burns") && it.outcome == "The tower fractures." && !it.canonized },
            )
        }
    }

    @Test
    fun `world context stays bounded for provider safety`() {
        val huge = project.copy(world = project.world.copy(notes = "x".repeat(80_000)))
        val context = engine.buildProjectContext(huge)
        assertTrue(context.length <= CreativeEngine.MAX_CONTEXT_CHARS)
        assertTrue(context.contains("Glass City"))
    }
}