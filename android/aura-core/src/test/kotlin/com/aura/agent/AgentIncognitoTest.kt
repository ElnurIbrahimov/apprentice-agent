package com.aura.agent

import com.aura.kg.ConversationKgExtractor
import com.aura.memory.MemoryStore
import com.aura.profile.UserProfileStore
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for the incognito gate in [MemoryAugmentedAgenticLoop.run].
 *
 * Incognito is a session-scoped no-write toggle: the agent still runs and
 * the user still gets the response, but every write path is blocked:
 *   - memoryStore.maybeStore() on the user message
 *   - extractProfileFromText() on the assistant reply
 *   - kgExtractor.extract() on the assistant turn
 *   - WRITE_LOCAL tools (e.g. `remember`) are refused by ToolExecutor
 *
 * READ_ONLY tools and the model's general reasoning still work.
 */
class AgentIncognitoTest {

    private val brain = mockk<Brain>(relaxed = true)
    private val toolRegistry = mockk<ToolRegistry>(relaxed = true)
    private val toolExecutor = mockk<ToolExecutor>(relaxed = true)
    private val memoryStore = mockk<MemoryStore>(relaxed = true)
    private val kgExtractor = mockk<ConversationKgExtractor>(relaxed = true)
    private val userProfileStore = mockk<UserProfileStore>(relaxed = true)

    private fun makeLoop(): MemoryAugmentedAgenticLoop =
        MemoryAugmentedAgenticLoop(
            brain = brain,
            toolRegistry = toolRegistry,
            toolExecutor = toolExecutor,
            memoryStore = memoryStore,
            kgExtractor = kgExtractor,
            userProfileStore = userProfileStore,
        )

    @Test
    fun `memoryEnabled=true (default) writes to memory after a turn`() = runTest {
        // Brain: emit a final text + finish. No tool calls.
        every { brain.stream(any(), any(), any(), any()) } returns flow {
            emit(BrainChunk.Text("hello back"))
            emit(BrainChunk.Finished("stop"))
        }
        // Recall returns nothing.
        coEvery { memoryStore.query(any(), any()) } returns emptyList()
        coEvery { memoryStore.maybeStore(any(), any()) } returns "new_id"
        coEvery { userProfileStore.update(any(), any(), any(), any()) } returns Unit
        coEvery { kgExtractor.extract(any()) } returns Unit

        val loop = makeLoop()
        val events = loop.run(
            conversation = Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(Turn(user = "hi", assistant = null)),
            ),
            model = "ollama:deepseek-v4-pro:cloud",
        ).toList()

        coVerify { brain.stream(any(), any(), any(), any()) }
        coVerify { memoryStore.maybeStore("hi", source = "user") }
        // KG extraction runs by default.
        coVerify { kgExtractor.extract("hello back") }
    }

    @Test
    fun `memoryEnabled=false skips the auto-store and profile extract`() = runTest {
        every { brain.stream(any(), any(), any(), any()) } returns flow {
            emit(BrainChunk.Text("hello back"))
            emit(BrainChunk.Finished("stop"))
        }
        coEvery { memoryStore.query(any(), any()) } returns emptyList()
        coEvery { kgExtractor.extract(any()) } returns Unit

        val loop = makeLoop()
        val events = loop.run(
            conversation = Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(Turn(user = "hi", assistant = null)),
            ),
            model = "ollama:deepseek-v4-pro:cloud",
            memoryEnabled = false,
        ).toList()

        coVerify { brain.stream(any(), any(), any(), any()) }
        coVerify(exactly = 0) { memoryStore.maybeStore(any(), any()) }
        coVerify(exactly = 0) { userProfileStore.update(name = any(), traits = any(), preferences = any(), facts = any()) }
        // KG extraction is gated by the same flag.
        coVerify(exactly = 0) { kgExtractor.extract(any()) }
    }

    @Test
    fun `memoryEnabled=false still returns the assistant text to the caller`() = runTest {
        every { brain.stream(any(), any(), any(), any()) } returns flow {
            emit(BrainChunk.Text("secret answer"))
            emit(BrainChunk.Finished("stop"))
        }
        coEvery { memoryStore.query(any(), any()) } returns emptyList()
        coEvery { kgExtractor.extract(any()) } returns Unit

        val loop = makeLoop()
        val events = loop.run(
            conversation = Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(Turn(user = "hi", assistant = null)),
            ),
            model = "ollama:deepseek-v4-pro:cloud",
            memoryEnabled = false,
        ).toList()

        val lastAssistant = events
            .filterIsInstance<AgentEvent.TextDelta>()
            .joinToString("") { it.text }
        assertEquals("secret answer", lastAssistant)
        assertTrue(
            events.any { it is AgentEvent.Done },
            "loop should still terminate with Done even in incognito mode",
        )
    }
}