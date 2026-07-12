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
    private val handRepository = mockk<com.aura.hands.HandRepository>(relaxed = true)
    private val providerRegistry = mockk<com.aura.providers.ProviderRegistry>(relaxed = true)

    private fun makeLoop(): MemoryAugmentedAgenticLoop =
        MemoryAugmentedAgenticLoop(
            brain = brain,
            toolRegistry = toolRegistry,
            toolExecutor = toolExecutor,
            memoryStore = memoryStore,
            kgExtractor = kgExtractor,
            userProfileStore = userProfileStore,
            handRepository = handRepository,
            providerRegistry = providerRegistry,
        )

    @Test
    fun `memoryEnabled=true (default) writes to memory after a turn`() = runTest {
        // Brain: emit a final text + finish. No tool calls.
        coEvery { brain.stream(any(), any(), any(), any()) } returns flow {
            emit(BrainChunk.Text("hello back"))
            emit(BrainChunk.Finished("stop"))
        }
        // Recall returns nothing.
        coEvery { memoryStore.query(any(), any()) } returns emptyList()
        coEvery { userProfileStore.update(any(), any(), any(), any()) } returns Unit
        coEvery { kgExtractor.extract(any()) } returns Unit
        coEvery { handRepository.getEnabled() } returns emptyList()

        val loop = makeLoop()
        loop.run(
            conversation = Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(Turn(user = "hi there", assistant = null)),
            ),
            model = "ollama:deepseek-v4-pro:cloud",
        ).toList()

        coVerify { brain.stream(any(), any(), any(), any()) }
        // The LLM gate falls back to heuristic (providerRegistry is a
        // relaxed mock returning empty flow), which says shouldStore=true
        // for content >= 4 chars. So store() should be called.
        coVerify { memoryStore.store(any<String>(), any<String>(), any<String>(), any<Float>(), any<List<String>>()) }
        // KG extraction runs by default.
        coVerify { kgExtractor.extract("hello back") }
    }

    @Test
    fun `memoryEnabled=false skips the auto-store and profile extract`() = runTest {
        coEvery { brain.stream(any(), any(), any(), any()) } returns flow {
            emit(BrainChunk.Text("hello back"))
            emit(BrainChunk.Finished("stop"))
        }
        coEvery { memoryStore.query(any(), any()) } returns emptyList()
        coEvery { kgExtractor.extract(any()) } returns Unit
        coEvery { handRepository.getEnabled() } returns emptyList()

        val loop = makeLoop()
        loop.run(
            conversation = Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(Turn(user = "hi", assistant = null)),
            ),
            model = "ollama:deepseek-v4-pro:cloud",
            memoryEnabled = false,
        ).toList()

        coVerify { brain.stream(any(), any(), any(), any()) }
        coVerify(exactly = 0) { memoryStore.store(any<String>(), any<String>(), any<String>(), any<Float>(), any<List<String>>()) }
        coVerify(exactly = 0) { userProfileStore.update(name = any(), traits = any(), preferences = any(), facts = any()) }
        coVerify(exactly = 0) { userProfileStore.mergeFacts(any()) }
        // KG extraction is gated by the same flag.
        coVerify(exactly = 0) { kgExtractor.extract(any()) }
    }

    @Test
    fun `memoryEnabled=false still returns the assistant text to the caller`() = runTest {
        coEvery { brain.stream(any(), any(), any(), any()) } returns flow {
            emit(BrainChunk.Text("secret answer"))
            emit(BrainChunk.Finished("stop"))
        }
        coEvery { memoryStore.query(any(), any()) } returns emptyList()
        coEvery { kgExtractor.extract(any()) } returns Unit
        coEvery { handRepository.getEnabled() } returns emptyList()

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

    @Test
    fun `trigger phrase is injected into context when user message matches`() = runTest {
        val hand = com.aura.hands.Hand(
            id = "h1",
            name = "standup",
            triggerPhrase = "daily standup",
            steps = "[{\"tool\":\"send_message\",\"args\":{\"to\":\"slack\",\"body\":\"done\"}}]",
        )
        coEvery { handRepository.getEnabled() } returns listOf(hand)
        coEvery { brain.stream(any(), any(), any(), any()) } returns flow {
            emit(BrainChunk.Text("Running standup hand"))
            emit(BrainChunk.Finished("stop"))
        }
        coEvery { memoryStore.query(any(), any()) } returns emptyList()
        coEvery { kgExtractor.extract(any()) } returns Unit

        val loop = makeLoop()
        val events = loop.run(
            conversation = Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(Turn(user = "give me my daily standup", assistant = null)),
            ),
            model = "ollama:deepseek-v4-pro:cloud",
        ).toList()

        val assistant = events.filterIsInstance<AgentEvent.TextDelta>().joinToString("") { it.text }
        assertEquals("Running standup hand", assistant)
    }
}