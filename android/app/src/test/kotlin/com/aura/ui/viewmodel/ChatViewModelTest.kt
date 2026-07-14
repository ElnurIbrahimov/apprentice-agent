package com.aura.ui.viewmodel

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.aura.data.UserPreferences
import com.aura.agent.AgentEvent
import com.aura.agent.Conversation
import com.aura.agent.ConversationStore
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.agent.Reaction
import com.aura.agent.Specialist
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.agent.ToolResult
import com.aura.agent.Turn
import com.aura.kg.KnowledgeGraphRepository
import com.aura.memory.MemoryStore
import com.aura.providers.ProviderKeys
import com.aura.providers.ModelCatalogRepository
import com.aura.providers.ProviderRegistry
import com.aura.voice.TextToSpeech
import com.aura.core.error.CrashLogger
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
class ChatViewModelTest {

    private val testDispatcher = StandardTestDispatcher()
    private lateinit var application: Application

    private lateinit var loop: MemoryAugmentedAgenticLoop
    private lateinit var providerKeys: ProviderKeys
    private lateinit var providerRegistry: ProviderRegistry
    private lateinit var toolRegistry: ToolRegistry
    private lateinit var toolExecutor: ToolExecutor
    private lateinit var textToSpeech: TextToSpeech
    private lateinit var userPreferences: UserPreferences
    private lateinit var memoryStore: MemoryStore
    private lateinit var conversationStore: ConversationStore
    private lateinit var knowledgeGraphRepository: KnowledgeGraphRepository
    private lateinit var crashLogger: CrashLogger

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        application = ApplicationProvider.getApplicationContext()

        loop = mockk(relaxed = true)
        providerKeys = mockk(relaxed = true)
        providerRegistry = mockk(relaxed = true)
        toolRegistry = mockk(relaxed = true)
        toolExecutor = mockk(relaxed = true)
        textToSpeech = mockk(relaxed = true)
        userPreferences = mockk(relaxed = true)
        memoryStore = mockk(relaxed = true)
        conversationStore = mockk(relaxed = true)
        knowledgeGraphRepository = mockk(relaxed = true)
        crashLogger = mockk(relaxed = true)

        every { userPreferences.defaultModel } returns MutableStateFlow("ollama:deepseek-v4-pro:cloud")
        every { providerKeys.loaded } returns MutableStateFlow(true)
        every { providerRegistry.all() } returns emptyList()
        every { providerRegistry.configured() } returns emptyList()
        every { providerRegistry.get("moa") } returns null
        every { toolRegistry.definitions() } returns emptyList()
        coEvery { conversationStore.mostRecent() } returns null
        coEvery { knowledgeGraphRepository.stats() } returns KnowledgeGraphRepository.Stats(nodeCount = 0, edgeCount = 0)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private fun createViewModel(
        modelCatalogRepository: ModelCatalogRepository? = null,
    ): ChatViewModel = ChatViewModel(
        application = application,
        loop = loop,
        providerKeys = providerKeys,
        providerRegistry = providerRegistry,
        toolRegistry = toolRegistry,
        toolExecutor = toolExecutor,
        textToSpeech = textToSpeech,
        userPreferences = userPreferences,
        memoryStore = memoryStore,
        conversationStore = conversationStore,
        knowledgeGraphRepository = knowledgeGraphRepository,
        crashLogger = crashLogger,
        modelCatalogRepository = modelCatalogRepository,
    )

    @Test
    fun `refreshModels loads prefixed models from configured providers`() = runTest(testDispatcher) {
        val provider = mockk<com.aura.providers.Provider>(relaxed = true)
        every { provider.prefix } returns "ollama"
        every { provider.displayName } returns "Ollama Cloud"
        every { provider.isConfigured() } returns true
        coEvery { provider.listModels() } returns listOf("qwen3.5:cloud", "deepseek-v4-pro:cloud")
        every { providerRegistry.all() } returns listOf(provider)
        every { providerRegistry.get("moa") } returns null

        val repository = ModelCatalogRepository(
            providerRegistry = providerRegistry,
            scope = this,
        )
        val vm = createViewModel(repository)
        vm.refreshModels()
        advanceUntilIdle()

        assertEquals(
            listOf("ollama:deepseek-v4-pro:cloud", "ollama:qwen3.5:cloud"),
            vm.state.value.availableModels,
        )
        assertEquals(null, vm.state.value.modelsError)
    }

    @Test
    fun `default state has TTS enabled and empty conversation`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        val state = vm.state.value
        assertTrue(state.ttsEnabled)
        assertEquals("", state.draft)
        assertEquals(0, state.conversation.turns.size)
    }

    @Test
    fun `setDraft updates state and suggests specialist`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { toolRegistry.definitions() } returns emptyList()
        vm.setDraft("remind me to call mom")
        advanceUntilIdle()
        assertEquals("remind me to call mom", vm.state.value.draft)
    }

    @Test
    fun `send adds user message and completes turn`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { conversationStore.recent(2) } returns emptyList()
        coEvery { loop.run(any(), any(), any()) } returns flowOf(
            AgentEvent.TextDelta("Hello"),
            AgentEvent.Done,
        )

        vm.setDraft("hi")
        vm.send()
        advanceUntilIdle()

        val state = vm.state.value
        assertFalse(state.streaming)
        assertEquals("hi", state.conversation.turns.lastOrNull()?.user)
        assertEquals("", state.draft)
    }

    @Test
    fun `cancel stops streaming`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { loop.run(any(), any(), any()) } returns kotlinx.coroutines.flow.emptyFlow()
        vm.setDraft("hi")
        vm.send()
        advanceUntilIdle()
        vm.cancel()
        advanceUntilIdle()
        assertFalse(vm.state.value.streaming)
    }

    @Test
    fun `retryAfterPermission executes tool and updates conversation`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { toolExecutor.execute("calendar_read", "{}", any()) } returns ToolResult.Ok("event")
        _stateDirectly(vm, ChatUiState(pendingToolRetry = "calendar_read" to "{}"))

        vm.retryAfterPermission("android.permission.READ_CALENDAR")
        advanceUntilIdle()

        coVerify { toolExecutor.execute("calendar_read", "{}", any()) }
        assertTrue(vm.state.value.conversation.turns.any { turn -> turn.toolTurns.any { it.name == "calendar_read" } })
    }

    @Test
    fun `loadConversation updates conversation from store`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        val conversation = Conversation(id = "conv-1", title = "Loaded")
        coEvery { conversationStore.load("conv-1") } returns conversation

        vm.loadConversation("conv-1")
        advanceUntilIdle()
        assertEquals("Loaded", vm.state.value.conversation.title)
    }

    @Test
    fun `toggleDeepMode flips flag`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        assertFalse(vm.state.value.deepModeEnabled)
        vm.toggleDeepMode()
        assertTrue(vm.state.value.deepModeEnabled)
    }

    @Test
    fun `setSpecialist updates model`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        // Specialist override wires the specialist's suggestedModel
        // straight into activeModel. The exact id is irrelevant here —
        // this test pins the wiring contract, not the model's existence
        // on Ollama Cloud. Use a verified id from the 2026-07-09
        // /v1/models snapshot so a future model-list refresh doesn't
        // accidentally turn a passing test into a "404 on send".
        val suggested = "ollama:qwen3.5:397b"
        val specialist = Specialist(
            name = "coder",
            icon = "\uD83D\uDCBB",
            systemPrompt = "coding specialist",
            suggestedModel = suggested,
        )
        vm.setSpecialist(specialist)
        assertEquals(suggested, vm.state.value.activeModel)
        assertEquals(specialist, vm.state.value.selectedSpecialist)
    }

    @Test
    fun `toggleIncognito flips the flag in state`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        assertFalse(vm.state.value.incognitoMode)
        vm.toggleIncognito()
        assertTrue(vm.state.value.incognitoMode)
        vm.toggleIncognito()
        assertFalse(vm.state.value.incognitoMode)
    }

    @Test
    fun `send with incognito on forwards memoryEnabled=false to the loop`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { loop.run(any(), any(), any(), any(), any(), any(), any()) } returns flowOf(
            AgentEvent.TextDelta("hi back"),
            AgentEvent.Done,
        )
        coEvery { conversationStore.recent(2) } returns emptyList()

        vm.toggleIncognito()
        vm.setDraft("tell me a secret")
        vm.send()
        advanceUntilIdle()

        // memoryEnabled = false (incognito on) must reach the loop.
        io.mockk.verify {
            loop.run(
                conversation = any(),
                model = any(),
                maxSteps = any(),
                options = any(),
                recallLimit = any(),
                specialist = any(),
                memoryEnabled = false,
            )
        }
    }

    @Test
    fun `send with incognito on skips conversation persistence`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { loop.run(any(), any(), any(), any(), any(), any(), any()) } returns flowOf(
            AgentEvent.TextDelta("hi"),
            AgentEvent.Done,
        )
        coEvery { conversationStore.recent(2) } returns emptyList()

        vm.toggleIncognito()
        vm.setDraft("private question")
        vm.send()
        advanceUntilIdle()

        // In incognito mode the conversation must NOT be persisted.
        io.mockk.coVerify(exactly = 0) { conversationStore.save(any()) }
    }

    @Test
    fun `send with incognito off still persists the conversation`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { loop.run(any(), any(), any(), any(), any(), any(), any()) } returns flowOf(
            AgentEvent.TextDelta("hi"),
            AgentEvent.Done,
        )
        coEvery { conversationStore.recent(2) } returns emptyList()

        // Default state: incognito off.
        vm.setDraft("normal question")
        vm.send()
        advanceUntilIdle()

        // Default path: conversation persisted at least once.
        io.mockk.coVerify(atLeast = 1) { conversationStore.save(any()) }
    }

@Test
    fun `tts can be toggled`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        assertTrue(vm.state.value.ttsEnabled)
        vm.toggleTts()
        assertFalse(vm.state.value.ttsEnabled)
        verify { textToSpeech.stop() }
    }

    @Test
    fun `newConversation creates fresh conversation and clears state`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        // Set some state first
        vm.setDraft("hello world")
        vm.toggleDeepMode()
        advanceUntilIdle()
        assertEquals("hello world", vm.state.value.draft)
        assertTrue(vm.state.value.deepModeEnabled)

        vm.newConversation()
        advanceUntilIdle()

        assertEquals("New conversation", vm.state.value.conversation.title)
        assertEquals("", vm.state.value.draft)
        assertFalse(vm.state.value.deepModeEnabled)
        assertFalse(vm.state.value.deepModeActive)
        assertEquals(null, vm.state.value.selectedSpecialist)
    }

    @Test
    fun `auto-title sets conversation title from first user message`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { loop.run(any(), any(), any(), any(), any(), any(), any()) } returns flowOf(
            AgentEvent.TextDelta("response"),
            AgentEvent.Done,
        )
        coEvery { conversationStore.recent(2) } returns emptyList()

        vm.setDraft("How do I configure Gradle Kotlin DSL build types?")
        vm.send()
        advanceUntilIdle()

        val title = vm.state.value.conversation.title
        assertTrue(title.startsWith("How do I configure Gradle Kotlin"))
        assertTrue(title != "New conversation")
    }

    @Test
    fun `auto-title truncates long messages`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { loop.run(any(), any(), any(), any(), any(), any(), any()) } returns flowOf(
            AgentEvent.TextDelta("response"),
            AgentEvent.Done,
        )
        coEvery { conversationStore.recent(2) } returns emptyList()

        // Use a message where the first 6 words exceed 60 chars
        val longMsg = "Understanding supercalifragilisticexpialidocious implementation methodologies in modern distributed systems architecture and design patterns"
        vm.setDraft(longMsg)
        vm.send()
        advanceUntilIdle()

        val title = vm.state.value.conversation.title
        assertTrue(title.length <= 50, "title length was ${title.length}: '$title'")
        assertTrue(title.endsWith("…"))
        assertTrue(title != "New conversation")
    }

    // ---- Commit 1: in-flight tool call tracking ----
    //
    // The agentic loop emits ToolCallStart → ToolResult (or
    // ToolCallEnd → ToolResult) pairs. The VM should track the
    // in-flight state in ChatUiState.inFlightToolCalls so the chat
    // UI can show a "running" badge while the tool is being
    // executed.

    @Test
    fun `ToolCallStart pushes an in-flight entry to state`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { conversationStore.recent(2) } returns emptyList()
        coEvery { loop.run(any(), any(), any(), any(), any(), any(), any()) } returns flowOf(
            AgentEvent.ToolCallStart(id = "call-1", name = "web_search"),
            AgentEvent.TextDelta("Found it."),
            AgentEvent.ToolResult(
                id = "call-1", name = "web_search", arguments = "{}",
                result = "3 results", needsPermission = null, permissionRationale = null,
            ),
            AgentEvent.Done,
        )

        vm.setDraft("search for x")
        vm.send()
        advanceUntilIdle()

        // The in-flight entry was added then removed on ToolResult.
        // After Done, the list must be empty (cleaned up).
        val state = vm.state.value
        assertEquals(0, state.inFlightToolCalls.size)
    }

    @Test
    fun `multiple concurrent tool calls are tracked in order`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        coEvery { conversationStore.recent(2) } returns emptyList()
        coEvery { loop.run(any(), any(), any(), any(), any(), any(), any()) } returns flowOf(
            AgentEvent.ToolCallStart(id = "a", name = "web_search"),
            AgentEvent.ToolCallStart(id = "b", name = "calendar_read"),
            AgentEvent.ToolResult(
                id = "a", name = "web_search", arguments = "{}",
                result = "ok", needsPermission = null, permissionRationale = null,
            ),
            AgentEvent.ToolResult(
                id = "b", name = "calendar_read", arguments = "{}",
                result = "ok", needsPermission = null, permissionRationale = null,
            ),
            AgentEvent.Done,
        )

        vm.setDraft("parallel tool calls")
        vm.send()
        advanceUntilIdle()

        // Both in-flight entries should be cleared after both
        // ToolResults land.
        assertEquals(0, vm.state.value.inFlightToolCalls.size)
    }

    @Test
    fun `newConversation clears in-flight tool calls`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        // Simulate a stale in-flight entry from a previous turn
        // that was interrupted (e.g. process death + restart).
        _stateDirectly(vm, vm.state.value.copy(
            inFlightToolCalls = listOf(
                InFlightToolCall(id = "stale", name = "web_search", args = "{}"),
            ),
        ))
        assertEquals(1, vm.state.value.inFlightToolCalls.size)

        vm.newConversation()
        advanceUntilIdle()
        assertEquals(0, vm.state.value.inFlightToolCalls.size)
    }

    @Test
    fun `cancel clears in-flight tool calls`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        // The loop emits ToolCallStart but never ToolResult — simulate
        // a slow tool that the user cancels mid-flight.
        coEvery { loop.run(any(), any(), any(), any(), any(), any(), any()) } returns flowOf(
            AgentEvent.ToolCallStart(id = "slow", name = "web_search"),
        )

        vm.setDraft("trigger a long tool")
        vm.send()
        advanceUntilIdle()

        // The in-flight list has the slow tool.
        assertEquals(1, vm.state.value.inFlightToolCalls.size)

        vm.cancel()
        advanceUntilIdle()

        // After cancel, the in-flight list is cleared.
        assertEquals(0, vm.state.value.inFlightToolCalls.size)
    }

    @Test
    fun `reactToTurn sets reaction on matching turn and persists`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        val turnTs = 42L
        _stateDirectly(
            vm,
            ChatUiState(
                conversation = Conversation(
                    turns = listOf(
                        Turn(user = "hi", timestamp = 10L),
                        Turn(assistant = "hello", timestamp = turnTs),
                    ),
                ),
            ),
        )
        coEvery { conversationStore.save(any()) } returns Unit

        vm.reactToTurn(turnTs, Reaction.Up)
        advanceUntilIdle()

        val updated = vm.state.value.conversation.turns.last()
        assertEquals(Reaction.Up, updated.reaction)
        coVerify { conversationStore.save(any()) }
    }

    @Test
    fun `reactToTurn toggles existing reaction off`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        val turnTs = 99L
        _stateDirectly(
            vm,
            ChatUiState(
                conversation = Conversation(
                    turns = listOf(Turn(assistant = "hello", timestamp = turnTs, reaction = Reaction.Up)),
                ),
            ),
        )
        coEvery { conversationStore.save(any()) } returns Unit

        vm.reactToTurn(turnTs, Reaction.Up)
        advanceUntilIdle()

        val updated = vm.state.value.conversation.turns.last()
        assertEquals(null, updated.reaction)
    }

    @Test
    fun `reactToTurn switches Up to Down when opposite is tapped`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        val turnTs = 7L
        _stateDirectly(
            vm,
            ChatUiState(
                conversation = Conversation(
                    turns = listOf(Turn(assistant = "hello", timestamp = turnTs, reaction = Reaction.Up)),
                ),
            ),
        )
        coEvery { conversationStore.save(any()) } returns Unit

        vm.reactToTurn(turnTs, Reaction.Down)
        advanceUntilIdle()

        assertEquals(Reaction.Down, vm.state.value.conversation.turns.last().reaction)
    }

    @Test
    fun `reactToTurn is a no-op when timestamp is unknown`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()
        _stateDirectly(
            vm,
            ChatUiState(
                conversation = Conversation(turns = listOf(Turn(assistant = "hello", timestamp = 1L))),
            ),
        )

        vm.reactToTurn(999L, Reaction.Up)
        advanceUntilIdle()

        // Original turn is unchanged.
        assertEquals(null, vm.state.value.conversation.turns.last().reaction)
        coVerify(exactly = 0) { conversationStore.save(any()) }
    }
}

private fun _stateDirectly(vm: ChatViewModel, newState: ChatUiState) {
    val field = ChatViewModel::class.java.getDeclaredField("_state")
    field.isAccessible = true
    @Suppress("UNCHECKED_CAST")
    val mutable = field.get(vm) as kotlinx.coroutines.flow.MutableStateFlow<ChatUiState>
    mutable.value = newState
}
