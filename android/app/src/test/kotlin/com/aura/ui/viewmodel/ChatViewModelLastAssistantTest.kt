package com.aura.ui.viewmodel

import android.app.Application
import com.aura.agent.Conversation
import com.aura.agent.ConversationStore
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.agent.Turn
import com.aura.data.UserPreferences
import com.aura.kg.KnowledgeGraphRepository
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import com.aura.voice.TextToSpeech
import com.aura.core.error.CrashLogger
import io.mockk.coEvery
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals

/**
 * Tests for [ChatViewModel.lastAssistantText]. Pure logic over the
 * conversation state, no Android plumbing involved.
 */
class ChatViewModelLastAssistantTest {

    private val store = mockk<ConversationStore>(relaxed = true)
    private val userPreferences = mockk<UserPreferences>(relaxed = true)
    private val app = mockk<Application>(relaxed = true)
    private val loop = mockk<MemoryAugmentedAgenticLoop>(relaxed = true)
    private val providerKeys = mockk<ProviderKeys>(relaxed = true)
    private val providerRegistry = mockk<ProviderRegistry>(relaxed = true)
    private val toolRegistry = mockk<ToolRegistry>(relaxed = true)
    private val toolExecutor = mockk<ToolExecutor>(relaxed = true)
    private val textToSpeech = mockk<TextToSpeech>(relaxed = true)
    private val memoryStore = mockk<com.aura.memory.MemoryStore>(relaxed = true)
    private val kgRepo = mockk<KnowledgeGraphRepository>(relaxed = true)
    private val crashLogger = mockk<CrashLogger>(relaxed = true)

    @Before
    fun setUp() { Dispatchers.setMain(UnconfinedTestDispatcher()) }
    @After
    fun tearDown() { Dispatchers.resetMain() }

    private fun makeVm(initialConv: Conversation): ChatViewModel {
        coEvery { userPreferences.defaultModel } returns kotlinx.coroutines.flow.flowOf("ollama:deepseek-v4-pro:cloud")
        io.mockk.every { providerKeys.loaded } returns kotlinx.coroutines.flow.MutableStateFlow(true)
        val vm = ChatViewModel(
            application = app,
            loop = loop,
            providerKeys = providerKeys,
            providerRegistry = providerRegistry,
            toolRegistry = toolRegistry,
            toolExecutor = toolExecutor,
            textToSpeech = textToSpeech,
            userPreferences = userPreferences,
            memoryStore = memoryStore,
            conversationStore = store,
            knowledgeGraphRepository = kgRepo,
        crashLogger = crashLogger,
        )
        // Replace the private _state with a Conversation containing
        // our test data. Done via reflection so we don't have to
        // wire up the entire VM to drive a full chat cycle.
        val stateField = ChatViewModel::class.java.getDeclaredField("_state").apply { isAccessible = true }
        val mf = stateField.get(vm) as kotlinx.coroutines.flow.MutableStateFlow<ChatUiState>
        mf.value = ChatUiState(conversation = initialConv)
        return vm
    }

    @Test
    fun `returns blank for an empty conversation`() = runTest {
        val vm = makeVm(Conversation())
        assertEquals("", vm.lastAssistantText())
    }

    @Test
    fun `returns the only assistant turn in a single-turn conversation`() = runTest {
        val vm = makeVm(
            Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(Turn(user = "hi", assistant = "hello there")),
            )
        )
        assertEquals("hello there", vm.lastAssistantText())
    }

    @Test
    fun `returns the most recent assistant when several exist`() = runTest {
        val vm = makeVm(
            Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(
                    Turn(user = "u1", assistant = "first"),
                    Turn(user = "u2", assistant = "second"),
                    Turn(user = "u3", assistant = "third"),
                ),
            )
        )
        assertEquals("third", vm.lastAssistantText())
    }

    @Test
    fun `skips over empty assistant text to find the most recent non-empty one`() = runTest {
        val vm = makeVm(
            Conversation(
                id = "c1", title = "t", createdAt = 0L, updatedAt = 0L,
                turns = listOf(
                    Turn(user = "u1", assistant = "first"),
                    // The model returned an empty assistant turn + a tool call.
                    Turn(user = "u2", assistant = "", toolTurns = listOf(
                        com.aura.agent.ToolTurn("t1", "echo", "{}", "echoed")
                    )),
                    Turn(user = "u3", assistant = null),
                ),
            )
        )
        assertEquals("first", vm.lastAssistantText())
    }
}
