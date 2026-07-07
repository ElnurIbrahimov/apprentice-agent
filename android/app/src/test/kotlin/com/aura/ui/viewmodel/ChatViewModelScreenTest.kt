package com.aura.ui.viewmodel

import com.aura.data.UserPreferences
import com.aura.agent.ConversationStore
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.kg.KnowledgeGraphRepository
import com.aura.providers.ProviderRegistry
import com.aura.voice.TextToSpeech
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestWatcher
import org.junit.runner.Description
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Robolectric-based tests for [ChatViewModel] screen-level state changes.
 *
 * These tests bypass Compose UI testing (which requires an activity host)
 * and verify the observable ViewModel state instead.
 */
@RunWith(RobolectricTestRunner::class)
@Config(application = android.app.Application::class, sdk = [34])
class ChatViewModelScreenTest {

    @get:Rule
    val mainDispatcherRule = MainDispatcherRule()

    private fun buildViewModel(): ChatViewModel {
        val application = org.robolectric.RuntimeEnvironment.getApplication()
        val loop: MemoryAugmentedAgenticLoop = mockk(relaxed = true)
        val providerRegistry: ProviderRegistry = mockk(relaxed = true)
        val toolRegistry: ToolRegistry = mockk(relaxed = true)
        val toolExecutor: ToolExecutor = mockk(relaxed = true)
        val textToSpeech: TextToSpeech = mockk(relaxed = true)
        val userPreferences: UserPreferences = mockk(relaxed = true)
        val memoryStore: com.aura.memory.MemoryStore = mockk(relaxed = true)
        val conversationStore: ConversationStore = mockk(relaxed = true)
        val knowledgeGraphRepository: KnowledgeGraphRepository = mockk(relaxed = true)

        every { userPreferences.defaultModel } returns MutableStateFlow("ollama:deepseek-v4-pro:cloud")
        every { providerRegistry.all() } returns emptyList()
        every { providerRegistry.configured() } returns emptyList()
        every { providerRegistry.get("moa") } returns null
        every { toolRegistry.definitions() } returns emptyList()
        coEvery { conversationStore.mostRecent() } returns null
        coEvery { knowledgeGraphRepository.stats() } returns KnowledgeGraphRepository.Stats(0, 0)

        return ChatViewModel(
            application = application,
            loop = loop,
            providerRegistry = providerRegistry,
            toolRegistry = toolRegistry,
            toolExecutor = toolExecutor,
            textToSpeech = textToSpeech,
            userPreferences = userPreferences,
            memoryStore = memoryStore,
            conversationStore = conversationStore,
            knowledgeGraphRepository = knowledgeGraphRepository,
        )
    }

    @Test
    fun `initial state has empty draft and assistant welcome prompt`() {
        val viewModel = buildViewModel()
        assertEquals("New conversation", viewModel.state.value.conversation.title)
        assertTrue(viewModel.state.value.conversation.systemPrompt.isNullOrEmpty() ||
            viewModel.state.value.conversation.systemPrompt?.contains("Aura") == true)
        assertEquals("", viewModel.state.value.draft)
    }

    @Test
    fun `setDraft updates draft and clears on send if empty`() {
        val viewModel = buildViewModel()
        viewModel.setDraft("hello")
        assertEquals("hello", viewModel.state.value.draft)
        viewModel.send()
        assertEquals("", viewModel.state.value.draft)
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class MainDispatcherRule : TestWatcher() {
    private val dispatcher = StandardTestDispatcher()
    override fun starting(description: Description?) {
        Dispatchers.setMain(dispatcher)
    }
    override fun finished(description: Description?) {
        Dispatchers.resetMain()
    }
}
