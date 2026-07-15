package com.aura.ui.viewmodel

import android.app.Application
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.aura.agent.ConversationStore
import com.aura.agent.MemoryAugmentedAgenticLoop
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolRegistry
import com.aura.core.error.CrashLogger
import com.aura.data.UserPreferences
import com.aura.documents.DocumentTextExtractor
import com.aura.documents.ExtractedDocument
import com.aura.kg.KnowledgeGraphRepository
import com.aura.memory.MemoryStore
import com.aura.providers.ModelCatalogRepository
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import com.aura.voice.TextToSpeech
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
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
class ChatViewModelDocumentTest {

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
    private lateinit var documentTextExtractor: DocumentTextExtractor

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
        documentTextExtractor = mockk(relaxed = true)

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

    private fun createViewModel(): ChatViewModel = ChatViewModel(
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
        documentTextExtractor = documentTextExtractor,
    )

    @Test
    fun `onDocumentPicked inserts extracted text as user message`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()

        val uri = Uri.parse("content://test/document.pdf")
        coEvery { documentTextExtractor.extract(uri) } returns ExtractedDocument(
            id = "abc",
            name = "notes.pdf",
            mimeType = "application/pdf",
            sourceUri = uri.toString(),
            text = "This is the document content.",
        )

        vm.onDocumentPicked(uri)
        advanceUntilIdle()

        val turns = vm.state.value.conversation.turns
        assertEquals(1, turns.size)
        assertTrue(turns.first().user?.contains("Attached document: notes.pdf") == true)
        assertTrue(turns.first().user?.contains("This is the document content.") == true)
        assertFalse(vm.state.value.streaming)
    }

    @Test
    fun `onDocumentPicked truncates oversized text`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()

        val uri = Uri.parse("content://test/long.txt")
        val longText = "x".repeat(15000)
        coEvery { documentTextExtractor.extract(uri) } returns ExtractedDocument(
            id = "def",
            name = "long.txt",
            mimeType = "text/plain",
            sourceUri = uri.toString(),
            text = longText,
        )

        vm.onDocumentPicked(uri)
        advanceUntilIdle()

        val userText = vm.state.value.conversation.turns.first().user
        assertTrue(userText?.contains("long.txt") == true)
        assertTrue((userText?.length ?: 0) < longText.length + 200)
        assertTrue(userText?.contains("more characters truncated") == true)
    }

    @Test
    fun `onDocumentPicked surfaces extraction errors`() = runTest(testDispatcher) {
        val vm = createViewModel()
        advanceUntilIdle()

        val uri = Uri.parse("content://test/broken.pdf")
        coEvery { documentTextExtractor.extract(uri) } throws IllegalStateException("Cannot open PDF")

        vm.onDocumentPicked(uri)
        advanceUntilIdle()

        assertEquals("Could not read document: Cannot open PDF", vm.state.value.error)
        assertFalse(vm.state.value.streaming)
        assertEquals(0, vm.state.value.conversation.turns.size)
    }
}
