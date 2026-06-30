package com.aura.ui.viewmodel

import com.aura.agent.Conversation
import com.aura.agent.ConversationStore
import com.aura.agent.Turn
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.coVerifyOrder
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
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for [HistoryViewModel.exportMarkdown] (synchronous, pure
 * function over a [Conversation]) and the load/search/delete state
 * machine.
 */
class HistoryViewModelTest {

    private val store = mockk<ConversationStore>(relaxed = true)

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `exportMarkdown produces a valid document with title and turns`() {
        val vm = HistoryViewModel(mockk(relaxed = true), store)
        val conv = Conversation(
            id = "c1",
            title = "Help with Kotlin",
            createdAt = 1_700_000_000_000L,
            updatedAt = 1_700_000_100_000L,
            systemPrompt = "be brief",
            model = "ollama:deepseek-v4-pro:cloud",
            turns = listOf(
                Turn(user = "How do I read a file?"),
                Turn(assistant = "Use kotlin.io.path.Path."),
                Turn(
                    user = "Show me",
                    assistant = "Here's an example:",
                    toolTurns = listOf(
                        com.aura.agent.ToolTurn(
                            id = "t1",
                            name = "read_file",
                            args = "{\"path\":\"/tmp/x\"}",
                            result = "hello world",
                        ),
                    ),
                ),
            ),
        )

        val md = vm.exportMarkdown(conv)

        // Title as a heading.
        assertTrue(md.startsWith("# Help with Kotlin"), "should start with title heading, got: $md")
        // Timestamps in `_..._` italics. The date is timezone-dependent
        // so we just check the format YYYY-MM-DD rather than a specific
        // value.
        assertTrue(
            Regex("""_Created \d{4}-\d{2}-\d{2} \d{2}:\d{2} · Updated \d{4}-\d{2}-\d{2} \d{2}:\d{2}_""")
                .containsMatchIn(md),
            "timestamps should be in YYYY-MM-DD HH:MM format, got: $md",
        )
        // Model annotation.
        assertTrue("`ollama:deepseek-v4-pro:cloud`" in md, "should include model")
        // Each turn is a section.
        assertTrue("## User\n\nHow do I read a file?" in md, "user turn should be a section")
        assertTrue("## Assistant\n\nUse kotlin.io.path.Path." in md, "assistant turn should be a section")
        assertTrue("## Tool: `read_file`" in md, "tool call should be a section")
        assertTrue("```json" in md, "tool args should be a code block")
        assertTrue("> hello world" in md, "tool result should be a blockquote")
    }

    @Test
    fun `exportMarkdown escapes an empty conversation gracefully`() {
        val vm = HistoryViewModel(mockk(relaxed = true), store)
        val conv = Conversation(id = "c1", title = "Empty", createdAt = 0L, updatedAt = 0L)
        val md = vm.exportMarkdown(conv)
        // Title and timestamps, but no per-turn sections.
        assertTrue("# Empty" in md)
        assertFalse("## User" in md, "no user turn so no User section")
        assertFalse("## Assistant" in md, "no assistant turn so no Assistant section")
    }

    @Test
    fun `load calls store_recent`() = runTest {
        coEvery { store.recent(50) } returns emptyList()
        val vm = HistoryViewModel(mockk(relaxed = true), store)
        vm.load()
        coVerify { store.recent(50) }
        assertEquals(emptyList(), vm.state.value.conversations)
        assertFalse(vm.state.value.loading)
    }

    @Test
    fun `delete calls store_delete then reloads`() = runTest {
        coEvery { store.delete("c1") } returns Unit
        coEvery { store.recent(50) } returns emptyList()
        val vm = HistoryViewModel(mockk(relaxed = true), store)
        vm.delete("c1")
        coVerifyOrder { store.delete("c1"); store.recent(50) }
    }
}
