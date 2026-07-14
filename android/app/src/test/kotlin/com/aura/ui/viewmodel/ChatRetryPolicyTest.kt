package com.aura.ui.viewmodel

import com.aura.agent.Conversation
import com.aura.agent.RecallSummary
import com.aura.agent.ToolTurn
import com.aura.agent.Turn
import com.aura.tools.Citation
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class ChatRetryPolicyTest {

    @Test
    fun `retry reuses last user turn and clears failed response artifacts`() {
        val conversation = Conversation(
            turns = listOf(
                Turn(user = "first", assistant = "done"),
                Turn(
                    user = "retry this",
                    assistant = "partial answer",
                    toolTurns = listOf(ToolTurn("tool-1", "search", "{}", "partial")),
                    citations = listOf(Citation(1, "Source", "https://example.com")),
                    recall = RecallSummary(memoryIds = listOf("memory-1")),
                ),
            ),
        )

        val retry = prepareConversationForRetry(conversation)

        assertNotNull(retry)
        assertEquals("retry this", retry.userText)
        assertEquals(2, retry.conversation.turns.size)
        assertEquals("retry this", retry.conversation.turns.last().user)
        assertNull(retry.conversation.turns.last().assistant)
        assertEquals(emptyList(), retry.conversation.turns.last().toolTurns)
        assertEquals(emptyList(), retry.conversation.turns.last().citations)
        assertNull(retry.conversation.turns.last().recall)
    }

    @Test
    fun `retry drops assistant-only turns after the last user turn`() {
        val conversation = Conversation(
            turns = listOf(
                Turn(user = "retry this", assistant = "failed"),
                Turn(assistant = "orphaned tail"),
            ),
            summaryThroughTurn = 2,
        )

        val retry = prepareConversationForRetry(conversation)

        assertNotNull(retry)
        assertEquals(1, retry.conversation.turns.size)
        assertEquals(1, retry.conversation.summaryThroughTurn)
    }

    @Test
    fun `retry is unavailable without a nonblank user turn`() {
        assertNull(prepareConversationForRetry(Conversation(turns = listOf(Turn(assistant = "hello")))))
    }
}
