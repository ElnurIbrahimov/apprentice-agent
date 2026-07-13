package com.aura.agent

import com.aura.core.error.CrashLogger
import com.aura.providers.FinishReason
import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

class ConversationCompactorTest {

    private val registry = mockk<ProviderRegistry>()
    private val crashLogger = mockk<CrashLogger>(relaxed = true)
    private val compactor = ConversationCompactor(registry, crashLogger)

    @Test
    fun `messages include durable summary and never resend summarized turns`() {
        val conversation = Conversation(
            systemPrompt = "Base system",
            contextSummary = "Elnur chose the teal design and Phase 1 is complete.",
            summaryThroughTurn = 2,
            turns = listOf(
                Turn(user = "old user", assistant = "old answer"),
                Turn(user = "also old", assistant = "also answered"),
                Turn(user = "recent question", assistant = "recent answer"),
            ),
        )

        val messages = conversation.toMessages()

        assertEquals(4, messages.size)
        assertEquals(ProviderMessage.Role.system, messages[0].role)
        assertEquals("Base system", messages[0].content)
        assertEquals(ProviderMessage.Role.system, messages[1].role)
        assertTrue(messages[1].content.contains("teal design"))
        assertTrue(messages.none { it.content.contains("old user") || it.content.contains("also old") })
        assertEquals("recent question", messages[2].content.substringBefore("recent answer").trim())
    }

    @Test
    fun `below threshold leaves conversation untouched without provider call`() = runTest {
        val conversation = conversationWithTurns(48)

        val result = compactor.compactIfNeeded(conversation, "test:model")

        assertSame(conversation, result)
        coVerify(exactly = 0) { registry.chat(any(), any(), any(), any()) }
    }

    @Test
    fun `first compaction summarizes oldest batch and preserves every stored turn`() = runTest {
        val conversation = conversationWithTurns(60)
        coEvery { registry.chat("test:model", any(), any(), any()) } returns flowOf(
            ProviderChunk(text = "The user is building Aura. Decisions one through thirty-six are preserved."),
            ProviderChunk(finishReason = FinishReason.stop),
        )

        val result = compactor.compactIfNeeded(conversation, "test:model")

        assertEquals(60, result.turns.size)
        assertEquals(36, result.summaryThroughTurn)
        assertTrue(result.contextSummary.contains("Decisions one through thirty-six"))
        assertEquals(conversation.turns, result.turns)
        coVerify(exactly = 1) { registry.chat("test:model", any(), any(), emptyList()) }
    }

    @Test
    fun `rolling compaction supplies previous summary and advances only new old turns`() = runTest {
        val conversation = conversationWithTurns(85).copy(
            contextSummary = "Existing durable facts.",
            summaryThroughTurn = 36,
        )
        var requestMessages: List<ProviderMessage> = emptyList()
        coEvery { registry.chat("test:model", any(), any(), any()) } answers {
            requestMessages = secondArg()
            flowOf(ProviderChunk(text = "Updated durable facts."), ProviderChunk(finishReason = FinishReason.stop))
        }

        val result = compactor.compactIfNeeded(conversation, "test:model")

        assertEquals(61, result.summaryThroughTurn)
        assertEquals("Updated durable facts.", result.contextSummary)
        val payload = requestMessages.joinToString("\n") { it.content }
        assertTrue(payload.contains("Existing durable facts"))
        assertTrue(payload.contains("user-36"))
        assertTrue(payload.contains("user-60"))
        assertTrue(!payload.contains("user-35"))
        assertTrue(!payload.contains("user-61"))
    }

    @Test
    fun `provider failure never blocks the actual conversation`() = runTest {
        val conversation = conversationWithTurns(60)
        coEvery { registry.chat(any(), any(), any(), any()) } returns flow {
            throw IllegalStateException("provider unavailable")
        }

        val result = compactor.compactIfNeeded(conversation, "test:model")

        assertSame(conversation, result)
    }

    @Test(expected = CancellationException::class)
    fun `cancellation is never swallowed`() = runTest {
        val conversation = conversationWithTurns(60)
        coEvery { registry.chat(any(), any(), any(), any()) } returns flow {
            throw CancellationException("stop")
        }
        compactor.compactIfNeeded(conversation, "test:model")
    }

    private fun conversationWithTurns(count: Int): Conversation = Conversation(
        turns = List(count) { index ->
            Turn(user = "user-$index", assistant = "assistant-$index")
        },
    )
}
