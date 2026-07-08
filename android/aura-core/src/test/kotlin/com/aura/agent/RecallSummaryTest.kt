package com.aura.agent

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Pure-data tests for [RecallSummary] and the [Turn] /
 * [Conversation] integration. The actual recall capture happens
 * in [MemoryAugmentedAgenticLoop] and is covered by the loop's
 * own integration tests — here we lock the shape of the data
 * model so chat UI / history UI can rely on it.
 */
class RecallSummaryTest {

    @Test
    fun `default RecallSummary is empty`() {
        val r = RecallSummary()
        assertEquals(emptyList(), r.memoryIds)
        assertEquals(emptyList(), r.handIds)
        // noResults defaults to false; the loop always recomputes
        // it based on whether the lists are empty, so callers
        // shouldn't rely on the default value.
    }

    @Test
    fun `recall with memories only is non-empty`() {
        val r = RecallSummary(memoryIds = listOf("m1", "m2", "m3"))
        assertEquals(3, r.memoryIds.size)
        assertEquals(0, r.handIds.size)
        assertFalse(r.noResults)
    }

    @Test
    fun `recall with hand only is non-empty`() {
        val r = RecallSummary(handIds = listOf("h1"))
        assertEquals(1, r.handIds.size)
        assertEquals(0, r.memoryIds.size)
        assertFalse(r.noResults)
    }

    @Test
    fun `recall with both is non-empty`() {
        val r = RecallSummary(
            memoryIds = listOf("m1", "m2"),
            handIds = listOf("h1"),
        )
        assertEquals(2, r.memoryIds.size)
        assertEquals(1, r.handIds.size)
        assertFalse(r.noResults)
    }

    @Test
    fun `Turn defaults to no recall`() {
        val turn = Turn()
        assertNull(turn.recall)
    }

    @Test
    fun `Turn can carry a recall summary`() {
        val r = RecallSummary(memoryIds = listOf("m1"))
        val turn = Turn(user = "hi", recall = r)
        assertEquals(r, turn.recall)
    }

    @Test
    fun `attachRecallToLastTurn is no-op on empty conversation`() {
        val conv = Conversation()
        val updated = conv.attachRecallToLastTurn(RecallSummary(memoryIds = listOf("m1")))
        // No turns means no-op — no exceptions, conversation unchanged.
        assertEquals(0, updated.turns.size)
    }

    @Test
    fun `attachRecallToLastTurn attaches to the last turn`() {
        val conv = Conversation(
            turns = listOf(
                Turn(user = "first", assistant = "first reply"),
                Turn(user = "second", assistant = "second reply"),
            ),
        )
        val recall = RecallSummary(memoryIds = listOf("m1", "m2"))
        val updated = conv.attachRecallToLastTurn(recall)
        assertEquals(2, updated.turns.size)
        // First turn untouched.
        assertNull(updated.turns[0].recall)
        // Last turn has the recall.
        assertEquals(recall, updated.turns[1].recall)
    }

    @Test
    fun `attachRecallToLastTurn preserves other turn fields`() {
        val original = Turn(
            user = "hi",
            assistant = "hello",
            citations = listOf(),
            timestamp = 1234L,
        )
        val conv = Conversation(turns = listOf(original))
        val updated = conv.attachRecallToLastTurn(RecallSummary(memoryIds = listOf("m1")))
        val last = updated.turns.last()
        assertEquals("hi", last.user)
        assertEquals("hello", last.assistant)
        assertEquals(1234L, last.timestamp)
        assertNotNull(last.recall)
    }
}
