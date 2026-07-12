package com.aura.memory

import com.aura.providers.ChatOptions
import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderRegistry
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class LlmWriteGateTest {

    private fun makeRegistry(text: String): ProviderRegistry {
        val registry = mockk<ProviderRegistry>(relaxed = true)
        coEvery { registry.chat(any(), any(), any(), any()) } returns flowOf(
            ProviderChunk(text = text),
        )
        return registry
    }

    @Test
    fun `heuristic rejects short content without LLM call`() = runTest {
        val registry = makeRegistry("") // no LLM call should happen
        val gate = LlmWriteGate(WriteGate(), registry, "test:model")
        val decision = gate.evaluate("hi", "user")
        assertFalse(decision.shouldStore)
        assertEquals("too_short", decision.reason)
    }

    @Test
    fun `heuristic rejects system messages without LLM call`() = runTest {
        val registry = makeRegistry("")
        val gate = LlmWriteGate(WriteGate(), registry, "test:model")
        val decision = gate.evaluate("system initialized", "system")
        assertFalse(decision.shouldStore)
        assertEquals("system_msg", decision.reason)
    }

    @Test
    fun `LLM says store with category and importance`() = runTest {
        val registry = makeRegistry("""{"store": true, "category": "preference", "importance": 0.9}""")
        val gate = LlmWriteGate(WriteGate(), registry, "test:model")
        val decision = gate.evaluate("I prefer dark mode for all my editors", "user")
        assertTrue(decision.shouldStore)
        assertEquals("preference", decision.category)
        assertEquals(0.9f, decision.importance, 0.01f)
        assertEquals("llm_classified", decision.reason)
    }

    @Test
    fun `LLM says do not store`() = runTest {
        val registry = makeRegistry("""{"store": false}""")
        val gate = LlmWriteGate(WriteGate(), registry, "test:model")
        val decision = gate.evaluate("what's the weather today?", "user")
        assertFalse(decision.shouldStore)
        assertEquals("llm_rejected", decision.reason)
    }

    @Test
    fun `falls back to heuristic when LLM returns garbage`() = runTest {
        val registry = makeRegistry("I think this is worth remembering")
        val gate = LlmWriteGate(WriteGate(), registry, "test:model")
        val decision = gate.evaluate("I prefer dark mode for all my editors", "user")
        // Should fall back to heuristic (which says store=true, category=preference)
        assertTrue(decision.shouldStore)
        assertEquals("preference", decision.category)
    }

    @Test
    fun `falls back to heuristic when LLM call throws`() = runTest {
        val registry = mockk<ProviderRegistry>(relaxed = true)
        coEvery { registry.chat(any(), any(), any(), any()) } throws RuntimeException("network error")
        val gate = LlmWriteGate(WriteGate(), registry, "test:model")
        val decision = gate.evaluate("I prefer dark mode for all my editors", "user")
        assertTrue(decision.shouldStore)
        assertEquals("preference", decision.category)
    }

    @Test
    fun `parses JSON wrapped in markdown code fence`() = runTest {
        val registry = makeRegistry("""```json
            {"store": true, "category": "fact", "importance": 0.7}
            ```""")
        val gate = LlmWriteGate(WriteGate(), registry, "test:model")
        val decision = gate.evaluate("My name is Elnur and I live in Baku", "user")
        assertTrue(decision.shouldStore)
        assertEquals("fact", decision.category)
    }

    @Test
    fun `importance is clamped to 0-1`() = runTest {
        val registry = makeRegistry("""{"store": true, "category": "fact", "importance": 1.5}""")
        val gate = LlmWriteGate(WriteGate(), registry, "test:model")
        val decision = gate.evaluate("I work as a software engineer", "user")
        assertTrue(decision.shouldStore)
        assertEquals(1.0f, decision.importance, 0.01f)
    }
}