package com.aura.providers

import com.aura.usage.UsageTracker
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ProviderRegistryTest {

    @Test
    fun `parse splits on colon`() = runTest {
        val p = mockk<Provider>(relaxed = true)
        val registry = ProviderRegistry(mapOf("foo" to p))
        val (prov, model) = registry.parse("foo:bar")
        assertEquals(p, prov)
        assertEquals("bar", model)
    }

    @Test
    fun `parse rejects unqualified model without discovery`() = runTest {
        val p = mockk<Provider>(relaxed = true) {
            every { prefix } returns "foo"
        }
        val registry = ProviderRegistry(mapOf("foo" to p))

        assertFailsWith<IllegalArgumentException> { registry.parse("default") }
        coVerify(exactly = 0) { p.listModels() }
    }

    @Test
    fun `parse throws on unknown prefix`() = runTest {
        val p = mockk<Provider>(relaxed = true)
        val registry = ProviderRegistry(mapOf("foo" to p))
        assertFailsWith<IllegalArgumentException> { registry.parse("bar:baz") }
    }

    @Test
    fun `chat records exact provider usage in central ledger`() = runTest {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "foo"
            every { chat("model", any(), any(), any()) } returns flowOf(
                ProviderChunk(text = "answer"),
                ProviderChunk(
                    finishReason = FinishReason.stop,
                    usage = Usage(promptTokens = 12, completionTokens = 7, totalTokens = 19),
                ),
            )
        }
        val tracker = UsageTracker()
        val registry = ProviderRegistry(mapOf("foo" to provider), tracker)

        registry.chat(
            "foo:model",
            listOf(ProviderMessage(ProviderMessage.Role.user, "question")),
        ).collect()

        val usage = tracker.snapshot.value
        assertEquals(12, usage.promptTokens)
        assertEquals(7, usage.completionTokens)
        assertEquals("foo:model", usage.models.single().modelId)
    }
}
