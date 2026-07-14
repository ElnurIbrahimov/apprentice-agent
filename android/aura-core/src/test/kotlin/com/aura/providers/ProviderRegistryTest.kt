package com.aura.providers

import com.aura.usage.UsageTracker
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import java.util.concurrent.Executors
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

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

    @Test
    fun `chat collects provider flow off the caller thread`() = runBlocking {
        val provider = mockk<Provider>(relaxed = true) {
            every { prefix } returns "foo"
            every { chat("model", any(), any(), any()) } returns flow {
                emit(ProviderChunk(text = Thread.currentThread().name))
            }
        }
        val registry = ProviderRegistry(mapOf("foo" to provider))
        val caller = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "ui-caller")
        }.asCoroutineDispatcher()

        try {
            val providerThread = withContext(caller) {
                registry.chat(
                    "foo:model",
                    listOf(ProviderMessage(ProviderMessage.Role.user, "question")),
                ).first().text.orEmpty()
            }

            assertFalse(
                providerThread.contains("ui-caller"),
                "provider flow ran on caller thread: $providerThread",
            )
        } finally {
            caller.close()
        }
    }
}
