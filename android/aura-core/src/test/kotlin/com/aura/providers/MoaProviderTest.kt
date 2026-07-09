package com.aura.providers

import app.cash.turbine.test
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * Unit tests for [MoaProvider] focused on cancellation and structure.
 */
class MoaProviderTest {

    @Test
    fun `cancel terminates in-flight MoA reference work`() = runTest {
        val hangFlow: Flow<ProviderChunk> = flow {
            delay(Long.MAX_VALUE)
        }

        val providers = mapOf(
            "ollama" to mockk<Provider>(relaxed = true) {
                every { this@mockk.prefix } returns "ollama"
                every { chat("glm-5.1:cloud", any(), any(), any()) } returns hangFlow
                every { chat("kimi-k2.6:cloud", any(), any(), any()) } returns hangFlow
            },
            "deepseek" to mockk<Provider>(relaxed = true) {
                every { prefix } returns "deepseek"
                every { displayName } returns "DeepSeek"
                every { chat("deepseek-v4-pro", any(), any(), any()) } returns flowOf()
            },
        )
        val registry = ProviderRegistry(providers)
        val moa = MoaProvider(registry = mockk { every { get() } returns registry })

        val job = launch {
            assertFailsWith<CancellationException> {
                moa.chat("default", emptyList(), ChatOptions(), emptyList()).collect {}
            }
        }

        advanceTimeBy(100L)
        moa.cancel()
        job.join()
    }

    @Test
    fun `aggregator is not called after cancellation`() = runTest {
        val hangFlow: Flow<ProviderChunk> = flow {
            delay(Long.MAX_VALUE)
        }

        val aggregator = mockk<Provider>(relaxed = true) {
            every { prefix } returns "deepseek"
            every { displayName } returns "DeepSeek"
            every { chat("deepseek-v4-pro", any(), any(), any()) } returns flowOf()
        }

        val providers = mapOf(
            "ollama" to mockk<Provider>(relaxed = true) {
                every { this@mockk.prefix } returns "ollama"
                every { chat("glm-5.1:cloud", any(), any(), any()) } returns hangFlow
                every { chat("kimi-k2.6:cloud", any(), any(), any()) } returns hangFlow
            },
            "deepseek" to aggregator,
        )
        val registry = ProviderRegistry(providers)
        val moa = MoaProvider(registry = mockk { every { get() } returns registry })

        val job = launch {
            assertFailsWith<CancellationException> {
                moa.chat("default", emptyList(), ChatOptions(), emptyList()).collect {}
            }
        }

        advanceTimeBy(100L)
        moa.cancel()
        job.join()

        verify(exactly = 0) { aggregator.chat("deepseek-v4-pro", any(), any(), any()) }
    }

    @Test
    fun `reference output propagates to aggregator message`() = runTest {
        val providers = mapOf(
            "ollama" to mockk<Provider>(relaxed = true) {
                every { this@mockk.prefix } returns "ollama"
                every { chat("glm-5.1:cloud", any(), any(), any()) } returns flowOf(
                    ProviderChunk(text = "Reference A output"),
                    ProviderChunk(finishReason = FinishReason.stop),
                )
                every { chat("kimi-k2.6:cloud", any(), any(), any()) } returns flowOf(
                    ProviderChunk(text = "Reference B output"),
                    ProviderChunk(finishReason = FinishReason.stop),
                )
            },
            "deepseek" to mockk<Provider>(relaxed = true) {
                every { prefix } returns "deepseek"
                every { displayName } returns "DeepSeek"
                every { chat("deepseek-v4-pro", any(), any(), any()) } returns flowOf(
                    ProviderChunk(text = "Aggregated"),
                    ProviderChunk(finishReason = FinishReason.stop),
                )
            },
        )
        val registry = ProviderRegistry(providers)
        val moa = MoaProvider(registry = mockk { every { get() } returns registry })

        moa.chat(
            "default",
            listOf(ProviderMessage(role = ProviderMessage.Role.user, content = "hi")),
            ChatOptions(),
            emptyList(),
        ).test {
            assertEquals("Aggregated", awaitItem().text)
            // The aggregator emits a finishReason=stop chunk after the text.
            awaitItem()
            awaitComplete()
        }
    }
}
