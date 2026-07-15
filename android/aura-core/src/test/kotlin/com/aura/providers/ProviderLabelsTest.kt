package com.aura.providers

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Locks the canonical label of every provider prefix Aura knows about.
 * Single source of truth — the chat header, model picker, onboarding
 * step, and history row all consume [providerLabel]. Adding a new
 * provider means either adding its label here or letting the `else`
 * branch handle it (in which case this test should be updated to
 * confirm the new prefix is covered).
 */
class ProviderLabelsTest {

    @Test
    fun `known providers get their canonical label`() {
        assertEquals("Ollama", providerLabel("ollama"))
        assertEquals("Anthropic", providerLabel("anthropic"))
        assertEquals("OpenAI", providerLabel("openai"))
        assertEquals("DeepSeek", providerLabel("deepseek"))
        assertEquals("Gemini", providerLabel("gemini"))
        assertEquals("Groq", providerLabel("groq"))
        assertEquals("OpenRouter", providerLabel("openrouter"))
        assertEquals("NVIDIA", providerLabel("nvidia"))
        assertEquals("MoA", providerLabel("moa"))
        assertEquals("xAI Grok", providerLabel("xai"))
        assertEquals("Together AI", providerLabel("together"))
        assertEquals("Cerebras", providerLabel("cerebras"))
        assertEquals("Meta Llama", providerLabel("llama"))
        assertEquals("ChatGPT", providerLabel("chatgpt"))
        assertEquals("Agnes AI", providerLabel("agnes"))
        assertEquals("Custom Endpoint", providerLabel("custom"))
        assertEquals("Mistral AI", providerLabel("mistral"))
    }

    @Test
    fun `tool and capability providers get a clean label`() {
        assertEquals("Brave Search", providerLabel("brave"))
        assertEquals("Tavily Search", providerLabel("tavily"))
        assertEquals("Firecrawl", providerLabel("firecrawl"))
        assertEquals("Exa Search", providerLabel("exa"))
        assertEquals("Jina Reader", providerLabel("jina"))
        assertEquals("ElevenLabs", providerLabel("elevenlabs"))
        assertEquals("Stability AI", providerLabel("stability"))
        assertEquals("Kling AI", providerLabel("kling"))
        assertEquals("World Labs", providerLabel("worldlabs"))
    }

    @Test
    fun `unknown prefix capitalizes the first letter`() {
        // Adding a brand new provider should not crash — the else
        // branch keeps the UX honest until a deliberate label is
        // added.
        assertEquals("Foobar", providerLabel("foobar"))
        assertEquals("Baz", providerLabel("baz"))
    }

    @Test
    fun `every prefix in ProviderKeys has a non-blank label`() {
        for (prefix in ProviderKeys.PREFIXES) {
            val label = providerLabel(prefix)
            org.junit.Assert.assertTrue(
                "Provider '$prefix' produced a blank label",
                label.isNotBlank(),
            )
        }
    }
}
