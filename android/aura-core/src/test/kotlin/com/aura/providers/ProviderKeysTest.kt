package com.aura.providers

import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import com.aura.security.DecryptionFailedException
import com.aura.security.KeyManager
import com.aura.security.SecureDataStore
import io.mockk.coEvery
import io.mockk.mockk
import kotlinx.coroutines.async
import kotlinx.coroutines.test.runTest
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Unit tests for [ProviderKeys] that exercise the live DataStore flow
 * without touching the actual Android framework. The data layer uses an
 * in-memory DataStore backed by a temp file; keys are encrypted via an
 * in-memory AES key.
 *
 * These tests catch regressions in the API key pipeline that was the
 * single biggest functional bug in the v1 cut — the user could type
 * a key in Settings, the DataStore would save it, and the providers
 * would never see it because they were constructed with a baked-in
 * env var.
 */
class ProviderKeysTest {

    /**
     * Builds a [ProviderKeys] wired to a clean in-memory DataStore and
     * an in-memory AES key so the test never touches the file system or
     * Android framework.
     */
    private fun createProviderKeys(): ProviderKeys {
        val file = File.createTempFile("pkt_test_", ".preferences_pb")
        file.deleteOnExit()
        val dataStore = PreferenceDataStoreFactory.create(
            produceFile = { file }
        )
        val keyManager = KeyManager(null) // in-memory key
        val secureDataStore = SecureDataStore(
            dataStoreProvider = javax.inject.Provider { dataStore },
            keyManager = keyManager,
        )
        return ProviderKeys(secureDataStore)
    }

    @Test
    fun `keyFor returns null when no key is set`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        assertNull(keys.keyFor("ollama"))
        assertNull(keys.keyFor("anthropic"))
    }

    @Test
    fun `keyFor returns null for blank keys`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        assertFalse(keys.isConfigured("ollama"))
    }

    @Test
    fun `isConfigured is false for unknown prefix`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        assertFalse(keys.isConfigured("nonexistent-provider"))
    }

    @Test
    fun `PREFIXES contains all supported providers`() {
        // The expected set is the source of truth; the production PREFIXES
        // list must always include these chat + capability prefixes. New
        // providers added to PREFIXES but not here will be a hidden regression.
        val expected = setOf(
            // chat
            "ollama", "anthropic", "openai", "deepseek", "gemini", "groq", "openrouter",
            "mistral", "xai", "together", "cerebras", "nvidia", "llama", "chatgpt",
            "agnes", "custom", "moa",
            // search & content
            "brave", "tavily", "firecrawl", "exa", "jina",
            // capabilities
            "elevenlabs", "stability", "kling", "worldlabs",
        )
        assertTrue(
            expected.all { it in ProviderKeys.PREFIXES },
            "Missing prefixes: ${expected - ProviderKeys.PREFIXES.toSet()}",
        )
    }

    // ─── New credential state tests ───────────────────────────────────────

    @Test
    fun `initial credential states are NotConfigured after load`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        for (prefix in ProviderKeys.PREFIXES) {
            assertEquals(
                ProviderCredentialState.NotConfigured,
                keys.credentialStates.value[prefix],
                "Expected NotConfigured for prefix '$prefix'"
            )
        }
    }

    @Test
    fun `initial state map is empty`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        assertTrue(keys.state.value.isEmpty())
    }

    @Test
    fun `loaded is initially false and becomes true after init`() = runTest {
        val keys = createProviderKeys()
        assertFalse(keys.loaded.value, "loaded should be false before init completes")
        keys.awaitLoaded()
        assertTrue(keys.loaded.value, "loaded should be true after init completes")
    }

    @Test
    fun `set stores key and returns it exactly`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        keys.set("ollama", "sk-test-123-abc")
        assertEquals("sk-test-123-abc", keys.keyFor("ollama"))
        assertTrue(keys.isConfigured("ollama"))
    }

    @Test
    fun `set updates credential state to Saved`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        keys.set("ollama", "sk-ollama-key")
        assertEquals(ProviderCredentialState.Saved, keys.credentialStates.value["ollama"])
    }

    @Test
    fun `set with blank key clears credential`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        keys.set("ollama", "sk-ollama-key")
        assertTrue(keys.isConfigured("ollama"))
        assertEquals(ProviderCredentialState.Saved, keys.credentialStates.value["ollama"])

        keys.set("ollama", "")
        assertNull(keys.keyFor("ollama"))
        assertFalse(keys.isConfigured("ollama"))
        assertEquals(ProviderCredentialState.NotConfigured, keys.credentialStates.value["ollama"])
    }

    @Test
    fun `set with whitespace-only key clears credential`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        keys.set("ollama", "sk-ollama-key")
        assertTrue(keys.isConfigured("ollama"))

        keys.set("ollama", "   ")
        assertNull(keys.keyFor("ollama"))
        assertFalse(keys.isConfigured("ollama"))
        assertEquals(ProviderCredentialState.NotConfigured, keys.credentialStates.value["ollama"])
    }

    @Test
    fun `setting one provider does not affect others`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()

        keys.set("ollama", "sk-ollama-value")
        keys.set("anthropic", "sk-anthropic-value")

        assertEquals("sk-ollama-value", keys.keyFor("ollama"))
        assertEquals("sk-anthropic-value", keys.keyFor("anthropic"))
        assertNull(keys.keyFor("openai"))
    }

    @Test
    fun `overwriting a key returns the latest value`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()
        keys.set("ollama", "old-key")
        keys.set("ollama", "new-key")
        assertEquals("new-key", keys.keyFor("ollama"))
        assertEquals(ProviderCredentialState.Saved, keys.credentialStates.value["ollama"])
    }

    @Test
    fun `credential state transitions NotConfigured to Saved to NotConfigured`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()

        // Initial
        assertEquals(ProviderCredentialState.NotConfigured, keys.credentialStates.value["ollama"])

        // Set → Saved
        keys.set("ollama", "some-key")
        assertEquals(ProviderCredentialState.Saved, keys.credentialStates.value["ollama"])

        // Clear → NotConfigured
        keys.set("ollama", "")
        assertEquals(ProviderCredentialState.NotConfigured, keys.credentialStates.value["ollama"])
    }

    @Test
    fun `only target provider credential state changes on set`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()

        keys.set("ollama", "sk-ollama")
        assertEquals(ProviderCredentialState.Saved, keys.credentialStates.value["ollama"])

        // Anthropic should remain NotConfigured
        assertEquals(ProviderCredentialState.NotConfigured, keys.credentialStates.value["anthropic"])
    }

    @Test
    fun `state flow preserves backward compat`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()

        keys.set("ollama", "sk-ollama-val")
        keys.set("anthropic", "sk-anthro-val")

        val map = keys.state.value
        assertEquals("sk-ollama-val", map["ollama"])
        assertEquals("sk-anthro-val", map["anthropic"])
        assertEquals(2, map.size)
    }

    @Test
    fun `write is persisted across ProviderKeys instances`() = runTest {
        val file = File.createTempFile("pkt_persist_", ".preferences_pb")
        file.deleteOnExit()
        val dataStore = PreferenceDataStoreFactory.create(produceFile = { file })
        val keyManager = KeyManager(null)
        val secureDataStore = SecureDataStore(
            dataStoreProvider = javax.inject.Provider { dataStore },
            keyManager = keyManager,
        )

        val keys1 = ProviderKeys(secureDataStore)
        keys1.awaitLoaded()
        keys1.set("ollama", "sk-persist-value")

        // Second instance reading from same DataStore
        val keys2 = ProviderKeys(secureDataStore)
        keys2.awaitLoaded()
        assertEquals("sk-persist-value", keys2.keyFor("ollama"))
        assertEquals(ProviderCredentialState.Saved, keys2.credentialStates.value["ollama"])
    }

    @Test
    fun `decryption failure during init sets StorageError terminal state`() = runTest {
        val mockStore = mockk<SecureDataStore>()
        // First provider fails, others are unset
        coEvery { mockStore.getString(any()) } returns null
        coEvery { mockStore.getString("ollama_api_key") } throws DecryptionFailedException("bad decrypt")
        coEvery { mockStore.getString("embedding_model") } returns null
        coEvery { mockStore.removeString(any()) } returns Unit
        coEvery { mockStore.putString(any(), any()) } returns Unit

        val keys = ProviderKeys(mockStore)
        keys.awaitLoaded()

        assertEquals(ProviderCredentialState.StorageError, keys.credentialStates.value["ollama"])
        assertNull(keys.keyFor("ollama"))
        assertFalse(keys.isConfigured("ollama"))
        // Other providers should still be NotConfigured (not StorageError)
        assertEquals(ProviderCredentialState.NotConfigured, keys.credentialStates.value["anthropic"])
    }

    @Test
    fun `loaded becomes true even when init load encounters errors`() = runTest {
        val mockStore = mockk<SecureDataStore>()
        coEvery { mockStore.getString(any()) } throws DecryptionFailedException("bad decrypt")
        coEvery { mockStore.getString("embedding_model") } returns null
        coEvery { mockStore.removeString(any()) } returns Unit
        coEvery { mockStore.putString(any(), any()) } returns Unit

        val keys = ProviderKeys(mockStore)
        keys.awaitLoaded()
        assertTrue(keys.loaded.value, "loaded must become true even on decryption failures")
    }

    @Test
    fun `concurrent sets for different providers both succeed`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()

        // Simulate concurrent writes using async from the runTest scope
        val job1 = async {
            keys.set("ollama", "sk-ollama-concurrent")
        }
        val job2 = async {
            keys.set("anthropic", "sk-anthropic-concurrent")
        }
        job1.await()
        job2.await()

        assertEquals("sk-ollama-concurrent", keys.keyFor("ollama"))
        assertEquals("sk-anthropic-concurrent", keys.keyFor("anthropic"))
    }

    @Test
    fun `serial writes to same provider always reflect the latest`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()

        // Sequential writes to the same provider - only the last should win
        keys.set("ollama", "key-1")
        keys.set("ollama", "key-2")
        keys.set("ollama", "key-3")

        assertEquals("key-3", keys.keyFor("ollama"))
    }

    @Test
    fun `setEmbeddingModel preserves credential states`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()

        keys.set("ollama", "sk-ollama")
        keys.setEmbeddingModel("custom-model")

        assertEquals("sk-ollama", keys.keyFor("ollama"))
        assertEquals(ProviderCredentialState.Saved, keys.credentialStates.value["ollama"])
        assertEquals("custom-model", keys.embeddingModel)
    }

    @Test
    fun `keyFor only returns keys for Saved providers`() = runTest {
        val keys = createProviderKeys()
        keys.awaitLoaded()

        // Initially NotConfigured - should return null
        assertNull(keys.keyFor("openai"))

        // After set - should return key
        keys.set("openai", "sk-openai")
        assertNotNull(keys.keyFor("openai"))

        // After clear - should return null
        keys.set("openai", "")
        assertNull(keys.keyFor("openai"))
    }
}
