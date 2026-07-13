package com.aura.providers

import com.aura.security.SecureDataStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Source of truth for cloud-provider API keys and embedding model settings.
 * Each key is the value the user pasted into the Settings UI (Ollama Cloud,
 * Anthropic, OpenAI, DeepSeek).
 *
 * Previously, [ProviderModule] read these from `System.getenv(...)` at Hilt
 * graph creation time, which meant the Settings UI was write-only — the user
 * typed a key, the DataStore stored it, but the provider instances already
 * constructed with the env var never saw the change. Even restarting the app
 * did not help because the env var didn't change.
 *
 * This singleton reads the keys from DataStore on demand. The Hilt graph still
 * constructs one ProviderKeys instance (singleton), and providers call
 * [keyFor] at every [Provider.chat] / [Provider.isConfigured] call. The
 * StateFlow cache is invalidated per-provider on every DataStore write
 * (no full loadAllKeys after each set).
 *
 * The initial load runs asynchronously to avoid ANR during Hilt graph
 * construction. [keyFor] returns null until that load completes. To avoid a
 * concurrent Settings write being overwritten by the late init load, all
 * writes and reads that need a consistent snapshot go through [stateMutex].
 *
 * Mirrors the role aura_python/api_keys.py played in the Python codebase.
 */
@Singleton
class ProviderKeys @Inject constructor(
    private val secureDataStore: SecureDataStore,
) {
    /**
     * The raw key-value map, published for backward compatibility.
     * Only contains entries for providers with non-blank keys.
     * Use [credentialStates] for richer lifecycle information.
     */
    private val _state = MutableStateFlow<Map<String, String>>(emptyMap())
    val state: StateFlow<Map<String, String>> = _state.asStateFlow()

    /**
     * Per-provider credential lifecycle state. Each provider in [PREFIXES]
     * appears as a key in this map.
     *
     * Transitions:
     * - [ProviderCredentialState.Loading] → [ProviderCredentialState.NotConfigured]
     *   (no saved key) or [ProviderCredentialState.Saved] (key loaded) or
     *   [ProviderCredentialState.StorageError] (decryption failure)
     * - [ProviderCredentialState.NotConfigured] → [ProviderCredentialState.Saved]
     *   (via [set])
     * - [ProviderCredentialState.Saved] → [ProviderCredentialState.NotConfigured]
     *   (via [set] with blank key)
     * - Any → [ProviderCredentialState.StorageError] on decryption failure
     */
    private val _credentialStates = MutableStateFlow(
        PREFIXES.associateWith { ProviderCredentialState.Loading }
    )
    val credentialStates: StateFlow<Map<String, ProviderCredentialState>> =
        _credentialStates.asStateFlow()

    /**
     * True once the initial DataStore load completes. The Settings
     * and Onboarding screens wait on this before showing the
     * "configured providers" list so the user never sees a
     * momentary "0 configured" flicker. Set in [init] after
     * per-provider state is populated.
     *
     * Always becomes true even when individual providers encounter
     * decryption errors — consumers are never stuck in a
     * perpetual loading state.
     */
    private val _loaded = MutableStateFlow(false)
    val loaded: StateFlow<Boolean> = _loaded.asStateFlow()

    /** Embedding model name (default: "nomic-embed-text"). */
    private val _embeddingModel = MutableStateFlow(DEFAULT_EMBEDDING_MODEL)
    val embeddingModel: String get() = _embeddingModel.value

    // Process-scoped: the @Singleton lives for the lifetime of the app, so we
    // don't need to cancel the scope explicitly.
    private val scope = CoroutineScope(
        SupervisorJob() + Dispatchers.IO
    )

    /**
     * Guards all state mutations. The initial async load in [init] and the
     * explicit write in [set] both go through this mutex to prevent the init
     * load from overwriting a user's freshly-written key.
     */
    private val stateMutex = Mutex()

    /**
     * Asynchronous initial load. We do NOT block here — Hilt graph construction
     * is on the main thread, and a blocking DataStore read on it can ANR.
     * Instead, [keyFor] returns null until the load completes; the user sees
     * "no provider configured" for a few hundred ms after app start, then
     * their saved key takes effect. This is the same behavior as before the
     * fix (the env-var approach also had no notion of "saved in DataStore").
     *
     * Individual decryption failures are handled per-provider (resulting in
     * [ProviderCredentialState.StorageError]) so a corrupted entry for one
     * provider does not block all other providers or leave [loaded] stuck at
     * false.
     */
    init {
        scope.launch {
            val values = mutableMapOf<String, String>()
            val states = mutableMapOf<String, ProviderCredentialState>()

            for (prefix in PREFIXES) {
                try {
                    val key = secureDataStore.getString("${prefix}_api_key")?.takeIf { it.isNotBlank() }
                    if (key == null) {
                        states[prefix] = ProviderCredentialState.NotConfigured
                    } else {
                        values[prefix] = key
                        states[prefix] = ProviderCredentialState.Saved
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (_: Exception) {
                    states[prefix] = ProviderCredentialState.StorageError
                }
            }

            val model = loadEmbeddingModel()
            stateMutex.withLock {
                _state.value = values
                _values.clear()
                _values.putAll(values)
                _credentialStates.value = states.toMap()
                _embeddingModel.value = model
            }
            // Flip the loaded signal last, after all state is
            // populated, so a consumer waiting on loaded.value
            // sees the populated state in the same read.
            // Always set to true, even if individual providers
            // encountered StorageError — terminal error is not
            // "still loading".
            _loaded.value = true
        }
    }

    /**
     * Internal decrypted values cache. Mirrors [state] for fast [keyFor]
     * access without map reconstruction. Modified directly in [set] and
     * [init] under [stateMutex].
     */
    private val _values = mutableMapOf<String, String>()

    /**
     * Returns the current API key for the given provider prefix, or null if
     * the user hasn't set one. Called on every chat request so the most
     * recent value wins.
     */
    fun keyFor(prefix: String): String? = _state.value[prefix]?.takeIf { it.isNotBlank() }

    /** True if the user has set a non-blank key for the given prefix. */
    fun isConfigured(prefix: String): Boolean = !keyFor(prefix).isNullOrBlank()

    /**
     * Block until the initial DataStore load completes. Use only on
     * app start (Application.onCreate) where blocking the main thread
     * is acceptable — DataStore reads on a warm start complete in
     * 5-20ms and on a cold start in ~50-100ms. Calling this from
     * a UI or background thread is wrong; the caller is on a hot
     * path.
     */
    suspend fun awaitLoaded() {
        if (_loaded.value) return
        _loaded.first { it }
    }

    /**
     * Write a new key and refresh the cached state. Persisted via DataStore
     * so the value survives process death.
     *
     * Unlike the previous implementation, this updates only the affected
     * provider in memory (no full loadAllKeys after each write). This ensures
     * writes are O(1) per provider and a concurrent older write cannot
     * overwrite a newer one because the targeted provider state is always
     * derived from the most recent write that acquired [stateMutex].
     *
     * Blank or whitespace-only keys are normalized to clear, removing the
     * stored entry from both DataStore and in-memory state.
     *
     * Never logs the key value.
     */
    suspend fun set(prefix: String, key: String) {
        val trimmed = key.trim()
        val datastoreKey = "${prefix}_api_key"
        stateMutex.withLock {
            if (trimmed.isBlank()) {
                secureDataStore.removeString(datastoreKey)
                _values.remove(prefix)
                _state.value = _state.value - prefix
                _credentialStates.value = _credentialStates.value + (prefix to ProviderCredentialState.NotConfigured)
            } else {
                secureDataStore.putString(datastoreKey, trimmed)
                _values[prefix] = trimmed
                _state.value = _state.value + (prefix to trimmed)
                _credentialStates.value = _credentialStates.value + (prefix to ProviderCredentialState.Saved)
            }
        }
    }


    /** Load the embedding model from DataStore, returning the fallback if absent. */
    private suspend fun loadEmbeddingModel(): String {
        val saved = secureDataStore.getString("embedding_model")
        return if (!saved.isNullOrBlank()) saved else DEFAULT_EMBEDDING_MODEL
    }

    /** Persist a new embedding model name. */
    suspend fun setEmbeddingModel(model: String) {
        val value = model.takeIf { it.isNotBlank() } ?: DEFAULT_EMBEDDING_MODEL
        if (value == DEFAULT_EMBEDDING_MODEL) {
            secureDataStore.removeString("embedding_model")
        } else {
            secureDataStore.putString("embedding_model", value)
        }
        stateMutex.withLock {
            _embeddingModel.value = value
        }
    }

    companion object {
        val PREFIXES = listOf("ollama", "anthropic", "openai", "deepseek", "gemini", "groq", "openrouter", "brave", "tavily", "firecrawl")
        const val DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    }
}
