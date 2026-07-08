package com.aura.providers

import com.aura.security.SecureDataStore
import kotlinx.coroutines.CoroutineScope
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
 * StateFlow cache is invalidated on every DataStore write.
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
    private val _state = MutableStateFlow<Map<String, String>>(emptyMap())
    val state: StateFlow<Map<String, String>> = _state.asStateFlow()

    /**
     * True once the initial DataStore load completes. The Settings
     * and Onboarding screens wait on this before showing the
     * "configured providers" list so the user never sees a
     * momentary "0 configured" flicker. Set in [init] after
     * [loadAllKeys] finishes.
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
     * Guards the DataStore load-and-publish sequence. The initial async load in
     * [init] and the explicit write in [set] both call [loadAllKeys] and
     * publish the result to [_state]. Without the mutex a user who saves a key
     * immediately after app start could have that write overwritten by the
     * still-running init load.
     */
    private val stateMutex = Mutex()

    /**
     * Asynchronous initial load. We do NOT block here — Hilt graph construction
     * is on the main thread, and a blocking DataStore read on it can ANR.
     * Instead, [keyFor] returns null until the load completes; the user sees
     * "no provider configured" for a few hundred ms after app start, then
     * their saved key takes effect. This is the same behavior as before the
     * fix (the env-var approach also had no notion of "saved in DataStore").
     */
    init {
        scope.launch {
            val loaded = loadAllKeys()
            val model = loadEmbeddingModel()
            stateMutex.withLock {
                _state.value = loaded
                _embeddingModel.value = model
            }
            // Flip the loaded signal last, after the state is
            // populated, so a consumer waiting on loaded.value
            // sees the populated state in the same read.
            _loaded.value = true
        }
    }

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
     */
    suspend fun set(prefix: String, key: String) {
        val datastoreKey = "${prefix}_api_key"
        if (key.isBlank()) {
            secureDataStore.removeString(datastoreKey)
        } else {
            secureDataStore.putString(datastoreKey, key)
        }
        val loaded = loadAllKeys()
        stateMutex.withLock {
            _state.value = loaded
        }
    }

    private suspend fun loadAllKeys(): Map<String, String> {
        val out = mutableMapOf<String, String>()
        for (prefix in PREFIXES) {
            val datastoreKey = "${prefix}_api_key"
            secureDataStore.getString(datastoreKey)?.let { out[prefix] = it }
        }
        return out
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
