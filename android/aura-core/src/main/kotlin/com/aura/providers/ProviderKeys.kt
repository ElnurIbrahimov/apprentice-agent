package com.aura.providers

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

private val Context.auraProviderKeys by preferencesDataStore(name = "aura_settings")

/**
 * Source of truth for cloud-provider API keys. Each key is the value the user
 * pasted into the Settings UI (Ollama Cloud, Anthropic, OpenAI, DeepSeek).
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
 * Mirrors the role aura_python/api_keys.py played in the Python codebase.
 */
@Singleton
class ProviderKeys @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val _state = MutableStateFlow<Map<String, String>>(emptyMap())
    val state: StateFlow<Map<String, String>> = _state.asStateFlow()

    // Process-scoped: the @Singleton lives for the lifetime of the app, so we
    // don't need to cancel the scope explicitly.
    private val scope = kotlinx.coroutines.CoroutineScope(
        kotlinx.coroutines.SupervisorJob() + kotlinx.coroutines.Dispatchers.IO
    )

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
            val prefs = runCatching { context.auraProviderKeys.data.first() }.getOrNull()
            if (prefs != null) _state.value = readFrom(prefs)
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
     * Write a new key and refresh the cached state. Persisted via DataStore
     * so the value survives process death.
     */
    suspend fun set(prefix: String, key: String) {
        context.auraProviderKeys.edit { prefs ->
            val datastoreKey = stringPreferencesKey("${prefix}_api_key")
            if (key.isBlank()) prefs.remove(datastoreKey) else prefs[datastoreKey] = key
        }
        val refreshed = runCatching {
            val prefs = context.auraProviderKeys.data.first()
            readFrom(prefs)
        }.getOrDefault(_state.value)
        _state.value = refreshed
    }

    private fun readFrom(prefs: androidx.datastore.preferences.core.Preferences): Map<String, String> {
        val out = mutableMapOf<String, String>()
        for (prefix in PREFIXES) {
            val datastoreKey = stringPreferencesKey("${prefix}_api_key")
            prefs[datastoreKey]?.let { out[prefix] = it }
        }
        return out
    }

    companion object {
        val PREFIXES = listOf("ollama", "anthropic", "openai", "deepseek")
    }
}
