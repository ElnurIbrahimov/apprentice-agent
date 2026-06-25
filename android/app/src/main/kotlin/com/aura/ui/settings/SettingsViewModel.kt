package com.aura.ui.settings

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.providers.ProviderRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

private val Context.auraPrefs by preferencesDataStore(name = "aura_settings")
private val KEY_OLLAMA_KEY = stringPreferencesKey("ollama_api_key")
private val KEY_ANTHROPIC_KEY = stringPreferencesKey("anthropic_api_key")
private val KEY_OPENAI_KEY = stringPreferencesKey("openai_api_key")
private val KEY_DEEPSEEK_KEY = stringPreferencesKey("deepseek_api_key")
private val KEY_DEFAULT_MODEL = stringPreferencesKey("default_model")
private val KEY_FIRST_RUN = stringPreferencesKey("first_run_complete")

data class SettingsUiState(
    val ollamaKey: String = "",
    val anthropicKey: String = "",
    val openaiKey: String = "",
    val deepseekKey: String = "",
    val defaultModel: String = "ollama:deepseek-v3.2:cloud",
    val firstRunComplete: Boolean = false,
    val configuredProviders: List<String> = emptyList(),
)

@HiltViewModel
class SettingsViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val providerRegistry: ProviderRegistry,
) : ViewModel() {

    private val _state = MutableStateFlow(SettingsUiState())
    val state: StateFlow<SettingsUiState> = _state.asStateFlow()

    init { reload() }

    fun reload() {
        viewModelScope.launch {
            val prefs = context.auraPrefs.data.first()
            val configured = providerRegistry.configured().map { "${it.prefix} (${it.displayName})" }
            _state.value = SettingsUiState(
                ollamaKey = prefs[KEY_OLLAMA_KEY] ?: "",
                anthropicKey = prefs[KEY_ANTHROPIC_KEY] ?: "",
                openaiKey = prefs[KEY_OPENAI_KEY] ?: "",
                deepseekKey = prefs[KEY_DEEPSEEK_KEY] ?: "",
                defaultModel = prefs[KEY_DEFAULT_MODEL] ?: "ollama:deepseek-v3.2:cloud",
                firstRunComplete = prefs[KEY_FIRST_RUN] == "true",
                configuredProviders = configured,
            )
        }
    }

    fun saveOllamaKey(k: String) = updateKey(KEY_OLLAMA_KEY, k) { _state.update { it.copy(ollamaKey = k) } }
    fun saveAnthropicKey(k: String) = updateKey(KEY_ANTHROPIC_KEY, k) { _state.update { it.copy(anthropicKey = k) } }
    fun saveOpenaiKey(k: String) = updateKey(KEY_OPENAI_KEY, k) { _state.update { it.copy(openaiKey = k) } }
    fun saveDeepseekKey(k: String) = updateKey(KEY_DEEPSEEK_KEY, k) { _state.update { it.copy(deepseekKey = k) } }

    fun setDefaultModel(model: String) {
        viewModelScope.launch {
            context.auraPrefs.edit { it[KEY_DEFAULT_MODEL] = model }
            _state.update { it.copy(defaultModel = model) }
        }
    }

    fun markFirstRunComplete() {
        viewModelScope.launch {
            context.auraPrefs.edit { it[KEY_FIRST_RUN] = "true" }
            _state.update { it.copy(firstRunComplete = true) }
        }
    }

    private fun updateKey(key: androidx.datastore.preferences.core.Preferences.Key<String>, value: String, refresh: () -> Unit) {
        viewModelScope.launch {
            context.auraPrefs.edit { it[key] = value }
            refresh()
            // Re-evaluate provider configuration
            // (Hilt singleton providers re-read env vars at startup; for runtime changes
            //  we'd need a refreshable provider registry, which v1.5 adds)
        }
    }
}
