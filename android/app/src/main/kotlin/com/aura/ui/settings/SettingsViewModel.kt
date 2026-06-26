package com.aura.ui.settings

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

private val Context.auraPrefs by preferencesDataStore(name = "aura_settings")
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
    private val providerKeys: ProviderKeys,
) : ViewModel() {

    private val _state = MutableStateFlow(SettingsUiState())
    val state: StateFlow<SettingsUiState> = _state.asStateFlow()

    init { reload() }

    fun reload() {
        viewModelScope.launch {
            val ollama = providerKeys.keyFor("ollama") ?: ""
            val anthropic = providerKeys.keyFor("anthropic") ?: ""
            val openai = providerKeys.keyFor("openai") ?: ""
            val deepseek = providerKeys.keyFor("deepseek") ?: ""
            val configured = providerRegistry.configured().map { "${it.prefix} (${it.displayName})" }
            val prefs = context.auraPrefs.data.first()
            _state.value = SettingsUiState(
                ollamaKey = ollama,
                anthropicKey = anthropic,
                openaiKey = openai,
                deepseekKey = deepseek,
                defaultModel = prefs[KEY_DEFAULT_MODEL] ?: "ollama:deepseek-v3.2:cloud",
                firstRunComplete = prefs[KEY_FIRST_RUN] == "true",
                configuredProviders = configured,
            )
        }
    }

    fun saveOllamaKey(k: String) = updateKey("ollama", k) { _state.update { it.copy(ollamaKey = k) } }
    fun saveAnthropicKey(k: String) = updateKey("anthropic", k) { _state.update { it.copy(anthropicKey = k) } }
    fun saveOpenaiKey(k: String) = updateKey("openai", k) { _state.update { it.copy(openaiKey = k) } }
    fun saveDeepseekKey(k: String) = updateKey("deepseek", k) { _state.update { it.copy(deepseekKey = k) } }

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

    private fun updateKey(prefix: String, value: String, refresh: () -> Unit) {
        viewModelScope.launch {
            providerKeys.set(prefix, value)
            refresh()
        }
    }
}
