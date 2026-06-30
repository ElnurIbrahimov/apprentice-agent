package com.aura.ui.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.data.UserPreferences
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SettingsUiState(
    val ollamaKey: String = "",
    val anthropicKey: String = "",
    val openaiKey: String = "",
    val deepseekKey: String = "",
    val groqKey: String = "",
    val openrouterKey: String = "",
    val defaultModel: String = "ollama:deepseek-v4-pro:cloud",
    val firstRunComplete: Boolean = false,
    val configuredProviders: List<String> = emptyList(),
)

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val providerRegistry: ProviderRegistry,
    private val providerKeys: ProviderKeys,
    private val userPreferences: UserPreferences,
) : ViewModel() {

    private val _state = MutableStateFlow(SettingsUiState())
    val state: StateFlow<SettingsUiState> = _state.asStateFlow()

    init { reload() }

    fun reload() {
        viewModelScope.launch {
            val configured = providerRegistry.configured().map { "${it.prefix} (${it.displayName})" }
            val defaultModel = userPreferences.defaultModel.first()
            val firstRunComplete = userPreferences.firstRunComplete.first()
            _state.value = SettingsUiState(
                ollamaKey = providerKeys.keyFor("ollama") ?: "",
                anthropicKey = providerKeys.keyFor("anthropic") ?: "",
                openaiKey = providerKeys.keyFor("openai") ?: "",
                deepseekKey = providerKeys.keyFor("deepseek") ?: "",
                groqKey = providerKeys.keyFor("groq") ?: "",
                openrouterKey = providerKeys.keyFor("openrouter") ?: "",
                defaultModel = defaultModel,
                firstRunComplete = firstRunComplete,
                configuredProviders = configured,
            )
        }
    }

    fun saveOllamaKey(k: String) = updateKey("ollama", k) { _state.update { it.copy(ollamaKey = k) } }
    fun saveAnthropicKey(k: String) = updateKey("anthropic", k) { _state.update { it.copy(anthropicKey = k) } }
    fun saveOpenaiKey(k: String) = updateKey("openai", k) { _state.update { it.copy(openaiKey = k) } }
    fun saveDeepseekKey(k: String) = updateKey("deepseek", k) { _state.update { it.copy(deepseekKey = k) } }
    fun saveGroqKey(k: String) = updateKey("groq", k) { _state.update { it.copy(groqKey = k) } }
    fun saveOpenrouterKey(k: String) = updateKey("openrouter", k) { _state.update { it.copy(openrouterKey = k) } }

    fun setDefaultModel(model: String) {
        viewModelScope.launch {
            userPreferences.setDefaultModel(model)
            _state.update { it.copy(defaultModel = model) }
        }
    }

    fun markFirstRunComplete() {
        viewModelScope.launch {
            userPreferences.setFirstRunComplete(true)
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
