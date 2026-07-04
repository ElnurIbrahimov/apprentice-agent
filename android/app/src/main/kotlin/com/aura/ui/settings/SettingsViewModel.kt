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
    /**
     * Whether the biometric app lock is on. When true, MainActivity
     * gates on a [androidx.biometric.BiometricPrompt] challenge
     * before showing the rest of the UI. Persisted via
     * [com.aura.data.UserPreferences] and round-tripped through
     * backup so the choice survives reinstalls onto the same device.
     */
    val appLockEnabled: Boolean = false,
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
            val appLockEnabled = userPreferences.appLockEnabled.first()
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
                appLockEnabled = appLockEnabled,
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

    /**
     * Toggle the biometric app lock. The actual gate that enforces
     * the lock lives in [com.aura.MainActivity]; this just persists
     * the choice. We don't run a biometric challenge on toggle —
     * the user is already in the app and authenticated to the OS
     * session. The challenge fires the next time the app is
     * launched (or resumed from background).
     */
    fun setAppLockEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferences.setAppLockEnabled(enabled)
            _state.update { it.copy(appLockEnabled = enabled) }
        }
    }

    private fun updateKey(prefix: String, value: String, refresh: () -> Unit) {
        viewModelScope.launch {
            providerKeys.set(prefix, value)
            refresh()
        }
    }
}
