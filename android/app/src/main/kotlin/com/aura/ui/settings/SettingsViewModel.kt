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
    val appLockEnabled: Boolean = false,
    val morningBriefEnabled: Boolean = true,
    val calendarMonitorEnabled: Boolean = true,
    /**
     * Current embedding model id, persisted via ProviderKeys. Used
     * to drive the Settings embedding-model picker and to restore
     * the choice after backup import.
     */
    val embeddingModel: String = ProviderKeys.DEFAULT_EMBEDDING_MODEL,
    val themeMode: String = "system",
    val customIdentity: String = "",
    val specialistOverrides: String = "{}",
    /**
     * Per-provider verify result: prefix → "✓ Verified — N models"
     * or "✗ Failed: ...". Null = not tested yet.
     */
    val verifyResults: Map<String, String> = emptyMap(),
    val verifying: String? = null,
    val morningBriefHour: Int = 7,
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
            // Wait for the initial DataStore load to finish so the
            // configured providers list doesn't show "0 configured"
            // on first launch while the keys are still being read.
            providerKeys.loaded.first { it }
            val configured = providerRegistry.configured().map { "${it.prefix} (${it.displayName})" }
            val defaultModel = userPreferences.defaultModel.first()
            val firstRunComplete = userPreferences.firstRunComplete.first()
            val appLockEnabled = userPreferences.appLockEnabled.first()
            val morningBriefEnabled = userPreferences.morningBriefEnabled.first()
            val calendarMonitorEnabled = userPreferences.calendarMonitorEnabled.first()
            val embeddingModel = providerKeys.embeddingModel
            val themeMode = userPreferences.themeMode.first()
            val customIdentity = userPreferences.customIdentity.first()
            val specialistOverrides = userPreferences.specialistOverrides.first()
            val morningBriefHour = userPreferences.morningBriefHour.first()
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
                morningBriefEnabled = morningBriefEnabled,
                calendarMonitorEnabled = calendarMonitorEnabled,
                embeddingModel = embeddingModel,
                themeMode = themeMode,
                customIdentity = customIdentity,
                specialistOverrides = specialistOverrides,
                morningBriefHour = morningBriefHour,
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

    /**
     * Toggle the morning-brief schedule. The actual cancel /
     * reschedule happens in [com.aura.proactive.ProactiveBootstrap]
     * on the next app launch — toggling in Settings persists the
     * choice, and the worker state converges when the app next
     * starts. This is intentional: the Settings VM has no business
     * touching WorkManager directly (it would couple the UI layer
     * to the proactive subsystem).
     */
    fun setMorningBriefEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferences.setMorningBriefEnabled(enabled)
            _state.update { it.copy(morningBriefEnabled = enabled) }
        }
    }

    fun setMorningBriefHour(hour: Int) {
        viewModelScope.launch {
            userPreferences.setMorningBriefHour(hour)
            _state.update { it.copy(morningBriefHour = hour) }
        }
    }

    /**
     * Toggle the calendar-monitor foreground service. Same pattern
     * as [setMorningBriefEnabled]: persists the choice, bootstrap
     * converges the actual FGS state on next app launch.
     */
    fun setCalendarMonitorEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferences.setCalendarMonitorEnabled(enabled)
            _state.update { it.copy(calendarMonitorEnabled = enabled) }
        }
    }

    fun setEmbeddingModel(model: String) {
        viewModelScope.launch {
            providerKeys.setEmbeddingModel(model)
            _state.update { it.copy(embeddingModel = model) }
        }
    }

    fun setThemeMode(mode: String) {
        viewModelScope.launch {
            userPreferences.setThemeMode(mode)
            _state.update { it.copy(themeMode = mode) }
        }
    }

    fun setCustomIdentity(identity: String) {
        viewModelScope.launch {
            userPreferences.setCustomIdentity(identity)
            _state.update { it.copy(customIdentity = identity) }
        }
    }

    fun setSpecialistOverrides(json: String) {
        viewModelScope.launch {
            userPreferences.setSpecialistOverrides(json)
            _state.update { it.copy(specialistOverrides = json) }
        }
    }

    private fun updateKey(prefix: String, value: String, refresh: () -> Unit) {
        viewModelScope.launch {
            providerKeys.set(prefix, value)
            refresh()
        }
    }

    /**
     * Verify the saved API key by hitting the provider's models endpoint.
     * If the call succeeds, the key is good. Same pattern as
     * OnboardingScreen.OnboardingViewModel.verifyKey.
     */
    fun verifyKey(prefix: String) {
        if (providerRegistry.configured().none { it.prefix == prefix }) {
            _state.update { it.copy(verifyResults = it.verifyResults + (prefix to "✗ No key saved for $prefix")) }
            return
        }
        _state.update { it.copy(verifying = prefix) }
        viewModelScope.launch {
            val provider = providerRegistry.all().firstOrNull { it.prefix == prefix }
            val result = if (provider == null) {
                "✗ Provider $prefix not found"
            } else {
                runCatching { provider.listModels() }
                    .map { "✓ Verified — ${it.size} models available" }
                    .getOrElse { "✗ Failed: ${it.message?.take(80)}" }
            }
            _state.update { it.copy(verifying = null, verifyResults = it.verifyResults + (prefix to result)) }
        }
    }
}