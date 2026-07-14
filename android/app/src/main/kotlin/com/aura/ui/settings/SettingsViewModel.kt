package com.aura.ui.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.agent.IdentityStore
import com.aura.data.UserPreferences
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import com.aura.providers.ProviderCredentialState
import com.aura.providers.ModelCatalog
import com.aura.providers.ModelCatalogRepository
import com.aura.providers.ProviderStatus
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.collectLatest
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
    val defaultModel: String = "",
    val visionModel: String = "",
    val backgroundModel: String = "",
    val deepModeModel: String = "",
    val moaReferenceModels: List<String> = emptyList(),
    val moaAggregatorModel: String = "",
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
    val embeddingModel: String = "",
    val themeMode: String = "system",
    /** Full identity text resolved from DataStore override or bundled asset. */
    val identityText: String = "",
    /** True when the user has a non-blank DataStore identity override. */
    val identityCustomized: Boolean = false,
    val specialistOverrides: String = "{}",
    /**
     * Per-provider verify result: prefix → "✓ Verified — N models"
     * or "✗ Failed: ...". Null = not tested yet.
     */
    val verifyResults: Map<String, String> = emptyMap(),
    val verifying: String? = null,
    val morningBriefHour: Int = 7,
    val availableModels: List<String> = emptyList(),
    val modelsLoading: Boolean = false,
    val modelsError: String? = null,
    val providerTests: Map<String, ProviderTestResult> = emptyMap(),
    val credentialStates: Map<String, ProviderCredentialState> = emptyMap(),
)

enum class ProviderTestPhase { Idle, Saving, Testing, Verified, Failed }

data class ProviderTestResult(
    val phase: ProviderTestPhase = ProviderTestPhase.Idle,
    val message: String? = null,
    val modelCount: Int = 0,
)

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val providerRegistry: ProviderRegistry,
    private val providerKeys: ProviderKeys,
    private val userPreferences: UserPreferences,
    private val identityStore: IdentityStore,
    private val modelCatalogRepository: ModelCatalogRepository,
) : ViewModel() {

    private fun configuredProviderLabels(): List<String> =
        providerRegistry.configured()
            .sortedBy { it.prefix }
            .map { "${it.prefix} (${it.displayName})" }

    private val _state = MutableStateFlow(SettingsUiState())
    val state: StateFlow<SettingsUiState> = _state.asStateFlow()

    init {
        reload()
        viewModelScope.launch {
            providerKeys.credentialStates.collectLatest { states ->
                _state.update { it.copy(credentialStates = states) }
            }
        }
        viewModelScope.launch {
            modelCatalogRepository.catalog.collectLatest(::applyCatalog)
        }
    }

    fun reload() {
        viewModelScope.launch {
            // Wait for the initial DataStore load to finish so the
            // configured providers list doesn't show "0 configured"
            // on first launch while the keys are still being read.
            providerKeys.loaded.first { it }
            val configured = configuredProviderLabels()
            val defaultModel = userPreferences.defaultModel.first()
            val visionModel = userPreferences.visionModel.first()
            val backgroundModel = userPreferences.backgroundModel.first()
            val deepModeModel = userPreferences.deepModeModel.first()
            val moaReferenceModels = userPreferences.moaReferenceModels.first()
            val moaAggregatorModel = userPreferences.moaAggregatorModel.first()
            val firstRunComplete = userPreferences.firstRunComplete.first()
            val appLockEnabled = userPreferences.appLockEnabled.first()
            val morningBriefEnabled = userPreferences.morningBriefEnabled.first()
            val calendarMonitorEnabled = userPreferences.calendarMonitorEnabled.first()
            val embeddingModel = providerKeys.embeddingModel
            val themeMode = userPreferences.themeMode.first()
            val identityText = identityStore.readCurrent()
            val identityCustomized = identityStore.hasOverride()
            val specialistOverrides = userPreferences.specialistOverrides.first()
            val morningBriefHour = userPreferences.morningBriefHour.first()
            _state.value = SettingsUiState(
                ollamaKey = providerKeys.keyFor("ollama") ?: "",
                anthropicKey = providerKeys.keyFor("anthropic") ?: "",
                openaiKey = providerKeys.keyFor("openai") ?: "",
                deepseekKey = providerKeys.keyFor("deepseek") ?: "",
                groqKey = providerKeys.keyFor("groq") ?: "",
                openrouterKey = providerKeys.keyFor("openrouter") ?: "",
                defaultModel = defaultModel.orEmpty(),
                visionModel = visionModel.orEmpty(),
                backgroundModel = backgroundModel.orEmpty(),
                deepModeModel = deepModeModel.orEmpty(),
                moaReferenceModels = moaReferenceModels,
                moaAggregatorModel = moaAggregatorModel.orEmpty(),
                firstRunComplete = firstRunComplete,
                configuredProviders = configured,
                appLockEnabled = appLockEnabled,
                morningBriefEnabled = morningBriefEnabled,
                calendarMonitorEnabled = calendarMonitorEnabled,
                embeddingModel = embeddingModel,
                themeMode = themeMode,
                identityText = identityText,
                identityCustomized = identityCustomized,
                specialistOverrides = specialistOverrides,
                morningBriefHour = morningBriefHour,
            )
        }
    }

    fun saveOllamaKey(k: String) = updateKeyDraft("ollama", k)
    fun saveAnthropicKey(k: String) = updateKeyDraft("anthropic", k)
    fun saveOpenaiKey(k: String) = updateKeyDraft("openai", k)
    fun saveDeepseekKey(k: String) = updateKeyDraft("deepseek", k)
    fun saveGroqKey(k: String) = updateKeyDraft("groq", k)
    fun saveOpenrouterKey(k: String) = updateKeyDraft("openrouter", k)

    fun setDefaultModel(model: String) {
        viewModelScope.launch {
            userPreferences.setDefaultModel(model)
            _state.update { it.copy(defaultModel = model) }
        }
    }

    fun setVisionModel(model: String) {
        viewModelScope.launch {
            userPreferences.setVisionModel(model)
            _state.update { it.copy(visionModel = model) }
        }
    }

    fun setBackgroundModel(model: String) {
        viewModelScope.launch {
            userPreferences.setBackgroundModel(model)
            _state.update { it.copy(backgroundModel = model) }
        }
    }

    fun setDeepModeModel(model: String) {
        viewModelScope.launch {
            userPreferences.setDeepModeModel(model)
            _state.update { it.copy(deepModeModel = model) }
        }
    }

    fun setMoaReferenceModels(models: List<String>) {
        viewModelScope.launch {
            val selected = models.distinct().take(4)
            userPreferences.setMoaReferenceModels(selected)
            _state.update { it.copy(moaReferenceModels = selected) }
            modelCatalogRepository.refreshProvider("moa", force = true)
        }
    }

    fun setMoaAggregatorModel(model: String) {
        viewModelScope.launch {
            userPreferences.setMoaAggregatorModel(model)
            _state.update { it.copy(moaAggregatorModel = model) }
            modelCatalogRepository.refreshProvider("moa", force = true)
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

    /** Persist the DataStore-backed identity used by the next chat send. */
    fun saveIdentity(text: String) {
        viewModelScope.launch {
            if (text.isBlank()) {
                identityStore.resetToDefault()
            } else {
                identityStore.save(text)
            }
            _state.update {
                it.copy(
                    identityText = identityStore.readCurrent(),
                    identityCustomized = identityStore.hasOverride(),
                )
            }
        }
    }

    /** Clear the custom identity and fall back to the bundled asset. */
    fun resetIdentity() {
        viewModelScope.launch {
            identityStore.resetToDefault()
            _state.update {
                it.copy(
                    identityText = identityStore.readCurrent(),
                    identityCustomized = identityStore.hasOverride(),
                )
            }
        }
    }

    fun setSpecialistOverrides(json: String) {
        viewModelScope.launch {
            userPreferences.setSpecialistOverrides(json)
            _state.update { it.copy(specialistOverrides = json) }
        }
    }

    private fun updateKeyDraft(prefix: String, value: String) {
        _state.update { current ->
            val updated = when (prefix) {
                "ollama" -> current.copy(ollamaKey = value)
                "anthropic" -> current.copy(anthropicKey = value)
                "openai" -> current.copy(openaiKey = value)
                "deepseek" -> current.copy(deepseekKey = value)
                "groq" -> current.copy(groqKey = value)
                "openrouter" -> current.copy(openrouterKey = value)
                else -> current
            }
            updated.copy(
                providerTests = updated.providerTests - prefix,
                verifyResults = updated.verifyResults - prefix,
            )
        }
    }

    private fun keyDraft(prefix: String): String = when (prefix) {
        "ollama" -> _state.value.ollamaKey
        "anthropic" -> _state.value.anthropicKey
        "openai" -> _state.value.openaiKey
        "deepseek" -> _state.value.deepseekKey
        "groq" -> _state.value.groqKey
        "openrouter" -> _state.value.openrouterKey
        else -> ""
    }

    fun saveAndTestProvider(prefix: String) {
        if (_state.value.verifying != null) return
        val value = keyDraft(prefix).trim()
        updateProviderTest(prefix, ProviderTestPhase.Saving, "Saving securely…")
        viewModelScope.launch {
            try {
                providerKeys.set(prefix, value)
                if (providerKeys.credentialStates.value[prefix] == ProviderCredentialState.StorageError) {
                    updateProviderTest(prefix, ProviderTestPhase.Failed, "Secure storage failed")
                    return@launch
                }
                if (value.isBlank()) {
                    modelCatalogRepository.refreshProvider(prefix, force = true)
                    updateProviderTest(prefix, ProviderTestPhase.Idle, "Credential removed")
                    return@launch
                }

                updateProviderTest(prefix, ProviderTestPhase.Testing, "Testing provider…")
                modelCatalogRepository.refreshProvider(prefix, force = true)
                val providerState = modelCatalogRepository.catalog.value.providers[prefix]
                val valid = providerState?.status == ProviderStatus.Ready &&
                    providerState.errorMessage == null &&
                    providerState.models.isNotEmpty()
                providerKeys.markValidation(prefix, valid)
                if (valid) {
                    updateProviderTest(
                        prefix,
                        ProviderTestPhase.Verified,
                        "Verified — ${providerState!!.models.size} models",
                        providerState.models.size,
                    )
                } else {
                    val message = providerState?.errorMessage
                        ?: providerState?.status?.name
                        ?: "Provider unavailable"
                    updateProviderTest(prefix, ProviderTestPhase.Failed, message)
                }
            } catch (cancelled: kotlinx.coroutines.CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                providerKeys.markValidation(prefix, false)
                updateProviderTest(
                    prefix,
                    ProviderTestPhase.Failed,
                    error.message?.take(100) ?: "Provider test failed",
                )
            } finally {
                _state.update {
                    it.copy(
                        verifying = null,
                        configuredProviders = configuredProviderLabels(),
                    )
                }
            }
        }
    }

    private fun updateProviderTest(
        prefix: String,
        phase: ProviderTestPhase,
        message: String,
        modelCount: Int = 0,
    ) {
        val legacy = when (phase) {
            ProviderTestPhase.Verified -> "✓ $message"
            ProviderTestPhase.Failed -> "✗ $message"
            else -> message
        }
        _state.update {
            it.copy(
                verifying = if (phase == ProviderTestPhase.Saving || phase == ProviderTestPhase.Testing) {
                    prefix
                } else null,
                providerTests = it.providerTests + (
                    prefix to ProviderTestResult(phase, message, modelCount)
                ),
                verifyResults = it.verifyResults + (prefix to legacy),
            )
        }
    }

    fun refreshModels() {
        if (_state.value.modelsLoading) return
        _state.update { it.copy(modelsLoading = true, modelsError = null) }
        viewModelScope.launch {
            modelCatalogRepository.refresh(force = true)
        }
    }

    fun verifyKey(prefix: String) = saveAndTestProvider(prefix)

    private fun applyCatalog(catalog: ModelCatalog) {
        val failures = catalog.providers.values
            .filter { provider ->
                provider.status !in setOf(
                    ProviderStatus.NotConfigured,
                    ProviderStatus.Loading,
                    ProviderStatus.Ready,
                )
            }
            .map { provider ->
                "${provider.providerPrefix}: ${provider.errorMessage ?: provider.status.name}"
            }
        _state.update {
            it.copy(
                configuredProviders = configuredProviderLabels(),
                availableModels = catalog.allModels.map { model -> model.id }.distinct().sorted(),
                modelsLoading = catalog.providers.values.any { provider ->
                    provider.status == ProviderStatus.Loading
                },
                modelsError = failures.firstOrNull(),
            )
        }
    }
}