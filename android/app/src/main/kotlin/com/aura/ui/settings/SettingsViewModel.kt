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
import com.aura.providers.CustomEndpointState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SettingsCredentialSpec(
    val prefix: String,
    val label: String,
    val helperText: String,
    val testsModelCatalog: Boolean,
)

val SETTINGS_CREDENTIAL_SPECS: List<SettingsCredentialSpec> = listOf(
    SettingsCredentialSpec("ollama", "Ollama Cloud", "Get a key at ollama.com/settings/keys", true),
    SettingsCredentialSpec("anthropic", "Anthropic", "Get a key at console.anthropic.com/settings/keys", true),
    SettingsCredentialSpec("openai", "OpenAI", "Get a key at platform.openai.com/api-keys", true),
    SettingsCredentialSpec("deepseek", "DeepSeek", "Get a key at platform.deepseek.com/api_keys", true),
    SettingsCredentialSpec("gemini", "Gemini", "Get a key at aistudio.google.com/apikey", true),
    SettingsCredentialSpec("groq", "Groq", "Get a key at console.groq.com/keys", true),
    SettingsCredentialSpec("openrouter", "OpenRouter", "Get a key at openrouter.ai/keys", true),
    SettingsCredentialSpec("mistral", "Mistral AI", "Get a key at console.mistral.ai/api-keys", true),
    SettingsCredentialSpec("xai", "xAI Grok", "Get a key at console.x.ai", true),
    SettingsCredentialSpec("together", "Together AI", "Get a key at api.together.xyz/settings/api-keys", true),
    SettingsCredentialSpec("cerebras", "Cerebras", "Get a key at cloud.cerebras.ai", true),
    SettingsCredentialSpec("nvidia", "NVIDIA NIM", "Get a key at build.nvidia.com/explore/discover", true),
    SettingsCredentialSpec("llama", "Meta Llama", "Get a key at llama.developer.meta.com", true),
    SettingsCredentialSpec("chatgpt", "ChatGPT Subscription", "Paste a token from `codex login` (OpenAI subscription auth)", true),
    SettingsCredentialSpec("agnes", "Agnes AI", "Get a key at agnes-ai.com/dashboard", true),
    // "Custom Endpoint" is now a dedicated card (CustomEndpointCard) — it
    // needs both a base URL and an API key, so it can't be a single
    // ProviderKeyField row. Don't add it back to this list.
    SettingsCredentialSpec("moa", "Mixture-of-Agents", "Configure MoA presets in code; no API key", true),
    SettingsCredentialSpec("brave", "Brave Search", "Used by Brave web search tools", false),
    SettingsCredentialSpec("tavily", "Tavily Search", "Used by Tavily research tools", false),
    SettingsCredentialSpec("firecrawl", "Firecrawl", "Used by Firecrawl page extraction", false),
    SettingsCredentialSpec("exa", "Exa Search", "Neural search — get a key at exa.ai/dashboard", false),
    SettingsCredentialSpec("jina", "Jina Reader", "URL-to-text search — get a key at jina.ai/reader", false),
    SettingsCredentialSpec("elevenlabs", "ElevenLabs", "TTS — get a key at elevenlabs.io/app/settings/api-keys", false),
    SettingsCredentialSpec("stability", "Stability AI", "Image generation — platform.stability.ai/account/keys", false),
    SettingsCredentialSpec("kling", "Kling AI", "Video generation — klingai.com/dev", false),
    SettingsCredentialSpec("worldlabs", "World Labs", "3D world generation — worldlabs.ai", false),
)

private val TOOL_CREDENTIAL_PREFIXES: Set<String> = SETTINGS_CREDENTIAL_SPECS
    .filterNot { it.testsModelCatalog }
    .mapTo(mutableSetOf()) { it.prefix }

data class SettingsUiState(
    val keyDrafts: Map<String, String> = ProviderKeys.PREFIXES.associateWith { "" },
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
    // Custom endpoint card state.
    val customBaseUrl: String = "",
    val customApiKey: String = "",
    val customIsConfigured: Boolean = false,
    val customTesting: Boolean = false,
    val customResult: String? = null,
    /** Distinct from credentialStates["custom"]: the URL/key are stored
     *  outside ProviderKeys, so this is a separate UI state. */
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
    private val customEndpointState: CustomEndpointState,
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
        viewModelScope.launch {
            // Reactive read so the card flips from "Unsaved" → "Configured"
            // the moment the user taps Save & Test.
            customEndpointState.state.collectLatest { (url, key, _) ->
                _state.update {
                    it.copy(
                        customBaseUrl = url,
                        customIsConfigured = url.isNotBlank() && key.isNotBlank(),
                    )
                }
            }
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
                keyDrafts = ProviderKeys.PREFIXES.associateWith { prefix ->
                    providerKeys.keyFor(prefix).orEmpty()
                },
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

    fun updateCredentialDraft(prefix: String, value: String) {
        require(prefix in ProviderKeys.PREFIXES) { "Unknown credential prefix: $prefix" }
        _state.update { current ->
            current.copy(
                keyDrafts = current.keyDrafts + (prefix to value),
                providerTests = current.providerTests - prefix,
                verifyResults = current.verifyResults - prefix,
            )
        }
    }

    private fun keyDraft(prefix: String): String = _state.value.keyDrafts[prefix].orEmpty()

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
                if (prefix in TOOL_CREDENTIAL_PREFIXES) {
                    updateProviderTest(
                        prefix,
                        ProviderTestPhase.Idle,
                        if (value.isBlank()) "Credential removed" else "Saved securely",
                    )
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

    /**
     * Persist the user's drafts for the custom endpoint and verify the
     * connection by hitting the live `/models` endpoint. The state is
     * written to [CustomEndpointState] (which is what the provider reads
     * from at chat time) and persisted to DataStore by the singleton.
     */
    fun saveAndTestCustomEndpoint() {
        if (_state.value.customTesting) return
        val url = _state.value.customBaseUrl.trim().trimEnd('/')
        val key = _state.value.customApiKey.trim()
        if (url.isBlank() || key.isBlank()) {
            _state.update { it.copy(customResult = "✗ Base URL and API key are required") }
            return
        }
        _state.update { it.copy(customTesting = true, customResult = "Testing…") }
        viewModelScope.launch {
            customEndpointState.setEndpoint(url, key)
            try {
                modelCatalogRepository.refreshProvider("custom", force = true)
                val providerState = modelCatalogRepository.catalog.value.providers["custom"]
                val valid = providerState?.status == ProviderStatus.Ready &&
                    providerState.errorMessage == null &&
                    providerState.models.isNotEmpty()
                if (valid) {
                    _state.update {
                        it.copy(
                            customTesting = false,
                            customResult = "✓ Verified — ${providerState!!.models.size} models",
                            customIsConfigured = true,
                        )
                    }
                } else {
                    val message = providerState?.errorMessage
                        ?: providerState?.status?.name
                        ?: "Provider unavailable"
                    _state.update {
                        it.copy(
                            customTesting = false,
                            customResult = "✗ $message",
                        )
                    }
                }
            } catch (cancelled: kotlinx.coroutines.CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                _state.update {
                    it.copy(
                        customTesting = false,
                        customResult = "✗ ${error.message?.take(80) ?: "Connection failed"}",
                    )
                }
            }
        }
    }

    fun updateCustomBaseUrl(value: String) {
        _state.update {
            it.copy(
                customBaseUrl = value,
                customResult = null,
                customIsConfigured = it.customIsConfigured && value.isNotBlank(),
            )
        }
    }

    fun updateCustomApiKey(value: String) {
        _state.update {
            it.copy(
                customApiKey = value,
                customResult = null,
                customIsConfigured = it.customIsConfigured && value.isNotBlank(),
            )
        }
    }

    fun clearCustomEndpoint() {
        viewModelScope.launch {
            customEndpointState.setEndpoint("", "", emptyList())
            _state.update {
                it.copy(
                    customBaseUrl = "",
                    customApiKey = "",
                    customIsConfigured = false,
                    customResult = null,
                )
            }
        }
    }

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