package com.aura.ui.screens.onboarding

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aura.FirstRunGate
import com.aura.data.UserPreferences
import com.aura.providers.ModelCatalog
import com.aura.providers.ModelCatalogRepository
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderStatus
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

enum class OnboardingStep { Intro, Provider, Model, Complete }

enum class OnboardingCredentialStatus { Empty, Draft, Saving, Verified, Invalid }

data class OnboardingUiState(
    val step: OnboardingStep = OnboardingStep.Intro,
    val keyDrafts: Map<String, String> = emptyMap(),
    val credentialStatus: Map<String, OnboardingCredentialStatus> = emptyMap(),
    val providerMessages: Map<String, String> = emptyMap(),
    val catalog: ModelCatalog = ModelCatalog(emptyMap(), emptyList()),
    val selectedDefaultModel: String? = null,
    val error: String? = null,
)

@HiltViewModel
class OnboardingViewModel @Inject constructor(
    private val firstRunGate: FirstRunGate,
    private val providerKeys: ProviderKeys,
    private val modelCatalogRepository: ModelCatalogRepository,
    private val userPreferences: UserPreferences,
) : ViewModel() {
    private val _state = MutableStateFlow(OnboardingUiState())
    val state: StateFlow<OnboardingUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            providerKeys.awaitLoaded()
            _state.value = _state.value.copy(
                keyDrafts = PROVIDERS.associateWith { providerKeys.keyFor(it).orEmpty() },
            )
        }
        viewModelScope.launch {
            modelCatalogRepository.catalog.collect { catalog ->
                val current = _state.value
                val statuses = current.credentialStatus.toMutableMap()
                val messages = current.providerMessages.toMutableMap()
                for ((prefix, provider) in catalog.providers) {
                    when (provider.status) {
                        ProviderStatus.Ready -> {
                            statuses[prefix] = OnboardingCredentialStatus.Verified
                            messages[prefix] = "Verified · ${provider.models.size} models"
                        }
                        ProviderStatus.Unauthorized -> {
                            statuses[prefix] = OnboardingCredentialStatus.Invalid
                            messages[prefix] = "The key was rejected."
                        }
                        ProviderStatus.RateLimit,
                        ProviderStatus.Network,
                        ProviderStatus.Timeout,
                        ProviderStatus.Malformed,
                        ProviderStatus.Empty -> {
                            if (statuses[prefix] == OnboardingCredentialStatus.Saving) {
                                statuses[prefix] = OnboardingCredentialStatus.Invalid
                                messages[prefix] = provider.errorMessage ?: provider.status.name
                            }
                        }
                        ProviderStatus.Loading -> statuses[prefix] = OnboardingCredentialStatus.Saving
                        ProviderStatus.NotConfigured -> Unit
                    }
                }
                _state.value = current.copy(
                    catalog = catalog,
                    credentialStatus = statuses,
                    providerMessages = messages,
                )
            }
        }
        viewModelScope.launch {
            userPreferences.defaultModel.collect { model ->
                _state.value = _state.value.copy(selectedDefaultModel = model)
            }
        }
    }

    fun updateKeyDraft(prefix: String, value: String) {
        if (prefix !in PROVIDERS) return
        _state.value = _state.value.copy(
            keyDrafts = _state.value.keyDrafts + (prefix to value),
            credentialStatus = _state.value.credentialStatus +
                (prefix to if (value.isBlank()) OnboardingCredentialStatus.Empty else OnboardingCredentialStatus.Draft),
            providerMessages = _state.value.providerMessages - prefix,
            error = null,
        )
    }

    fun saveAndTest(prefix: String) {
        if (prefix !in PROVIDERS) return
        val draft = _state.value.keyDrafts[prefix].orEmpty().trim()
        if (draft.isBlank()) {
            _state.value = _state.value.copy(
                credentialStatus = _state.value.credentialStatus + (prefix to OnboardingCredentialStatus.Invalid),
                providerMessages = _state.value.providerMessages + (prefix to "Enter an API key first."),
            )
            return
        }
        _state.value = _state.value.copy(
            credentialStatus = _state.value.credentialStatus + (prefix to OnboardingCredentialStatus.Saving),
            providerMessages = _state.value.providerMessages - prefix,
            error = null,
        )
        viewModelScope.launch {
            runCatching {
                providerKeys.set(prefix, draft)
                modelCatalogRepository.refreshProvider(prefix, force = true)
            }.onFailure { error ->
                _state.value = _state.value.copy(
                    credentialStatus = _state.value.credentialStatus + (prefix to OnboardingCredentialStatus.Invalid),
                    providerMessages = _state.value.providerMessages +
                        (prefix to (error.message ?: "Could not test this provider.")),
                )
            }
        }
    }

    fun selectDefaultModel(modelId: String) {
        if (_state.value.catalog.allModels.none { it.id == modelId }) {
            _state.value = _state.value.copy(error = "Choose an available model.")
            return
        }
        _state.value = _state.value.copy(selectedDefaultModel = modelId, error = null)
        viewModelScope.launch { userPreferences.setDefaultModel(modelId) }
    }

    fun next() {
        val current = _state.value
        _state.value = when (current.step) {
            OnboardingStep.Intro -> current.copy(step = OnboardingStep.Provider, error = null)
            OnboardingStep.Provider -> if (current.catalog.allModels.isNotEmpty()) {
                current.copy(step = OnboardingStep.Model, error = null)
            } else {
                current.copy(error = "Save and verify a provider, or choose Skip for local-only mode.")
            }
            OnboardingStep.Model -> if (current.selectedDefaultModel != null) {
                current.copy(step = OnboardingStep.Complete, error = null)
            } else {
                current.copy(error = "Choose a default chat model.")
            }
            OnboardingStep.Complete -> current
        }
    }

    fun back() {
        val previous = when (_state.value.step) {
            OnboardingStep.Intro -> OnboardingStep.Intro
            OnboardingStep.Provider -> OnboardingStep.Intro
            OnboardingStep.Model -> OnboardingStep.Provider
            OnboardingStep.Complete -> OnboardingStep.Model
        }
        _state.value = _state.value.copy(step = previous, error = null)
    }

    fun skip(onComplete: () -> Unit) = complete(onComplete)

    fun finish(onComplete: () -> Unit) = complete(onComplete)

    private fun complete(onComplete: () -> Unit) {
        viewModelScope.launch {
            firstRunGate.markComplete()
            onComplete()
        }
    }

    private companion object {
        val PROVIDERS = listOf("ollama", "anthropic")
    }
}

@Composable
fun OnboardingRoute(
    onComplete: () -> Unit,
    viewModel: OnboardingViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    OnboardingContent(
        state = state,
        onBack = viewModel::back,
        onSkip = { viewModel.skip(onComplete) },
        onNext = viewModel::next,
        onKeyDraftChanged = viewModel::updateKeyDraft,
        onSaveAndTest = viewModel::saveAndTest,
        onModelSelected = viewModel::selectDefaultModel,
        onFinish = { viewModel.finish(onComplete) },
    )
}
