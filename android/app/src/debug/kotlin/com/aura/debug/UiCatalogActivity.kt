package com.aura.debug

import android.os.Bundle
import android.graphics.Color
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Text
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import com.aura.ui.components.AuraCard
import com.aura.ui.components.AuraEmptyState
import com.aura.ui.components.AuraErrorState
import com.aura.ui.components.AuraInlineStatus
import com.aura.ui.components.AuraLoadingState
import com.aura.ui.components.InlineStatusTone
import com.aura.ui.components.ModelPickerSheet
import com.aura.ui.components.ResponsiveContainer
import com.aura.providers.ModelCatalog
import com.aura.providers.ModelDescriptor
import com.aura.providers.ProviderModelList
import com.aura.providers.ProviderStatus
import com.aura.ui.screens.onboarding.OnboardingContent
import com.aura.ui.screens.onboarding.OnboardingCredentialStatus
import com.aura.ui.screens.onboarding.OnboardingStep
import com.aura.ui.screens.onboarding.OnboardingUiState
import com.aura.ui.screens.home.HomeContent
import com.aura.ui.viewmodel.HomeLoadState
import com.aura.ui.viewmodel.HomeUiState
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraTheme
import com.aura.ui.theme.AuraThemeTokens

class UiCatalogActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val surface = intent.getStringExtra(EXTRA_SURFACE)
            ?.takeIf(CatalogSurfaces::contains) ?: "home"
        val state = intent.getStringExtra(EXTRA_STATE)
            ?.takeIf(CatalogStates::contains) ?: "content"
        val theme = intent.getStringExtra(EXTRA_THEME)
            ?.takeIf(CatalogThemes::contains) ?: "dark"
        val barStyle = if (theme == "dark") {
            SystemBarStyle.dark(Color.TRANSPARENT)
        } else {
            SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT)
        }
        enableEdgeToEdge(statusBarStyle = barStyle, navigationBarStyle = barStyle)
        setContent {
            AuraTheme(themeMode = theme) {
                UiCatalog(surface = surface, state = state)
            }
        }
    }

    companion object {
        const val EXTRA_SURFACE: kotlin.String = "surface"
        const val EXTRA_STATE: kotlin.String = "state"
        const val EXTRA_THEME: kotlin.String = "theme"
    }
}

val CatalogSurfaces = setOf(
    "startup", "onboarding", "home", "chat", "model-picker", "settings",
    "memory", "history", "tasks", "reminders", "hands", "tools",
    "proactive", "graph", "profile", "identity", "diagnostics", "voice",
    "quick-ask", "widget-config",
)

val CatalogStates = setOf(
    "content", "loading", "empty", "error", "no-provider", "selected", "partial-error",
)

val CatalogThemes = setOf("light", "dark")

@Composable
private fun UiCatalog(surface: kotlin.String, state: kotlin.String) {
    val colors = AuraThemeTokens.colors
    if (surface == "onboarding") {
        val descriptor = ModelDescriptor("ollama:model-a", "model-a", "ollama")
        val catalog = ModelCatalog(
            providers = mapOf(
                "ollama" to ProviderModelList(
                    providerPrefix = "ollama",
                    status = ProviderStatus.Ready,
                    models = listOf(descriptor),
                ),
            ),
            allModels = listOf(descriptor),
        )
        val onboardingState = when (state) {
            "selected" -> OnboardingUiState(
                step = OnboardingStep.Complete,
                credentialStatus = mapOf("ollama" to OnboardingCredentialStatus.Verified),
                catalog = catalog,
                selectedDefaultModel = descriptor.id,
            )
            "empty", "no-provider" -> OnboardingUiState(step = OnboardingStep.Provider)
            "loading" -> OnboardingUiState(
                step = OnboardingStep.Provider,
                keyDrafts = mapOf("ollama" to "test-key"),
                credentialStatus = mapOf("ollama" to OnboardingCredentialStatus.Saving),
            )
            else -> OnboardingUiState(
                step = OnboardingStep.Provider,
                keyDrafts = mapOf("ollama" to "test-key"),
                credentialStatus = mapOf("ollama" to OnboardingCredentialStatus.Verified),
                providerMessages = mapOf("ollama" to "✓ Verified · 1 model"),
                catalog = catalog,
            )
        }
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background,
            contentColor = MaterialTheme.colorScheme.onBackground,
        ) {
            OnboardingContent(
                state = onboardingState,
                onBack = {},
                onSkip = {},
                onNext = {},
                onKeyDraftChanged = { _, _ -> },
                onSaveAndTest = {},
                onModelSelected = {},
                onFinish = {},
            )
        }
        return
    }
    if (surface == "home") {
        val homeState = when (state) {
            "loading" -> HomeUiState(loadState = HomeLoadState.Loading)
            "empty", "no-provider" -> HomeUiState(loadState = HomeLoadState.Empty, toolsCount = 38)
            "error" -> HomeUiState(
                loadState = HomeLoadState.Error(
                    "Home data is unavailable. Check permissions and try again.",
                    hasPartialContent = false,
                ),
            )
            "partial-error" -> HomeUiState(
                loadState = HomeLoadState.Error("Calendar is unavailable", hasPartialContent = true),
                pendingTasks = listOf("Ship the next Aura build"),
                toolsCount = 38,
            )
            else -> HomeUiState(
                loadState = HomeLoadState.Content,
                today = listOf("14:00 · Product review"),
                pendingTasks = listOf("Ship the next Aura build", "Review release notes"),
                upcomingReminders = listOf("16:30 · Call Alex"),
                handsCount = 3,
                toolsCount = 38,
                proactiveCount = 2,
            )
        }
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background,
            contentColor = MaterialTheme.colorScheme.onBackground,
        ) {
            Box(Modifier.fillMaxSize().safeDrawingPadding()) {
                HomeContent(
                    state = homeState,
                    greeting = "Good afternoon, Elnur",
                    dateLabel = "Tuesday, July 14",
                )
            }
        }
        return
    }
    if (surface == "model-picker") {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(colors.background)
                .testTag("ui-catalog-root"),
        ) {
            ModelPickerSheet(
                currentModel = if (state == "selected") "test:primary-model" else "",
                models = if (state in setOf("content", "selected", "partial-error")) {
                    listOf("test:primary-model", "test:secondary-model", "other:long-model-name-for-layout-testing")
                } else {
                    emptyList()
                },
                isLoading = state == "loading",
                errorMessage = when (state) {
                    "error", "partial-error" -> "Test provider could not refresh."
                    "no-provider" -> "Connect a provider in Settings."
                    else -> null
                },
                onPick = {},
                onDismiss = {},
            )
        }
        return
    }
    ResponsiveContainer(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
            .safeDrawingPadding()
            .testTag("ui-catalog-root"),
    ) {
        Column(
            modifier = Modifier.padding(vertical = AuraSpacing.lg),
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.md),
        ) {
            Text(
                text = surface.replace('-', ' ').replaceFirstChar { it.uppercase() },
                color = colors.textPrimary,
                modifier = Modifier.testTag("ui-catalog-surface"),
            )
            when (state) {
                "loading" -> AuraLoadingState(label = "Loading $surface…")
                "empty", "no-provider" -> AuraEmptyState(
                    title = if (state == "no-provider") "Connect a provider" else "Nothing here yet",
                    message = if (state == "no-provider") {
                        "Add and verify a provider key to choose a model."
                    } else {
                        "Create the first item to populate this surface."
                    },
                    actionLabel = "Primary action",
                    onAction = {},
                )
                "error" -> AuraErrorState(
                    title = "Could not load $surface",
                    message = "A deterministic test failure occurred.",
                    onRetry = {},
                )
                else -> {
                    AuraInlineStatus("$state state", tone = InlineStatusTone.Success)
                    repeat(3) { index ->
                        AuraCard(elevated = index == 0) {
                            Text("Representative $surface item ${index + 1}", color = colors.textPrimary)
                            Text("Secondary information and action hierarchy", color = colors.textSecondary)
                        }
                    }
                }
            }
        }
    }
}
