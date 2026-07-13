package com.aura.debug

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Text
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
