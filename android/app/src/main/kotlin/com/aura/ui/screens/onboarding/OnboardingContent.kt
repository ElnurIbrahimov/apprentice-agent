package com.aura.ui.screens.onboarding

import androidx.compose.animation.AnimatedContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size

import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Hub
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.aura.ui.components.AuraIconButton
import com.aura.ui.components.AuraInlineStatus
import com.aura.ui.components.AuraPrimaryButton
import com.aura.ui.components.InlineStatusTone
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.util.modelDisplayName

import com.aura.ui.theme.AuraThemeTokens
@Composable
fun OnboardingContent(
    state: OnboardingUiState,
    onBack: () -> Unit,
    onSkip: () -> Unit,
    onNext: () -> Unit,
    onKeyDraftChanged: (String, String) -> Unit,
    onSaveAndTest: (String) -> Unit,
    onModelSelected: (String) -> Unit,
    onFinish: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .imePadding(),
    ) {
        OnboardingHeader(
            step = state.step,
            onBack = onBack,
            onSkip = onSkip,
        )
        LinearProgressIndicator(
            progress = { (state.step.ordinal + 1) / OnboardingStep.entries.size.toFloat() },
            modifier = Modifier
                .fillMaxWidth()
                .testTag("onboarding-progress"),
        )

        AnimatedContent(
            targetState = state.step,
            label = "onboarding-step",
            modifier = Modifier.weight(1f),
        ) { step ->
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = AuraSpacing.md, vertical = AuraSpacing.sm),
            ) {
                when (step) {
                    OnboardingStep.Intro -> IntroStep(
                        modifier = Modifier.verticalScroll(rememberScrollState()),
                    )
                    OnboardingStep.Provider -> ProviderSetupStep(
                        state = state,
                        onKeyDraftChanged = onKeyDraftChanged,
                        onSaveAndTest = onSaveAndTest,
                        modifier = Modifier.verticalScroll(rememberScrollState()),
                    )
                    OnboardingStep.Model -> ModelSelectionStep(
                        catalog = state.catalog,
                        selectedModel = state.selectedDefaultModel,
                        error = state.error,
                        onModelSelected = onModelSelected,
                        onReturnToProviders = onBack,
                    )
                    OnboardingStep.Complete -> CompleteStep(
                        state = state,
                        modifier = Modifier.verticalScroll(rememberScrollState()),
                    )
                }
            }
        }

        if (state.error != null && state.step != OnboardingStep.Model) {
            AuraInlineStatus(
                text = state.error,
                tone = InlineStatusTone.Error,
                modifier = Modifier.padding(horizontal = AuraSpacing.md, vertical = AuraSpacing.xs),
            )
        }

        OnboardingBottomAction(
            state = state,
            onNext = onNext,
            onFinish = onFinish,
        )
    }
}

@Composable
private fun OnboardingHeader(
    step: OnboardingStep,
    onBack: () -> Unit,
    onSkip: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .padding(horizontal = AuraSpacing.xs),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (step != OnboardingStep.Intro) {
            AuraIconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
            }
        } else {
            Spacer(Modifier.size(48.dp))
        }
        Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Set up Aura",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Step ${step.ordinal + 1} of ${OnboardingStep.entries.size}",
                style = MaterialTheme.typography.labelSmall,
                color = AuraThemeTokens.colors.textPrimary,
            )
        }
        if (step != OnboardingStep.Complete) {
            TextButton(onClick = onSkip) { Text("Skip") }
        } else {
            Spacer(Modifier.size(48.dp))
        }
    }
}

@Composable
private fun OnboardingBottomAction(
    state: OnboardingUiState,
    onNext: () -> Unit,
    onFinish: () -> Unit,
) {
    val enabled = when (state.step) {
        OnboardingStep.Intro -> true
        OnboardingStep.Provider -> state.catalog.allModels.isNotEmpty()
        OnboardingStep.Model -> state.selectedDefaultModel != null
        OnboardingStep.Complete -> true
    }
    val label = when (state.step) {
        OnboardingStep.Intro -> "Get started"
        OnboardingStep.Provider -> "Choose a model"
        OnboardingStep.Model -> "Review setup"
        OnboardingStep.Complete -> "Start chatting"
    }
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(AuraSpacing.md)
            .testTag("onboarding-bottom-actions"),
    ) {
        AuraPrimaryButton(
            onClick = if (state.step == OnboardingStep.Complete) onFinish else onNext,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(label) }
    }
}

@Composable
private fun IntroStep(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(AuraSpacing.md),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            imageVector = Icons.Filled.AutoAwesome,
            contentDescription = null,
            tint = AuraThemeTokens.colors.actionPrimary,
            modifier = Modifier.size(56.dp),
        )
        Text(
            text = "Your assistant, set up your way",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
        Text(
            text = "Connect one provider, verify it, and choose the model Aura should use for new chats.",
            style = MaterialTheme.typography.bodyLarge,
            color = AuraThemeTokens.colors.textPrimary,
            textAlign = TextAlign.Center,
        )
        FeatureLine(Icons.Filled.Memory, "Remembers what matters")
        FeatureLine(Icons.Filled.TaskAlt, "Handles tasks and reminders")
        FeatureLine(Icons.Filled.Hub, "Uses the providers you choose")
        FeatureLine(Icons.Filled.Lock, "Keeps credentials on this device")
    }
}

@Composable
private fun FeatureLine(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = AuraThemeTokens.colors.actionPrimary)
        Text(label, style = MaterialTheme.typography.bodyLarge)
    }
}

@Composable
private fun CompleteStep(state: OnboardingUiState, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(AuraSpacing.md),
    ) {
        Icon(
            imageVector = Icons.Filled.CheckCircle,
            contentDescription = null,
            tint = AuraThemeTokens.colors.actionPrimary,
            modifier = Modifier.size(56.dp),
        )
        Text(
            text = "Aura is ready",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Default chat model",
            style = MaterialTheme.typography.labelMedium,
            color = AuraThemeTokens.colors.textPrimary,
        )
        Text(
            text = state.selectedDefaultModel?.let(::modelDisplayName) ?: "Local-only mode",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
            modifier = Modifier.testTag("onboarding-selected-model"),
        )
        val verified = state.credentialStatus.values.count {
            it == OnboardingCredentialStatus.Verified
        }
        Text(
            text = "$verified verified provider${if (verified == 1) "" else "s"}",
            style = MaterialTheme.typography.bodyMedium,
            color = AuraThemeTokens.colors.textPrimary,
        )
    }
}
