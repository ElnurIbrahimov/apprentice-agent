package com.aura.ui.screens.onboarding

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import com.aura.providers.ProviderCredentialState
import com.aura.ui.settings.ProviderKeyField
import com.aura.ui.theme.AuraSpacing

import com.aura.ui.theme.AuraThemeTokens
@Composable
fun ProviderSetupStep(
    state: OnboardingUiState,
    onKeyDraftChanged: (String, String) -> Unit,
    onSaveAndTest: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(AuraSpacing.md),
    ) {
        Icon(
            imageVector = Icons.Filled.Cloud,
            contentDescription = null,
            tint = AuraThemeTokens.colors.actionPrimary,
        )
        Text(
            text = "Connect a provider",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Start with one provider. Your key is stored on this device and tested only when you tap Save & Test.",
            style = MaterialTheme.typography.bodyMedium,
            color = AuraThemeTokens.colors.textPrimary,
        )

        ProviderField(
            prefix = "ollama",
            label = "Ollama Cloud",
            helper = "Recommended · get a key at ollama.com/settings/keys",
            state = state,
            onKeyDraftChanged = onKeyDraftChanged,
            onSaveAndTest = onSaveAndTest,
        )
        ProviderField(
            prefix = "anthropic",
            label = "Anthropic (optional)",
            helper = "Add later in Settings if you prefer.",
            state = state,
            onKeyDraftChanged = onKeyDraftChanged,
            onSaveAndTest = onSaveAndTest,
        )

        val verifiedCount = state.credentialStatus.values.count {
            it == OnboardingCredentialStatus.Verified
        }
        Text(
            text = if (verifiedCount == 0) {
                "No verified providers yet"
            } else {
                "$verifiedCount provider${if (verifiedCount == 1) "" else "s"} verified"
            },
            style = MaterialTheme.typography.labelMedium,
            color = if (verifiedCount > 0) AuraThemeTokens.colors.actionPrimary
            else AuraThemeTokens.colors.textPrimary,
        )
    }
}

@Composable
private fun ProviderField(
    prefix: String,
    label: String,
    helper: String,
    state: OnboardingUiState,
    onKeyDraftChanged: (String, String) -> Unit,
    onSaveAndTest: (String) -> Unit,
) {
    val status = state.credentialStatus[prefix] ?: OnboardingCredentialStatus.Empty
    ProviderKeyField(
        label = label,
        value = state.keyDrafts[prefix].orEmpty(),
        onValueChange = { onKeyDraftChanged(prefix, it) },
        helperText = helper,
        onVerify = { onSaveAndTest(prefix) },
        verifyResult = state.providerMessages[prefix],
        verifying = status == OnboardingCredentialStatus.Saving,
        credentialState = when (status) {
            OnboardingCredentialStatus.Verified -> ProviderCredentialState.Valid
            OnboardingCredentialStatus.Invalid -> ProviderCredentialState.Invalid
            OnboardingCredentialStatus.Saving -> ProviderCredentialState.Saved
            OnboardingCredentialStatus.Empty,
            OnboardingCredentialStatus.Draft -> null
        },
    )
}
