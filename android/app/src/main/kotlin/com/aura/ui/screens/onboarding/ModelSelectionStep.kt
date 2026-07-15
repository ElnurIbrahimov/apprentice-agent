package com.aura.ui.screens.onboarding

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import com.aura.providers.ModelCatalog
import com.aura.providers.ModelDescriptor
import com.aura.providers.ProviderStatus
import com.aura.ui.components.AuraEmptyState
import com.aura.ui.components.AuraInlineStatus
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.util.modelDisplayName

@Composable
fun ModelSelectionStep(
    catalog: ModelCatalog,
    selectedModel: String?,
    error: String?,
    onModelSelected: (String) -> Unit,
    modifier: Modifier = Modifier,
    onReturnToProviders: () -> Unit = {},
) {
    if (catalog.allModels.isEmpty()) {
        AuraEmptyState(
            icon = Icons.Filled.SmartToy,
            title = "No models available",
            message = "Return to provider setup, save a valid key, and test it.",
            actionLabel = "Back to providers",
            onAction = onReturnToProviders,
            modifier = modifier.fillMaxSize(),
        )
        return
    }

    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .testTag("onboarding-model-list"),
        verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
    ) {
        item {
            Text(
                text = "Choose your default model",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "This model will be selected when a new chat starts. You can change it anytime.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = AuraSpacing.xs, bottom = AuraSpacing.sm),
            )
            error?.let {
                AuraInlineStatus(
                    text = it,
                    tone = com.aura.ui.components.InlineStatusTone.Error,
                    modifier = Modifier.padding(bottom = AuraSpacing.sm),
                )
            }
        }

        for ((prefix, provider) in catalog.providers) {
            if (provider.models.isEmpty()) continue
            item(key = "header-$prefix") {
                ProviderHeader(
                    prefix = prefix,
                    count = provider.models.size,
                    status = provider.status,
                )
            }
            items(provider.models, key = ModelDescriptor::id) { model ->
                ModelChoiceRow(
                    model = model,
                    selected = model.id == selectedModel,
                    onClick = { onModelSelected(model.id) },
                )
            }
        }
    }
}

@Composable
private fun ProviderHeader(prefix: String, count: Int, status: ProviderStatus) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = AuraSpacing.xs),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = providerName(prefix),
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.weight(1f),
        )
        Text(
            text = "$count · ${statusLabel(status)}",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun ModelChoiceRow(
    model: ModelDescriptor,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        color = if (selected) MaterialTheme.colorScheme.primaryContainer
        else MaterialTheme.colorScheme.surfaceContainerLow,
        shape = RoundedCornerShape(AuraDimensions.controlRadius),
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .testTag("onboarding-model-${model.id}"),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = AuraSpacing.sm, vertical = AuraSpacing.xs),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            RadioButton(selected = selected, onClick = onClick)
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = modelDisplayName(model.id),
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Medium,
                )
                Text(
                    text = model.id,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                )
            }
            if (selected) {
                Text(
                    text = "Default",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}

private fun providerName(prefix: String): String = com.aura.providers.providerLabel(prefix)

private fun statusLabel(status: ProviderStatus): String = when (status) {
    ProviderStatus.Ready -> "Ready"
    ProviderStatus.Loading -> "Loading"
    ProviderStatus.Unauthorized -> "Invalid key"
    ProviderStatus.RateLimit -> "Rate limited"
    ProviderStatus.Network -> "Offline"
    ProviderStatus.Timeout -> "Timed out"
    ProviderStatus.Malformed -> "Unavailable"
    ProviderStatus.Empty -> "No models"
    ProviderStatus.NotConfigured -> "Not configured"
}
