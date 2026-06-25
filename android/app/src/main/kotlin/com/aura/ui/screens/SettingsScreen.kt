package com.aura.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.ui.settings.ProviderKeyField
import com.aura.ui.settings.SettingsViewModel

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun SettingsScreen(viewModel: SettingsViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = "Settings",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Text(
            text = "Connect model providers. Keys are stored locally and never leave the device.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(modifier = Modifier.height(8.dp))

        // Status banner
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = if (state.configuredProviders.isEmpty()) "No providers configured yet." else "Configured providers:",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Medium,
                )
                if (state.configuredProviders.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(4.dp))
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        state.configuredProviders.forEach { name ->
                            AssistChip(
                                onClick = {},
                                label = { Text(name) },
                                colors = AssistChipDefaults.assistChipColors(
                                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                                    labelColor = MaterialTheme.colorScheme.onPrimaryContainer,
                                ),
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "API Keys",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Restart the app after adding a key so the provider can pick it up.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )

        ProviderKeyField(
            label = "Ollama Cloud",
            value = state.ollamaKey,
            onValueChange = viewModel::saveOllamaKey,
            helperText = "Get a key at ollama.com/settings/keys",
        )
        ProviderKeyField(
            label = "Anthropic",
            value = state.anthropicKey,
            onValueChange = viewModel::saveAnthropicKey,
            helperText = "Get a key at console.anthropic.com/settings/keys",
        )
        ProviderKeyField(
            label = "OpenAI",
            value = state.openaiKey,
            onValueChange = viewModel::saveOpenaiKey,
            helperText = "Get a key at platform.openai.com/api-keys",
        )
        ProviderKeyField(
            label = "DeepSeek",
            value = state.deepseekKey,
            onValueChange = viewModel::saveDeepseekKey,
            helperText = "Get a key at platform.deepseek.com/api_keys",
        )

        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "Default model",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Default: ${state.defaultModel}",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
        )
        Spacer(modifier = Modifier.height(4.dp))
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            listOf(
                "ollama:deepseek-v3.2:cloud" to "DeepSeek V3.2 (cheap, fast)",
                "ollama:kimi-k2.6:cloud" to "Kimi K2.6 (tool calls)",
                "anthropic:claude-sonnet-4-5" to "Claude Sonnet 4.5",
                "ollama:minimax-m2.7:cloud" to "MiniMax M2.7 (code)",
            ).forEach { (id, label) ->
                AssistChip(
                    onClick = { viewModel.setDefaultModel(id) },
                    label = { Text(label) },
                    colors = AssistChipDefaults.assistChipColors(
                        containerColor = if (state.defaultModel == id) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant,
                        labelColor = if (state.defaultModel == id) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
                    ),
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Aura Android v0.1.0 — Day 3 build",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.4f),
        )
    }
}
