package com.aura.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.ui.settings.BackupViewModel
import com.aura.ui.settings.ProviderKeyField
import com.aura.ui.settings.SettingsViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel = hiltViewModel(),
    backupViewModel: BackupViewModel = hiltViewModel(),
) {
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
            text = "API keys are stored locally and never leave your device. Changes take effect immediately.",
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
        ProviderKeyField(
            label = "Groq",
            value = state.groqKey,
            onValueChange = viewModel::saveGroqKey,
            helperText = "Get a key at console.groq.com/keys",
        )
        ProviderKeyField(
            label = "OpenRouter",
            value = state.openrouterKey,
            onValueChange = viewModel::saveOpenrouterKey,
            helperText = "Get key at openrouter.ai/keys",
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
                "ollama:deepseek-v4-pro:cloud" to "DeepSeek V4 Pro (fast, cheap)",
                "ollama:kimi-k2.7-code:cloud" to "Kimi K2.7 Code (tool use)",
                "anthropic:claude-sonnet-4-5" to "Claude Sonnet 4.5",
                "ollama:minimax-m2.7:cloud" to "MiniMax M2.7 (code)",
                "ollama:gemma4:31b:cloud" to "Gemma 4 31B",
                "ollama:qwen3.5:cloud" to "Qwen 3.5",
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

        // ── Backup & restore ──
        val backupState by backupViewModel.state.collectAsState()
        val context = androidx.compose.ui.platform.LocalContext.current
        val importLauncher = androidx.activity.compose.rememberLauncherForActivityResult(
            contract = androidx.activity.result.contract.ActivityResultContracts.OpenDocument(),
        ) { uri: android.net.Uri? ->
            if (uri != null) backupViewModel.stageImport(uri)
        }
        val coroutineScope = androidx.compose.runtime.rememberCoroutineScope()

        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "Backup & restore",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Export everything (memories, conversations, knowledge graph, " +
                "hands, tasks, user profile) to a JSON file. Imports add to " +
                "what's already here — turn on 'purge first' to start fresh.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(Modifier.height(8.dp))

        OutlinedButton(
            onClick = {
                coroutineScope.launch {
                    val file = backupViewModel.prepareExportFile()
                    val uri = androidx.core.content.FileProvider.getUriForFile(
                        context,
                        "${context.packageName}.fileprovider",
                        file,
                    )
                    val share = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                        type = "application/json"
                        putExtra(android.content.Intent.EXTRA_STREAM, uri)
                        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    context.startActivity(
                        android.content.Intent.createChooser(share, "Share Aura backup")
                            .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                    )
                }
            },
            enabled = !backupState.exportInFlight,
        ) {
            Text(if (backupState.exportInFlight) "Exporting…" else "Export to JSON")
        }

        Spacer(Modifier.height(8.dp))

        OutlinedButton(
            onClick = {
                importLauncher.launch(arrayOf("application/json", "*/*"))
            },
            enabled = !backupState.importInFlight,
        ) {
            Text(if (backupState.importInFlight) "Restoring…" else "Restore from JSON")
        }

        backupState.lastResult?.let { result ->
            Spacer(Modifier.height(8.dp))
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = androidx.compose.foundation.shape.RoundedCornerShape(8.dp),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = result,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.weight(1f),
                    )
                    TextButton(onClick = { backupViewModel.clearResult() }) {
                        Text("Dismiss")
                    }
                }
            }
        }

        if (backupState.showImportConfirm) {
            AlertDialog(
                onDismissRequest = { backupViewModel.cancelImport() },
                title = { Text("Restore from backup?") },
                text = {
                    Text(
                        "This will add the rows from the backup file to " +
                            "your existing data. Existing rows with the " +
                            "same id are replaced; new rows are added. " +
                            "Embeddings are NOT included — after restoring, " +
                            "go to the Memory tab and tap 'Rebuild embeddings' " +
                            "to re-embed everything in one pass.",
                    )
                },
                confirmButton = {
                    TextButton(onClick = { backupViewModel.confirmImport(purgeFirst = false) }) {
                        Text("Add to existing")
                    }
                },
                dismissButton = {
                    TextButton(onClick = { backupViewModel.cancelImport() }) {
                        Text("Cancel")
                    }
                },
            )
        }

        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Aura Android v" + com.aura.BuildConfig.VERSION_NAME,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.4f),
        )
    }
}
