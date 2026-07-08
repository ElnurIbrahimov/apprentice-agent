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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.rememberTimePickerState
import androidx.compose.material3.TimePicker
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.ui.settings.BackupViewModel
import com.aura.ui.settings.ProviderKeyField
import com.aura.ui.settings.SettingsViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalLayoutApi::class, androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateProfile: () -> Unit,
    onOpenIdentityEditor: () -> Unit = {},
    viewModel: SettingsViewModel = hiltViewModel(),
    backupViewModel: BackupViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp),
    ) {
        // ── Header ──────────────────────────────────────────────────────
        Column(modifier = Modifier.padding(top = 16.dp, bottom = 12.dp)) {
            Text(
                text = "Settings",
                style = MaterialTheme.typography.displaySmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Connect providers, manage memory, customize Aura",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
        }
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
            onVerify = { viewModel.verifyKey("ollama") },
            verifyResult = state.verifyResults["ollama"],
            verifying = state.verifying == "ollama",
        )
        ProviderKeyField(
            label = "Anthropic",
            value = state.anthropicKey,
            onValueChange = viewModel::saveAnthropicKey,
            helperText = "Get a key at console.anthropic.com/settings/keys",
            onVerify = { viewModel.verifyKey("anthropic") },
            verifyResult = state.verifyResults["anthropic"],
            verifying = state.verifying == "anthropic",
        )
        ProviderKeyField(
            label = "OpenAI",
            value = state.openaiKey,
            onValueChange = viewModel::saveOpenaiKey,
            helperText = "Get a key at platform.openai.com/api-keys",
            onVerify = { viewModel.verifyKey("openai") },
            verifyResult = state.verifyResults["openai"],
            verifying = state.verifying == "openai",
        )
        ProviderKeyField(
            label = "DeepSeek",
            value = state.deepseekKey,
            onValueChange = viewModel::saveDeepseekKey,
            helperText = "Get a key at platform.deepseek.com/api_keys",
            onVerify = { viewModel.verifyKey("deepseek") },
            verifyResult = state.verifyResults["deepseek"],
            verifying = state.verifying == "deepseek",
        )
        ProviderKeyField(
            label = "Groq",
            value = state.groqKey,
            onValueChange = viewModel::saveGroqKey,
            helperText = "Get a key at console.groq.com/keys",
            onVerify = { viewModel.verifyKey("groq") },
            verifyResult = state.verifyResults["groq"],
            verifying = state.verifying == "groq",
        )
        ProviderKeyField(
            label = "OpenRouter",
            value = state.openrouterKey,
            onValueChange = viewModel::saveOpenrouterKey,
            helperText = "Get key at openrouter.ai/keys",
            onVerify = { viewModel.verifyKey("openrouter") },
            verifyResult = state.verifyResults["openrouter"],
            verifying = state.verifying == "openrouter",
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

        Text(
            text = "Embedding model",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Used to turn memories into vectors locally. Current: ${state.embeddingModel}",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
        )
        Spacer(modifier = Modifier.height(4.dp))
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            listOf(
                "nomic-embed-text" to "Nomic Embed (default)",
                "mxbai-embed-large" to "mixed-bread large",
                "all-minilm" to "all-MiniLM",
                "snowflake-arctic-embed" to "Snowflake Arctic",
            ).forEach { (id, label) ->
                AssistChip(
                    onClick = { viewModel.setEmbeddingModel(id) },
                    label = { Text(label) },
                    colors = AssistChipDefaults.assistChipColors(
                        containerColor = if (state.embeddingModel == id) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant,
                        labelColor = if (state.embeddingModel == id) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
                    ),
                )
            }
        }

        // ── Appearance: theme mode ──
        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "Appearance",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Choose light, dark, or follow the system theme.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(modifier = Modifier.height(4.dp))
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            listOf(
                "system" to "System",
                "light" to "Light",
                "dark" to "Dark",
            ).forEach { (id, label) ->
                AssistChip(
                    onClick = { viewModel.setThemeMode(id) },
                    label = { Text(label) },
                    colors = AssistChipDefaults.assistChipColors(
                        containerColor = if (state.themeMode == id) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant,
                        labelColor = if (state.themeMode == id) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
                    ),
                )
            }
        }

        // ── Persona: custom identity + specialist overrides ──
        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "Persona",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Custom system prompt prepended to Aura's built-in identity. Use this to set language, tone, or persona. Leave blank to use the default.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(modifier = Modifier.height(8.dp))

        var identityText by remember(state.customIdentity) { mutableStateOf(state.customIdentity) }
        OutlinedTextField(
            value = identityText,
            onValueChange = { identityText = it },
            label = { Text("Custom identity prompt") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3,
            maxLines = 8,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
        ) {
            TextButton(onClick = { identityText = ""; viewModel.setCustomIdentity("") }) {
                Text("Reset")
            }
            Spacer(modifier = Modifier.width(8.dp))
            Button(onClick = { viewModel.setCustomIdentity(identityText.trim()) }) {
                Text("Save")
            }
        }

        // ── Specialist prompt overrides ──
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "Specialist prompts",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Override the built-in system prompt for each specialist. Tap a specialist to edit its prompt.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(modifier = Modifier.height(8.dp))

        // Parse current overrides
        val overridesMap = remember(state.specialistOverrides) {
            try { Json.decodeFromString<Map<String, String>>(state.specialistOverrides) }
            catch (e: Exception) { emptyMap() }
        }
        var editingSpecialist by remember { mutableStateOf<com.aura.agent.Specialist?>(null) }

        for (specialist in com.aura.agent.Specialist.ALL) {
            val hasOverride = overridesMap.containsKey(specialist.name) && !overridesMap[specialist.name].isNullOrBlank()
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(text = "${specialist.icon} ${specialist.name.replaceFirstChar { it.uppercase() }}", style = MaterialTheme.typography.bodyLarge)
                Spacer(modifier = Modifier.weight(1f))
                if (hasOverride) Text("✏️", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
                TextButton(onClick = { editingSpecialist = specialist }) { Text(if (hasOverride) "Edit" else "Default") }
            }
        }

        // Edit dialog
        editingSpecialist?.let { specialist ->
            var promptText by remember(specialist) {
                mutableStateOf(overridesMap[specialist.name] ?: specialist.systemPrompt)
            }
            AlertDialog(
                onDismissRequest = { editingSpecialist = null },
                title = { Text("${specialist.icon} ${specialist.name.replaceFirstChar { it.uppercase() }}") },
                text = {
                    OutlinedTextField(
                        value = promptText,
                        onValueChange = { promptText = it },
                        label = { Text("System prompt") },
                        modifier = Modifier.fillMaxWidth(),
                        minLines = 4,
                        maxLines = 12,
                    )
                },
                confirmButton = {
                    Button(onClick = {
                        val updated = overridesMap.toMutableMap().apply { put(specialist.name, promptText.trim()) }
                        viewModel.setSpecialistOverrides(Json.encodeToString(MapSerializer(String.serializer(), String.serializer()), updated))
                        editingSpecialist = null
                    }) { Text("Save") }
                },
                dismissButton = {
                    Row {
                        TextButton(onClick = {
                            val updated = overridesMap.toMutableMap().apply { remove(specialist.name) }
                            viewModel.setSpecialistOverrides(Json.encodeToString(MapSerializer(String.serializer(), String.serializer()), updated))
                            editingSpecialist = null
                        }) { Text("Reset to default") }
                        Spacer(modifier = Modifier.width(8.dp))
                        TextButton(onClick = { editingSpecialist = null }) { Text("Cancel") }
                    }
                },
            )
        }

        // ── Privacy: biometric app lock + proactive worker toggles ──
        Spacer(modifier = Modifier.height(8.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "Privacy",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Require biometric authentication to open Aura. Toggle the proactive workers off if you don't want the 7am brief or the calendar monitor.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(modifier = Modifier.height(8.dp))

        // App lock row.
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(text = "App lock", style = MaterialTheme.typography.bodyLarge)
                Text(
                    text = if (state.appLockEnabled)
                        "Enabled — biometric required to open Aura"
                    else
                        "Off — Aura opens straight to chat",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                )
            }
            Switch(
                checked = state.appLockEnabled,
                onCheckedChange = { viewModel.setAppLockEnabled(it) },
            )
        }

        // Profile row.
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(text = "Profile", style = MaterialTheme.typography.bodyLarge)
                Text(
                    text = "Name, traits, and facts Aura has learned",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                )
            }
            TextButton(onClick = onNavigateProfile) { Text("Edit") }
        }

        // Proactive worker toggles. Each is opt-out with a true
        // default — a fresh install gets both workers. The actual
        // worker state converges on the next app launch via
        // ProactiveBootstrap; the Settings VM only persists the
        // choice.
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(text = "Morning brief", style = MaterialTheme.typography.bodyLarge)
                Text(
                    text = if (state.morningBriefEnabled)
                        "On — %02d:00 daily summary".format(state.morningBriefHour)
                    else
                        "Off — no morning notification",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                )
            }
            Switch(
                checked = state.morningBriefEnabled,
                onCheckedChange = { viewModel.setMorningBriefEnabled(it) },
            )
        }
        if (state.morningBriefEnabled) {
            Spacer(modifier = Modifier.height(4.dp))
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Brief at:", style = MaterialTheme.typography.bodySmall)
                var showBriefTimePicker by remember { mutableStateOf(false) }
                OutlinedButton(onClick = { showBriefTimePicker = true }) {
                    Text("%02d:00".format(state.morningBriefHour))
                }
                if (showBriefTimePicker) {
                    val tpState = rememberTimePickerState(
                        initialHour = state.morningBriefHour,
                        initialMinute = 0,
                    )
                    AlertDialog(
                        onDismissRequest = { showBriefTimePicker = false },
                        title = { Text("Morning brief time") },
                        text = { TimePicker(state = tpState) },
                        confirmButton = {
                            TextButton(onClick = {
                                viewModel.setMorningBriefHour(tpState.hour)
                                showBriefTimePicker = false
                            }) { Text("Set") }
                        },
                        dismissButton = {
                            TextButton(onClick = { showBriefTimePicker = false }) { Text("Cancel") }
                        },
                    )
                }
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(text = "Calendar monitor", style = MaterialTheme.typography.bodyLarge)
                Text(
                    text = if (state.calendarMonitorEnabled)
                        "On — runs in background, shows notification"
                    else
                        "Off — stops the persistent foreground service",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                )
            }
            Switch(
                checked = state.calendarMonitorEnabled,
                onCheckedChange = { viewModel.setCalendarMonitorEnabled(it) },
            )
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

        Spacer(modifier = Modifier.height(32.dp))

        // ── Footer ─────────────────────────────────────────────────────
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp, top = 8.dp),
        ) {
            Surface(
                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                shape = androidx.compose.foundation.shape.CircleShape,
            ) {
                Text(
                    text = "✦",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(12.dp),
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Aura",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            )
            Text(
                text = "v" + com.aura.BuildConfig.VERSION_NAME,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.4f),
            )
        }
    }
}